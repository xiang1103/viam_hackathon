# viam_hackathon

Python script that drives a Viam-managed UFactory arm and gripper (pick-and-place) from this laptop.
The same code in `move_arm.py` also runs as a Viam module service in the Viam console.

## Running

- Run the script with the project venv: `.venv/bin/python move_arm.py`. Don't use the conda Python (`/opt/anaconda3/...`), because `viam-sdk` is installed only in `.venv`.
- Running the script **moves the physical arm**. Never run it unless the user explicitly asks.
  - For checks that don't touch the robot, parse or import the file instead, for example by constructing `MyGenericService("t")` without calling `main()`.
- Dependencies: `viam-sdk`, `python-dotenv`, `typing_extensions` (installed in `.venv`).

## Connection

- `connect()` reads `API_KEY`, `API_KEY_ID`, and `ROBOT_ADDRESS` from `.env`, which `load_dotenv()` loads.
- `ROBOT_ADDRESS` must be the machine's **cloud address** (`<machine>-main.<id>.viam.cloud`), copied from the Viam console under CONNECT → Code sample.
  - Don't use a LAN IP. `192.168.1.x` is the arm controller, not viam-server, and it isn't reachable from this laptop. Using it fails with `Unable to establish a connection`.
- Never hardcode or print the API key. Never commit `.env`. When you add a new setting, add it to `.env.example` too.

## Code structure (`move_arm.py`)

- `MyGenericService` serves two purposes:
  - **Module mode:** viam-server calls `validate_config` and `new()`, which inject the gripper and motion service as dependencies.
  - **Local mode:** `main()` constructs the service directly and wires `gripper_name`, `gripper` (`Gripper.from_robot`), and `motion` (`MotionClient.from_robot(robot, "builtin")`) by hand.
  - When you add a new dependency, wire it in **both** `new()` and `main()`.
- Don't rename the class or change its `MODEL` / `ModelFamily` line. The Viam platform uses them to identify the module.
- Resource names: the gripper is `"gripper"` (`DEFAULT_GRIPPER_NAME`), and the motion service is `"builtin"`. `main()` prints `robot.resource_names`, so check new names against that output.
- New behaviours go in as a new `action` branch in `do_command`. `main()` picks the action to run. The current actions are `static-cycle`, `go-to-pick`, and `go-to-place`.

## Motion rules

- Move the arm through `self.motion.move(component_name=gripper_name, destination=PoseInFrame("world", pose))`, which is `move_gripper_to`. Don't command arm joints directly.
- Poses are the gripper frame origin in the `world` frame, in mm. `o_z=-1` points the gripper down.
- Always approach a pose from above (`offset_pose(pose, APPROACH_MM)`) before descending onto it.
- After grasping, lift by `LIFT_MM` before any lateral move.
- Gripper (UFactory): `open()` and `grab()` block until the jaws finish moving, and `grab()` returns whether something is held.
  - Don't poll `is_moving` after these calls. It is always false once the call returns.
- Before changing `PICK_POSE` / `PLACE_POSE` or offsets, confirm the new values are within the arm's reachable, collision-free workspace. Test with `go-to-pick` before `static-cycle`.

## Sorting package (`recycle_sorter/`)

Multi-object, multi-class sorter that sits alongside `move_arm.py` and follows the motion rules above.
It assesses everything in the unsorted zone, creates one pile per class, then picks and places. See `README.md`.

- `move_arm.py` is the hardware-proven reference. When a number here disagrees with it, `move_arm.py` wins.
  - Derived from it: `table_top: 0`, `gripper.tcp_offset: 70`, `bounds.frame_z_min: 60`, `pick.approach: 150`, `pick.grasp_speed: 8`, the `survey` pose (= `WATCH_POSE`), camera `cam`.
- Perception is switchable (`machine.yaml` → `perception`). `depth` (default, for color): OpenCV finds anything above the table, HSV names its color. `viam`: Viam vision services, one per class (`vision.segmenters`), intended for later trained detectors.
- Layout is dynamic (`unsorted_zone: auto`, `sorted_areas: auto` in `workspace.yaml`): `policy/zones.py` draws the zone around what the first look finds and generates sorted areas in the free table beyond it. Grasp points come from each item's real footprint (`perception/segment.py: choose_grasp`).
- `--mode brand` (cans/bottles): `perception: yolo` (YOLOE open-vocabulary, `perception/yolo.py`, weights in git-ignored `models/`) draws each item's box; depth inside the box gives 3D; the crop is read by Claude (`classify/brand_claude.py`, needs `ANTHROPIC_API_KEY`) into the `categories` of `config/sort.yaml`. YOLO's own label ("bottle") is ignored. `detect_cans.py` is the standalone detector. Tests use fakes - never load the model or call the API from tests.
- Frame: the package works in `machine.yaml` → `reference_frame: arm_origin` (the arm's base), not `world`. The machine's `world` was recalibrated on 2026-09-18 (arm at (-65, 35, 26), turned -90.8°); every number in `config/` was measured from the arm's base. The real walls go to the planner from `workspace.yaml` → `obstacles`. `move_arm.py` still sends its poses in `world`, so its fixed poses are no longer where they were - re-check before running it.
- Orders: `python -m recycle_sorter.cli --mode label --want coke=2 sparkling_water` (or `--order "2 cokes and a sparkling water"`, parsed by `llm/parse_order.py`) picks ONLY the requested classes - surest reads first, one pile per class - and logs what is missing. It is `run_sort(..., wanted={class: n})`: the arm half that `pipeline.py` does not have. Works with any mode (`--mode color --want red=2` needs no Ollama).
- `--mode label` (free, the team's direction): `classify/label_vlm.py` hands each YOLO crop to `llm/classify_image.py` (Xiang's local Ollama vision model + `reference_pics/`); piles = the categories in `llm/categories.py`. Needs Ollama reachable (`OLLAMA_URL`). `--mode brand` (Claude API) is kept but unused - the team does not want paid APIs.
- Camera calibration lives in config: `machine.yaml` `depth_scale` (this RealSense reads ~6 % long) and `workspace.yaml` `gripper.xy_offset`, both measured on the arm on 2026-09-18. To re-measure x/y: run `scripts/02_hover_test.py`, and while it hovers take a read-only snapshot - the camera looks straight down at the lid.
- `choose_next` (`perception/select.py`) leaves an item where it is, with a warning, when its grasp target is outside `bounds`. The target is the item's position plus `gripper.xy_offset`, for example a can at the table's -y edge. The next item is picked instead of `UnsafeTarget` stopping the run.
  `bounds.named_*_min` loosen the limits only for taught poses (`survey`, `scan`), never for picks.
- z values in the package mean **fingertip** height. `Manipulator.move_to` adds `tcp_offset` to get the gripper-frame pose for `motion.move`.
- Safe to run (no motion): `.venv/bin/python -m pytest -q`, and `python -m recycle_sorter.cli --replay <frames dir>`.
- These **move the arm**: `python -m recycle_sorter.cli` (without `--dry-run` / `--replay`), `scripts/02_hover_test.py`. Use `--step` on first runs.
- Everything tuned at the table lives in `config/*.yaml`, not in code.

## Drink order pipeline (`pipeline.py`, `llm/`)

A typed order such as "2 cokes and a sparkling water" goes in, and the arm fetches those cans.
It runs entirely on this laptop through Ollama, so it needs no API key.

- `pipeline.py` is a front end on top of `recycle_sorter`. For each command:
  1. `llm/parse_order.py` (`qwen2.5:1.5b`) turns the text into `{category: count}` plus a list of `not_supported` items.
     The result is written to `data/orders/<timestamp>.json` as `command`, `items`, and `not_supported`.
  2. No confirmation: the arm starts fetching as soon as the order is parsed. Use `--step` or `--dry-run` to check first.
  3. `run_sort(robot, "label", wanted=items)` does the rest: survey picture, YOLO boxes, the local VLM label for each crop (`classify/label_vlm.py` → `llm/classify_image.py`), then pick and place, surest reads first, one pile per category.
  4. `result`, `fetched`, and `missing` are added to the same JSON, or `error` if the run failed. The prompt then returns for the next command.
  5. The last annotated survey picture (boxes and labels) is copied to `data/scans/<timestamp>.png`, the same timestamp as the order JSON, and its path is recorded as `picture`.
     The last scan picture, with each can's crop box, is copied to `<timestamp>-scan.png` (`scan_picture`).
     `run_sort` draws every picture it takes in `data/debug/`. Nothing else is saved: no crops and no per-box results.
  - The robot connects once per session. Ctrl-C stops the arm, the same way the CLI does.
- Run it:
  - `python pipeline.py` prompts for commands, and **the arm moves**.
  - Use `--step` on first runs to press Enter before every motion.
  - `python pipeline.py --dry-run` plans and logs the moves without moving anything.
    Dry-run skips the move to `survey`, so its picture is taken from wherever the arm is. When the arm is not at `survey`, the zone or layout can fail with errors like "no room for sorted piles".
  - `python pipeline.py "2 cokes"` runs one command and exits.
  - Two pictures per look in `--mode label` (`sort.yaml` → `view: scan`):
    - **Positions**, and so every grasp, come from `survey`, the original `cans_view` pose. `depth_scale` and `gripper.xy_offset` were measured there.
      Planning grasps from the lower pose made every grab miss on 2026-09-18.
    - **Labels** come from both pictures. Each can is sent to the VLM as two crops in one request: its YOLO box in the survey picture (from above) and its box in the `scan` picture (the lower `cam_lower_pos` pose, closer and from the side). `LocalLabelClassifier.classify_scan` does this, and `llm/classify_image.py` accepts a list of views with `view_names`.
      The scan-picture box is the YOLO box there nearest the can's projected 3D box (`perception/scan.py`), linked one-to-one (`LocalLabelClassifier._link`). The projection alone lands tens of pixels off and caught two cans per crop on 2026-09-18.
      Every YOLO box in either picture is sent to the VLM. A can hidden behind a nearer one still gets its own scan box if YOLO found one, and visible cans pick their boxes first, so a can hidden behind another can't take the front can's box.
      A survey can with no scan box is sent with its survey crop alone. A scan box that no survey can links to is sent alone, as `scan-N`. It has no position, so it is logged and drawn but never picked (`last_scan_only`). Nothing is deferred. A can that hasn't moved keeps its last read.
      Links: each read's `meta["link"]` holds `can`, `survey_box`, `scan_box`, `scan_yolo_index`, and `views`. `#i` in the survey picture is `#i` in the scan picture. The links are written to `data/debug/<ts>-scan-links.json`, logged per can, and copied into the order JSON as `links`.
      `num_ctx` always leaves room for two views (`MAX_VIEWS`). Ollama reloads the model, about 50 s, whenever `num_ctx` changes, so mixing one-view and two-view requests must not change it.
    - `--cam-pos` (mode `label_cam_pos`) reads labels from the survey picture too: one picture per look. `recycle_sorter.cli` has the same flag.
    The joint angles are in `config/joint_positions.json`, but the code always moves to the stored gripper poses through the planner, never to raw joint angles.
  - The same arm path without the prompt: `python -m recycle_sorter.cli --mode label --order "..."` or `--want coke=2`.
  - The image half only, with no arm: `python scan_drinks.py --camera|<pic>`. It writes `data/scans/<timestamp>/` with `picture.jpg`, `detected.jpg`, `crops/`, and `results.json`.
  - The text half only: `python -m llm.parse_order`.
- Needs `ollama serve` running, with `qwen2.5:1.5b` and `qwen2.5vl:7b` pulled.
  - Override the models with the `ORDER_MODEL` and `VISION_MODEL` environment variables.
  - Every request sends `keep_alive: -1`, so both models stay loaded until Ollama stops. Ollama's default unloads them after 5 minutes. Override with `OLLAMA_KEEP_ALIVE_MODELS`, a number or a duration like `"30m"`.
  - Loading the vision model and having it read the reference photos takes about 55 s once per Ollama start.
    - `python -m llm.warmup &` does this in the background right after `ollama serve`.
    - `pipeline.py` also starts `warm_up()` in the background on launch, overlapping with connecting to the robot.
    - After that, a crop takes about 12 s even on the first order. Measured 2026-09-19, with both models loaded together (7.2 GB).
  - The 7B model needs about 7 GB of memory.
- **Categories are shared.** They live only in `llm/categories.py`, and both models read them from there:
  - `CATEGORIES`: each category has an `order` description (for the parser) and a `visual` description (for the VLM).
  - `BRAND_KEYWORDS`: ordered, first match wins, whole words only.
  - The current categories are `coke`, `diet_coke`, `water`, `sparkling_water`, `energy_drink`, `coconut_water`, `ginger_ale`, `non_listed_drinks`, and `not_supported`.
  - To add a product, edit only this file.
- Small models are wrong more often about labels than about words. So after the model answers, a brand keyword in the request text, or in the VLM's `visible_text`, overrides the model's label.
  - The plain "Coca-Cola" logo does not override a `diet_coke` answer.
  - Avoid generic keywords that other drinks share, such as `cola` or `zero sugar` alone.
- The VLM answers `regular_coke` internally, and the code maps it back to `coke`. Under the strict schema the model confused `coke` with `coconut_water`, so don't rename it back.
- `reference_pics/` is tracked in git. Every photo in it is sent with each crop, labelled with its category.
  - The category comes from the file or folder name. That is either a category name (`coke.JPG`) or a brand that `BRAND_KEYWORDS` knows (`canada_dry/` → `ginger_ale`).
  - `num_ctx` grows with the number of photos, about 1200 tokens per image. Ollama's default of 4096 fails with even a few photos.
- `scan_drinks.py` and `detect_cans.py` never move the arm. With `--camera` they connect through `LiveRobot.create(dry_run=True)`, which skips every motion call, including `set_speed`.
- Timing on this laptop, measured 2026-09-18:
  - Parsing the order takes about 2.5 s, and YOLO about 0.2 s.
  - The VLM takes about 10 s per crop. The first crop after the model loads takes about 55 s, because it has to process the reference photos.
- Known weakness: the survey crops are about 100 px, taken from above, and blurry.
  - The VLM sometimes invents label text, confuses Diet Coke with Coke, and confuses Canada Dry with other drinks.
  - On the 2026-09-18 table it got about 8 of 10 cans right.
  - Planned fixes: close-up verification at the approach pose before each grasp, and top-down reference photos.
