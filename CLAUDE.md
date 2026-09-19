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
- Orders: `python -m recycle_sorter.cli --mode label --want coke=2 water` (or `--order "2 cokes and a sparkling water"`, parsed by `llm/parse_order.py`) picks ONLY the requested classes - surest reads first, one pile per class - and logs what is missing. It is `run_sort(..., wanted={class: n})`: the arm half that `pipeline.py` does not have. Works with any mode (`--mode color --want red=2` needs no Ollama).
- `--mode label` (free, the team's direction): `classify/label_vlm.py` hands each YOLO crop to `llm/classify_image.py` (Xiang's local Ollama vision model + `reference_pics/`); piles = the categories in `llm/categories.py`. Needs Ollama reachable (`OLLAMA_URL`). `--mode brand` (Claude API) is kept but unused - the team does not want paid APIs.
- Camera calibration lives in config: `machine.yaml` `depth_scale` (this RealSense reads ~6 % long) and `workspace.yaml` `gripper.xy_offset`, both measured on the arm on 2026-09-18. To re-measure x/y: run `scripts/02_hover_test.py`, and while it hovers take a read-only snapshot - the camera looks straight down at the lid.
- `choose_next` (`perception/select.py`) leaves an item where it is, with a warning, when its grasp target is outside `bounds`. The target is the item's position plus `gripper.xy_offset`, for example a can at the table's -y edge. The next item is picked instead of `UnsafeTarget` stopping the run.
  `bounds.named_*_min` loosen the limits only for taught poses (`survey`, `scan`), never for picks.
- Picks have their own limits: `pick.max_reach` (655) and `pick.y_min` (-545), used by `choose_next` and `pick_at`.
  Reach is measured to the arm's actual target: the grasp point plus `xy_offset`.
  Pile layout and placing keep `sorted_layout.max_reach` (650) and `bounds.y` (-460), so piles are never laid out at the table's edge.
  - `pick.max_reach` is the arm's real straight-down reach. It was computed on 2026-09-19 from the arm's kinematics, whose forward kinematics matched Viam's reported gripper pose exactly.
  - The reach is about 710 mm at grasp height, 668 mm at approach height (+150), and 664 mm at lift height (+160). It's the same in every direction.
  - A pick passes through all three heights, so a higher `pick.lift` or `pick.approach` reduces the reach. At +200 the reach was 644 mm.
  - Beyond that, the planner answers "zero IK solutions", so cans out there must be moved closer.
- `survey` is a **top view** since 2026-09-19 (computed from the arm's kinematics, then stood at): camera at (326, -355, 713) looking 10° off straight down, so cans are seen 10–29° off vertical instead of 35–49°. The old tilted view is kept as `survey_tilted`. From above, YOLO needs the `round metal object` prompt (`machine.yaml`). Labels are not visible from there, so `--cam-pos` / `label_cam_pos` cannot work with it, and `sort.yaml` `survey_crop: false` reads the scan crop alone. `poses.yaml` `joint_moves` no longer lists `survey` (its `cans_view` joints are the old view). Try another pose with `--action reset --pose NAME`.
- Label reads are kept for the whole session and reading stops early (`classify/label_vlm.py`). `SESSION_MEMORY` outlives the classifier, which `run_sort` rebuilds for every order: a can is not read again when it stands within 25 mm of where it was read AND its crop still looks the same (`signature()` colour histogram, `SAME_LOOK` 0.85 - measured: same can 0.92, different cans 0.59), so a can swapped by hand is re-read. For an order, `run_sort` sets `classifier.wanted`; reading stops once reads of confidence ≥ 0.9 cover every ordered item, the rest come back as `unread` (confidence 0: never picked, not remembered), and scan-only boxes are skipped. A sort (`wanted` None) still reads everything. One read takes ~4 s on the M3 Max and ~10 s on the 16 GB Mac.
- `pick.scan_first: true` (`workspace.yaml`): in scan modes the label picture is taken first and the survey picture last, right before the pick. A look that expects an empty table goes to the survey alone.
- Standing cans are located from their **outline in the colour picture** (`perception/silhouette.py`, `workspace.yaml` → `silhouette`), not from depth: depth drops out on a shiny lid, a different part in every picture, and the depth-based centre jumped up to 18 mm between two pictures of an untouched can (2026-09-19). Depth still gives the height. The depth position is kept when the fit is poor (`min_iou`, `max_shift`).
- One can boxed twice (the lid by `round metal object`, the whole can by `soda_can`; YOLO's NMS only compares boxes of one class) is one item: `observations_from_boxes` keeps the surer box when two land within `segmentation.same_item_mm` (40) and draws the other in red. As two items on 2026-09-19, the duplicate borrowed another can's scan box and a Red Bull was fetched as a coke.
- Standing cans are grasped at a fixed wrist angle, `gripper.upright_theta` (0): a round lid has no short axis, so its measured yaw is noise. `gripper.xy_offset` is only valid at that angle - re-measure it with the hover test after changing either.
- z values in the package mean **fingertip** height. `Manipulator.move_to` adds `tcp_offset` to get the gripper-frame pose for `motion.move`.
- Safe to run (no motion): `.venv/bin/python -m pytest -q`, and `python -m recycle_sorter.cli --replay <frames dir>`.
- These **move the arm**: `python -m recycle_sorter.cli` (without `--dry-run` / `--replay`), `scripts/02_hover_test.py`. Use `--step` on first runs.
- Everything tuned at the table lives in `config/*.yaml`, not in code.

## Drink order pipeline (`pipeline.py`, `llm/`)

A typed order such as "2 cokes and a sparkling water" goes in, and the arm fetches those cans.
It runs entirely on this laptop through Ollama, so it needs no API key.

- `pipeline.py` is a front end on top of `recycle_sorter`. For each command:
  1. `llm/parse_order.py` (`qwen2.5:1.5b`) turns the text into `{category: count}` plus a list of `not_supported` items. It is printed, not saved.
  2. No confirmation: the arm starts fetching as soon as the order is parsed. Use `--step` or `--dry-run` to check first.
  3. `run_sort(robot, "label", wanted=items)` does the rest: survey picture, YOLO boxes, the local VLM label for each crop (`classify/label_vlm.py` → `llm/classify_image.py`), then pick and place, surest reads first, one pile per category.
  4. `fetched` and `missing` are printed, or the error if the run failed. The prompt then returns for the next command.
  5. Files: only `data/scans/<launch time>/`, made once per launch, holding the latest look's two annotated YOLO pictures, `survey.png` and `scan.png`. Each look overwrites them. `pipeline.py` takes ONE look per command (`--look once` is its default: the scan picture and the survey picture, then every ordered item is fetched from them with no new picture between picks or at the end); `--look when_needed` brings back the re-checks. `order.json` there gets one entry per command: the parsed order, `not_available`, what was fetched (or `planned` in a dry run) and what was `missing`. An ordered category that no can was read as is skipped and reported as missing; nothing else on the table is touched (`tests/test_pipeline.py` runs this end to end with fakes).
     Both are drawn the same way: each YOLO box with `#i label confidence`, and `#i` is the same can in both. `survey.png` also shows, in red with the reason, every YOLO box that didn't become an item (`observations_from_boxes(..., rejected)`: nothing above the table, too few depth points, outside the unsorted zone, too big). It also shows the unsorted zone in magenta. `scan.png` shows cans seen only there in orange, as `scan-N`.
     A `--dry-run` doesn't move to `scan`, so its two pictures show the same view.
     This is `run_sort(..., pictures=<folder>)`. With `pictures`, nothing goes to `data/orders/`, `data/debug/` or `data/frames/`.
     Without it (`recycle_sorter.cli`, `scripts/`), `run_sort` still keeps every look for debugging: raw frames in `data/frames/`, and overlays, links and pile layouts in `data/debug/`.
  - The robot connects once per session, in the background: `command>` shows at once, and a command waits for the connection only when it has something to fetch. Parsing doesn't wait. Ctrl-C stops the arm, the same way the CLI does.
  - Nothing prints into the prompt on its own:
    - The model warm-up runs silently in a daemon thread, so quitting never waits for it.
    - `pipeline.py` sets `VIAM_LOG_LEVEL=WARNING`, which `recycle_sorter/io/robot.py: connect()` passes to the SDK. The SDK resets its log level on every connect, so setting it anywhere else doesn't work.
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
      Links: each read's `meta["link"]` holds `can`, `survey_box`, `scan_box`, `scan_yolo_index`, and `views`. `#i` in the survey picture is `#i` in the scan picture. The links are logged per can. Without `pictures`, they are also written to `data/debug/<ts>-scan-links.json`.
      `num_ctx` always leaves room for two views (`MAX_VIEWS`). Ollama reloads the model, about 50 s, whenever `num_ctx` changes, so mixing one-view and two-view requests must not change it.
    - `--cam-pos` (mode `label_cam_pos`) reads labels from the survey picture too: one picture per look. `recycle_sorter.cli` has the same flag.
    The joint angles are in `config/joint_positions.json`. Survey ↔ scan moves by those fixed joint angles, so it takes the same path every time (`poses.yaml` → `joint_moves`, `Manipulator._joint_move`). Every other move goes through the planner.
      It moves to and from the **exact taught numbers**, and only when the arm is already on one of them, within `tolerance_deg` on every joint. A raw joint move skips the planner's obstacle check, so from anywhere else the planner is used.
      A whole turn off on joint 1, 4 or 6 is the same pose but winds the camera cable differently, so it doesn't count and the planner is used.
      `cam_lower_pos` is stored in the version that pairs with `cans_view`: `[-81.862, -27.512, -31.533, -317.105, 64.803, 252.209]`. It was hand-taught (2026-09-19, fourth time) as `[-81.862, -27.512, -31.533, 42.895, 64.803, -107.791]`, the same pose with joints 4 and 6 a whole turn round. When re-teaching, store it the same way, or the base turns 372° every trip.
      Each trip turns joints 4, 5 and 6 by about 144–167°, because the two poses hold the wrist in different arrangements. The user accepted this on 2026-09-19.
  - The same arm path without the prompt: `python -m recycle_sorter.cli --mode label --order "..."` or `--want coke=2`.
  - The image half only, with no arm: `python scan_drinks.py --camera|<pic>`. It writes `data/scans/<timestamp>/` with `picture.jpg`, `detected.jpg`, `crops/`, and `results.json`.
  - The text half only: `python -m llm.parse_order`.
- Model loading: nothing loads during an order, and nothing slow runs on the event loop.
  - At launch, background threads load the two Ollama models (`llm/warmup.py`) and YOLO (`shared_detector(...).warm_up()`). An order that arrives earlier waits for them.
  - YOLO is **one** instance per process (`perception/yolo.py: shared_detector`), used for the survey picture, the scan picture and the warm-up. Don't create a `YoloDetector` in pipeline code: `run_sort` makes a new classifier for every order, so each order would reload it (about 2 s).
  - Every blocking call runs in a thread through `asyncio.to_thread`: YOLO, each VLM read, and order parsing. `YoloDetector` has a lock, so a warm-up and a detection never run at the same time.
- Needs `ollama serve` running, with `qwen2.5:1.5b` and `qwen2.5vl:7b` pulled.
  - Override the models with the `ORDER_MODEL` and `VISION_MODEL` environment variables.
  - Every request sends `keep_alive: -1`, so both models stay loaded until Ollama stops. Ollama's default unloads them after 5 minutes. Override with `OLLAMA_KEEP_ALIVE_MODELS`, a number or a duration like `"30m"`.
  - Loading the vision model and having it read the reference photos takes about 55 s once per Ollama start.
    - `python -m llm.warmup &` does this in the background right after `ollama serve`. `ollama serve` alone loads nothing.
    - `pipeline.py` also runs the same `warm_all()` in the background on launch, overlapping with connecting to the robot.
    - Order matters: the vision model loads first, then the order model. Loading the 7B model second makes Ollama unload the order model on this 16 GB Mac. In this order both stay loaded (about 8.6 GB).
    - After that, a crop takes about 12 s even on the first order. Measured 2026-09-19, with both models loaded together (7.2 GB).
  - The 7B model needs about 7 GB of memory.
- **Categories are shared.** They live only in `llm/categories.py`, and both models read them from there:
  - `CATEGORIES`: each category has an `order` description (for the parser) and a `visual` description (for the VLM).
  - `BRAND_KEYWORDS`: ordered, first match wins, whole words only.
  - The current categories are `coke`, `diet_coke`, `water` (still and sparkling), `energy_drink`, `coconut_water`, `ginger_ale`, `non_listed_drinks`, and `not_supported`.
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
