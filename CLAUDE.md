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
- z values in the package mean **fingertip** height. `Manipulator.move_to` adds `tcp_offset` to get the gripper-frame pose for `motion.move`.
- Safe to run (no motion): `.venv/bin/python -m pytest -q`, and `python -m recycle_sorter.cli --replay <frames dir>`.
- These **move the arm**: `python -m recycle_sorter.cli` (without `--dry-run` / `--replay`), `scripts/02_hover_test.py`. Use `--step` on first runs.
- Everything tuned at the table lives in `config/*.yaml`, not in code.

## Drink order pipeline (`pipeline.py`, `llm/`)

A typed order such as "2 cokes and a sparkling water" goes in. Out comes JSON naming which detected cans to pick, by YOLO box index.
It runs entirely on this laptop through Ollama, so it needs no API key. It does not move the arm yet.

- Flow, one fresh picture per command:
  1. `llm/parse_order.py` (`qwen2.5:1.5b`) turns the text into `{category: count}` plus a list of `not_supported` items.
  2. The picture comes from the camera (`--camera`) or from a file (`--image`).
  3. `detect_cans.py` (YOLOE) draws numbered boxes.
  4. `scan_drinks.py: scan()` crops each box (8% padding, enlarged to 640 px), and `llm/classify_image.py` (`qwen2.5vl:7b`) labels each crop.
  5. `pipeline.py: choose()` picks the most confident matches for each category. `low` confidence items are never picked. A shortfall goes to `missing`.
- Run it:
  - `python pipeline.py --camera` prompts for commands.
  - `python pipeline.py --image pic.jpg "2 cokes"` runs one command and exits.
  - `python scan_drinks.py --camera|<pic>` runs only the image half.
  - `python -m llm.parse_order` runs only the text half.
- Needs `ollama serve` running, with `qwen2.5:1.5b` and `qwen2.5vl:7b` pulled.
  - Override the models with the `ORDER_MODEL` and `VISION_MODEL` environment variables.
  - The 7B model needs about 7 GB of memory.
- **Categories are shared.** They live only in `llm/categories.py`, and both models read them from there:
  - `CATEGORIES`: each category has an `order` description (for the parser) and a `visual` description (for the VLM).
  - `BRAND_KEYWORDS`: ordered, first match wins, whole words only.
  - The current categories are `coke`, `diet_coke`, `water`, `sparkling_water`, `energy_drink`, `coconut_water`, `general_soda`, and `not_supported`.
  - To add a product, edit only this file.
- Small models are wrong more often about labels than about words. So after the model answers, a brand keyword in the request text, or in the VLM's `visible_text`, overrides the model's label.
  - The plain "Coca-Cola" logo does not override a `diet_coke` answer.
  - Avoid generic keywords that other drinks share, such as `cola` or `zero sugar` alone.
- The VLM answers `regular_coke` internally, and the code maps it back to `coke`. Under the strict schema the model confused `coke` with `coconut_water`, so don't rename it back.
- `reference_pics/` is tracked in git. Every photo in it is sent with each crop, labelled with its category.
  - The category comes from the file or folder name. That is either a category name (`coke.JPG`) or a brand that `BRAND_KEYWORDS` knows (`canada_dry/` → `general_soda`).
  - `num_ctx` grows with the number of photos, about 1200 tokens per image. Ollama's default of 4096 fails with even a few photos.
- Output: each command writes `data/scans/<timestamp>/`, which is git-ignored. It holds `picture.jpg`, `detected.jpg`, `crops/NN.jpg`, `results.json`, and `order.json`.
  - With `--camera`, the depth frame is also saved to `data/frames/`, for the 3D grasp later.
- Safe to run, because it never moves the arm: `pipeline.py`, `scan_drinks.py`, and `detect_cans.py`.
  - With `--camera` they connect through `LiveRobot.create(dry_run=True)`, which skips every motion call, including `set_speed`.
- Timing on this laptop, measured 2026-09-18:
  - Parsing the order takes about 2.5 s, and YOLO about 0.2 s.
  - The VLM takes about 10 s per crop. The first crop after the model loads takes about 55 s, because it has to process the reference photos.
- Known weakness: the survey crops are about 100 px, taken from above, and blurry.
  - The VLM sometimes invents label text, confuses Diet Coke with Coke, and confuses Canada Dry with other drinks.
  - On the 2026-09-18 table it got about 8 of 10 cans right.
  - Planned fixes: close-up verification at the approach pose before each grasp, and top-down reference photos.
