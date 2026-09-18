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
- `--mode brand` (cans/bottles): top-down survey for positions, then an eye-level `scan` pose (taught, `poses.yaml` → `named.scan`) whose crops are read by Claude (`classify/brand_claude.py`, needs `ANTHROPIC_API_KEY` in `.env`). Items hidden from the side are deferred to the next look. Tests use a fake client - never call the real API from tests.
- z values in the package mean **fingertip** height. `Manipulator.move_to` adds `tcp_offset` to get the gripper-frame pose for `motion.move`.
- Safe to run (no motion): `.venv/bin/python -m pytest -q`, and `python -m recycle_sorter.cli --replay <frames dir>`.
- These **move the arm**: `python -m recycle_sorter.cli` (without `--dry-run` / `--replay`), `scripts/02_hover_test.py`. Use `--step` on first runs.
- Everything tuned at the table lives in `config/*.yaml`, not in code.
