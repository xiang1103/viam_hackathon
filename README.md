# viam_hackathon — Recycle Sorter

A Viam arm with a wrist RealSense picks items out of one unsorted pile and places them into
sorted piles. One pick-and-place loop; only the **classifier** changes per stage:

| Stage | Sort by | Classifier | Status |
|---|---|---|---|
| 1 | color | HSV inside depth-segmented masks | built, tested offline |
| 2 | shape | contours + height profile | next |
| 3 | material | Claude vision → bootstraps a Viam-trained TFLite model | planned |
| 4 | brand (soda vs water) | Claude vision close-up before the pick | stretch |

```
survey pose → RGB-D snapshot → segment (above table plane, inside the unsorted zone) → choose topmost
  → classify → pick → set down in that class's pile (created on demand) → re-survey
```

## Zones and piles

Nothing about the piles is fixed in advance — only where they are *allowed* to go.

1. **Assess.** From the survey pose, find and classify everything in the **unsorted zone**
   → e.g. `{red: 2, blue: 2, green: 1, grey: 1}` plus the largest item in each class.
2. **Plan.** Create one pile per class inside the **sorted areas** (free table space you define):
   slot count from how many were seen, slot spacing from how big they are. A **reject** pile is
   always reserved first, for low-confidence items and classes that don't fit.
3. **Sort.** Pick topmost → set it down in its pile's next slot → re-survey, until the zone is empty.
   A heap hides things, so it stays dynamic: a class first seen mid-run gets a new pile from the
   remaining space, and a pile that fills up grows an extension.

Every run writes `data/debug/layout-plan.png` (the plan, before anything moves) and
`data/debug/layout.png` (live: slots fill in as items land) — a top-down map of the table.
`--dry-run` and `--replay` show the assessment and the plan without moving the arm.

- **What to sort by:** `--mode` (one per classifier). **Merge classes into one pile:**
  `groups: {warm: [red, orange, yellow]}` in [config/sort.yaml](config/sort.yaml).
- **Where piles may go:** `sorted_areas` in [config/workspace.yaml](config/workspace.yaml) are untested
  guesses. Teach real ones by jogging the gripper to two opposite corners of free table space and
  running `python scripts/01_teach_pose.py --area left` at each. Taught areas override the guesses.
- **Spacing:** `piles.slot_gap`, `spare_slots`, `pile_gap` in `workspace.yaml`.
- Startup refuses to run if a sorted area overlaps the unsorted zone, another area, or the safety
  bounds. Only objects inside the unsorted zone are ever picked, so sorted items are never re-picked.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # fill in from the Viam app: machine → CONNECT → API keys
.venv/bin/python -m pytest -q
```

## Day-1 bring-up (in order — each step gates the next)

| # | Command | Moves arm? | What you learn / must pass |
|---|---|---|---|
| 0 | `python move_arm.py --step` | **yes**, Enter per move | Blind pick→place between two fixed poses (`config/poses.yaml` → `static`). Proves arm + gripper + motion before any vision. |
| 1 | `python scripts/00_discover.py` | no | Resource names, frame tree, camera streams. Color and depth must be the **same resolution** — if not, set `align_color_depth: true` on the RealSense (fragment mod). |
| 2 | Jog in the Viam app, then `python scripts/01_teach_pose.py home` and `... survey` | no | `survey` = camera pointing straight down, ~400 mm above the pile (RealSense can't see closer than ~280 mm). |
| 3 | Clear the table, go to survey, `python scripts/03_record_frames.py --table` | no | Prints the measured `table_top` → put it in `config/workspace.yaml`. |
| 4 | One block on the table: `python scripts/02_hover_test.py` | **yes**, Enter per move | **Gate: gripper hovers within 15 mm of the block in x/y, and fingertips sit `--hover` (60) mm above its top.** A height error goes into `gripper.tcp_offset`; finger rotation into `yaw_offset_deg`. If it fails, fix the camera frame / `move_frame` before anything else. |
| 5 | Jog to two opposite corners of free table space, running `python scripts/01_teach_pose.py --area left` at each (repeat for `right`) | no | Replaces the guessed sorted areas with ones the arm can really reach. Check the magenta unsorted zone in `data/debug/` covers where you dump items. |
| 6 | `python -m recycle_sorter.cli --mode color --dry-run` | no | Assessment + pile plan + first pick logged, nothing moves. Check `data/debug/layout-plan.png`. |
| 7 | `python -m recycle_sorter.cli --mode color --step` | **yes**, Enter per move | First real sort. Drop `--step` once it behaves, then raise `arm_speed`. |

Also measure the gripper's real max opening → `workspace.yaml` `gripper.max_open`. It decides
which recycling items are feasible later (a full-size can is ~66 mm).

## Adding a classifier (where the color gate plugs in)

The loop never changes; a stage is just a class with one method. `segment()` has already found
each object, so a classifier only answers "what is this one?":

```python
# recycle_sorter/classify/my_color.py
class MyColorClassifier:
    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        # obs.crop = BGR crop, obs.mask = HxW bool mask into frame.color, obs.width/length/height in mm
        return Classification(label="red", confidence=0.9)
```

1. Return it from `make_classifier()` in [recycle_sorter/app.py](recycle_sorter/app.py) for your mode.
2. Add the mode under `modes:` in [config/sort.yaml](config/sort.yaml). That's all — piles are created from whatever labels you return.
3. Test it offline: `python -m recycle_sorter.cli --mode <mode> --replay data/frames`.

`classify/color_hsv.py` is a working baseline for the color gate — replace it, or keep it to compare against.

## Working without the robot

Every survey saves RGB + depth + intrinsics + camera→world to `data/frames/<ts>/`.

```bash
python scripts/03_record_frames.py -n 10          # at the table: bank frames of varied piles
python -m recycle_sorter.cli --replay data/frames  # anywhere: perception + labels + overlays
```

## Config (everything tuned at the table lives here, not in code)

- `config/machine.yaml` — resource names, `move_frame`, arm speed
- `config/workspace.yaml` — `table_top`, safety bounds, unsorted zone, sorted areas, gripper + pick parameters
- `config/sort.yaml` — per mode: min confidence and class groups; color hues
- `config/poses.yaml` — static pick/place poses, plus home / survey / sorted areas written by `01_teach_pose.py`

Safety: every target is bounds-checked before it reaches the planner (`manipulation/safety.py`);
Ctrl-C stops the arm; the workcell fragment's walls/table/ceiling are in every motion plan.

## Layout

```
move_arm.py                      entry point: blind static cycle (no camera)
recycle_sorter/
  service.py                     MyGenericService = the code-1 module; do_command → package
  app.py / cli.py                run_sort loop, --dry-run / --step / --replay
  io/          robot.py (connect, snapshot)   recorder.py (save / load frames)
  perception/  frames.py  segment.py  select.py
  classify/    base.py (Protocol)  color_hsv.py        ← one file per stage
  manipulation/ motion.py (bounds, tcp offset)  pickplace.py  safety.py
  policy/      sort_policy.py (class → pile key)  piles.py (creates + grows piles in the sorted areas)
  viz.py       top-down table map
```

## Optional: drive the machine directly from Claude (MCP)

[erh/viam-mcp-server](https://github.com/erh/viam-mcp-server) runs **on the machine** as a generic
service and exposes `<component>__<method>` tools. Handy for discovery, jogging, and debugging.

1. Viam app → CONFIGURE → add service → `erh:viam-mcp-server:mcp-server`, attributes:
   ```json
   { "components": ["arm", "gripper", "<camera name>"], "address": ":8765" }
   ```
2. Check reachability, then register it:
   ```bash
   nc -zv <machine-host> 8765
   claude mcp add --transport http viam-mcp http://<machine-host>:8765
   ```

⚠️ The endpoint has **no authentication** — anyone on the same network can move the arm while it
is enabled. Fine on a trusted LAN; remove the service when you're done. Direct MCP moves also
bypass this repo's workspace bounds checks, so keep the speed low.
