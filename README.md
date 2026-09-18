# viam_hackathon — Recycle Sorter

A Viam arm with a wrist RealSense picks items out of one unsorted pile and places them into
sorted piles. One pick-and-place loop; only the **classifier** changes per stage:

| Stage | Sort by | Classifier | Status |
|---|---|---|---|
| 1 | color | OpenCV: objects from the depth image, color by HSV | built, tested offline |
| 2 | shape | contours + height profile | next |
| 3 | material | Claude vision → bootstraps a Viam-trained TFLite model | planned |
| 4 | brand (soda vs water) | Claude vision close-up before the pick | stretch |

```
survey pose → RGB-D snapshot → find the objects (inside the unsorted zone) → choose topmost
  → classify → pick → set down in that class's pile (created on demand) → re-survey
```

## Perception: two options (`perception:` in [config/machine.yaml](config/machine.yaml))

**`depth` (default, used for color).** OpenCV on the depth image: anything standing above the table
inside the unsorted zone is an object; its color is read afterwards with HSV. It is class-agnostic, which
is what "assess the zone, then create piles to fit" needs: whatever colors turn up get piles, including
black / white / grey, with nothing to configure per color (hues live in [config/sort.yaml](config/sort.yaml)).
Needs the RealSense depth **aligned** to color — see *Camera alignment* below; step 3 checks it.

**`viam`.** Viam vision services (`get_object_point_clouds`), as `move_arm.py` does. A color detector +
segmenter only reports its **one** target color, so it needs one service per class (`vision.segmenters`:
label → service) and never sees a color that has no service. It earns its place in the later stages:
with a trained ML detector that names what it finds, set `use_detection_label: true`.

Both paths build objects through the same code, so size, **top height** and grasp angle are computed from
the object's points — not from a box center, which sits too low on tall items (why `move_arm.py`
hard-codes `OBJECT_HEIGHT_MM`).

### Camera alignment (required for `depth`)

The RealSense's depth sensor has a wider lens (~87°) than its color sensor (~70°). By default the two
images are the same *size* but **not aligned**, and Viam only reports the color lens. Unaligned, 3D
positions come out distorted (the table looks tilted ~8°) and colors are read from the wrong pixels.
Measured on this machine on 2026-09-18: tilt 7.8° → 0.5° once the right lens is assumed.

Fix — one attribute on the camera. In the Viam app, add to the `fragment_mods` of the arm/camera fragment
(next to the existing `use_urdfs` mod):

```json
{ "$set": { "components.cam.attributes.align_color_depth": true } }
```

It only changes the depth image returned by `GetImages`; the camera's point cloud (what `move_arm.py`'s
vision service uses) is already aligned, so that pipeline is unaffected. If the white table gives noisy
depth, `"depth_visual_preset": "high_accuracy"` on the same camera may help.

Two safeguards are built in regardless: heights are measured relative to the table *as seen in each frame*
(so a calibration a few cm or degrees off still works, and grasp heights stay in the arm's coordinates),
and a blob must also look different from the table, or be over 40 mm tall, to count as an object — a plain
white table produces depth-noise bumps as tall as a block.

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
The map is drawn as you see the table standing behind the arm (`map_view: behind`; use `front` if you face it).

- **What to sort by:** `--mode` (one per classifier). **Merge classes into one pile:**
  `groups: {warm: [red, orange, yellow]}` in [config/sort.yaml](config/sort.yaml).
- **Where piles may go:** the arm sits near the table's **+y edge** (measured from depth: the table ends
  ~220–265 mm from the arm on that side and runs past −530 on the other), so both `sorted_areas` in
  [config/workspace.yaml](config/workspace.yaml) are on the open **−y** side — to your right when you stand behind the arm.
  Their exact positions are untested guesses. Teach real ones by jogging the gripper to two opposite corners of
  free table space and running `python scripts/01_teach_pose.py --area near` at each. Taught areas override the guesses.
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
| 0 | `python -m recycle_sorter.cli --action static-cycle --step` | **yes**, Enter per move | Blind pick→place between `move_arm.py`'s two fixed poses, but through this package. Proves our motion layer matches theirs before any vision. |
| 1 | `python scripts/00_discover.py` | no | Resource names, frame tree, camera streams, and what each vision service in `machine.yaml` sees. |
| 2 | *(optional)* Jog in the Viam app, then `python scripts/01_teach_pose.py survey` / `home` | no | `survey` ships as `move_arm.py`'s proven `WATCH_POSE`, so only re-teach it if the magenta unsorted zone isn't fully in view. |
| 3 | Arm at survey, `python scripts/03_record_frames.py --table` | no | **Gate: must say `OK`.** Reports the table's tilt as the camera sees it. Several degrees = the depth image is not aligned to color (see *Camera alignment* below) and nothing downstream can work. |
| 4 | One block on the table: `python scripts/02_hover_test.py` | **yes**, Enter per move | **Gate: gripper hovers within 15 mm of the block in x/y, and fingertips sit `--hover` (60) mm above its top.** A height error goes into `gripper.tcp_offset`; finger rotation into `yaw_offset_deg`. If it fails, fix the camera frame / `move_frame` before anything else. |
| 5 | Jog to two opposite corners of free table space, running `python scripts/01_teach_pose.py --area left` at each (repeat for `right`) | no | Replaces the guessed sorted areas with ones the arm can really reach. Check the magenta unsorted zone in `data/debug/` covers where you dump items. |
| 6 | `python -m recycle_sorter.cli --mode color --dry-run` | no | Assessment + pile plan + first pick logged, nothing moves. Check `data/debug/layout-plan.png`. |
| 7 | `python -m recycle_sorter.cli --mode color --step` | **yes**, Enter per move | First real sort. Drop `--step` once it behaves, then raise `arm_speed`. |

Also measure the gripper's real max opening → `workspace.yaml` `gripper.max_open`. It decides
which recycling items are feasible later (a full-size can is ~66 mm).

## Adding a classifier (where the color gate plugs in)

The loop never changes between stages; a stage is a class with one method. The objects are already found, so a
classifier only answers "what is this one?":

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

`classify/color_hsv.py` is the color stage. `vision_label.py` takes the class from a Viam vision service instead, when `perception: viam`.

## Working without the robot

Every survey saves RGB + depth + intrinsics + camera→world **and the vision services' objects** to
`data/frames/<ts>/`, so a Viam-vision run replays offline exactly as it was seen.

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
move_arm.py  connect.py          Xiang's working module code: single-object vision pick via the Viam
                                 vision service. Hardware-proven. NOT edited on this branch.
recycle_sorter/                  multi-object, multi-class sorter built around the same motion rules
  service.py                     same do_command interface as move_arm.py, backed by the package
  app.py / cli.py                run_sort loop, --dry-run / --step / --replay
  io/          robot.py (connect, snapshot)   recorder.py (save / load frames)
  perception/  viam_vision.py (Viam services → objects)  pcd.py  segment.py (shared geometry + OpenCV fallback)
               frames.py  select.py
  classify/    base.py (Protocol)  vision_label.py  color_hsv.py   ← one file per stage
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
