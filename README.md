# viam_hackathon — Recycle Sorter

A Viam arm with a wrist RealSense picks items out of one unsorted pile and places them into
sorted piles. One pick-and-place loop; only the **classifier** changes per stage:

| Stage | Sort by | Classifier | Status |
|---|---|---|---|
| 1 | color | OpenCV: objects from the depth image, color by HSV | built, tested offline |
| 2 | shape | contours + height profile | next |
| 3 | material | Claude vision → bootstraps a Viam-trained TFLite model | planned |
| 4 | **brand** (cans, bottles) | top-down survey for positions + eye-level scan read by Claude | built, tested offline |

```
survey pose → RGB-D snapshot → find the objects (inside the unsorted zone) → choose topmost
  → classify → pick → set down in that class's pile (created on demand) → re-survey
```

## Cans: find them with YOLO, read their labels with Claude

```bash
python -m recycle_sorter.cli --mode brand --step
```

1. **Find (YOLO).** `perception: yolo` runs YOLOE, an open-vocabulary YOLO that looks for the *words*
   "can" and "bottle" — no training. It only draws the boundary; whatever it calls the item is ignored.
   On the real table it found 9/9 cans at 0.55–0.82 confidence in 0.16 s, ignoring the laptop, power strip and
   tape. (Standard pretrained YOLO has no "can" class: 5/9 at 0.2–0.35. Depth alone: 6/9 plus the clutter.)
2. **Locate (depth).** The aligned depth pixels inside each box give the 3D position, height and grasp —
   the same object the pile and motion code always used.
3. **Read (Claude).** Each box is cropped, enlarged, and all crops go to `claude-opus-5` in one request. It
   reads text, logo and colors, so a can turned partly away still works. With the camera tilted as it is
   now the labels are visible in the survey picture itself (`view: survey`) — one picture per look.
4. **Pile.** `categories` in [config/sort.yaml](config/sort.yaml) are the piles, each with a description. A pile can
   span brands ("energy-drink") or split one ("coke" / "diet-coke"). Readable but none of them → `other`;
   unreadable or under `min_confidence` → reject. Delete `categories` to get one pile per brand instead.

`detect_cans.py` is the detector on its own, for any pipeline — load once, then ~0.2 s a picture:

```python
from detect_cans import CanDetector
detector = CanDetector()           # ~1.5 s, once
boxes = detector(image_bgr)        # .x0 .y0 .x1 .y1 .confidence .mask
```

```bash
python detect_cans.py data/frames/<timestamp>/color.png    # writes color.detected.jpg next to it
python detect_cans.py --camera                              # live picture (moves nothing) + 3D positions and reach
```

Setup: `ANTHROPIC_API_KEY=...` in `.env`. The first YOLO run downloads weights into `models/` (28 MB, plus a
600 MB text encoder needed only that once; the prompts are then cached). To use your own trained model:
`yolo: {kind: yolo, weights: models/yours.pt}` in [config/machine.yaml](config/machine.yaml).

**Keep the cans within reach.** The arm (xArm 5/6/7 family) reaches ~650 mm top-down. Items further out are
reported and left. Piles are made in whatever table is free around the cans — either side of them, or
beyond — so a starting group that fills the whole reachable table leaves nowhere to sort *to*, and the run
refuses with a message saying so. `--look-only` shows what it found and the plan without picking anything.

`view: scan` (a second, eye-level picture from a taught `scan` pose, with cans hidden behind others deferred
to the next look) is still there for a camera that looks straight down and cannot see the labels.

## Perception options (`perception:` in [config/machine.yaml](config/machine.yaml))

**`yolo` (default, for cans).** See above.

**`depth` (for the colored blocks).** OpenCV on the depth image: anything standing above the table
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

## Zones and piles — all dynamic

Nothing about the layout is fixed in advance. Dump a pile of any size anywhere in the `search_region`:

1. **Find the pile.** On the first look the robot finds everything there and draws the **unsorted zone**
   around it (+ a margin for hidden or nudged items). The zone is then fixed for the run; only items
   inside it are picked, so sorted items are never re-picked.
2. **Make room.** The **sorted areas** are generated from the free table around that zone: strips on the
   open side, plus `front` (beyond the zone). Each is cut short so its far corner stays within reach.
3. **Assess + plan.** Classify everything → e.g. `{red: 2, blue: 2, green: 1}` → one pile per class, slot
   count from how many were seen, slot spacing from how big they are. **Reject** is reserved first.
4. **Sort.** Pick topmost → set it down in its pile's next slot → re-survey, until the zone is empty.
   A class first seen mid-run gets a new pile from the remaining space; a full pile grows an extension.

**How many pictures.** Each picture is a camera download (2–30 s depending on the link), so `--look` sets
how often one is taken. For four spread-out blocks:

| `--look` | Pictures | Behaviour |
|---|---|---|
| `once` | 1 | Every pick planned from a single picture. Fastest; nothing is re-checked, so an item that was hidden, nudged or dropped stays put. |
| `when_needed` (default) | 2 | One picture per batch. Looks again only after a failed grasp, after picking something that had a close neighbour, and once at the end to confirm the zone is empty. |
| `every_pick` | 5 | A new picture after every pick. Slowest, most careful — use it for heaps. |

**Any shape, any orientation.** Size, height and rotation are measured per item from its 3D points. The
grasp is chosen from the item's real footprint: across a part that is solid from edge to edge over a
finger's width, preferring the item's full width — so an arch is gripped across a leg, not through its
hollow. The red line in `data/debug/*.png` shows exactly where the fingers will close.

To pin things instead of `auto`, give `unsorted_zone` / `sorted_areas` explicit rectangles in
[config/workspace.yaml](config/workspace.yaml), or teach areas with `scripts/01_teach_pose.py --area NAME`.

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

## Resetting

```bash
python -m recycle_sorter.cli --action reset          # arm back to the survey pose (add --step to confirm first)
python -m recycle_sorter.cli --action open-gripper   # let go of whatever it is holding
```

A run that finishes returns there by itself; after a crash or Ctrl-C the arm stays where it stopped.
If it is holding a block, reset first and open the gripper over the table — not where it stopped mid-air.

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
  perception/  segment.py (objects from depth)  scan.py (eye-level crops, hidden cans)  viam_vision.py  pcd.py
               frames.py  select.py
  classify/    base.py (Protocol)  brand_claude.py  color_hsv.py  vision_label.py   ← one file per stage
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
