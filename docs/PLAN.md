# Recycle Sorter — Viam Hackathon Plan

## Context

Goal: a robot arm that picks items out of one unsorted pile and places them into sorted piles, growing in difficulty: **color → shape → material → brand (soda vs water)**. Items start as blocks/toys and move to mixed real recycling if time allows. Timeline is multi-day; code is Python.

What we have (from the machine config):
- Networked arm `arm` @ 192.168.1.210 (URDF kinematics, likely UFactory xArm/Lite6) + `gripper`
- Wrist-mounted Intel RealSense (serial 051122071371) — eye-in-hand, RGB + depth
- Workcell obstacle fragment (table z = −123 mm, walls x=740 / y=500, ceiling z=1050) — automatically included in every motion plan
- `code-1` generic service from the hackathon module (takes `arm`, `gripper`)
- Repo is empty (README only). Laptop: no Viam SDK/CLI yet; Python 3.14 default, `python3.12` available.

**Core design idea:** build ONE reliable pick-and-place loop, and make the *classifier* the only thing that changes between stages. Every stage is then a demo-able checkpoint, and harder stages never put the working demo at risk.

```
survey pose → capture RGB-D → segment objects (depth) → choose next object
   → Classifier.classify(crop) → SortPolicy: label → bin pose
   → pick (pre-grasp, linear descend, grab, verify, lift) → place → re-survey → repeat
```

## Approach

### Code strategy
Develop as a remote Python script from the laptop (API key + machine address) for fast iteration. All logic lives in an importable package `recycle_sorter/`; on the last day, wrap `run_sort()` into the `code-1` generic service as `do_command({"cmd": "sort", "mode": "color"})`.

### Environment
- `python3.12 -m venv .venv` (viam-sdk 0.80.0 installs on 3.14, but 3.12 is the safer, tested target)
- Deps: `viam-sdk`, `numpy`, `opencv-python`, `pillow`, `pyyaml`, `python-dotenv`, `pydantic`, `anthropic`, `pytest`, `pytest-asyncio`
- Secrets in `.env` (gitignored): `API_KEY`, `API_KEY_ID`, `ROBOT_ADDRESS`, `ANTHROPIC_API_KEY`
- Note: viam-sdk hard-pins `protobuf==6.33.5` — watch for conflicts when adding packages.

### Repo layout
```
pyproject.toml  .env.example  .gitignore (data/, .env, .venv)
config/   poses.yaml (home, survey, bins)  colors.yaml  bins.yaml  workspace.yaml
recycle_sorter/
  app.py            run_sort() main loop          cli.py   --mode --dry-run --step
  types.py          ObjectObservation, Classification
  io/               robot.py (LiveRobot)  replay.py (ReplayRobot)  recorder.py
  perception/       frames.py (deproject, cam→world)  segment.py  select.py
  classify/         base.py (Protocol)  color_hsv.py  shape_contour.py
                    vlm_claude.py  viam_ml.py  cascade.py
  manipulation/     motion.py  grasp.py  pickplace.py  safety.py  recovery.py
  policy/           sort_policy.py (label → bin, plus "unknown" bin)
  data/             viam_dataset.py (upload + tag crops for training)
scripts/  00_discover.py  01_teach_pose.py  02_hover_test.py
          03_record_frames.py  04_calib_yaw.py
module/   main.py (Generic service → run_sort)  meta.json  run.sh
tests/    test_segment_replay.py  test_classifiers.py  test_safety_bounds.py  fixtures/frames/
```

### Perception + pick pipeline (constant across stages)
1. **Survey**: `arm.move_to_joint_positions` to a taught top-down pose ~350–450 mm above the pile (RealSense min range ≈ 280 mm). Wait for `is_moving()` false + 0.3 s.
2. **Capture**: `camera.get_images()` → color + depth (`VIAM_RAW_DEPTH` → `bytes_to_depth_array()` → numpy); intrinsics from `get_properties()`. Depth is only pixel-aligned to color if the RealSense `align_color_depth` attribute is true — check on day 1; add via fragment mod if missing.
3. **Camera→world**: build a 4×4 numerically by calling `machine.transform_pose` on 4 camera-frame points (origin, +100 mm x/y/z). Avoids orientation-vector convention mistakes.
4. **Segment**: deproject depth → world; keep points with `z > table_top + 8 mm` inside the pile ROI; morphological open + connected components. Measure `table_top` empirically (median z of bare table) rather than trusting −123.
5. **Select**: topmost / most isolated / finger clearance; reject objects wider than gripper opening.
6. **Grasp pose**: top-down (`o_z = −1`), yaw `theta` from `cv2.minAreaRect` short axis + a once-calibrated offset; `z = max(top_z − finger_depth, table_top + 5)`.
7. **Execute** (`MotionClient.move` with `PoseInFrame`): open → pre-grasp (+100 mm, free plan) → descend with `LinearConstraint(line_tolerance_mm≈2)` → `grab()` → verify (`is_holding_something` / gripper position) → lift 150 mm → pre-place above bin → descend → `open()` → retreat → **re-survey every pick**.
8. **Recovery**: plan fail → retry theta+180 → higher pre-grasp → skip. Empty grasp ×2 at same spot → blacklist 40 mm radius. 3 consecutive failures → go home and stop.
9. **Safety**: low arm speed on day 1 (`do_command({"set_speed": 25})`); clamp all targets to workspace bounds in code before planning; `--step` mode (Enter before each move); SIGINT → `arm.stop()` + `machine.stop_all()`; timeouts on every RPC; bins added as `WorldState` obstacles.

### Classifier per stage (the only piece that swaps)
| Stage | Classifier | Why |
|---|---|---|
| 1 Color | `HSVColorClassifier` — median HSV/LAB inside each depth mask, nearest prototype from `colors.yaml` | One path for N colors, handles black/white/grey, works offline. Viam `color_detector` = one service per hue, boxes only, no greys. |
| 2 Shape | `ContourShapeClassifier` — `approxPolyDP` vertex count, circularity, aspect ratio, 3D height profile on the depth-derived silhouette | Lighting-independent; no training needed for blocks. Train only if toys are irregular. |
| 3 Material | **Hybrid**: `ClaudeVLMClassifier` (zero-shot on crop, structured output `{material, confidence}`, model `claude-sonnet-5`) → auto-upload every crop + label to a Viam dataset → human-correct labels → train TFLite `single_label_classification` → `ViamMLClassifier` → `CascadeClassifier` (ML if conf ≥ 0.8 else VLM) | Working demo on day 2, platform-native trained model by day 3. Training needs ≥15 images, ≥2 labels, can take 1 h+. |
| 4 Brand | VLM zero-shot with a **close-up look before the pick** (~250–300 mm above chosen object; wrist cam can't see the object once held). Schema `{category: soda_can\|water_bottle\|other, brand, text_seen, confidence}` | Flashy stretch demo without a custom detector. |

VLM latency is 2–5 s — run classification concurrently while the arm moves.

### Offline / no-robot development
`RobotIO` interface with `LiveRobot` and `ReplayRobot`. Every live snapshot saves RGB, depth `.npy`, intrinsics, cam→world matrix, and arm pose to `data/frames/<ts>/`. Perception and classifiers are developed and regression-tested against these frames, so teammates can work without the arm.

## Milestones

**Day 1 AM — bring-up**
- venv + deps; `.env`; `00_discover.py` prints `machine.resource_names`, frame system config, camera intrinsics → learn camera name, arm/gripper model, whether depth is aligned
- Gripper open/grab test; identify gripper type and max opening (decides which recycling items are feasible)
- `01_teach_pose.py`: record home, survey, bin poses into `poses.yaml`
- `02_hover_test.py`: hover TCP over a taped X detected by the camera → **pass = error ≤ 15 mm**, otherwise fix camera frame. Also settles whether `motion.move` should target `"arm"` or `"gripper"`.

**Day 1 PM — ✅ Checkpoint 1: color sort** of 6+ blocks into 2–3 piles
- Frame recorder, segmentation on saved frames, HSV classifier, first `--step` pick-and-place

**Day 2 AM — ✅ Checkpoint 2: shape sort**
- Clutter scoring, recovery logic, contour classifier, replay regression tests

**Day 2 PM — ✅ Checkpoint 3a: material sort via VLM**
- Claude classifier; crops auto-uploaded to a Viam dataset from now on

**Day 3 — ✅ Checkpoint 3b + stretch 4**
- Label-correct → train → deploy `tflite_cpu` + `mlmodel` vision service → cascade
- Brand / soda-vs-water close-up stage with real cans and bottles
- Wrap into `code-1` module `do_command`; simple status output (counts per bin, confusion log)
- **Record a backup demo video the moment each checkpoint works.**

## Risks
1. **Hand-eye calibration error** (top risk) → hover test gate on day 1 before anything else
2. Depth holes on shiny/clear items (cans, bottles) → RGB mask + neighborhood-median depth, or default height; depth-hole ratio is itself a useful material hint
3. Gripper stroke too small for full-size cans → pick items to suit (mini cans, crushed cans, caps) or grasp across the narrow axis
4. Linear constraint returns "no path" → widen tolerance to 2–5 mm; if gripper-vs-table collision blocks the descent, add a one-move `CollisionSpecification` allow
5. Training queue time → start dataset collection day 2, VLM path is the fallback demo

## Verification
- `pytest` on replay frames: object count, labels, centroid within tolerance; safety-bounds unit tests
- `--dry-run`: plans and logs moves, nothing executes; `--step`: human confirms each move
- Hover-test calibration gate (≤ 15 mm) before any grasp
- Per-stage metrics CSV: pick success rate, seconds/pick, classification confusion matrix from logged crops
- End-to-end per checkpoint: scatter N items, run `python -m recycle_sorter.cli --mode <color|shape|material|brand>`, confirm all land in the right pile with no human intervention
- After `pip install`, confirm SDK signatures with `help()` (e.g. whether `Camera.get_image` still exists alongside `get_images`)
