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
survey pose → RGB-D snapshot → segment (above table plane, inside pile ROI) → choose topmost
  → classify → label→bin → pick (pre-grasp, linear descend, grab, lift) → place → re-survey
```

Full staged plan, milestones, and risks: [docs/PLAN.md](docs/PLAN.md).

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # fill in from the Viam app: machine → CONNECT → API keys
.venv/bin/python -m pytest -q
```

## Day-1 bring-up (in order — each step gates the next)

| # | Command | Moves arm? | What you learn / must pass |
|---|---|---|---|
| 1 | `python scripts/00_discover.py` | no | Resource names, frame tree, camera streams. Color and depth must be the **same resolution** — if not, set `align_color_depth: true` on the RealSense (fragment mod). |
| 2 | Jog in the Viam app, then `python scripts/01_teach_pose.py home` and `... survey` | no | `survey` = camera pointing straight down, ~400 mm above the pile (RealSense can't see closer than ~280 mm). |
| 3 | Clear the table, go to survey, `python scripts/03_record_frames.py --table` | no | Prints the measured `table_top` → put it in `config/workspace.yaml`. |
| 4 | One block on the table: `python scripts/02_hover_test.py` | **yes**, Enter per move | **Gate: gripper hovers within 15 mm of the block.** Also gives `yaw_offset_deg`. If it fails, fix the camera frame / `move_frame` before anything else. |
| 5 | Jog over each pile spot: `python scripts/01_teach_pose.py --bin bin_a` (`bin_b`, `bin_c`, `reject`) | no | Drop locations. Keep them **outside** `pile_roi` so sorted items aren't re-picked. |
| 6 | `python -m recycle_sorter.cli --mode color --dry-run` | no | Full plan logged, nothing moves. Check `data/debug/*.png`. |
| 7 | `python -m recycle_sorter.cli --mode color --step` | **yes**, Enter per move | First real sort. Drop `--step` once it behaves, then raise `arm_speed`. |

Also measure the gripper's real max opening → `workspace.yaml` `gripper.max_open`. It decides
which recycling items are feasible later (a full-size can is ~66 mm).

## Working without the robot

Every survey saves RGB + depth + intrinsics + camera→world to `data/frames/<ts>/`.

```bash
python scripts/03_record_frames.py -n 10          # at the table: bank frames of varied piles
python -m recycle_sorter.cli --replay data/frames  # anywhere: perception + labels + overlays
```

## Config (everything tuned at the table lives here, not in code)

- `config/machine.yaml` — resource names, `move_frame`, arm speed
- `config/workspace.yaml` — `table_top`, safety bounds, pile ROI, gripper + pick parameters
- `config/sort.yaml` — label → bin per mode, color hues
- `config/poses.yaml` — written by `01_teach_pose.py`

Safety: every target is bounds-checked before it reaches the planner (`manipulation/safety.py`);
Ctrl-C stops the arm; the workcell fragment's walls/table/ceiling are in every motion plan.

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
