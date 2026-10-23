# RoboWeave: class-aware semantic navigation for a warehouse robot

Team **1hillToAnother** (Keesha, Shimsha Bhagavandas)

A simulated warehouse robot that decides *what* is in front of it before it decides *how* to react:

| It sees | It does |
|---|---|
| a **person** | stops and yields inside a 2 m zone, resumes when the way is clear |
| a **pillar / static object** | plans a path around it (A*) and keeps moving |
| something **unclear** | slows down, asks a vision-language model (VLM) in the background, and never waits for the answer |

Every decision is logged in plain English ("Person detected, yielding 2 m", "Pillar ahead, routing around").
It is compared against two lidar-only baselines that cannot tell objects apart, and watched through an operator dashboard.

Everything is simulated (2D warehouse, lidar, first-person camera). No real robot or real footage is used.

## Results

Fine-tuned YOLOv8n, 8 trials per cell (seed 0 is the nominal world; the other seeds change lidar noise and the walking person's start delay, speed and direction). Full tables are in `eval/EVAL.md` after you run the evaluation.

| | RoboWeave | Lidar-only: route around | Lidar-only: stop & wait |
|---|---|---|---|
| Collisions, scenarios with a person (`crossing`, `mixed`) | **0 of 16 trials** | 3 of 16 | 2 of 8 (all 8 `mixed` runs never finished) |
| Closest approach to a person | **1.47 m or more** | contact | contact |
| Finished `pillar`, `mixed`, `unclear` | 24 of 24 | 24 of 24 | **0 of 24** (stuck at the first pillar or crate) |
| Time, `crossing` | 25.8 s | 22.8 s | 23.1 s |
| Time, `mixed` | 26.5 s | 22.0 s | never finished |
| Time, `pillar` | 22.8 s | 21.6 s | never finished |
| Time, `unclear` | 39.3 s | 24.2 s | never finished |
| Person recall (tracker) | about 97% | | |

Read this honestly:

- RoboWeave is **not faster**. It trades roughly 3 to 4.5 s per person encounter for keeping its distance. Its time advantage is only over the policy that gets stuck.
- 0 of 16 against 3 of 16 collisions is suggestive, not statistically conclusive (one-sided p of about 0.11).
- Simulated people walk fixed routes and never avoid the robot, so a robot that stops in their way can be walked into.
- The `unclear` scenario costs +15 s because YOLO was never trained on a crate and calls it a person. The robot treats unknown objects as possible people: that costs time, not safety.
- The detector is fine-tuned on simulated frames, not real footage.
- A real VLM (Ollama, `gemma3:4b`) answers in about 2.5 s once warm and returns valid JSON, but on the simulator's flat-coloured frames it says "other / proceed", which is ignored. No speed-up from a real VLM is claimed. The `scripted` VLM is a simulator oracle for testing logic only.

## How it works

```
 lidar (360 beams) ─┐
                    ├─> perception ─> semantic grid ─> A* planner ─┐
 camera ─> YOLOv8n ─┘   (tracker,      (person zone,    (8-connected,├─> decision rules ─> motion
                         lidar fusion)  pillar margin,   shortened)  │   (yield / route /   (v, w)
                                        20 s memory)                 │    slow / stop)
                                                       VLM (non-blocking worker thread) ─┘
                                                                     └─> plain-English decision log
```

- **Perception** (`perception.py`): YOLOv8n boxes are tracked with greedy IoU matching and a per-class smoothed score with hysteresis, so labels do not flicker. Position comes from the lidar range at the box's bearing (monocular ground-contact fallback beyond lidar range). Velocity is only taken from lidar-sourced positions.
- **Semantic grid** (`grid.py`): 0.2 m cells. Static floor plan, plus unexplained lidar hits, plus labelled people and pillars. People are labelled but do not block planning; pillars get a safety margin. Unexplained objects stay on the map for 20 s unless a lidar beam sees through them.
- **Decision rules** (`decision.py`):
  - Yield if a remembered person is within 2.0 m of the robot or of the next 1.5 m of path (with hysteresis and a 2 s memory).
  - Emergency stop 0.15 m from the robot body for anything the map does not explain.
  - Weak labels with some "person" evidence are treated as people.
  - A person who has not moved for 8 s (or a mislabelled static object) is passed slowly at reduced speed.
  - Unclear objects within 6 m slow the robot to half speed and trigger one VLM question.
- **VLM** (`vlm.py`): runs in a worker thread with a 4 s timeout and cooldown. If it is slow or down, the robot stays cautious and carries on. It can only make the robot more careful about a person, or relax a weak label to "static". Safety rules never depend on it.
- **Baselines**: `box` (lidar only, routes around everything, 0.7 m passes) and `stop_all` (lidar only, stops within 1.5 m, plans on the floor plan only).

Scenarios (`scenarios.py`): `pillar` (static pillar in the aisle), `crossing` (a person walks across), `mixed` (both), `unclear` (a crate that is not on the floor plan and not a YOLO class).

## Free tools only

Python, pygame, NumPy, Pillow, Ultralytics YOLOv8, FastAPI, Uvicorn, Streamlit, Plotly, pandas. The optional VLM runs locally with [Ollama](https://ollama.com) (free, offline). Gemini is supported through `GEMINI_API_KEY` (free tier, rate limited, may use submitted data for training; only simulated frames are sent), but that backend has not been tested against the live API.

Licence note: YOLOv8 / Ultralytics is AGPL-3.0. Fine for a hackathon or research; a commercial product would need an Ultralytics licence or a permissive detector (RT-DETR, YOLOX).

## Setup (Windows PowerShell)

```powershell
cd "C:\vs code\roboweave-semantic"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install ultralytics
```

Keep the project out of OneDrive or other synced folders (slow image access). If you move the folder, delete and recreate `venv`.

## Run

All commands run from the project root with the venv active.

**1. Try the simulator by hand** (arrows/WASD drive, Space pauses people, Esc quits)
```powershell
python -m roboweave.demo
```

**2. Build the detector** (skip if `runs\detect\roboweave\weights\best.pt` already exists)
```powershell
python -m roboweave.make_dataset --train 1200 --val 250
python -m roboweave.train --epochs 25
python -m roboweave.yolo_check --model runs\detect\roboweave\weights\best.pt
```
CPU training is slow (about 25 epochs; it converged near epoch 11 here). Plug in the laptop and use a high-performance power mode.

**3. Drive one scenario with a policy**
```powershell
python -m roboweave.navigate --scenario crossing --mode roboweave --weights runs\detect\roboweave\weights\best.pt
python -m roboweave.navigate --scenario crossing --mode box       --weights runs\detect\roboweave\weights\best.pt
python -m roboweave.navigate --scenario crossing --mode stop_all  --weights runs\detect\roboweave\weights\best.pt
```
Scenarios: `pillar`, `crossing`, `mixed`, `unclear`. Add `--headless` for no window. Without `--weights` a noisy simulated detector stands in for YOLO (logic checks only, not a result).

**4. Compare every scenario and policy (one run each)**
```powershell
python -m roboweave.compare --weights runs\detect\roboweave\weights\best.pt
```

**5. Repeated-trial evaluation** (writes `eval\EVAL.md` and `eval\eval_results.json`)
```powershell
python -m roboweave.evaluate --weights runs\detect\roboweave\weights\best.pt --seeds 3
```
With real YOLO on a CPU this takes a long time (the 8-trial run took hours). Start with `--seeds 3`. `--reuse` re-aggregates saved runs without rerunning.

**6. Dashboard** (two terminals)
```powershell
# terminal 1, project root
python -m roboweave.server --weights runs\detect\roboweave\weights\best.pt
# terminal 2
cd scada_dashboard
streamlit run dashboard.py
```
Tabs: **Live run** (map with 2 m yield rings, robot camera with tracker boxes, what it believes, decision log, speed and person-gap charts), **Benchmark** (runs all scenarios under all policies), **Evaluation** (shows `eval\EVAL.md` results). Start the server from the project root so it finds `eval\`.

**7. Optional VLM**
```powershell
ollama pull gemma3:4b
ollama run gemma3:4b "hi"          # warm it up: the first load can take over a minute on CPU
python -m roboweave.vlm_live --model gemma3:4b
python -m roboweave.navigate --scenario unclear --mode roboweave --weights runs\detect\roboweave\weights\best.pt --vlm ollama --vlm-model gemma3:4b --vlm-timeout 20
```
`--vlm scripted` uses the simulator oracle instead (logic test only, never report it as a model result).

**8. Tests**
```powershell
python -m roboweave.test_perception --weights runs\detect\roboweave\weights\best.pt
python -m roboweave.test_vlm
python -m unittest tests.test_tracker
```

## Layout

```
roboweave/
  world.py  sensors.py  scenarios.py     simulator: world, lidar, camera, scenarios
  make_dataset.py  train.py  yolo_check.py   detector data, training, sanity check
  perception.py  grid.py  planner.py  decision.py   the robot's brain
  vlm.py  vlm_live.py                    VLM client and a one-call latency test
  navigate.py  compare.py                single run, one-run-per-cell table
  session.py  metrics.py  evaluate.py    reusable run, measurements, repeated trials
  server.py                              FastAPI + WebSocket sim server for the dashboard
scada_dashboard/
  dashboard.py  core/api.py  components/rw_charts.py  .streamlit/config.toml
tests/  eval/  (eval/ is generated)
```

The server exposes `GET /state`, `WS /ws`, `GET /frame.jpg`, `POST /control`, `GET|POST /compare`, `GET /eval`, `GET /health`. The dashboard polls over HTTP; `/ws` carries the same data for other clients.

The `src/`, `build/`, `benchmark.py`, `scada_dashboard/core/bridge.py`, `core/cad_parser.py` and `components/ui_charts.py` files left over from the original fleet-coordination project are not used by RoboWeave.

## Metric definitions (`roboweave/metrics.py`)

All metrics come from simulator ground truth, never from what the robot believes.

- **Collisions**: distinct contact events between the robot and anything. **Near misses**: separate episodes with a person gap under 0.5 m.
- **Moving inside 2 m person zone**: seconds per trial the robot moved while a real person was within 2.0 m in front of it (after 0.5 s). 0.0 also happens when the robot never met a person, so read it next to closest approach.
- **False yield**: seconds yielding with no real person within 3.6 m. **Stuck**: seconds in STOP or WAIT.
- **Person recall**: frames where a real person was visible within 8 m and the tracker had a person within 1 m of them. **False person tracks**: seconds per trial holding a person-class object with no real person near it.
- **Decisions logged**: state changes that produced a log line (a coverage check, since logging is built into each state change).

## Limitations and scale-up path

- One robot, one 24 x 14 m simulated warehouse, four hand-built scenarios.
- Detector trained on simulated frames; an untrained object class is treated as a possible person.
- People do not react to the robot. Real deployment needs real sensor noise, acceleration limits, and a validated detector on real footage.
- Scale-up: more object classes (crate, pallet, forklift), a detector trained on real warehouse images, and multiple robots sharing the semantic map.