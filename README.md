# Hotel Robot — Aria

An AI-powered web application for controlling a physical hotel service robot through natural language and voice commands. A locally-hosted large language model (Qwen) acts as an intelligent agent that reasons step-by-step, issues robot commands, reads live status, and adapts to real-world outcomes.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Running the App](#running-the-app)
6. [Remote Access](#remote-access)
7. [Usage Guide](#usage-guide)
8. [How the Agent Works](#how-the-agent-works)
9. [Robot API Reference](#robot-api-reference)
10. [Web App API Reference](#web-app-api-reference)
11. [Configuration](#configuration)
12. [Deployment Workflow](#deployment-workflow)
13. [Troubleshooting](#troubleshooting)
14. [Known Limitations](#known-limitations)

---

## Overview

Aria is a single-file Flask application (`jing_app.py`) that connects three systems:

- **A local LLM** (Qwen 3.6 35B via MLX) that interprets commands and reasons about actions
- **A physical robot** (WATER/Shuidi chassis) controlled over HTTP
- **A web interface** with live status, mapping, camera feed, and a task queue

The user types or speaks a command like *"go to the front desk, wait ten seconds, then return to the charger."* The agent breaks this into actions, executes each against the robot, waits for physical completion, and moves to the next — all while streaming live updates to the browser.

---

## Architecture

```
┌─────────────┐     HTTP/SSE      ┌──────────────────────┐
│   Browser   │ ◄───────────────► │   Flask (jing_app)   │
│  (UI + JS)  │                   │                      │
└─────────────┘                   │  ┌────────────────┐  │
                                  │  │  Agent Loop    │  │
                                  │  │  (run_agent)   │  │
                                  │  └───────┬────────┘  │
                                  │          │           │
                    ┌─────────────┼──────────┼───────────┼─────────────┐
                    │             │           │           │             │
                    ▼             │           ▼           │             ▼
          ┌──────────────────┐    │  ┌─────────────────┐  │   ┌──────────────────┐
          │  Qwen LLM (MLX)  │    │  │  Robot HTTP API │  │   │  Camera (MJPEG)  │
          │  127.0.0.1:8080  │    │  │  10.1.17.225    │  │   │ webcam.urbot.ai  │
          └──────────────────┘    │  └─────────────────┘  │   └──────────────────┘
                                  └───────────────────────┘
```

**Request flow:**

1. Browser sends a command to Flask's `/command` endpoint
2. Flask launches the agent loop in a background thread
3. The agent asks Qwen what to do, given the command + live robot status
4. Qwen calls one tool (move, wait, charge, etc.)
5. Flask executes it against the robot API and polls until it physically completes
6. The real result is fed back to Qwen, which decides the next step
7. Live updates stream to the browser via Server-Sent Events (SSE)
8. Loop repeats until Qwen calls `finish()`

---

## Prerequisites

**Hardware**
- Mac Studio (or any machine capable of running the MLX model)
- The robot on the same reachable network (`10.1.17.225`)

**Software**
- Python 3.11+
- `conda` with an `mlx` environment
- The following Python packages: `flask`, `flask-cors`, `requests`, `openai`
- `mlx_lm` for serving the model
- `cloudflared` (optional, for external HTTPS access)
- Tailscale (optional, for cross-network access)

---

## Installation & Setup

### 1. Set up the Python environment

```bash
conda create -n mlx python=3.11
conda activate mlx
pip install flask flask-cors requests openai mlx-lm
```

### 2. Start the Qwen model server

The app expects an OpenAI-compatible model server on port 8080:

```bash
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-6bit --port 8080
```

Verify it's running:

```bash
curl http://127.0.0.1:8080/v1/models
```

You should see the model listed.

### 3. Place the app

Copy `jing_app.py` to the home directory (`~/jing_app.py`).

---

## Running the App

```bash
conda activate mlx
python3 ~/jing_app.py
```

You should see:

```
====================================================
  Hotel Robot — Aria
====================================================
  local   → http://localhost:5050
  network → http://10.1.17.202:5050
  tailnet → http://100.115.171.5:5050
====================================================
```

Open any of those URLs in a browser.

**To restart after changes:**

```bash
pkill -f "python3.*jing_app.py"
python3 ~/jing_app.py
```

---

## Remote Access

### Tailscale (recommended for the team)

Tailscale gives a permanent IP that works from any network.

```bash
ssh urbot@100.115.171.5
```

Once connected to the tailnet, open `http://100.115.171.5:5050` from anywhere.

Check the connection:

```bash
tailscale status
```

### Cloudflare Tunnel (for external HTTPS / voice)

For a temporary public HTTPS URL (needed for microphone access in some browsers):

```bash
cloudflared tunnel --url http://localhost:5050
```

This prints a `https://<random>.trycloudflare.com` URL. Note that it changes on every restart. If you see **Error 1033**, the tunnel process died — restart it.

---

## Usage Guide

### Chat commands

Type or speak natural language into the chat box. Examples:

| Command | What happens |
|---|---|
| `go to the front desk` | Robot navigates to Frontdesk |
| `go to the kitchen then come back to the charger` | Two-step sequence |
| `wait 30 seconds then go to the meeting room` | Timed sequence |
| `what's your battery level?` | Reports status |
| `stop` / `pause` | Halts current movement |
| `resume` / `continue` | Resumes paused tasks |

The agent understands loose phrasing — "head over to reception," "go charge," "take a break for a minute" all work.

### Location buttons

The left sidebar lists all navigation markers. Clicking one adds a move task to the queue. The button flashes green to confirm.

### Quick actions

- **Go home** — returns the robot to its charger
- **Cancel** — cancels the current task
- **Pause** — pauses the queue
- **Resume** — resumes the queue

### Task queue

The right panel shows every task with a live status badge:

- **PLANNED** — queued, waiting to run
- **IN PROGRESS** — currently executing (shows live position, battery, and the API being called)
- **DONE** — completed (animates out)
- **PAUSED** — waiting for resume
- **CANCELLED** — removed

### Live map & camera

Click **expand** on the mini-map to open a popup with three tabs:

- **Map** — full floor map with clickable markers
- **Camera** — live MJPEG video feed
- **Both** — split view

### Voice input

Click the microphone button and speak. In Chrome, if the mic is blocked on the non-HTTPS address, enable it once via:

```
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Add `http://100.115.171.5:5050`, set to **Enabled**, and relaunch. Alternatively, use the Cloudflare HTTPS URL where the mic works without any flags.

---

## How the Agent Works

The intelligence lives in `run_agent()`. Rather than pre-scripting every action, it hands control to Qwen and lets the model reason step by step.

### The loop

1. **Context** — Qwen receives the user's request plus a JSON snapshot of the robot's current state (battery, position, movement status)
2. **Decide** — Qwen calls exactly one tool
3. **Execute** — the chosen tool runs against the real robot; the app polls `/api/robot_status` until the action physically completes
4. **Report** — the real result (`{"success": true/false, "reason": ..., "robot_status": {...}}`) is fed back to Qwen
5. **Repeat** — Qwen reads the result and decides the next action, or calls `finish()`

### Agent tools

| Tool | Purpose |
|---|---|
| `move_to(marker)` | Navigate to a named location |
| `go_charge()` | Return to and dock at the charger |
| `wait(seconds)` | Pause before the next action |
| `check_status()` | Read battery, position, movement state |
| `cancel_move()` | Stop current movement |
| `finish(message)` | End the task with a summary to the user |

### Why this design

The agent interprets results itself. When a move succeeds, Qwen sees `success: true` and moves on. When it fails, Qwen decides whether to retry, try something else, or explain the problem to the user. This makes the system resilient to the robot's real-world quirks rather than brittle to them.

---

## Robot API Reference

All robot endpoints return JSON. Success is indicated by `code: 0` **or** the presence of a `task_id` in the results (the `/api/move` endpoint uses the latter).

Base: `http://10.1.17.225`

| Action | Method & Endpoint | Notes |
|---|---|---|
| Move to marker | `GET :9001/api/move?marker=<name>` | Returns `task_id` on success |
| Robot status | `GET :9001/api/robot_status` | Battery, pose, movement state |
| List markers | `GET :9001/api/markers/query_list` | All named locations |
| Return to charger | `POST :19001/api/tools/operation/task/go-back` | |
| Cancel task | `POST :19001/api/tools/operation/task/cancel` | |
| Lift cabin up | `GET :19001/api/tools/operation/lift/up` | |
| Lift cabin down | `POST :19001/api/tools/operation/lift/down` | |
| Cabin status | `GET :19001/api/tools/device/around` | Cleaning cabin battery/status |

**Key status fields** (from `/api/robot_status` → `results`):

- `power_percent` — battery level
- `running_status` — `idle` / `moving`
- `move_status` — `succeeded` / `failed` / `cancelled`
- `charge_state` — `true` when docked and charging
- `current_pose` — `{x, y, theta}`
- `move_target` — current destination
- `estop_state` — emergency stop active

### Known markers

**Navigation (Floor 1):** `Frontdesk`, `front_desk`, `Meetingroom`, `Kitchen`, `steakhouse`, `waiting`, `waiting1`, `Demotest`, `securitycheck`, `toReception`, `summon_point_5`, `destination`

**Chargers:** `charge_point_1F_40300423` (default), `charge_point_1F_40300165`, `charge_point_1F_1`

**Floor 2:** `point1`, `point2`, `point3`, `point4`

Marker names are **case-sensitive**. "front desk" maps to `Frontdesk`, not `frontdesk`.

---

## Web App API Reference

Endpoints served by Flask on port 5050:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/command` | POST | Send a chat command `{"text": "..."}` |
| `/status` | GET | Robot status for the sidebar |
| `/position` | GET | Robot position for the map |
| `/markers` | GET | List of navigation markers |
| `/stream` | GET | Server-Sent Events stream for live updates |
| `/tq/add` | POST | Add a task to the queue |
| `/tq/pause` | POST | Pause the queue |
| `/tq/resume` | POST | Resume the queue |
| `/tq/cancel_current` | POST | Cancel the in-progress task |
| `/tq/clear` | POST | Clear all tasks |
| `/tq/state` | GET | Current queue state |
| `/test_qwen` | GET | Diagnostic: is Qwen reachable? |
| `/test_move/<marker>` | GET | Diagnostic: raw move API response |

### Diagnostic endpoints

Two endpoints help debug without going through the full UI:

```bash
# Is the model responding?
curl http://100.115.171.5:5050/test_qwen

# What does a raw move return?
curl http://100.115.171.5:5050/test_move/Frontdesk
```

---

## Configuration

Key constants at the top of `jing_app.py`:

```python
QWEN_BASE_URL  = "http://127.0.0.1:8080/v1"
QWEN_MODEL     = "mlx-community/Qwen3.6-35B-A3B-6bit"

ROBOT_IP       = "10.1.17.225"
ROBOT_PORT     = 9001
TIMEOUT        = 8

DEFAULT_CHARGER_MARKER = "charge_point_1F_40300423"
```

Adjust `ROBOT_IP` if the robot's address changes. Adjust `QWEN_MODEL` to swap models (must be served on the same port).

---

## Deployment Workflow

**Always deploy with `scp`, never by copy-pasting into an editor.** Pasting a ~2000-line file into vi reliably introduces corruption.

From your laptop (a fresh terminal, not the SSH session):

```bash
scp ~/Downloads/jing_app.py urbot@100.115.171.5:~/jing_app.py
```

Then on the Mac Studio:

```bash
# Verify the file parses before running
python3 -c "import ast; ast.parse(open('/Users/urbot/jing_app.py').read()); print('Syntax OK')"

# Restart
pkill -f "python3.*jing_app.py"
python3 ~/jing_app.py
```

The syntax check catches corruption before you waste time on a broken run.

---

## Troubleshooting

**"Couldn't connect to server" on port 5050**
The app isn't running. Start it with `python3 ~/jing_app.py` and check for errors.

**404 on `/test_move` or other routes**
An old version of the file is running. Deploy the latest via `scp` and restart.

**Agent replies "On it!" but nothing happens**
The agent thread is crashing. Check the terminal for a traceback. A common cause is a syntax error from a corrupted file — run the syntax check above.

**"Could not move to X: unknown error" but the robot moves anyway**
The `/api/move` endpoint returns a `task_id` on success rather than `code: 0`. The app's `safe_json()` handles this — if you see this error, confirm you're running the latest version.

**Location stuck showing an old marker**
The `/status` endpoint only shows a target while actively moving. If it's stuck, the running file is outdated.

**Qwen returns empty responses**
Remove any `extra_body={"thinking": False}` parameters — this version of the MLX server handles them poorly. The current code does not use them.

**Verify the model is up:**

```bash
curl http://127.0.0.1:8080/v1/models
```

**Watch the agent reason** — the terminal prints its decisions:

```
[AGENT] === Step 1: asking Qwen ===
[AGENT] Qwen chose: move_to(Frontdesk)
[AGENT] Result: {"success": true, "reason": "arrived", ...}
[AGENT] === Step 2: asking Qwen ===
[AGENT] Qwen chose: go_charge()
[AGENT] FINISHED: ...
```

If Step 2 never appears, the model stopped early. If it appears but chooses `finish` prematurely, the model needs a stronger reminder of remaining steps.

---

## Known Limitations

- **Multi-step reliability** — the local model occasionally stops after the first step of a sequence. The agent reminds it of the original goal after each action, but this remains the hardest part of the system.
- **Floor cleaning / sweeping** — not yet implemented. Requires the cleaning cabin serial number and zone coordinates.
- **Voice input** — needs an HTTPS origin or a Chrome flag to work in all browsers.
- **Single robot** — the app controls one robot. Multi-robot support would need a robot registry and scheduler.
- **Development server** — runs on Flask's built-in server, which is fine for a demo but not for production traffic.

---

## Project Structure

Everything lives in one file, `jing_app.py`, organized top to bottom:

1. **Config** — model, robot, and charger constants
2. **SSE helpers** — `push_update`, `tq_push`
3. **Robot API functions** — `robot_move_to`, `robot_get_status`, etc.
4. **Task queue** — `_tq_runner` and helpers
5. **Agent** — `robot_snapshot`, `_execute_agent_tool`, `run_agent`, `handle_command`
6. **Flask routes** — UI, commands, status, task queue, diagnostics
7. **HTML/CSS/JS** — the full frontend as a string
8. **Main** — server startup

---

*Built for the Q-bay hotel robot project.*
