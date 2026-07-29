"""
jing_app.py — Hotel Robot Web App
Mac Studio: conda activate mlx && python3 ~/jing_app.py
Browser:    http://100.115.171.5:5050
"""

# ── Imports ──────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
import json
import threading
import time as time_module
import queue
import sys
import re as re_module
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
QWEN_BASE_URL  = "http://127.0.0.1:8080/v1"
QWEN_API_KEY   = "not-needed"
QWEN_MODEL     = "mlx-community/Qwen3.6-35B-A3B-6bit"

ROBOT_IP       = "10.1.17.225"
ROBOT_PORT     = 9001
ROBOT_BASE_URL = f"http://{ROBOT_IP}:{ROBOT_PORT}"
TIMEOUT        = 8

DEFAULT_CHARGER_MARKER = "charge_point_1F_40300423"
DEFAULT_CHARGER_ID     = 40300423

app    = Flask(__name__)
CORS(app)
client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)

# ── SSE Queue ─────────────────────────────────────────────────────────────────
sse_queue = queue.Queue()

def push_update(reply, action=None):
    print(f"[SSE] {action}: {reply}", file=sys.stderr, flush=True)
    sse_queue.put({"reply": reply, "action": action, "background": True})

def tq_push(update_type, task_id=None, status=None, extra=None, detail=None, api=None):
    msg = {"tq_update": True, "update_type": update_type}
    if task_id is not None: msg["task_id"] = task_id
    if status   is not None: msg["status"]  = status
    if detail   is not None: msg["detail"]  = detail   # live status text
    if api      is not None: msg["api"]     = api      # API endpoint called
    if extra    is not None: msg.update(extra)
    print(f"[TQ] {update_type} task={task_id} status={status} {detail or ''}", file=sys.stderr, flush=True)
    sse_queue.put(msg)

# ── Robot API Functions ───────────────────────────────────────────────────────
def safe_json(r):
    if not r.content or r.content.strip() in (b"", b"null"):
        return {"code": 0 if r.status_code < 300 else -1,
                "status": "OK" if r.status_code < 300 else "ERROR",
                "message": "SUCCESS" if r.status_code < 300 else "ERROR"}
    try:
        data = r.json()

        # Success detection — this robot uses MULTIPLE response formats:
        # 1. {"code": 0} — standard success
        # 2. {"results": {"task_id": "..."}} — move accepted, task started (SUCCESS!)
        # 3. {"status": "OK"} — some endpoints
        has_task_id = bool(data.get("results", {}).get("task_id")) if isinstance(data.get("results"), dict) else False
        code_ok     = data.get("code", None) == 0
        # A move that returns a task_id has been ACCEPTED even if status says ERROR
        is_success  = code_ok or has_task_id

        data["status"] = "OK" if is_success else "ERROR"
        data["_accepted"] = has_task_id  # flag that a task was created

        if not data.get("customerErrorMessage") and data.get("error_message"):
            data["customerErrorMessage"] = data["error_message"]
        elif not data.get("customerErrorMessage") and data.get("message"):
            data["customerErrorMessage"] = data["message"]
        return data
    except Exception:
        return {"code": -1, "status": "ERROR", "raw": r.text[:200],
                "customerErrorMessage": f"HTTP {r.status_code}: {r.text[:100]}"}

def robot_get_status():
    r = requests.get(f"{ROBOT_BASE_URL}/api/robot_status", timeout=TIMEOUT)
    return r.json()

def robot_get_markers():
    r = requests.get(f"{ROBOT_BASE_URL}/api/markers/query_list", timeout=TIMEOUT)
    return r.json()

def robot_move_to(marker_name):
    r = requests.get(f"{ROBOT_BASE_URL}/api/move",
                     params={"marker": marker_name}, timeout=TIMEOUT)
    return safe_json(r)

def robot_cancel_task():
    r = requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/task/cancel",
                      json={}, timeout=TIMEOUT)
    return safe_json(r)

def robot_return_to_charger():
    r = requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/task/go-back",
                      json={}, timeout=TIMEOUT)
    return safe_json(r)

def robot_lift_up():
    r = requests.get(f"http://{ROBOT_IP}:19001/api/tools/operation/lift/up", timeout=TIMEOUT)
    return safe_json(r)

def robot_lift_down():
    r = requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/lift/down",
                      json={}, timeout=TIMEOUT)
    return safe_json(r)

def robot_get_cabin_status():
    r = requests.get(f"http://{ROBOT_IP}:19001/api/tools/device/around", timeout=TIMEOUT)
    return safe_json(r)

def parse_markers(api_response):
    if isinstance(api_response, dict) and "results" in api_response:
        return list(api_response["results"].keys())
    return api_response if isinstance(api_response, list) else []

def nav_markers(all_markers):
    return [m for m in all_markers
            if not any(m.startswith(p) for p in ("sweep_start_","map_","charge_point_"))]

# ── Task Queue ────────────────────────────────────────────────────────────────
task_queue        = []
task_queue_lock   = threading.Lock()
tq_runner_event   = threading.Event()
tq_cancel_event   = threading.Event()
tq_paused_by_user = [False]
tq_thread         = None
tq_id_counter     = [0]

def tq_new_id():
    tq_id_counter[0] += 1
    return tq_id_counter[0]

def _ensure_tq_runner():
    global tq_thread
    tq_runner_event.set()
    tq_cancel_event.clear()
    if tq_thread and tq_thread.is_alive():
        return
    tq_paused_by_user[0] = False
    tq_thread = threading.Thread(target=_tq_runner, daemon=True)
    tq_thread.start()

def _tq_runner():
    """Executes task queue in order. Each task completes before the next starts."""
    while True:
        # Only block if explicitly paused
        if not tq_runner_event.is_set():
            tq_runner_event.wait()

        with task_queue_lock:
            next_task = next(
                (t for t in task_queue if t["status"] in ("planned", "paused")), None)

        if not next_task:
            time_module.sleep(0.5)
            continue

        tid     = next_task["id"]
        ttype   = next_task["type"]
        success = False
        was_paused = False

        # Mark in-progress
        with task_queue_lock:
            for t in task_queue:
                if t["id"] == tid:
                    t["status"] = "in_progress"
        tq_push("status", task_id=tid, status="in_progress", detail="Starting...")
        tq_cancel_event.clear()

        try:
            # ── MOVE ──────────────────────────────────────────────────────────
            if ttype == "move":
                marker = next_task.get("marker", "")
                r   = requests.get(f"{ROBOT_BASE_URL}/api/move",
                                   params={"marker": marker}, timeout=TIMEOUT)
                res = safe_json(r)
                ok  = res.get("code", -1) == 0 or res.get("status") == "OK"
                if not ok:
                    err = res.get("customerErrorMessage", res.get("message", "unknown error"))
                    print(f"[MOVE] Failed: {res}", file=sys.stderr, flush=True)
                    push_update(f"Could not move to {marker}: {err}", "error")
                    # Task failed — will be removed so next task can run
                else:
                    push_update(f"Heading to {marker}...", "moving")
                    deadline = time_module.time() + 300
                    while time_module.time() < deadline:
                        # Cancel
                        if tq_cancel_event.is_set():
                            try: robot_cancel_task()
                            except Exception: pass
                            break
                        # Pause
                        if not tq_runner_event.is_set():
                            try: robot_cancel_task()
                            except Exception: pass
                            was_paused = True
                            with task_queue_lock:
                                for t in task_queue:
                                    if t["id"] == tid: t["status"] = "paused"
                            tq_push("status", task_id=tid, status="paused")
                            push_update(f"Paused on the way to {marker}.", "paused")
                            tq_runner_event.wait()
                            if not tq_cancel_event.is_set():
                                was_paused = False
                                with task_queue_lock:
                                    for t in task_queue:
                                        if t["id"] == tid: t["status"] = "in_progress"
                                tq_push("status", task_id=tid, status="in_progress")
                                push_update(f"Resuming move to {marker}...", "moving")
                                try:
                                    requests.get(f"{ROBOT_BASE_URL}/api/move",
                                                 params={"marker": marker}, timeout=TIMEOUT)
                                except Exception: pass
                            continue
                        time_module.sleep(2)
                        try:
                            sr   = requests.get(f"{ROBOT_BASE_URL}/api/robot_status", timeout=TIMEOUT)
                            rres = sr.json().get("results", {})
                            run_st = rres.get("running_status","")
                            mv_st  = rres.get("move_status","")
                            pose   = rres.get("current_pose",{})
                            x      = round(pose.get("x",0),1)
                            y      = round(pose.get("y",0),1)
                            bat    = rres.get("power_percent","?")
                            if run_st == "idle" and mv_st == "succeeded":
                                success = True
                                tq_push("status", task_id=tid, status="in_progress",
                                        detail=f"✅ Arrived! pos ({x},{y})", api=api_url)
                                push_update(f"Arrived at {marker}!", "arrived")
                                break
                            if mv_st in ("failed", "cancelled"):
                                tq_push("status", task_id=tid, status="in_progress",
                                        detail=f"❌ Move {mv_st}", api=api_url)
                                push_update(f"Move to {marker} failed.", "error")
                                break
                            # Still moving — show live position
                            tq_push("status", task_id=tid, status="in_progress",
                                    detail=f"🤖 Moving... pos ({x},{y}) bat:{bat}%", api=api_url)
                        except Exception:
                            pass

            # ── GO HOME ───────────────────────────────────────────────────────
            elif ttype == "go_home":
                go_api = f"POST /api/tools/operation/task/go-back (port 19001)"
                tq_push("status", task_id=tid, status="in_progress",
                        detail="Calling go-back API...", api=go_api)
                r   = requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/task/go-back",
                                    json={}, timeout=TIMEOUT)
                res = safe_json(r)
                ok  = res.get("code", -1) == 0 or res.get("status") == "OK"
                if not ok:
                    tq_push("status", task_id=tid, status="in_progress",
                            detail=f"❌ {res.get('message','error')}", api=go_api)
                    push_update(f"Go-home failed: {res.get('message','error')}", "error")
                else:
                    tq_push("status", task_id=tid, status="in_progress",
                            detail="Heading to charger...", api=go_api)
                    push_update("Heading to charging station...", "moving")
                    deadline = time_module.time() + 300
                    while time_module.time() < deadline:
                        if tq_cancel_event.is_set():
                            try: robot_cancel_task()
                            except Exception: pass
                            break
                        if not tq_runner_event.is_set():
                            try: robot_cancel_task()
                            except Exception: pass
                            was_paused = True
                            with task_queue_lock:
                                for t in task_queue:
                                    if t["id"] == tid: t["status"] = "paused"
                            tq_push("status", task_id=tid, status="paused")
                            push_update("Paused — will resume go-home when ready.", "paused")
                            tq_runner_event.wait()
                            if not tq_cancel_event.is_set():
                                was_paused = False
                                with task_queue_lock:
                                    for t in task_queue:
                                        if t["id"] == tid: t["status"] = "in_progress"
                                tq_push("status", task_id=tid, status="in_progress")
                                push_update("Resuming go-home...", "moving")
                                try:
                                    requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/task/go-back",
                                                  json={}, timeout=TIMEOUT)
                                except Exception: pass
                            continue
                        time_module.sleep(2)
                        try:
                            sr   = requests.get(f"{ROBOT_BASE_URL}/api/robot_status", timeout=TIMEOUT)
                            rres = sr.json().get("results", {})
                            bat  = rres.get("power_percent","?")
                            pose = rres.get("current_pose",{})
                            x    = round(pose.get("x",0),1)
                            y    = round(pose.get("y",0),1)
                            if rres.get("charge_state", False):
                                success = True
                                tq_push("status", task_id=tid, status="in_progress",
                                        detail=f"✅ Docked! bat:{bat}%", api=go_api)
                                push_update("Docked at charging station!", "arrived")
                                break
                            if rres.get("running_status") == "idle" and rres.get("move_status") == "succeeded":
                                success = True
                                tq_push("status", task_id=tid, status="in_progress",
                                        detail=f"✅ Home! pos ({x},{y})", api=go_api)
                                push_update("Back at home position.", "arrived")
                                break
                            tq_push("status", task_id=tid, status="in_progress",
                                    detail=f"🤖 Going home... pos ({x},{y}) bat:{bat}%", api=go_api)
                        except Exception:
                            pass

            # ── WAIT ──────────────────────────────────────────────────────────
            elif ttype == "wait":
                secs    = int(next_task.get("seconds", 5))
                elapsed = next_task.get("_elapsed", 0)
                wait_api = f"internal wait ({secs}s)"
                tq_push("status", task_id=tid, status="in_progress",
                        detail=f"⏳ Waiting {secs}s...", api=wait_api)
                push_update(f"Waiting {secs - elapsed}s...", "wait")
                while elapsed < secs:
                    if tq_cancel_event.is_set():
                        break
                    if not tq_runner_event.is_set():
                        was_paused = True
                        with task_queue_lock:
                            for t in task_queue:
                                if t["id"] == tid:
                                    t["status"]   = "paused"
                                    t["_elapsed"] = elapsed
                        tq_push("status", task_id=tid, status="paused")
                        push_update(f"Paused — {secs - elapsed}s remaining.", "paused")
                        tq_runner_event.wait()
                        if not tq_cancel_event.is_set():
                            was_paused = False
                            with task_queue_lock:
                                for t in task_queue:
                                    if t["id"] == tid: t["status"] = "in_progress"
                            tq_push("status", task_id=tid, status="in_progress")
                            push_update(f"Resuming wait — {secs - elapsed}s remaining...", "wait")
                        continue
                    time_module.sleep(1)
                    elapsed += 1
                    remaining = secs - elapsed
                    if remaining > 0 and secs <= 60:
                        push_update(f"⏳ {remaining}s remaining...", "wait")
                        tq_push("status", task_id=tid, status="in_progress",
                                detail=f"⏳ {remaining}s remaining...", api=wait_api)
                if not tq_cancel_event.is_set():
                    success = True
                    push_update("Done waiting.", "wait")
                    with task_queue_lock:
                        for t in task_queue:
                            if t["id"] == tid: t.pop("_elapsed", None)

            # ── STATUS ────────────────────────────────────────────────────────
            elif ttype == "status":
                status_api = "GET /api/robot_status (port 9001)"
                tq_push("status", task_id=tid, status="in_progress",
                        detail="Checking robot status...", api=status_api)
                sr  = requests.get(f"{ROBOT_BASE_URL}/api/robot_status", timeout=TIMEOUT)
                res = sr.json().get("results", {})
                bat = res.get("power_percent", "?")
                run = res.get("running_status", "unknown")
                chg = res.get("charge_state", False)
                detail_str = f"bat:{bat}% {'⚡' if chg else ''} {run}"
                tq_push("status", task_id=tid, status="in_progress",
                        detail=detail_str, api=status_api)
                push_update(f"Battery {bat}%, {'charging' if chg else 'not charging'}, {run}.", "get_robot_status")
                success = True

        except Exception as e:
            push_update(f"Task error: {str(e)}", "error")

        # ── Outcome ───────────────────────────────────────────────────────────
        if tq_cancel_event.is_set() and not was_paused:
            with task_queue_lock:
                task_queue[:] = [t for t in task_queue if t["id"] != tid]
            tq_push("status", task_id=tid, status="cancelled")
            if tq_paused_by_user[0]:
                with task_queue_lock:
                    nxt = next((t for t in task_queue if t["status"] == "planned"), None)
                    if nxt:
                        nxt["status"] = "paused"
                        tq_push("status", task_id=nxt["id"], status="paused")
        elif success:
            with task_queue_lock:
                for t in task_queue:
                    if t["id"] == tid: t["status"] = "completed"
            tq_push("status", task_id=tid, status="completed")
            time_module.sleep(0.8)
            with task_queue_lock:
                task_queue[:] = [t for t in task_queue if t["id"] != tid]
            tq_push("remove", task_id=tid)
        elif was_paused:
            pass  # stays as paused in queue
        else:
            # Failed without cancel/pause — remove so queue continues
            with task_queue_lock:
                task_queue[:] = [t for t in task_queue if t["id"] != tid]
            tq_push("status", task_id=tid, status="cancelled")

        tq_cancel_event.clear()
        if not tq_paused_by_user[0]:
            tq_runner_event.set()
        time_module.sleep(0.2)

# ── Sequence Planning ─────────────────────────────────────────────────────────
def _plan_steps_from_text(user_text):
    """
    Single Qwen call: read full command, output complete ordered step list.
    Returns list of step dicts or None.
    """
    known = ["Frontdesk","front_desk","Meetingroom","Kitchen","steakhouse",
             "waiting","waiting1","Demotest","securitycheck","toReception",
             "summon_point_5","destination"]
    try:
        r     = robot_get_markers()
        all_m = parse_markers(r)
        known = [m for m in all_m
                 if not any(m.startswith(p) for p in ("sweep_start_","map_"))
                 and not m.startswith("charge_point_")]
    except Exception:
        pass

    system = (
        "You are a robot task planner. Output ONLY a JSON array of tasks — no explanation, no markdown.\n"
        "Types:\n"
        '  {"type":"move","marker":"<exact>","label":"Go to <name>"}\n'
        '  {"type":"go_home","label":"Return to charger"}\n'
        '  {"type":"wait","seconds":<N>,"label":"Wait <N>s"}\n'
        '  {"type":"status","label":"Check status"}\n'
        f"Valid markers: {', '.join(known)}\n"
        "Rules:\n"
        "  front desk/reception -> Frontdesk\n"
        "  meeting room -> Meetingroom\n"
        "  kitchen -> Kitchen\n"
        "  restaurant/dining -> steakhouse\n"
        "  waiting area -> waiting\n"
        "  security -> securitycheck\n"
        "  home/charger/charge -> go_home type (NOT move)\n"
        "  1 minute = 60 seconds\n"
        "Output ONLY the JSON array. Example:\n"
        '[{"type":"move","marker":"Frontdesk","label":"Go to Frontdesk"},'
        '{"type":"wait","seconds":10,"label":"Wait 10s"},'
        '{"type":"go_home","label":"Return to charger"}]'
    )
    try:
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_text}
            ],
            max_tokens=500,
        )
        raw = resp.choices[0].message.content or ""
        m   = re_module.search(r'\[.*\]', raw, re_module.DOTALL)
        if not m:
            return None
        steps = json.loads(m.group())
        valid = []
        for s in steps:
            t = s.get("type", "")
            if t == "move" and s.get("marker"):
                valid.append({"type":"move","marker":s["marker"],
                              "label":s.get("label", f"Go to {s['marker']}")})
            elif t == "go_home":
                valid.append({"type":"go_home","label":s.get("label","Return to charger")})
            elif t == "wait" and s.get("seconds") is not None:
                secs = int(s["seconds"])
                valid.append({"type":"wait","seconds":secs,
                              "label":s.get("label", f"Wait {secs}s")})
            elif t == "status":
                valid.append({"type":"status","label":"Check status"})
        return valid or None
    except Exception as e:
        print(f"[PLAN] Error: {e}", file=sys.stderr, flush=True)
        return None

# ── Complexity Detection ──────────────────────────────────────────────────────
SEQUENCE_TRIGGERS = [
    "then", "after that", "wait", "and then", "come back",
    "return after", "go back after", "然后", "等一下", "等待", "分钟后", "之后", "接着",
]
TIME_WORDS = ["minute", "second", "hour", "分钟", "秒", "小时"]

def is_complex(text):
    t = text.lower().strip()
    if len(t.split()) <= 3 and len(t) <= 15:
        return False
    return any(k in t for k in SEQUENCE_TRIGGERS) or any(k in t for k in TIME_WORDS)

# ── Qwen Agent Tools ──────────────────────────────────────────────────────────
TOOLS = [
    {"type":"function","function":{"name":"move_to_marker",
     "description":"Move robot to a named location. front desk->Frontdesk, meeting room->Meetingroom, kitchen->Kitchen, restaurant->steakhouse, waiting->waiting, security->securitycheck.",
     "parameters":{"type":"object","properties":{"marker_name":{"type":"string"}},"required":["marker_name"]}}},
    {"type":"function","function":{"name":"return_to_charger",
     "description":"Send robot back to charging station. Use for: go home, go charge, return to base, dock.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"get_robot_status",
     "description":"Get robot battery, position, movement state.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"cancel_movement",
     "description":"Stop current movement.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"list_markers",
     "description":"List all named locations on the robot map.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"lift_cabin_up",
     "description":"Lift the cleaning cabin up.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"lift_cabin_down",
     "description":"Lower the cleaning cabin down.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"get_cabin_status",
     "description":"Get sweeping cabin status and battery.",
     "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"ask_clarification",
     "description":"Ask user to clarify when command is too vague.",
     "parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
]

TOOL_DISPATCH = {
    "move_to_marker":    lambda a: robot_move_to(a["marker_name"]),
    "return_to_charger": lambda a: robot_return_to_charger(),
    "get_robot_status":  lambda a: robot_get_status(),
    "cancel_movement":   lambda a: robot_cancel_task(),
    "list_markers":      lambda a: robot_get_markers(),
    "lift_cabin_up":     lambda a: robot_lift_up(),
    "lift_cabin_down":   lambda a: robot_lift_down(),
    "get_cabin_status":  lambda a: robot_get_cabin_status(),
}

SYSTEM_PROMPT = f"""You are Aria, an intelligent hotel service robot assistant and concierge.
You control a physical robot and respond warmly and professionally.

Your personality:
- Warm, professional, proactive — like a 5-star hotel concierge
- Intelligent: think carefully before acting, notice context
- Multilingual: always reply in the same language the user speaks
- Concise but warm — natural sentences, never robotic
- Proactive: warn if battery < 20% before long moves

EXACT marker names (case-sensitive, use these exactly):
  "Frontdesk" — front desk
  "front_desk" — alternate front desk
  "Meetingroom" — meeting room
  "Kitchen" — kitchen
  "steakhouse" — restaurant/dining
  "waiting" — waiting area
  "waiting1" — second waiting area
  "Demotest" — demo point
  "securitycheck" — security check
  "toReception" — reception corridor
  "summon_point_5" — summon point
  "destination" — delivery point
  "{DEFAULT_CHARGER_MARKER}" — charger (use return_to_charger, NOT move_to_marker)

Rules:
1. Single clear command → call tool immediately, confirm warmly in 1 sentence
2. Use conversation history for "try again", "repeat", "do that again"
3. Unclear → make reasonable inference, state assumption, act
4. Out of scope → decline warmly, offer what you CAN do
5. Never use move_to_marker for going home — use return_to_charger
6. Never navigate to sweep_start_ or map_ markers
7. Battery < 20% → warn user before long moves
8. Reply in whatever language the user speaks
"""

# ── Conversation History ──────────────────────────────────────────────────────
conversation_history = []
MAX_HISTORY = 10

def add_to_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY * 2:
        del conversation_history[:-MAX_HISTORY * 2]

def build_reply(name, args, robot_result):
    ok  = robot_result.get("code", -1) == 0 or robot_result.get("status") == "OK"
    res = robot_result.get("results", {})

    if name == "get_robot_status":
        bat     = res.get("power_percent", "?")
        running = res.get("running_status", "unknown")
        charging= res.get("charge_state", False)
        target  = res.get("move_target", "")
        estop   = res.get("estop_state", False)
        if estop:   return f"⚠️ Emergency stop active! Battery at {bat}%."
        if charging: return f"I'm charging at my station — battery at {bat}%. Ready when you need me!"
        if running == "idle" and not target:
            return f"I'm idle and ready. Battery at {bat}%."
        if running in ("moving","running") or target:
            return f"I'm on my way{f' to {target}' if target else ''}. Battery at {bat}%."
        return f"Status: {running}. Battery at {bat}%."

    elif name == "move_to_marker":
        dest = args.get("marker_name","destination")
        if ok:
            # Add to task queue display
            tid  = tq_new_id()
            task = {"id":tid,"type":"move","label":f"Go to {dest} (chat)",
                    "status":"in_progress","marker":dest}
            with task_queue_lock: task_queue.append(task)
            tq_push("add", task_id=tid, status="in_progress",
                    extra={"label":task["label"],"ttype":"move"})
            return f"Of course! Heading to {dest} now."
        else:
            err = robot_result.get("customerErrorMessage", robot_result.get("message","error"))
            return f"I couldn't navigate to {dest}. {err}"

    elif name == "return_to_charger":
        if ok:
            tid  = tq_new_id()
            task = {"id":tid,"type":"go_home","label":"Return to charger (chat)",
                    "status":"in_progress","marker":""}
            with task_queue_lock: task_queue.append(task)
            tq_push("add", task_id=tid, status="in_progress",
                    extra={"label":task["label"],"ttype":"go_home"})
            return "Of course! Heading back to my charging station now."
        else:
            return f"I couldn't return to the charger: {robot_result.get('message','error')}"

    elif name == "cancel_movement":
        msg = robot_result.get("customerErrorMessage", robot_result.get("message",""))
        if "不存在" in msg or "not exist" in msg.lower():
            return "Robot is already idle — nothing to cancel."
        return "Movement cancelled." if ok else f"Cancel failed: {msg}"

    elif name == "list_markers":
        nav = nav_markers(parse_markers(robot_result))
        return f"Available locations: {', '.join(nav)}."

    elif name == "lift_cabin_up":
        return "Lifting cabin up!" if ok else f"Lift failed: {robot_result.get('message','')}"

    elif name == "lift_cabin_down":
        return "Cabin lowered." if ok else f"Lower failed: {robot_result.get('message','')}"

    elif name == "get_cabin_status":
        devices = robot_result.get("data", [])
        cabins  = [d for d in devices if d.get("lastDeviceStatus") and d.get("type") in (2,3)]
        if cabins:
            parts = []
            for c in cabins:
                st  = c.get("lastDeviceStatus", {})
                pct = st.get("powerPercent","?")
                chg = st.get("isCharging", False)
                svc = st.get("serviceStatus","unknown")
                parts.append(f"{pct}% ({'charging' if chg else 'not charging'}), {svc}")
            return "Cabin: " + "; ".join(parts)
        return "No cabin devices found."

    return f"Done: {name}."

def run_single_step(user_text, messages_so_far):
    messages_so_far.append({"role":"user","content":user_text})
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages_so_far,
        tools=TOOLS,
        max_tokens=400,
    )
    message = response.choices[0].message
    messages_so_far.append({"role":"assistant","content": message.content or ""})
    add_to_history("user", user_text)
    if message.content: add_to_history("assistant", message.content)

    if not message.tool_calls:
        text = (message.content or "").strip()
        if not text:
            # retry once
            try:
                messages_so_far.append({"role":"user","content":"Please respond to my request."})
                retry = client.chat.completions.create(
                    model=QWEN_MODEL, messages=messages_so_far, tools=TOOLS, max_tokens=300)
                message = retry.choices[0].message
                text    = (message.content or "").strip()
            except Exception:
                pass
        if not message.tool_calls:
            return text or "I'm here — how can I help?", None, None, messages_so_far

    # Execute tool calls
    reply_text   = ""
    action_taken = None
    robot_result = None

    for call in (message.tool_calls or []):
        name = call.function.name
        args = json.loads(call.function.arguments) if call.function.arguments else {}

        if name == "ask_clarification":
            reply_text   = args.get("question","Could you clarify?")
            action_taken = "ask_clarification"
            continue

        action_taken = name
        if name not in TOOL_DISPATCH:
            reply_text = f"Unknown action: {name}"
            continue

        try:
            robot_result = TOOL_DISPATCH[name](args)
            reply_text   = build_reply(name, args, robot_result)
        except requests.exceptions.ConnectionError:
            reply_text   = "Cannot reach the robot — is it powered on?"
            robot_result = {"error":"connection"}
        except requests.exceptions.Timeout:
            reply_text   = "Robot timed out — it may be busy."
            robot_result = {"error":"timeout"}
        except Exception as e:
            reply_text   = f"Something went wrong: {str(e)}"
            robot_result = {"error":str(e)}

    return reply_text, action_taken, robot_result, messages_so_far

# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT AGENT — Qwen controls everything
# ══════════════════════════════════════════════════════════════════════════════

def robot_snapshot():
    """Return a clean dict of the robot's current state for the agent to read."""
    try:
        res  = robot_get_status().get("results", {})
        pose = res.get("current_pose", {})
        return {
            "battery_percent":  res.get("power_percent", "unknown"),
            "is_charging":      res.get("charge_state", False),
            "running_status":   res.get("running_status", "unknown"),
            "move_status":      res.get("move_status", "unknown"),
            "move_target":      res.get("move_target", "none"),
            "position":         {"x": round(pose.get("x", 0), 2),
                                 "y": round(pose.get("y", 0), 2)},
            "emergency_stop":   res.get("estop_state", False),
        }
    except Exception as e:
        return {"error": f"Could not read robot status: {e}"}


def wait_for_arrival(target_desc, cancel_event, timeout=300, expect_charge=False):
    """
    Poll robot status until it physically arrives/completes.
    Returns (success: bool, final_status: dict, reason: str).
    Pushes live position to the task queue via tq_push using the caller's tid.
    """
    deadline = time_module.time() + timeout
    last_pos = None
    still_count = 0

    while time_module.time() < deadline:
        if cancel_event.is_set():
            return False, robot_snapshot(), "cancelled by user"
        time_module.sleep(2)
        snap = robot_snapshot()
        if "error" in snap:
            continue

        run = snap["running_status"]
        mv  = snap["move_status"]
        pos = snap["position"]

        # Success conditions
        if expect_charge and snap["is_charging"]:
            return True, snap, "docked and charging"
        if run == "idle" and mv == "succeeded":
            return True, snap, "arrived successfully"

        # Failure conditions
        if mv in ("failed", "cancelled"):
            return False, snap, f"move {mv}"

        # Detect if stuck (position unchanged for a while while idle)
        if last_pos == pos and run == "idle":
            still_count += 1
            if still_count >= 4:  # 8 seconds idle without success flag
                # Idle + not moving + reached target area — treat as arrived
                return True, snap, "reached destination (idle)"
        else:
            still_count = 0
        last_pos = pos

        yield_live = f"pos ({pos['x']},{pos['y']}) bat:{snap['battery_percent']}%"
        globals()['_last_live'] = yield_live  # for the caller to read

    return False, robot_snapshot(), "timed out"


# Agent tool definitions — these map to real robot capabilities
AGENT_TOOLS = [
    {"type":"function","function":{
        "name":"move_to",
        "description":"Navigate the robot to a named marker location. Returns the robot status after arrival so you can verify success.",
        "parameters":{"type":"object","properties":{
            "marker":{"type":"string","description":"Exact marker name, e.g. Frontdesk, Kitchen, Meetingroom"}
        },"required":["marker"]}}},
    {"type":"function","function":{
        "name":"go_charge",
        "description":"Send the robot back to its charging station and dock. Returns status after docking.",
        "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{
        "name":"wait",
        "description":"Pause for a number of seconds before the next action.",
        "parameters":{"type":"object","properties":{
            "seconds":{"type":"number","description":"How many seconds to wait"}
        },"required":["seconds"]}}},
    {"type":"function","function":{
        "name":"check_status",
        "description":"Read the robot's current battery, position, and movement state.",
        "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{
        "name":"cancel_move",
        "description":"Stop the robot's current movement immediately.",
        "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{
        "name":"finish",
        "description":"Call this when the user's entire request has been completed, or if it cannot be completed. Provide a warm summary message for the user.",
        "parameters":{"type":"object","properties":{
            "message":{"type":"string","description":"Friendly summary to tell the user"}
        },"required":["message"]}}},
]


def _execute_agent_tool(name, args, cancel_event):
    """
    Execute one agent tool against the real robot.
    Returns a dict result that gets fed back to Qwen so IT can interpret it.
    Handles all task-queue UI updates.
    """
    if name == "move_to":
        marker = args.get("marker", "").strip()
        tid = tq_new_id()
        with task_queue_lock:
            task_queue.append({"id":tid,"type":"move","label":f"Go to {marker}",
                               "status":"in_progress","marker":marker})
        tq_push("add", task_id=tid, status="in_progress",
                extra={"label":f"Go to {marker}","ttype":"move"},
                api=f"GET /api/move?marker={marker}")

        # Send move command
        r   = requests.get(f"{ROBOT_BASE_URL}/api/move",
                           params={"marker":marker}, timeout=TIMEOUT)
        res = safe_json(r)
        accepted = res.get("status") == "OK" or res.get("_accepted") or \
                   bool(res.get("results",{}).get("task_id"))
        print(f"[AGENT] move_to({marker}) accepted={accepted} raw={res}", file=sys.stderr, flush=True)

        if not accepted:
            err = res.get("customerErrorMessage", "robot rejected the move")
            tq_push("status", task_id=tid, status="cancelled", detail=f"❌ {err}")
            time_module.sleep(0.4)
            with task_queue_lock:
                task_queue[:] = [t for t in task_queue if t["id"] != tid]
            tq_push("remove", task_id=tid)
            return {"success":False,"marker":marker,"reason":err,
                    "robot_status":robot_snapshot()}

        push_update(f"Heading to {marker}...", "moving")

        # Poll until arrival
        deadline = time_module.time() + 300
        last_pos = None; still = 0
        while time_module.time() < deadline:
            if cancel_event.is_set():
                return {"success":False,"marker":marker,"reason":"cancelled",
                        "robot_status":robot_snapshot()}
            time_module.sleep(2)
            snap = robot_snapshot()
            if "error" in snap: continue
            pos = snap["position"]
            tq_push("status", task_id=tid, status="in_progress",
                    detail=f"🤖 pos({pos['x']},{pos['y']}) bat:{snap['battery_percent']}%",
                    api=f"GET /api/move?marker={marker}")
            if snap["running_status"]=="idle" and snap["move_status"]=="succeeded":
                tq_push("status", task_id=tid, status="completed", detail=f"✅ Arrived at {marker}")
                time_module.sleep(0.5)
                with task_queue_lock:
                    task_queue[:] = [t for t in task_queue if t["id"] != tid]
                tq_push("remove", task_id=tid)
                push_update(f"Arrived at {marker}!", "arrived")
                return {"success":True,"marker":marker,"reason":"arrived",
                        "robot_status":snap}
            if snap["move_status"] in ("failed","cancelled"):
                tq_push("status", task_id=tid, status="cancelled", detail=f"❌ {snap['move_status']}")
                time_module.sleep(0.4)
                with task_queue_lock:
                    task_queue[:] = [t for t in task_queue if t["id"] != tid]
                tq_push("remove", task_id=tid)
                return {"success":False,"marker":marker,"reason":snap["move_status"],
                        "robot_status":snap}
            # stuck detection
            if last_pos==pos and snap["running_status"]=="idle":
                still += 1
                if still >= 4:
                    tq_push("status", task_id=tid, status="completed", detail=f"✅ Reached {marker}")
                    time_module.sleep(0.5)
                    with task_queue_lock:
                        task_queue[:] = [t for t in task_queue if t["id"] != tid]
                    tq_push("remove", task_id=tid)
                    push_update(f"Reached {marker}.", "arrived")
                    return {"success":True,"marker":marker,"reason":"reached (idle)",
                            "robot_status":snap}
            else:
                still = 0
            last_pos = pos

        # timeout
        with task_queue_lock:
            task_queue[:] = [t for t in task_queue if t["id"] != tid]
        tq_push("remove", task_id=tid)
        return {"success":False,"marker":marker,"reason":"timed out",
                "robot_status":robot_snapshot()}

    elif name == "go_charge":
        tid = tq_new_id()
        with task_queue_lock:
            task_queue.append({"id":tid,"type":"go_home","label":"Return to charger",
                               "status":"in_progress","marker":""})
        tq_push("add", task_id=tid, status="in_progress",
                extra={"label":"Return to charger","ttype":"go_home"},
                api="POST /api/tools/operation/task/go-back")

        r = requests.post(f"http://{ROBOT_IP}:19001/api/tools/operation/task/go-back",
                          json={}, timeout=TIMEOUT)
        res = safe_json(r)
        print(f"[AGENT] go_charge raw={res}", file=sys.stderr, flush=True)
        push_update("Heading to charging station...", "moving")

        deadline = time_module.time() + 300
        while time_module.time() < deadline:
            if cancel_event.is_set():
                return {"success":False,"reason":"cancelled","robot_status":robot_snapshot()}
            time_module.sleep(2)
            snap = robot_snapshot()
            if "error" in snap: continue
            pos = snap["position"]
            tq_push("status", task_id=tid, status="in_progress",
                    detail=f"🤖 pos({pos['x']},{pos['y']}) bat:{snap['battery_percent']}%",
                    api="POST .../task/go-back")
            if snap["is_charging"]:
                tq_push("status", task_id=tid, status="completed", detail="✅ Docked & charging")
                time_module.sleep(0.5)
                with task_queue_lock:
                    task_queue[:] = [t for t in task_queue if t["id"] != tid]
                tq_push("remove", task_id=tid)
                push_update("Docked at charging station!", "arrived")
                return {"success":True,"reason":"docked","robot_status":snap}
            if snap["running_status"]=="idle" and snap["move_status"]=="succeeded":
                tq_push("status", task_id=tid, status="completed", detail="✅ Home")
                time_module.sleep(0.5)
                with task_queue_lock:
                    task_queue[:] = [t for t in task_queue if t["id"] != tid]
                tq_push("remove", task_id=tid)
                push_update("Back home.", "arrived")
                return {"success":True,"reason":"home","robot_status":snap}

        with task_queue_lock:
            task_queue[:] = [t for t in task_queue if t["id"] != tid]
        tq_push("remove", task_id=tid)
        return {"success":False,"reason":"timed out","robot_status":robot_snapshot()}

    elif name == "wait":
        secs = int(args.get("seconds", 5))
        tid = tq_new_id()
        with task_queue_lock:
            task_queue.append({"id":tid,"type":"wait","label":f"Wait {secs}s",
                               "status":"in_progress","seconds":secs})
        tq_push("add", task_id=tid, status="in_progress",
                extra={"label":f"Wait {secs}s","ttype":"wait"})
        push_update(f"Waiting {secs}s...", "wait")
        for i in range(secs):
            if cancel_event.is_set():
                return {"success":False,"reason":"cancelled"}
            remaining = secs - i
            tq_push("status", task_id=tid, status="in_progress", detail=f"⏳ {remaining}s left")
            if remaining <= 30:
                push_update(f"⏳ {remaining}s remaining...", "wait")
            time_module.sleep(1)
        tq_push("status", task_id=tid, status="completed", detail="✅ Done")
        time_module.sleep(0.4)
        with task_queue_lock:
            task_queue[:] = [t for t in task_queue if t["id"] != tid]
        tq_push("remove", task_id=tid)
        push_update("Done waiting.", "wait")
        return {"success":True,"waited_seconds":secs}

    elif name == "check_status":
        snap = robot_snapshot()
        push_update(f"Battery {snap.get('battery_percent')}%, {snap.get('running_status')}.", "get_robot_status")
        return {"success":True,"robot_status":snap}

    elif name == "cancel_move":
        try:
            robot_cancel_task()
        except Exception: pass
        return {"success":True,"reason":"movement cancelled","robot_status":robot_snapshot()}

    return {"success":False,"reason":f"unknown tool {name}"}


def run_agent(user_text, cancel_event):
    """
    The intelligent agent. Qwen reads the command, decides actions one at a time,
    reads the REAL robot response after each, and decides what to do next.
    Qwen interprets success/failure itself — we just report facts.
    """
    print(f"[AGENT] >>> run_agent STARTED for: {user_text}", file=sys.stderr, flush=True)
    print(f"[AGENT] >>> cancel_event.is_set()={cancel_event.is_set()}", file=sys.stderr, flush=True)
    # Make sure we start fresh — clear any leftover cancel flag
    cancel_event.clear()

    known = ["Frontdesk","front_desk","Meetingroom","Kitchen","steakhouse",
             "waiting","waiting1","Demotest","securitycheck","toReception",
             "summon_point_5","destination"]
    try:
        all_m = parse_markers(robot_get_markers())
        known = [m for m in all_m
                 if not any(m.startswith(p) for p in ("sweep_start_","map_"))
                 and not m.startswith("charge_point_")]
    except Exception:
        pass

    marker_list = ", ".join(known)
    system = (
        "You are Aria, an intelligent hotel service robot. You control a real physical robot and decide its actions.\n\n"
        "You work step by step:\n"
        "1. Read the user's request and the current robot status\n"
        "2. Call ONE tool to take the next action\n"
        "3. Read the tool's result (which includes the real robot status)\n"
        "4. Decide the next action, or call finish() when the whole request is done\n\n"
        "You interpret results yourself:\n"
        '- Each tool returns a JSON result with "success" (true/false), "reason", and "robot_status"\n'
        "- If success is true, the action worked — move on to the next step\n"
        "- If success is false, decide whether to retry, try something else, or finish with an explanation\n"
        "- Trust the robot_status position and battery to understand what's happening\n\n"
        "Available markers (use exact names): " + marker_list + "\n\n"
        "Interpretation rules:\n"
        '- "front desk" = Frontdesk, "meeting room" = Meetingroom, "restaurant"/"dining" = steakhouse\n'
        '- "home"/"charge"/"charger" = use go_charge (NOT move_to)\n'
        '- "security" = securitycheck\n'
        '- Convert time: "1 minute" = wait(60)\n\n'
        "Be warm and concise in your finish() message. Reply in the user's language.\n"
        "Always call finish() at the end with a friendly summary."
    )

    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"Request: {user_text}\n\nCurrent robot status: {json.dumps(robot_snapshot())}"}
    ]

    for step in range(20):
        if cancel_event.is_set():
            push_update("Okay, I've stopped.", "cancelled")
            return

        print(f"[AGENT] === Step {step+1}: asking Qwen ===", file=sys.stderr, flush=True)
        try:
            resp = client.chat.completions.create(
                model=QWEN_MODEL, messages=messages,
                tools=AGENT_TOOLS, max_tokens=400)
            msg = resp.choices[0].message
        except Exception as e:
            print(f"[AGENT] Qwen error: {e}", file=sys.stderr, flush=True)
            push_update("I had trouble thinking that through. Could you try again?", "error")
            return

        # No tool call — Qwen is talking, treat as finish
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            if text:
                push_update(text, "done")
                add_to_history("assistant", text)
            else:
                # empty — nudge once
                messages.append({"role":"user","content":"Please continue — call a tool or finish()."})
                continue
            return

        call = msg.tool_calls[0]
        name = call.function.name
        try:
            args = json.loads(call.function.arguments) if call.function.arguments else {}
        except Exception:
            args = {}

        print(f"[AGENT] Qwen chose: {name}({args})", file=sys.stderr, flush=True)

        # Record Qwen's decision in the message history
        messages.append({
            "role":"assistant","content":msg.content or "",
            "tool_calls":[{"id":call.id,"type":"function",
                           "function":{"name":name,"arguments":call.function.arguments or "{}"}}]
        })

        # finish() ends the loop
        if name == "finish":
            final = args.get("message","All done!")
            push_update(final, "done")
            add_to_history("assistant", final)
            print(f"[AGENT] FINISHED: {final}", file=sys.stderr, flush=True)
            return

        # Execute the tool against the real robot
        result = _execute_agent_tool(name, args, cancel_event)
        print(f"[AGENT] Result: {json.dumps(result)[:150]}", file=sys.stderr, flush=True)

        # Feed the REAL result back to Qwen, and remind it of the original goal
        # so it continues with remaining steps instead of stopping early
        result_msg = json.dumps(result)
        reminder = (
            f" [Original request was: '{user_text}'. "
            f"If there are still steps remaining in that request, call the next tool now. "
            f"Only call finish() when EVERYTHING in the original request is complete.]"
        )
        messages.append({
            "role":"tool","tool_call_id":call.id,
            "content": result_msg + reminder
        })

    push_update("I've done everything I can for that request!", "done")


def handle_command(user_text):
    t_lower = user_text.lower().strip()
    add_to_history("user", user_text)

    # Immediate control commands (don't go through agent)
    if any(k in t_lower for k in ["stop","pause","halt","暂停","停止"]) and len(t_lower.split()) <= 3:
        tq_cancel_event.set()
        try: robot_cancel_task()
        except Exception: pass
        try: requests.post("http://127.0.0.1:5050/tq/pause", timeout=2)
        except Exception: pass
        reply = "Stopped. Let me know when you'd like to continue."
        add_to_history("assistant", reply)
        return reply, "cancelled", {}

    if any(k in t_lower for k in ["resume","continue","继续"]) and len(t_lower.split()) <= 3:
        try: requests.post("http://127.0.0.1:5050/tq/resume", timeout=2)
        except Exception: pass
        reply = "Resuming!"
        add_to_history("assistant", reply)
        return reply, "resumed", {}

    # EVERYTHING else goes through the intelligent agent
    print(f"[AGENT] New request: {user_text}", file=sys.stderr, flush=True)
    tq_cancel_event.clear()
    thread = threading.Thread(target=run_agent, args=(user_text, tq_cancel_event), daemon=True)
    thread.start()
    return "On it! Working through your request now.", "sequence_started", {}


# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML

@app.route("/command", methods=["POST"])
def command():
    data = request.json
    text = data.get("text","").strip()
    if not text: return jsonify({"error":"empty"}), 400
    reply, action, result = handle_command(text)
    return jsonify({"reply":reply,"action":action,"robot_result":result})

@app.route("/test_qwen")
def test_qwen():
    """Test if Qwen is reachable and responding."""
    try:
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role":"user","content":"Say 'hello' and nothing else."}],
            max_tokens=20)
        return jsonify({"ok":True,"reply":resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/test_move/<marker>")
def test_move(marker):
    """Directly test a move API call, bypassing the agent."""
    try:
        r = requests.get(f"{ROBOT_BASE_URL}/api/move", params={"marker":marker}, timeout=TIMEOUT)
        return jsonify({"ok":True,"raw":r.json(),"parsed":safe_json(r)})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/status")
def status():
    try:
        res  = robot_get_status().get("results",{})
        pose = res.get("current_pose",{})
        running = res.get("running_status","unknown")
        target  = res.get("move_target","")
        charging = res.get("charge_state",False)
        # Only show target as location when actually moving toward it
        if charging:
            location = "Charging station"
        elif running in ("moving","running") and target:
            location = target
        else:
            location = "—"
        return jsonify({
            "ok":True,"battery":res.get("power_percent","?"),
            "charging":charging,
            "moving":running,
            "move_status":res.get("move_status",""),
            "move_target":target,
            "location":location,
            "estop":res.get("estop_state",False),
            "x":pose.get("x"),"y":pose.get("y"),"theta":pose.get("theta"),
        })
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/position")
def position():
    try:
        res  = robot_get_status().get("results",{})
        pose = res.get("current_pose",{})
        return jsonify({
            "ok":True,"x":pose.get("x"),"y":pose.get("y"),"theta":pose.get("theta"),
            "running":res.get("running_status","idle"),
            "move_target":res.get("move_target",""),
            "move_status":res.get("move_status",""),
        })
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/markers")
def markers():
    try:
        r     = robot_get_markers()
        all_m = parse_markers(r)
        return jsonify({"ok":True,"nav":nav_markers(all_m),
                        "charge":[m for m in all_m if m.startswith("charge_point_")]})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route("/stream")
def stream():
    def event_stream():
        while True:
            try:
                msg = sse_queue.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield 'data: {"ping":true}\n\n'
    return Response(stream_with_context(event_stream()),
                    content_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Task Queue Routes ─────────────────────────────────────────────────────────
@app.route("/tq/add", methods=["POST"])
def tq_add():
    data  = request.json
    ttype = data.get("type","move")
    label = data.get("label","")
    tid   = tq_new_id()
    task  = {"id":tid,"type":ttype,"label":label,"status":"planned",
             "marker":data.get("marker",""),"seconds":data.get("seconds",0)}
    with task_queue_lock: task_queue.append(task)
    tq_push("add", task_id=tid, status="planned", extra={"label":label,"ttype":ttype})
    _ensure_tq_runner()
    return jsonify({"ok":True,"task_id":tid})

@app.route("/tq/pause", methods=["POST"])
def tq_pause_route():
    tq_runner_event.clear()
    tq_paused_by_user[0] = True
    tq_push("paused_all")
    return jsonify({"ok":True})

@app.route("/tq/resume", methods=["POST"])
def tq_resume_route():
    tq_paused_by_user[0] = False
    tq_cancel_event.clear()
    tq_runner_event.set()
    tq_push("resumed")
    _ensure_tq_runner()
    return jsonify({"ok":True})

@app.route("/tq/cancel_current", methods=["POST"])
def tq_cancel_current():
    try: robot_cancel_task()
    except Exception: pass
    tq_cancel_event.set()
    return jsonify({"ok":True})

@app.route("/tq/clear", methods=["POST"])
def tq_clear():
    with task_queue_lock: task_queue.clear()
    tq_cancel_event.set()
    tq_push("cleared")
    return jsonify({"ok":True})

@app.route("/tq/state")
def tq_state():
    with task_queue_lock:
        return jsonify({"ok":True,"tasks":list(task_queue),"paused":tq_paused_by_user[0]})

@app.route("/cancel_sequence", methods=["POST"])
def cancel_sequence():
    try: robot_cancel_task()
    except Exception: pass
    with task_queue_lock: task_queue.clear()
    tq_cancel_event.set()
    tq_push("cleared")
    return jsonify({"ok":True})

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Hotel Robot</title>
<style>
:root{--g:#2E7D4F;--gd:#1B5C38;--gl:#EAF7EE;--glb:#C8EDCF;
  --bg:#F5F7F5;--s1:#FFFFFF;--s2:#F0F4F1;--bd:#D8E4DA;
  --tx:#1a2e1f;--tx2:#4a6655;--txm:#7a9685;--ra:8px;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--tx);height:100vh;overflow:hidden;display:flex;flex-direction:column}
.topbar{height:48px;background:var(--s1);border-bottom:1px solid var(--bd);display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0}
.robot-icon{width:28px;height:28px;background:var(--g);border-radius:var(--ra);display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.topbar-title{font-size:14px;font-weight:600}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.status-pill{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2);background:var(--s2);border:1px solid var(--bd);border-radius:99px;padding:3px 10px}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--g);flex-shrink:0}
.sdot.off{background:#ef4444}
.state-bar{padding:6px 16px;font-size:12px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--bd);transition:all .3s;display:none}
.state-bar.idle{background:var(--gl);color:var(--g);border-color:var(--glb)}
.state-bar.moving{background:#E6F1FB;color:#185FA5;border-color:#B5D4F4}
.state-bar.charging{background:var(--gl);color:var(--g);border-color:var(--glb)}
.state-bar.error{background:#FCEBEB;color:#A32D2D;border-color:#F7C1C1}
.state-bar.arrived{background:var(--gl);color:var(--g);border-color:var(--glb)}
.layout{display:grid;grid-template-columns:230px 1fr 260px;flex:1;overflow:hidden}
.left{border-right:1px solid var(--bd);background:var(--s1);display:flex;flex-direction:column;overflow:hidden}
.right{border-left:1px solid var(--bd);background:var(--s1);display:flex;flex-direction:column;overflow:hidden}
.middle{display:flex;flex-direction:column;overflow:hidden}
.sec-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--txm);padding:12px 14px 5px;font-weight:600;flex-shrink:0}
.stat-card{background:var(--s2);border:1px solid var(--bd);border-radius:var(--ra);padding:8px 10px;margin:0 12px 5px}
.bat-top{display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px}
.bat-bar{height:4px;background:var(--bd);border-radius:2px;overflow:hidden}
.bat-fill{height:100%;background:var(--g);border-radius:2px;transition:width .5s}
.bat-fill.low{background:#ef4444}.bat-fill.med{background:#f59e0b}
.srow{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--tx2);padding:3px 0}
.divider{height:1px;background:var(--bd);margin:3px 0}
.sval{color:var(--tx);font-weight:500;font-size:11px}
.nav-list{padding:0 10px;display:flex;flex-direction:column;gap:2px;overflow-y:auto;flex:1}
.nav-btn{padding:7px 10px;border-radius:6px;font-size:12px;color:var(--tx2);cursor:pointer;display:flex;align-items:center;gap:8px;border:1px solid transparent;background:none;text-align:left;width:100%;transition:all .12s}
.nav-btn:hover{background:var(--s2);border-color:var(--g);color:var(--g)}
.qa-grid{padding:6px 10px 10px;display:grid;grid-template-columns:1fr 1fr;gap:5px;flex-shrink:0}
.qa-btn{padding:7px 6px;border-radius:6px;border:1px solid var(--bd);background:var(--s2);font-size:11px;color:var(--tx2);cursor:pointer;text-align:center;transition:all .12s;line-height:1.3}
.qa-btn:hover{border-color:var(--g);color:var(--g);background:rgba(46,125,79,.08)}
.qa-btn.danger:hover{border-color:#ef4444;color:#ef4444;background:rgba(239,68,68,.08)}
.msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.mwrap{display:flex;gap:8px;align-items:flex-start}
.mwrap.user{flex-direction:row-reverse}
.av{width:28px;height:28px;border-radius:50%;background:var(--s2);border:1px solid var(--bd);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;color:var(--tx2)}
.mwrap.user .av{background:var(--g);border-color:var(--g);color:#fff;font-size:11px;font-weight:600}
.bubble{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:8px 12px;font-size:13px;line-height:1.5;max-width:82%}
.mwrap.user .bubble{background:var(--gl);border-color:var(--glb);color:#1a2e1f}
.tag{font-size:10px;color:var(--txm);margin-top:4px;display:flex;align-items:center;gap:3px}
.tag-dot{width:5px;height:5px;border-radius:50%;background:var(--g);flex-shrink:0}
.typing-bub span{width:6px;height:6px;border-radius:50%;background:var(--g);opacity:.5;animation:bop 1.2s infinite;display:inline-block;margin:0 2px}
.typing-bub span:nth-child(2){animation-delay:.2s}.typing-bub span:nth-child(3){animation-delay:.4s}
@keyframes bop{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}
.input-area{border-top:1px solid var(--bd);background:var(--s1);padding:10px 14px;display:flex;flex-direction:column;gap:6px;flex-shrink:0}
.voice-hint{font-size:11px;color:var(--txm);min-height:15px}
.input-row{display:flex;gap:8px;align-items:flex-end}
#txt{flex:1;background:var(--s2);border:1px solid var(--bd);border-radius:var(--ra);padding:8px 12px;color:var(--tx);font-size:13px;font-family:inherit;resize:none;line-height:1.4;outline:none;max-height:120px;transition:border-color .15s}
#txt:focus{border-color:var(--g)}
#txt::placeholder{color:var(--txm)}
.mic{width:36px;height:36px;border-radius:var(--ra);border:1px solid var(--bd);background:var(--s2);color:var(--tx2);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;transition:all .15s}
.mic:hover{border-color:var(--g);color:var(--g)}
.mic.on{background:var(--g);border-color:var(--g);color:#fff;animation:pulse .9s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(46,125,79,.4)}50%{box-shadow:0 0 0 7px rgba(46,125,79,0)}}
.send{width:36px;height:36px;border-radius:var(--ra);border:none;background:var(--g);color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.send:hover{background:var(--gd)}.send:disabled{opacity:.45}
.log-area{overflow-y:auto;padding:0 12px 6px;flex-shrink:0;max-height:90px}
.log-item{font-size:11px;color:var(--txm);padding:4px 7px;border-radius:5px;background:var(--s2);border:1px solid var(--bd);display:flex;align-items:center;gap:5px;margin-bottom:3px}
.log-dot{width:5px;height:5px;border-radius:50%;background:var(--g);flex-shrink:0}
.log-dot.err{background:#ef4444}.log-time{margin-left:auto;font-size:10px;color:var(--txm)}
/* Task Queue */
.tq-item{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:6px;font-size:11px;border:1px solid var(--bd);background:var(--s2);transition:all .3s}
.tq-item.in-progress{background:#E6F1FB;border-color:#B5D4F4}
.tq-item.paused{background:#FAEEDA;border-color:#FAC775}
.tq-item.completed{background:var(--gl);border-color:var(--glb)}
.tq-item.cancelled{background:#FCEBEB;border-color:#F7C1C1}
.tq-badge{font-size:9px;padding:1px 5px;border-radius:4px;font-weight:600;white-space:nowrap;flex-shrink:0}
.tq-badge.in-progress{background:#185FA5;color:#fff}
.tq-badge.planned{background:var(--bd);color:var(--tx2)}
.tq-badge.completed{background:var(--g);color:#fff}
.tq-badge.paused{background:#BA7517;color:#fff}
.tq-badge.cancelled{background:#A32D2D;color:#fff}
</style>
</head>
<body>
<div class="topbar">
  <div class="robot-icon">🤖</div>
  <span class="topbar-title">Hotel Robot — Aria</span>
  <div class="topbar-right">
    <span style="font-size:12px;color:var(--txm)">Q-bay · floor 1</span>
    <div class="status-pill"><div class="sdot" id="sdot"></div><span id="conn-label">Connecting...</span></div>
  </div>
</div>

<div id="state-bar" class="state-bar idle">
  <span id="state-icon">🟢</span>
  <span id="state-text" style="flex:1;font-weight:500">Robot idle — ready</span>
  <span id="state-sub" style="font-size:11px;opacity:.7"></span>
</div>

<div class="layout">
  <!-- LEFT -->
  <div class="left">
    <div class="sec-label">Status</div>
    <div class="stat-card">
      <div class="bat-top"><span style="color:var(--tx2)">Battery</span><span id="s-bat" style="color:var(--g);font-weight:500">—</span></div>
      <div class="bat-bar"><div class="bat-fill" id="s-bar" style="width:0%"></div></div>
    </div>
    <div class="stat-card">
      <div class="srow"><span>Moving</span><span class="sval" id="s-mov">—</span></div>
      <div class="divider"></div>
      <div class="srow"><span>Charging</span><span class="sval" id="s-chg">—</span></div>
      <div class="divider"></div>
      <div class="srow"><span>E-stop</span><span class="sval" id="s-es">—</span></div>
      <div class="divider"></div>
      <div class="srow"><span>Location</span><span class="sval" id="s-loc" style="font-size:10px;max-width:100px;text-align:right;overflow:hidden;text-overflow:ellipsis">—</span></div>
    </div>
    <div class="sec-label">Go to</div>
    <div class="nav-list" id="nav-list">
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'Frontdesk')"><span>🏢</span>Frontdesk</button>
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'Kitchen')"><span>🍳</span>Kitchen</button>
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'waiting')"><span>🛋️</span>Waiting</button>
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'steakhouse')"><span>🍽️</span>Steakhouse</button>
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'Meetingroom')"><span>👥</span>Meeting room</button>
      <button class="nav-btn" onclick="tqAddMoveBtn(this,'securitycheck')"><span>🛡️</span>Security</button>
    </div>
    <div class="sec-label">Quick</div>
    <div class="qa-grid">
      <button class="qa-btn" onclick="tqAddGoHome()">🔋 Go home</button>
      <button class="qa-btn danger" onclick="tqCancelCurrent()" title="Cancel current task">✕ Cancel</button>
      <button class="qa-btn danger" onclick="tqPause()" title="Pause queue">⏸ Pause</button>
      <button class="qa-btn" style="border-color:var(--g);color:var(--g)" onclick="tqStart()" title="Resume queue">▶ Resume</button>
    </div>
  </div>

  <!-- MIDDLE -->
  <div class="middle">
    <div class="msgs" id="msgs">
      <div class="mwrap">
        <div class="av">🤖</div>
        <div><div class="bubble">Hi! I'm Aria, your hotel robot assistant. Type a command, click a location, or speak using the mic.</div></div>
      </div>
    </div>
    <div class="input-area">
      <div class="voice-hint" id="hint"></div>
      <div class="input-row">
        <textarea id="txt" rows="1" placeholder="Type a command or speak…"></textarea>
        <button class="mic" id="mic">🎤</button>
        <button class="send" id="send">↑</button>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="right">
    <div class="sec-label" style="display:flex;align-items:center;justify-content:space-between;padding-right:12px">
      <span>Live Map</span>
      <button onclick="openMap()" style="font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--bd);background:var(--s2);color:var(--txm);cursor:pointer">expand</button>
    </div>
    <div style="padding:0 10px 8px;flex-shrink:0">
      <canvas id="miniMapCanvas" width="240" height="150" style="width:100%;border-radius:8px;border:1px solid var(--bd);background:var(--s2);display:block"></canvas>
    </div>

    <div class="sec-label">Log</div>
    <div class="log-area" id="log-area">
      <div class="log-item"><div class="log-dot"></div>Agent started<span class="log-time" id="start-time"></span></div>
    </div>

    <div class="sec-label" style="display:flex;align-items:center;justify-content:space-between;padding-right:12px;flex-shrink:0">
      <span>Task Queue</span>
      <div style="display:flex;gap:4px">
        <button id="tq-start-btn" onclick="tqStart()" style="display:none;font-size:10px;padding:2px 7px;border-radius:4px;border:none;background:var(--g);color:#fff;cursor:pointer;font-weight:500">▶</button>
        <button id="tq-pause-btn" onclick="tqPause()" style="display:none;font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--bd);background:var(--s2);color:var(--txm);cursor:pointer">⏸</button>
        <button onclick="tqClearAll()" style="font-size:10px;padding:2px 5px;border-radius:4px;border:1px solid var(--bd);background:var(--s2);color:var(--txm);cursor:pointer">✕</button>
      </div>
    </div>
    <div id="task-queue-list" style="overflow-y:auto;padding:0 10px;display:flex;flex-direction:column;gap:3px;flex:1;min-height:0"></div>
    <div style="padding:5px 10px 8px;font-size:10px;color:var(--txm);text-align:center;flex-shrink:0">Click locations above to add tasks</div>
  </div>
</div>

<!-- MAP MODAL -->
<div id="map-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:100;align-items:center;justify-content:center">
  <div style="background:var(--s1);border:1px solid var(--bd);border-radius:12px;width:900px;max-width:96vw;overflow:hidden;display:flex;flex-direction:column;max-height:92vh">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--bd);flex-shrink:0">
      <div style="display:flex;gap:4px">
        <button id="tab-map" onclick="switchTab('map')" style="padding:5px 14px;border-radius:6px;border:none;background:var(--g);color:#fff;cursor:pointer;font-size:12px;font-weight:500">🗺 Map</button>
        <button id="tab-cam" onclick="switchTab('cam')" style="padding:5px 14px;border-radius:6px;border:none;background:var(--s2);color:var(--tx2);cursor:pointer;font-size:12px">📷 Camera</button>
        <button id="tab-both" onclick="switchTab('both')" style="padding:5px 14px;border-radius:6px;border:none;background:var(--s2);color:var(--tx2);cursor:pointer;font-size:12px">⊞ Both</button>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:11px;color:var(--txm);background:var(--s2);border:1px solid var(--bd);border-radius:6px;padding:3px 8px" id="map-pill">idle</span>
        <button onclick="closeMap()" style="width:28px;height:28px;border-radius:6px;border:1px solid var(--bd);background:var(--s2);cursor:pointer;font-size:14px;color:var(--tx2)">✕</button>
      </div>
    </div>
    <div id="view-map" style="display:flex;flex-direction:column;flex:1;overflow:hidden">
      <canvas id="mapCanvas" width="900" height="460" style="display:block;background:var(--s2);flex:1"></canvas>
      <div style="padding:8px 16px;border-top:1px solid var(--bd);display:flex;gap:14px;flex-wrap:wrap;flex-shrink:0">
        <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2)"><span style="width:8px;height:8px;border-radius:50%;background:#2E7D4F;display:inline-block"></span>Robot</span>
        <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2)"><span style="width:8px;height:8px;border-radius:50%;background:#185FA5;display:inline-block"></span>Navigation</span>
        <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2)"><span style="width:8px;height:8px;border-radius:50%;background:#A32D2D;display:inline-block"></span>Charger</span>
        <span style="margin-left:auto;font-size:11px;color:var(--txm)">Click marker to navigate</span>
      </div>
    </div>
    <div id="view-cam" style="display:none;flex:1;background:#000;align-items:center;justify-content:center;min-height:400px">
      <img id="cam-feed" src="" style="max-width:100%;max-height:520px;object-fit:contain">
    </div>
    <div id="view-both" style="display:none;flex:1;flex-direction:row;overflow:hidden;min-height:400px">
      <div style="flex:1;display:flex;flex-direction:column;border-right:1px solid var(--bd)">
        <canvas id="mapCanvas2" width="440" height="400" style="display:block;background:var(--s2);flex:1"></canvas>
      </div>
      <div style="flex:1;background:#000;display:flex;align-items:center;justify-content:center">
        <img id="cam-feed2" src="" style="max-width:100%;max-height:100%;object-fit:contain">
      </div>
    </div>
  </div>
</div>

<script>
// ── DOM refs ─────────────────────────────────────────────────
const msgsEl=document.getElementById("msgs"),txtEl=document.getElementById("txt");
const micEl=document.getElementById("mic"),hintEl=document.getElementById("hint");
const logEl=document.getElementById("log-area");
document.getElementById("start-time").textContent=new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});

// ── Chat ──────────────────────────────────────────────────────
function addMsg(text,role,action){
  const w=document.createElement("div");w.className="mwrap "+role;
  const av=document.createElement("div");av.className="av";av.textContent=role==="user"?"J":"🤖";
  const r=document.createElement("div");
  const b=document.createElement("div");b.className="bubble";b.textContent=text;r.appendChild(b);
  if(action&&role==="bot"){const t=document.createElement("div");t.className="tag";t.innerHTML='<div class="tag-dot"></div>'+action.replace(/_/g," ");r.appendChild(t);}
  w.appendChild(av);w.appendChild(r);msgsEl.appendChild(w);msgsEl.scrollTop=msgsEl.scrollHeight;
}
function addLog(text,type){
  const el=document.createElement("div");el.className="log-item";
  const dot=document.createElement("div");dot.className="log-dot"+(type==="err"?" err":"");
  const time=document.createElement("span");time.className="log-time";
  time.textContent=new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
  el.appendChild(dot);el.appendChild(document.createTextNode(text));el.appendChild(time);
  logEl.insertBefore(el,logEl.firstChild);
  // keep max 20 log items
  while(logEl.children.length>20)logEl.removeChild(logEl.lastChild);
}
function showTyping(){const w=document.createElement("div");w.className="mwrap";w.id="typing";w.innerHTML='<div class="av">🤖</div><div class="bubble"><div class="typing-bub"><span></span><span></span><span></span></div></div>';msgsEl.appendChild(w);msgsEl.scrollTop=msgsEl.scrollHeight;}
function hideTyping(){const e=document.getElementById("typing");if(e)e.remove();}
function setUI(on){txtEl.disabled=!on;document.getElementById("send").disabled=!on;}

async function cmd(text){
  if(!text.trim())return;
  addMsg(text,"user");txtEl.value="";txtEl.style.height="auto";
  showTyping();setUI(false);
  try{
    const res=await fetch("/command",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
    const data=await res.json();
    hideTyping();
    addMsg(data.reply,"bot",data.action);
    if(data.action&&data.action!=="ask_clarification")addLog(data.action.replace(/_/g," "));
    fetchStatus();
  }catch(e){hideTyping();addMsg("Cannot reach the server — is jing_app.py running?","bot");addLog("connection error","err");}
  setUI(true);
}
txtEl.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();cmd(txtEl.value);}});
txtEl.addEventListener("input",()=>{txtEl.style.height="auto";txtEl.style.height=txtEl.scrollHeight+"px";});
document.getElementById("send").addEventListener("click",()=>cmd(txtEl.value));

// ── Voice ─────────────────────────────────────────────────────
let listening=false,recog=null;
if("webkitSpeechRecognition"in window||"SpeechRecognition"in window){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  recog=new SR();recog.continuous=false;recog.interimResults=true;recog.lang="en-US";
  recog.onstart=()=>{listening=true;micEl.classList.add("on");hintEl.textContent="Listening…";};
  recog.onresult=e=>{let fin="",int="";for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)fin+=e.results[i][0].transcript;else int+=e.results[i][0].transcript;}txtEl.value=fin||int;if(fin){stopListen();cmd(fin);}};
  recog.onerror=()=>stopListen();recog.onend=()=>stopListen();
  micEl.addEventListener("click",()=>{listening?stopListen():recog.start();});
}else{micEl.disabled=true;hintEl.textContent="Voice needs Chrome or Safari";}
function stopListen(){listening=false;micEl.classList.remove("on");hintEl.textContent="";try{recog&&recog.stop();}catch(e){}}

// ── Status polling ────────────────────────────────────────────
async function fetchStatus(){
  try{
    const res=await fetch("/status");const d=await res.json();
    if(!d.ok)throw new Error(d.error);
    document.getElementById("sdot").className="sdot";
    document.getElementById("conn-label").textContent="Connected";
    const bat=d.battery!==null&&d.battery!=="?"?Math.round(Number(d.battery)):null;
    document.getElementById("s-bat").textContent=bat!==null?bat+"%":"—";
    document.getElementById("s-bat").style.color=bat===null?"var(--txm)":bat<20?"#ef4444":bat<40?"#f59e0b":"var(--g)";
    const bar=document.getElementById("s-bar");
    bar.style.width=(bat||0)+"%";
    bar.className="bat-fill"+(bat&&bat<20?" low":bat&&bat<40?" med":"");
    document.getElementById("s-mov").textContent=d.moving||"idle";
    document.getElementById("s-chg").textContent=d.charging?"Yes":"No";
    document.getElementById("s-chg").style.color=d.charging?"var(--g)":"";
    document.getElementById("s-es").textContent=d.estop?"ACTIVE":"Off";
    document.getElementById("s-es").style.color=d.estop?"#ef4444":"";
    document.getElementById("s-loc").textContent=d.location!=="unknown"?d.location:"—";
    // State bar
    const sb=document.getElementById("state-bar");
    const si=document.getElementById("state-icon");
    const st=document.getElementById("state-text");
    const ss=document.getElementById("state-sub");
    sb.style.display="flex";
    if(d.charging){sb.className="state-bar charging";si.textContent="🔋";st.textContent="Charging at station";ss.textContent=bat+"%";}
    else if(d.moving==="moving"||d.moving==="running"){sb.className="state-bar moving";si.textContent="🤖";st.textContent=d.location&&d.location!=="unknown"?"On the way to "+d.location:"Moving...";ss.textContent=bat+"%";}
    else if(d.estop){sb.className="state-bar error";si.textContent="⚠️";st.textContent="Emergency stop active!";ss.textContent="";}
    else{sb.className="state-bar idle";si.textContent="🟢";st.textContent="Robot idle — ready";ss.textContent=bat+"%";}
  }catch(e){
    document.getElementById("sdot").className="sdot off";
    document.getElementById("conn-label").textContent="Robot offline";
  }
}
async function fetchMarkers(){
  try{
    const res=await fetch("/markers");const d=await res.json();
    if(!d.ok||!d.nav.length)return;
    const icons={Frontdesk:"🏢",front_desk:"🏢",Kitchen:"🍳",steakhouse:"🍽️",Meetingroom:"👥",waiting:"🛋️",waiting1:"🛋️",securitycheck:"🛡️",Demotest:"🔬",destination:"📦",toReception:"🚪",summon_point_5:"📍"};
    const list=document.getElementById("nav-list");list.innerHTML="";
    d.nav.forEach(m=>{
      const btn=document.createElement("button");btn.className="nav-btn";
      btn.innerHTML='<span>'+(icons[m]||"📍")+"</span>"+m;
      btn.onclick=()=>tqAddMoveBtn(btn,m);
      list.appendChild(btn);
    });
  }catch(e){}
}
fetchStatus();fetchMarkers();setInterval(fetchStatus,2000);

// ── Task Queue UI ─────────────────────────────────────────────
const TQ_ICONS={move:"🧭",go_home:"🔋",wait:"⏳",status:"📊"};
const TQ_LABELS={in_progress:"IN PROGRESS",planned:"PLANNED",completed:"DONE",paused:"PAUSED",cancelled:"CANCELLED"};

function tqAddMoveBtn(btn,marker){
  tqAddMove(marker);
  const orig={bg:btn.style.background,bc:btn.style.borderColor,c:btn.style.color};
  btn.style.background="var(--gl)";btn.style.borderColor="var(--g)";btn.style.color="var(--g)";
  setTimeout(()=>{btn.style.background=orig.bg;btn.style.borderColor=orig.bc;btn.style.color=orig.c;},600);
}
async function tqAddMove(marker){
  await fetch("/tq/add",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({type:"move",label:"Go to "+marker,marker})});
}
async function tqAddGoHome(){
  await fetch("/tq/add",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({type:"go_home",label:"Return to charger"})});
}
async function tqPause(){
  try{
    await fetch("/tq/pause",{method:"POST"});
    // Immediate UI feedback — don't wait for SSE
    document.getElementById("tq-pause-btn").style.display="none";
    document.getElementById("tq-start-btn").style.display="inline-block";
    // Mark all planned/in-progress items as paused
    document.querySelectorAll(".tq-item.planned,.tq-item.in-progress").forEach(item=>{
      item.className="tq-item paused";
      const b=item.querySelector(".tq-badge");
      if(b){b.className="tq-badge paused";b.textContent="PAUSED";}
    });
    addLog("Queue paused","");
  }catch(e){console.error("Pause failed:",e);}
}
async function tqStart(){
  try{
    await fetch("/tq/resume",{method:"POST"});
    // Immediate UI feedback
    document.getElementById("tq-start-btn").style.display="none";
    document.getElementById("tq-pause-btn").style.display="inline-block";
    document.querySelectorAll(".tq-item.paused").forEach(item=>{
      item.className="tq-item planned";
      const b=item.querySelector(".tq-badge");
      if(b){b.className="tq-badge planned";b.textContent="PLANNED";}
    });
    addLog("Queue resumed","");
  }catch(e){console.error("Resume failed:",e);}
}
async function tqCancelCurrent(){
  try{
    await fetch("/tq/cancel_current",{method:"POST"});
    addLog("Current task cancelled","err");
    // In-progress item will be removed via SSE — just log it
  }catch(e){console.error("Cancel failed:",e);}
}
async function tqClearAll(){
  if(!confirm("Clear all tasks?"))return;
  try{
    await fetch("/tq/clear",{method:"POST"});
    addLog("Queue cleared","");
  }catch(e){console.error("Clear failed:",e);}
}

function renderTQItem(task,animate){
  const el=document.createElement("div");
  el.id="tq-"+task.id;
  el.className="tq-item "+(task.status||"planned");
  if(animate){el.style.opacity="0";setTimeout(()=>{el.style.opacity="1";el.style.transition="opacity .3s";},10);}
  el.innerHTML=
    '<span style="font-size:12px">'+(TQ_ICONS[task.type]||"⚡")+'</span>'+
    '<span style="flex:1;color:var(--tx);font-size:11px">'+(task.label||task.type)+'</span>'+
    '<span class="tq-badge '+(task.status||"planned")+'">'+(TQ_LABELS[task.status]||task.status)+'</span>';
  return el;
}

function handleTQUpdate(data){
  const list=document.getElementById("task-queue-list");
  const id=data.task_id;
  const el=id?document.getElementById("tq-"+id):null;

  if(data.update_type==="add"){
    const empty=list.querySelector("div:not(.tq-item)");if(empty)empty.remove();
    const task={id,type:data.ttype,label:data.label,status:data.status||"planned"};
    list.appendChild(renderTQItem(task,true));
    // Show pause button, hide start button when items added
    document.getElementById("tq-pause-btn").style.display="inline-block";
    document.getElementById("tq-start-btn").style.display="none";
    addLog((data.label||data.ttype)+" added","");
  }
  else if(data.update_type==="status"&&el){
    el.className="tq-item "+(data.status||"planned");
    el.style.cssText="flex-direction:column;align-items:stretch;gap:3px;";
    const badge=document.getElementById("tq-badge-"+id);
    if(badge){badge.className="tq-badge "+(data.status||"planned");badge.textContent=TQ_LABELS[data.status]||data.status;}
    // Update live detail text
    const detailEl=document.getElementById("tq-detail-"+id);
    if(detailEl&&data.detail!==undefined)detailEl.textContent=data.detail;
    // Update API endpoint shown
    const apiEl=document.getElementById("tq-api-"+id);
    if(apiEl&&data.api)apiEl.textContent=data.api;
    else if(!apiEl&&data.api){
      const d=document.createElement("div");d.id="tq-api-"+id;
      d.style="font-size:9px;color:var(--txm);font-family:monospace;padding-left:18px";
      d.textContent=data.api;el.insertBefore(d,detailEl);
    }
    if(data.status==="completed"||data.status==="cancelled"){
      setTimeout(()=>{
        if(el){el.style.opacity="0";el.style.transform=data.status==="completed"?"translateX(16px)":"translateX(-16px)";el.style.transition="all .35s";
          setTimeout(()=>{el.remove();checkTQEmpty();},350);}
      },data.status==="completed"?700:150);
    }
  }
  else if(data.update_type==="remove"&&el){el.remove();checkTQEmpty();}
  else if(data.update_type==="cleared"){
    list.innerHTML='<div style="font-size:11px;color:var(--txm);padding:6px 2px">No tasks yet</div>';
    document.getElementById("tq-start-btn").style.display="none";
    document.getElementById("tq-pause-btn").style.display="none";
  }
  else if(data.update_type==="paused_all"){
    document.getElementById("tq-pause-btn").style.display="none";
    document.getElementById("tq-start-btn").style.display="inline-block";
    list.querySelectorAll(".tq-item.planned").forEach(item=>{
      item.className="tq-item paused";
      const b=item.querySelector(".tq-badge");if(b){b.className="tq-badge paused";b.textContent="PAUSED";}
    });
  }
  else if(data.update_type==="resumed"){
    document.getElementById("tq-start-btn").style.display="none";
    document.getElementById("tq-pause-btn").style.display="inline-block";
    list.querySelectorAll(".tq-item.paused").forEach(item=>{
      item.className="tq-item planned";
      const b=item.querySelector(".tq-badge");if(b){b.className="tq-badge planned";b.textContent="PLANNED";}
    });
  }
}

function checkTQEmpty(){
  const list=document.getElementById("task-queue-list");
  if(!list.querySelector(".tq-item")){
    list.innerHTML='<div style="font-size:11px;color:var(--txm);padding:6px 2px">No tasks yet</div>';
    document.getElementById("tq-start-btn").style.display="none";
    document.getElementById("tq-pause-btn").style.display="none";
  }
}

async function loadTQState(){
  try{
    const res=await fetch("/tq/state");const d=await res.json();
    const list=document.getElementById("task-queue-list");list.innerHTML="";
    if(!d.tasks||!d.tasks.length){checkTQEmpty();return;}
    d.tasks.forEach(t=>list.appendChild(renderTQItem(t,false)));
    document.getElementById("tq-pause-btn").style.display=d.paused?"none":"inline-block";
    document.getElementById("tq-start-btn").style.display=d.paused?"inline-block":"none";
  }catch(e){}
}
loadTQState();

// ── SSE ───────────────────────────────────────────────────────
const evtSource=new EventSource("/stream");
evtSource.onmessage=e=>{
  const data=JSON.parse(e.data);
  if(data.ping)return;
  if(data.tq_update===true){handleTQUpdate(data);return;}
  if(data.background){
    addMsg(data.reply,"bot",data.action);
    // Update state bar from SSE events
    const sb=document.getElementById("state-bar");
    const si=document.getElementById("state-icon");
    const st=document.getElementById("state-text");
    if(data.action==="moving"||data.action==="move_to_marker"||data.action==="return_to_charger"){
      sb.style.display="flex";sb.className="state-bar moving";si.textContent="🤖";st.textContent=data.reply;
    }else if(data.action==="arrived"){
      sb.style.display="flex";sb.className="state-bar arrived";si.textContent="✅";st.textContent=data.reply;
    }else if(data.action==="wait"){
      sb.style.display="flex";sb.className="state-bar idle";si.textContent="⏳";st.textContent=data.reply;
    }else if(data.action==="done"||data.action==="paused"){
      fetchStatus();
    }
  }
};

// ── Mini Map (robot centered) ─────────────────────────────────
const MMARKERS=[
  {name:"Frontdesk",x:1.05,y:-2.87,type:"nav"},{name:"front_desk",x:1.23,y:-5.12,type:"nav"},
  {name:"Meetingroom",x:3.39,y:11.41,type:"nav"},{name:"Kitchen",x:0.48,y:-14.87,type:"nav"},
  {name:"steakhouse",x:1.8,y:-11,type:"nav"},{name:"waiting",x:1.78,y:-7.15,type:"nav"},
  {name:"Demotest",x:2.43,y:-8,type:"nav"},{name:"securitycheck",x:-16.14,y:9.4,type:"nav"},
  {name:"toReception",x:-19.05,y:-12.35,type:"nav"},{name:"summon_point_5",x:-0.27,y:-9.98,type:"system"},
  {name:"charge_point_1F_40300423",x:4.25,y:-8.19,type:"charge"},
];
const MCOLS={nav:"#185FA5",charge:"#A32D2D",system:"#888780"};
let mRobot={x:4.25,y:-8.19,tx:4.25,ty:-8.19,theta:0};
const mCanvas=document.getElementById("miniMapCanvas");
const mCtx=mCanvas.getContext("2d");
const MW=mCanvas.width,MH=mCanvas.height,VR=12;

function drawMini(){
  mCtx.fillStyle="#f5f7f5";mCtx.fillRect(0,0,MW,MH);
  const sc=Math.min(MW,MH)/(VR*2);
  mCtx.strokeStyle="rgba(0,0,0,0.05)";mCtx.lineWidth=1;
  for(let gx=-VR;gx<=VR;gx+=2){const cx=MW/2+gx*sc;mCtx.beginPath();mCtx.moveTo(cx,0);mCtx.lineTo(cx,MH);mCtx.stroke();}
  for(let gy=-VR;gy<=VR;gy+=2){const cy=MH/2-gy*sc;mCtx.beginPath();mCtx.moveTo(0,cy);mCtx.lineTo(MW,cy);mCtx.stroke();}
  MMARKERS.forEach(m=>{
    const cx=MW/2+(m.x-mRobot.x)*sc,cy=MH/2-(m.y-mRobot.y)*sc;
    if(cx<-10||cx>MW+10||cy<-10||cy>MH+10)return;
    const col=MCOLS[m.type],r=m.type==="system"?3:5;
    mCtx.beginPath();mCtx.arc(cx,cy,r,0,Math.PI*2);mCtx.strokeStyle=col;mCtx.lineWidth=1.5;mCtx.stroke();mCtx.fillStyle=col+"44";mCtx.fill();
    mCtx.fillStyle="#94a3b8";mCtx.font="8px -apple-system,sans-serif";mCtx.textAlign="center";
    mCtx.fillText(m.name.length>12?m.name.slice(0,10)+"…":m.name,cx,cy-r-2);
  });
  const rx=MW/2,ry=MH/2;
  mCtx.beginPath();mCtx.arc(rx,ry,10,0,Math.PI*2);mCtx.fillStyle="rgba(46,125,79,0.15)";mCtx.fill();
  mCtx.beginPath();mCtx.arc(rx,ry,6,0,Math.PI*2);mCtx.fillStyle="#2E7D4F";mCtx.fill();mCtx.strokeStyle="#fff";mCtx.lineWidth=2;mCtx.stroke();
  mCtx.save();mCtx.translate(rx,ry);mCtx.rotate(-(mRobot.theta||0)+Math.PI/2);
  mCtx.beginPath();mCtx.moveTo(0,-9);mCtx.lineTo(-3,-4);mCtx.lineTo(3,-4);mCtx.closePath();mCtx.fillStyle="#fff";mCtx.fill();mCtx.restore();
}
function animMini(){mRobot.x+=(mRobot.tx-mRobot.x)*.1;mRobot.y+=(mRobot.ty-mRobot.y)*.1;drawMini();requestAnimationFrame(animMini);}
animMini();
async function pollMiniPos(){try{const r=await fetch("/position");const d=await r.json();if(d.ok&&d.x!==null){mRobot.tx=d.x;mRobot.ty=d.y;mRobot.theta=d.theta||0;}}catch(e){}}
setInterval(pollMiniPos,1500);pollMiniPos();

// ── Full Map Modal ────────────────────────────────────────────
const MARKERS=[
  {name:"front_desk",x:1.23,y:-5.12,type:"nav"},{name:"Frontdesk",x:1.05,y:-2.87,type:"nav"},
  {name:"Meetingroom",x:3.39,y:11.41,type:"nav"},{name:"Kitchen",x:0.48,y:-14.87,type:"nav"},
  {name:"steakhouse",x:1.8,y:-11,type:"nav"},{name:"waiting",x:1.78,y:-7.15,type:"nav"},
  {name:"waiting1",x:0.76,y:-0.86,type:"nav"},{name:"Demotest",x:2.43,y:-8,type:"nav"},
  {name:"securitycheck",x:-16.14,y:9.4,type:"nav"},{name:"toReception",x:-19.05,y:-12.35,type:"nav"},
  {name:"summon_point_5",x:-0.27,y:-9.98,type:"system"},{name:"destination",x:1.65,y:-3.74,type:"nav"},
  {name:"charge_point_1F_40300423",x:4.25,y:-8.19,type:"charge"},
  {name:"charge_point_1F_40300165",x:4.43,y:-6.91,type:"charge"},
  {name:"charge_point_1F_1",x:4.11,y:-9.06,type:"charge"},
];
const MCOLORS={nav:"#185FA5",charge:"#A32D2D",system:"#888780"};
let robot={x:4.25,y:-8.19,tx:4.25,ty:-8.19,theta:0};
let mapInterval=null,animFrame=null,hovered=null,currentTab="map";
const canvas=document.getElementById("mapCanvas");
const ctx=canvas.getContext("2d");
const W=900,H=460,PAD=44;
canvas.width=W;canvas.height=H;
const allX=MARKERS.map(m=>m.x).concat([-22,7]);
const allY=MARKERS.map(m=>m.y).concat([14,-18]);
const minX=Math.min(...allX),maxX=Math.max(...allX),minY=Math.min(...allY),maxY=Math.max(...allY);

function toC(rx,ry){return[PAD+(rx-minX)/(maxX-minX)*(W-PAD*2),H-PAD-(ry-minY)/(maxY-minY)*(H-PAD*2)];}

function drawMap(){
  ctx.fillStyle="#f5f7f5";ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="rgba(0,0,0,0.04)";ctx.lineWidth=1;
  for(let gx=Math.ceil(minX);gx<=maxX;gx+=2){const[cx]=toC(gx,0);ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();}
  for(let gy=Math.ceil(minY);gy<=maxY;gy+=2){const[,cy]=toC(0,gy);ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(W,cy);ctx.stroke();}
  MARKERS.forEach(m=>{
    const[cx,cy]=toC(m.x,m.y);const col=MCOLORS[m.type];const r=hovered===m.name?9:m.type==="system"?5:7;
    ctx.beginPath();ctx.arc(cx,cy,r+4,0,Math.PI*2);ctx.fillStyle=col+"22";ctx.fill();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle=col+"55";ctx.fill();
    ctx.beginPath();ctx.arc(cx,cy,r*.45,0,Math.PI*2);ctx.fillStyle=col;ctx.fill();
    ctx.fillStyle=hovered===m.name?col:"#94a3b8";ctx.font=(hovered===m.name?"bold ":"")+"10px -apple-system,sans-serif";ctx.textAlign="center";
    ctx.fillText(m.name.length>15?m.name.slice(0,13)+"…":m.name,cx,cy-r-4);
  });
  const[rx,ry]=toC(robot.x,robot.y);
  ctx.beginPath();ctx.arc(rx,ry,16,0,Math.PI*2);ctx.fillStyle="rgba(46,125,79,0.12)";ctx.fill();
  ctx.beginPath();ctx.arc(rx,ry,10,0,Math.PI*2);ctx.fillStyle="#2E7D4F";ctx.fill();ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.stroke();
  ctx.save();ctx.translate(rx,ry);if(robot.theta!==undefined)ctx.rotate(-robot.theta+Math.PI/2);
  ctx.beginPath();ctx.moveTo(0,-13);ctx.lineTo(-4,-7);ctx.lineTo(4,-7);ctx.closePath();ctx.fillStyle="#fff";ctx.fill();ctx.restore();
  if(currentTab==="both"){const c2=document.getElementById("mapCanvas2");if(c2)drawOnCanvas(c2,c2.width,c2.height);}
}

function drawOnCanvas(c,w,h){
  const ct=c.getContext("2d");const p=30;
  function tc(rx,ry){return[p+(rx-minX)/(maxX-minX)*(w-p*2),h-p-(ry-minY)/(maxY-minY)*(h-p*2)];}
  ct.fillStyle="#f5f7f5";ct.fillRect(0,0,w,h);
  MARKERS.forEach(m=>{
    const[cx,cy]=tc(m.x,m.y);const col=MCOLORS[m.type];const r=5;
    ct.beginPath();ct.arc(cx,cy,r,0,Math.PI*2);ct.strokeStyle=col;ct.lineWidth=1.5;ct.stroke();ct.fillStyle=col+"44";ct.fill();
    ct.fillStyle="#94a3b8";ct.font="9px -apple-system,sans-serif";ct.textAlign="center";ct.fillText(m.name.length>12?m.name.slice(0,10)+"…":m.name,cx,cy-r-3);
  });
  const[rx,ry]=tc(robot.x,robot.y);
  ct.beginPath();ct.arc(rx,ry,9,0,Math.PI*2);ct.fillStyle="#2E7D4F";ct.fill();ct.strokeStyle="#fff";ct.lineWidth=2;ct.stroke();
  ct.save();ct.translate(rx,ry);if(robot.theta)ct.rotate(-robot.theta+Math.PI/2);
  ct.beginPath();ct.moveTo(0,-12);ct.lineTo(-3,-6);ct.lineTo(3,-6);ct.closePath();ct.fillStyle="#fff";ct.fill();ct.restore();
}

function animateRobot(){
  const sp=0.08,dx=robot.tx-robot.x,dy=robot.ty-robot.y;
  if(Math.abs(dx)>.01||Math.abs(dy)>.01){robot.x+=dx*sp;robot.y+=dy*sp;}else{robot.x=robot.tx;robot.y=robot.ty;}
  drawMap();animFrame=requestAnimationFrame(animateRobot);
}
async function pollPosition(){
  try{const res=await fetch("/position");const d=await res.json();
    if(d.ok&&d.x!==null){robot.tx=d.x;robot.ty=d.y;robot.theta=d.theta;robot.running=d.running;
      const pill=document.getElementById("map-pill");
      pill.textContent=d.running+(d.move_target?" → "+d.move_target:"");
      pill.style.color=d.running==="moving"?"var(--g)":"var(--txm)";
    }
  }catch(e){}
}

function switchTab(tab){
  currentTab=tab;
  document.getElementById("view-map").style.display=tab==="map"?"flex":"none";
  document.getElementById("view-cam").style.display=tab==="cam"?"flex":"none";
  document.getElementById("view-both").style.display=tab==="both"?"flex":"none";
  ["map","cam","both"].forEach(t=>{const b=document.getElementById("tab-"+t);b.style.background=t===tab?"var(--g)":"var(--s2)";b.style.color=t===tab?"#fff":"var(--tx2)";});
  if(tab==="cam"||tab==="both"){const u="https://webcam.urbot.ai/video";document.getElementById("cam-feed").src=u;document.getElementById("cam-feed2").src=u;}
}
function openMap(){
  document.getElementById("map-modal").style.display="flex";
  switchTab(currentTab||"map");
  animFrame=requestAnimationFrame(animateRobot);
  mapInterval=setInterval(pollPosition,1000);pollPosition();
}
function closeMap(){
  document.getElementById("map-modal").style.display="none";
  clearInterval(mapInterval);cancelAnimationFrame(animFrame);
  document.getElementById("cam-feed").src="";document.getElementById("cam-feed2").src="";
}
canvas.addEventListener("mousemove",e=>{
  const rect=canvas.getBoundingClientRect();const mx=(e.clientX-rect.left)*(W/rect.width);const my=(e.clientY-rect.top)*(H/rect.height);
  let found=null;MARKERS.forEach(m=>{const[cx,cy]=toC(m.x,m.y);if(Math.hypot(cx-mx,cy-my)<14)found=m.name;});
  hovered=found;canvas.style.cursor=found?"pointer":"default";
});
canvas.addEventListener("click",e=>{
  const rect=canvas.getBoundingClientRect();const mx=(e.clientX-rect.left)*(W/rect.width);const my=(e.clientY-rect.top)*(H/rect.height);
  MARKERS.forEach(m=>{if(m.type==="system")return;const[cx,cy]=toC(m.x,m.y);if(Math.hypot(cx-mx,cy-my)<14){tqAddMove(m.name);closeMap();}});
});
document.getElementById("map-modal").addEventListener("click",e=>{if(e.target===document.getElementById("map-modal"))closeMap();});
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"

    print("\n" + "="*52)
    print("  Hotel Robot — Aria")
    print("="*52)
    print(f"  local   → http://localhost:5050")
    print(f"  network → http://{local_ip}:5050")
    print(f"  tailnet → http://100.115.171.5:5050")
    print("="*52 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=False)
