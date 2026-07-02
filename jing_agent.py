"""
jing_agent.py
-------------
Full pipeline: text command -> Qwen3.6-35B (MLX, local) -> WATER/Shuidi robot HTTP API.

Voice/STT left out for now -- add later on top of handle_command().

Run order:
  1. python3 jing_agent.py --check       # confirm Qwen + robot both respond
  2. python3 jing_agent.py --markers     # list real marker names on the robot
  3. python3 jing_agent.py               # start the interactive agent loop

Requirements:
  pip install openai requests
"""

import json
import argparse
import requests
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
QWEN_BASE_URL  = "http://127.0.0.1:8080/v1"
QWEN_API_KEY   = "not-needed"
QWEN_MODEL     = "mlx-community/Qwen3.6-35B-A3B-6bit"

ROBOT_IP       = "10.1.17.225"
ROBOT_PORT     = 9001
ROBOT_BASE_URL = f"http://{ROBOT_IP}:{ROBOT_PORT}"
TIMEOUT        = 8
# ============================================================

client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)


# ============================================================
# HELPERS
# ============================================================

def parse_markers(api_response: dict) -> list:
    """
    Robot returns markers as a dict under 'results', not a flat list.
    e.g. { "status": "OK", "results": { "Frontdesk": {...}, "Kitchen": {...} } }
    """
    if isinstance(api_response, dict) and "results" in api_response:
        return list(api_response["results"].keys())
    if isinstance(api_response, list):
        return api_response
    return []

SYSTEM_MARKER_PREFIXES = ("charge_point_", "sweep_start_", "sweep_", "map_")

def navigation_markers(all_markers: list) -> list:
    """Filter out internal/system markers, keep only real destinations."""
    return [m for m in all_markers
            if not any(m.startswith(p) for p in SYSTEM_MARKER_PREFIXES)]


# ============================================================
# ROBOT FUNCTIONS
# ============================================================

def robot_get_status():
    r = requests.get(f"{ROBOT_BASE_URL}/api/robot_status", timeout=TIMEOUT)
    return r.json()

def robot_get_markers():
    r = requests.get(f"{ROBOT_BASE_URL}/api/markers/query_list", timeout=TIMEOUT)
    return r.json()

def robot_move_to(marker_name: str):
    r = requests.get(f"{ROBOT_BASE_URL}/api/move",
                     params={"marker": marker_name}, timeout=TIMEOUT)
    return r.json()

def robot_cancel_move():
    r = requests.get(f"{ROBOT_BASE_URL}/api/move/cancel", timeout=TIMEOUT)
    return r.json()

def robot_emergency_stop(engage: bool):
    r = requests.get(f"{ROBOT_BASE_URL}/api/estop",
                     params={"flag": "true" if engage else "false"}, timeout=TIMEOUT)
    return r.json()

TOOL_DISPATCH = {
    "move_to_marker":   lambda a: robot_move_to(a["marker_name"]),
    "get_robot_status": lambda a: robot_get_status(),
    "cancel_movement":  lambda a: robot_cancel_move(),
    "emergency_stop":   lambda a: robot_emergency_stop(a["engage"]),
    "list_markers":     lambda a: robot_get_markers(),
}


# ============================================================
# TOOL DEFINITIONS
# ============================================================

def build_tools(known_markers: list):
    nav = navigation_markers(known_markers)
    marker_hint = (
        f"Valid destination markers: {', '.join(nav)}"
        if nav else
        "No markers loaded yet — use list_markers to fetch them."
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "move_to_marker",
                "description": (
                    "Move the robot to a named location on its map. "
                    "Pick the closest matching marker from the list. " + marker_hint
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "marker_name": {
                            "type": "string",
                            "description": "Exact marker name from the robot map."
                        }
                    },
                    "required": ["marker_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_robot_status",
                "description": "Check the robot's current status: battery level, whether moving, position, emergency stop state.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_movement",
                "description": "Stop or cancel the robot's current movement task.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "emergency_stop",
                "description": "Engage or release the robot's emergency stop. Use with caution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engage": {
                            "type": "boolean",
                            "description": "true to engage, false to release."
                        }
                    },
                    "required": ["engage"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_markers",
                "description": "Fetch all named locations on the robot's map.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
    ]


SYSTEM_PROMPT = """You are a hotel service robot assistant controlling a physical robot.

Known locations on this robot's map:

Floor 1:
- Frontdesk / front_desk — front desk area
- Meetingroom — meeting room
- Kitchen — kitchen
- steakhouse — steakhouse / restaurant
- waiting / waiting1 — waiting areas
- destination — generic delivery destination
- Demotest — demo / test point
- securitycheck — security check area
- toReception — reception area
- summon_point_5 — summon point

Floor 2:
- point1, point2, point3, point4 — numbered locations

Rules:
- Use tools to take real actions when asked.
- When asked to go somewhere, pick the closest matching marker name.
- Do NOT navigate to charge_point or sweep_start markers — those are system markers.
- If the user asks something unrelated to robot operation, politely decline and do not call any tool.
- If unsure which marker matches, ask for clarification rather than guessing.
- Always confirm what action you are about to take.
"""


# ============================================================
# CORE AGENT FUNCTION
# ============================================================

def handle_command(user_text: str, known_markers: list):
    tools = build_tools(known_markers)

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text}
        ],
        tools=tools,
    )

    message = response.choices[0].message

    if not message.tool_calls:
        print(f"\n🤖  {message.content}")
        return

    for call in message.tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)
        print(f"\n🔧  Calling: {name}({args})")

        if name not in TOOL_DISPATCH:
            print(f"⚠️   Unknown tool '{name}'")
            continue

        try:
            result = TOOL_DISPATCH[name](args)
            print(f"✅  Robot response: {json.dumps(result, indent=2)}")

            if name == "list_markers":
                new_markers = parse_markers(result)
                known_markers.clear()
                known_markers.extend(new_markers)
                print(f"📍  Navigation markers: {navigation_markers(new_markers)}")

        except requests.exceptions.ConnectionError:
            print(f"❌  Cannot reach robot at {ROBOT_BASE_URL}")
            print("    → Is the robot powered on and on the network?")
        except requests.exceptions.Timeout:
            print(f"❌  Robot timed out after {TIMEOUT}s")
        except Exception as e:
            print(f"❌  Error: {e}")


# ============================================================
# CHECK MODE
# ============================================================

def run_checks():
    print("=" * 52)
    print("  CONNECTION CHECKS")
    print("=" * 52)

    print(f"\n1. Qwen at {QWEN_BASE_URL} ...")
    try:
        r = requests.get(f"{QWEN_BASE_URL}/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"   ✅ Responding. Models: {models}")
    except Exception as e:
        print(f"   ❌ Not responding: {e}")
        print("   → Check: ps aux | grep mlx_lm")
        print("   → Start: conda activate mlx && nohup mlx_lm.server "
              "--model mlx-community/Qwen3.6-35B-A3B-6bit "
              "--host 127.0.0.1 --port 8080 > ~/mlx_server.log 2>&1 &")
        return False

    print(f"\n2. Robot at {ROBOT_BASE_URL} ...")
    try:
        result = robot_get_status()
        print(f"   ✅ Responding: {json.dumps(result, indent=6)}")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot reach {ROBOT_BASE_URL}")
        print("   → Try: ping 10.1.17.225")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    print("\n✅  All checks passed. Ready to run.")
    return True


# ============================================================
# MARKERS MODE
# ============================================================

def run_markers():
    print(f"\nFetching markers from {ROBOT_BASE_URL} ...")
    try:
        result = robot_get_markers()
        all_markers = parse_markers(result)
        nav = navigation_markers(all_markers)
        print(f"\n📍  All markers ({len(all_markers)} total):")
        for m in sorted(all_markers):
            tag = "" if m in nav else "  ← system"
            print(f"    {m}{tag}")
        print(f"\n🧭  Navigation destinations ({len(nav)}):")
        for m in sorted(nav):
            print(f"    {m}")
    except Exception as e:
        print(f"❌  Could not fetch markers: {e}")


# ============================================================
# INTERACTIVE AGENT LOOP
# ============================================================

def run_agent():
    print("\n" + "=" * 52)
    print("  🤖  Hotel Robot Agent")
    print("=" * 52)
    print("Type a command and press Enter. Type 'quit' to exit.")
    print("\nExamples:")
    print("  go to the front desk")
    print("  what is the battery level")
    print("  stop the robot")
    print("  go to the kitchen")
    print("  list markers\n")

    known_markers = []
    print("Fetching markers from robot...")
    try:
        result = robot_get_markers()
        known_markers = parse_markers(result)
        nav = navigation_markers(known_markers)
        print(f"📍  Destinations: {nav}\n")
    except Exception as e:
        print(f"⚠️   Could not fetch markers: {e}")
        print("    Robot may be offline — continuing anyway.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        handle_command(user_input, known_markers)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotel Robot Agent — Jing")
    parser.add_argument("--check",   action="store_true", help="Test Qwen + robot connections")
    parser.add_argument("--markers", action="store_true", help="List all markers on the robot map")
    args = parser.parse_args()

    if args.check:
        run_checks()
    elif args.markers:
        run_markers()
    else:
        run_agent()