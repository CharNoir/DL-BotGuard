import time
import json
import uuid
import os
from datetime import datetime
from pynput import mouse, keyboard
from pathlib import Path


try:
    import win32gui
    import win32process
    import psutil
    WINDOWS_SUPPORT = True
except Exception:
    WINDOWS_SUPPORT = False

# -------- Configuration ----------
HASH_WINDOW_TITLES = False   # If True, store sha256(title) + length instead of raw title
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
print(ROOT_DIR)
LOGS_DIR = os.path.join(ROOT_DIR, "data", "raw", "our_bot")
print(LOGS_DIR)
SCHEMA_VERSION = "v1" 
# --------------------------------

# Create unique run ID
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_id = str(uuid.uuid4())

# File paths
log_dir = os.path.join(LOGS_DIR, SCHEMA_VERSION)
os.makedirs(log_dir, exist_ok=True)

mouse_dir = os.path.join(log_dir, "mouse")
key_dir = os.path.join(log_dir, "key")
os.makedirs(mouse_dir, exist_ok=True)
os.makedirs(key_dir, exist_ok=True)

mouse_file = os.path.join(mouse_dir, f"mouse_events_{run_id}.jsonl")
keyboard_file = os.path.join(key_dir, f"key_events_{run_id}.jsonl")

def event_base(event_type):
    """Generate the base event metadata shared by all events."""
    return {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,
        "event_type": event_type,
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        "wall_time_ms": int(time.time() * 1000),
    }

def write_event(filepath, event):
    """Append event dictionary as a JSONL line."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

_cached_hwnd = None
_cached_window_info = None

def get_foreground_window_info():
    """
    Returns a minimal dict with info about the current foreground window:
    {
      "hwnd": int or None,
      "pid": int or None,
      "process_name": str or None
    }
    """
    global _cached_hwnd, _cached_window_info

    if not WINDOWS_SUPPORT:
        return {"hwnd": None, "pid": None, "process_name": None}

    try:
        hwnd = win32gui.GetForegroundWindow()
    except Exception:
        hwnd = None

    # If unchanged, return cached info
    if hwnd and hwnd == _cached_hwnd and _cached_window_info is not None:
        return _cached_window_info

    info = {
        #"hwnd": None, 
        "pid": None, 
        "process_name": None}

    if not hwnd:
        _cached_hwnd = None
        _cached_window_info = info
        return info

    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        #info["hwnd"] = int(hwnd)
        info["pid"] = int(pid)
        try:
            proc = psutil.Process(pid)
            info["process_name"] = proc.name()
        except Exception:
            info["process_name"] = None
    except Exception:
        # keep None fields if something fails
        pass

    _cached_hwnd = hwnd
    _cached_window_info = info
    return info

# ---------- Mouse Handlers ----------
def on_move(x, y):
    e = event_base("pointermove")
    e.update({"x_screen": x, "y_screen": y})
    e.update({"foreground": get_foreground_window_info()})
    write_event(mouse_file, e)

def on_click(x, y, button, pressed):
    e = event_base("click" if pressed else "release")
    e.update({"x_screen": x, "y_screen": y, "button": str(button), "pressed": pressed})
    e.update({"foreground": get_foreground_window_info()})
    write_event(mouse_file, e)

def on_scroll(x, y, dx, dy):
    e = event_base("scroll")
    e.update({"x_screen": x, "y_screen": y, "wheel_dx": dx, "wheel_dy": dy})
    e.update({"foreground": get_foreground_window_info()})
    write_event(mouse_file, e)

# ---------- Keyboard Handlers ----------
def on_press(key):
    e = event_base("key_down")
    e.update({"foreground": get_foreground_window_info()})
    write_event(keyboard_file, e)

def on_release(key):
    e = event_base("key_up")
    e.update({"foreground": get_foreground_window_info()})
    write_event(keyboard_file, e)

print(f"Logging mouse → {mouse_file}")
print(f"Logging keyboard (timings only) → {keyboard_file}")
if not WINDOWS_SUPPORT:
    print("Note: pywin32 and/or psutil not available — foreground window info will be empty.")
print("Tracking started... Press Ctrl+C to stop.")

# Start
with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as ml, \
     keyboard.Listener(on_press=on_press, on_release=on_release) as kl:
    try:
        ml.join()
        kl.join()
    except KeyboardInterrupt:
        print("\nLogging stopped.")
