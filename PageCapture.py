"""Repeatedly simulate Win+Screenshot followed by the Right Arrow key.

WARNING: Win+Screenshot saves a PNG into Pictures\Screenshots every time it is
pressed. Running this for many iterations can create a lot of files quickly.

Usage:
    python win_ss_right.py
    python win_ss_right.py -n 10 -d 0.5

Stop early with Ctrl+C in the terminal.
"""

import argparse
import ctypes
import time

# --- Windows constants ---
INPUT_KEYBOARD = 1

VK_LWIN = 0x5B      # Left Windows key
VK_SNAPSHOT = 0x2C  # Print Screen / Snapshot key
VK_RIGHT = 0x27     # Right Arrow key

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

PUL = ctypes.POINTER(ctypes.c_ulong)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_I(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", INPUT_I),
    ]


def _send_key(vk, flags=0):
    """Send one keyboard input event."""
    extra = ctypes.c_ulong(0)
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ii.ki.wVk = vk
    inp.ii.ki.wScan = 0
    inp.ii.ki.dwFlags = flags
    inp.ii.ki.time = 0
    inp.ii.ki.dwExtraInfo = ctypes.pointer(extra)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


def key_down(vk, extended=False):
    flags = KEYEVENTF_EXTENDEDKEY if extended else 0
    _send_key(vk, flags)


def key_up(vk, extended=False):
    flags = KEYEVENTF_KEYUP
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    _send_key(vk, flags)


def key_tap(vk, extended=False, hold=0.05):
    key_down(vk, extended)
    time.sleep(hold)
    key_up(vk, extended)


def press_win_screenshot():
    """Press and release Left Windows + Snapshot."""
    key_down(VK_LWIN, extended=True)
    key_down(VK_SNAPSHOT, extended=True)
    time.sleep(0.05)
    key_up(VK_SNAPSHOT, extended=True)
    key_up(VK_LWIN, extended=True)


def release_all():
    """Safety: make sure the hot keys are not left held down."""
    key_up(VK_RIGHT, extended=True)
    key_up(VK_SNAPSHOT, extended=True)
    key_up(VK_LWIN, extended=True)


def main():
    parser = argparse.ArgumentParser(
        description="Repeatedly press Win+Screenshot, then Right Arrow."
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=None,
        help="Number of cycles (Win+Screenshot + Right Arrow). Default: infinite.",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between actions. Default: 1.0.",
    )
    args = parser.parse_args()

    print("This script will repeatedly press Win+Screenshot, then Right Arrow.")
    print(f"Delay between actions: {args.delay}s")
    if args.count:
        print(f"Cycles: {args.count}")
    else:
        print("Cycles: infinite (stop with Ctrl+C)")
    print()
    input("Press Enter to start...")

    try:
        count = 0
        while args.count is None or count < args.count:
            press_win_screenshot()
            time.sleep(args.delay)
            key_tap(VK_RIGHT, extended=True)
            count += 1
            print(f"Cycle #{count}")
            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        release_all()


if __name__ == "__main__":
    main()
