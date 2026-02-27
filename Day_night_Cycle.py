"""
Shahriar Akbari 25.02.2026 

Day night cycle for bacteria growth with projector on windows

The code works through changes screen from internal to extended
"""
import os
import time
import subprocess
from datetime import datetime, timedelta

ON_TIME = (8, 0)     # 08:00  Day ---------> Change these two line for different timeframes for day and night cycle
OFF_TIME = (20, 0)   # 20:00  Night

ON_MODE = "/extend"   # change to /clone or /external if you prefer
OFF_MODE = "/internal"


def find_displayswitch():
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        os.path.join(windir, "Sysnative", "DisplaySwitch.exe"),
        os.path.join(windir, "System32", "DisplaySwitch.exe"),
        os.path.join(windir, "SysWOW64", "DisplaySwitch.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("DisplaySwitch.exe not found.")


EXE = find_displayswitch()


def ds(mode: str):
    # Launch without cmd/start parsing issues
    subprocess.Popen([EXE, mode])


def next_occurrence(hour: int, minute: int) -> datetime:
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def sleep_until(dt: datetime):
    while True:
        remaining = (dt - datetime.now()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))  # wake up at least every minute


def is_between(now: datetime, start_hm, end_hm) -> bool:
    """Return True if now time is in [start, end) when start < end in same day."""
    start = now.replace(hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0)
    end = now.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
    return start <= now < end


def main():
    print("Starting display scheduler: ON at 08:00, OFF at 20:00")

    while True:
        now = datetime.now()

        # Decide what state we should be in right now
        if is_between(now, ON_TIME, OFF_TIME):
            # We are in the ON window (08:00–20:00)
            print("Setting ON mode now:", ON_MODE)
            ds(ON_MODE)
            next_switch = next_occurrence(OFF_TIME[0], OFF_TIME[1])
            print("Next switch (OFF) at:", next_switch)
        else:
            # We are in the OFF window (20:00–08:00)
            print("Setting OFF mode now:", OFF_MODE)
            ds(OFF_MODE)
            next_switch = next_occurrence(ON_TIME[0], ON_TIME[1])
            print("Next switch (ON) at:", next_switch)

        sleep_until(next_switch)


if __name__ == "__main__":
    main()
