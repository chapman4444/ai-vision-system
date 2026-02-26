"""Auto-calibration: finds and clicks the 9 red targets automatically.

Requires the viewer.pyw calibration tool to be open (System Menu → Test Coordinate System).
Uses scipy connected-component labeling to find the largest red blob (the active target)
and clicks its center. Waits for the target to move before clicking the next one.

Usage:
    python auto_calibrate.py                # Default: monitor 1
    python auto_calibrate.py --monitor 2    # Use monitor 2
"""
import argparse
import os
import time
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

SCREENSHOT = "llm_screenshots/current_view.png"
TARGETS = 9


def find_red_bullseye():
    """Find center of the largest red blob (the active calibration target).

    Returns (x, y) in image pixel coordinates, or None if no target found.
    Filters by blob size (>500px) to ignore the cursor crosshair.
    """
    try:
        img = Image.open(SCREENSHOT).convert("RGB")
        img.load()
    except Exception:
        return None
    arr = np.array(img)
    red_mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    if not np.any(red_mask):
        return None
    labeled, n = ndimage.label(red_mask)
    if n == 0:
        return None
    sizes = ndimage.sum(red_mask, labeled, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    if sizes[biggest - 1] < 500:
        return None
    cy, cx = ndimage.center_of_mass(red_mask, labeled, biggest)
    return int(cx), int(cy)


def click(x, y, monitor=1):
    """Send a click at screen coordinates via viewer_cli.py."""
    subprocess.run(
        ["python", "viewer_cli.py", "click", "--monitor", str(monitor),
         "--x", str(x), "--y", str(y)],
        capture_output=True
    )


def wait_for_new_target(prev_center, timeout=8.0):
    """Wait until the target moves to a new position or calibration ends.

    Returns the new target center, or None if calibration appears to have ended
    (no large red blob found after timeout).
    """
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(0.2)
        center = find_red_bullseye()
        if center is None:
            # Could be mid-write screenshot or calibration ended.
            # Keep waiting briefly to distinguish.
            continue
        if prev_center is None:
            return center
        dx = abs(center[0] - prev_center[0])
        dy = abs(center[1] - prev_center[1])
        if dx > 30 or dy > 30:
            return center
    # Timeout — check one more time
    center = find_red_bullseye()
    if center is not None and prev_center is not None:
        dx = abs(center[0] - prev_center[0])
        dy = abs(center[1] - prev_center[1])
        if dx <= 30 and dy <= 30:
            return None  # Same position after timeout = probably done
    return center


def main():
    parser = argparse.ArgumentParser(description="Auto-calibrate by clicking 9 red targets")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (default: 1)")
    args = parser.parse_args()

    print(f"Auto-calibration: monitor {args.monitor}, waiting for {TARGETS} targets...")
    hits = 0
    prev_center = None

    for i in range(TARGETS):
        if i == 0:
            center = None
            for _ in range(30):
                center = find_red_bullseye()
                if center:
                    break
                time.sleep(0.2)
            if not center:
                print("No calibration target found. Is the calibration tool open?")
                return
        else:
            center = wait_for_new_target(prev_center)

        if center is None:
            if hits > 0:
                print(f"Calibration complete ({hits}/{TARGETS} targets hit)")
            else:
                print(f"Target {i+1}: no target found")
            return

        print(f"  {i+1}/{TARGETS}: ({center[0]}, {center[1]})")
        click(center[0], center[1], args.monitor)
        hits += 1
        prev_center = center

    print(f"Calibration complete ({hits}/{TARGETS} targets hit)")


if __name__ == "__main__":
    main()
