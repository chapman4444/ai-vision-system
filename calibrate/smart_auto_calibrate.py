#!/usr/bin/env python3
"""
Smart Automated Calibration - Better Target Detection
Scans entire screen systematically to find red targets
"""

import time
import pyautogui
import random
import numpy as np
from PIL import Image

def find_red_targets_on_screen():
    """Scan entire screen to find red target circles"""
    print("Scanning screen for red targets...")
    
    screenshot = pyautogui.screenshot()
    screen_array = np.array(screenshot)
    
    # Find red pixels (R > 200, G < 50, B < 50)
    red_mask = (screen_array[:, :, 0] > 200) & (screen_array[:, :, 1] < 50) & (screen_array[:, :, 2] < 50)
    
    # Find coordinates where red pixels exist
    red_coords = np.where(red_mask)
    
    if len(red_coords[0]) == 0:
        return []
    
    # Group nearby red pixels into target centers
    red_points = list(zip(red_coords[1], red_coords[0]))  # (x, y) format
    
    targets = []
    used_points = set()
    
    for x, y in red_points:
        if (x, y) in used_points:
            continue
            
        # Find all points within 30 pixels (target radius)
        cluster = []
        for px, py in red_points:
            if (px, py) not in used_points:
                distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if distance < 30:
                    cluster.append((px, py))
                    used_points.add((px, py))
        
        if len(cluster) >= 5:  # Need at least 5 red pixels to be a target
            # Calculate center of cluster
            center_x = sum(px for px, py in cluster) // len(cluster)
            center_y = sum(py for px, py in cluster) // len(cluster)
            targets.append((center_x, center_y))
    
    print(f"Found {len(targets)} potential targets")
    return targets

def run_smart_calibration():
    """Run smart automated calibration with better target detection"""
    
    print("🎯 Starting Smart Automated Calibration")
    print("=" * 50)
    
    # Launch the calibration GUI first
    print("Step 1: Launching fullscreen calibration GUI...")
    import subprocess
    gui_process = subprocess.Popen(['python', 'fullscreen_calibration.py'])
    
    # Configure pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.05
    
    print("Step 2: Waiting 5 seconds for GUI to initialize and auto-start...")
    time.sleep(5)
    
    try:
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width} × {screen_height}")
        
        click_count = 0
        max_clicks = 50
        no_target_count = 0
        
        while click_count < max_clicks and no_target_count < 15:
            # Scan for red targets
            targets = find_red_targets_on_screen()
            
            if targets:
                no_target_count = 0  # Reset counter
                
                # Click the first target found
                target_x, target_y = targets[0]
                
                # Add small random offset for realistic clicking
                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-3, 3)
                
                click_x = target_x + offset_x
                click_y = target_y + offset_y
                
                print(f"Click {click_count + 1}: Target at ({target_x}, {target_y}) -> Click at ({click_x}, {click_y})")
                
                # Perform the click
                pyautogui.click(click_x, click_y)
                click_count += 1
                
                # Wait for target to change
                time.sleep(1.8)
                
            else:
                no_target_count += 1
                print(f"No targets found (attempt {no_target_count}/15)")
                
                if no_target_count >= 15:
                    print("No targets found for 15 consecutive scans - calibration complete!")
                    break
                    
                # Wait a bit and try again
                time.sleep(0.5)
                
        print(f"\n✓ Smart calibration completed!")
        print(f"Total accurate clicks: {click_count}")
        print("Press ESC in GUI to view results")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pyautogui.FAILSAFE = True

if __name__ == "__main__":
    # Check if numpy is available
    try:
        import numpy as np
        run_smart_calibration()
    except ImportError:
        print("NumPy not available - falling back to basic detection")
        print("Install with: pip install numpy")
        
        # Fallback without numpy
        print("🎯 Starting Basic Automated Calibration")
        pyautogui.FAILSAFE = False
        time.sleep(3)
        
        try:
            click_count = 0
            # Just try clicking at some expected target positions with delays
            expected_positions = [
                (50, 50), (960, 50), (1870, 50),    # Top row
                (50, 540), (960, 540), (1870, 540), # Middle row  
                (50, 1030), (960, 1030), (1870, 1030), # Bottom row
            ]
            
            for x, y in expected_positions:
                print(f"Trying click {click_count + 1} at ({x}, {y})")
                pyautogui.click(x, y)
                click_count += 1
                time.sleep(2)
                if click_count >= 15:
                    break
                    
        finally:
            pyautogui.FAILSAFE = True