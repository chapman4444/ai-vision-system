#!/usr/bin/env python3
"""
Automated Calibration Test Runner
Automatically performs the fullscreen calibration test
"""

import time
import pyautogui
import random
import json

def run_automated_calibration():
    """Run automated calibration clicking"""
    
    print("🎯 Starting Automated Fullscreen Calibration Test")
    print("=" * 60)
    
    # Disable failsafe and set timing
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.1
    
    print("Waiting 5 seconds for fullscreen GUI to initialize...")
    time.sleep(5)
    
    try:
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width} × {screen_height}")
        
        # Expected target sequence based on the GUI logic
        margin = 50
        
        # Predict target positions (same logic as GUI)
        expected_targets = []
        
        # Corners
        expected_targets.extend([
            (margin, margin),
            (screen_width - margin, margin),
            (margin, screen_height - margin),
            (screen_width - margin, screen_height - margin)
        ])
        
        # Edges
        expected_targets.extend([
            (screen_width // 2, margin),
            (screen_width // 2, screen_height - margin),
            (margin, screen_height // 2),
            (screen_width - margin, screen_height // 2)
        ])
        
        # Center
        expected_targets.append((screen_width // 2, screen_height // 2))
        
        # Quarters
        expected_targets.extend([
            (screen_width // 4, screen_height // 4),
            (3 * screen_width // 4, screen_height // 4),
            (screen_width // 4, 3 * screen_height // 4),
            (3 * screen_width // 4, 3 * screen_height // 4)
        ])
        
        # Grid points
        for x_ratio in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
            for y_ratio in [0.25, 0.5, 0.75]:
                x = int(screen_width * x_ratio)
                y = int(screen_height * y_ratio)
                expected_targets.append((x, y))
        
        print(f"Generated {len(expected_targets)} expected target positions")
        
        # Wait for calibration to auto-start
        print("\nWaiting for auto-start...")
        time.sleep(3)
        
        print("\n🤖 Beginning automated clicking sequence...")
        
        # Since targets are randomized, we'll use a different approach
        # Look for red pixels to find targets, then click them
        
        click_count = 0
        max_clicks = 40  # Maximum expected targets
        
        while click_count < max_clicks:
            try:
                # Take screenshot to find red target
                screenshot = pyautogui.screenshot()
                
                # Look for red pixels (targets)
                found_target = False
                
                # Sample key positions where targets likely appear
                sample_positions = expected_targets + [
                    # Add some random positions in case we miss any
                    (random.randint(100, screen_width-100), 
                     random.randint(100, screen_height-100))
                    for _ in range(5)
                ]
                
                for target_x, target_y in sample_positions:
                    try:
                        # Check multiple pixels around this position for red target
                        found_red = False
                        for dx in [-10, -5, 0, 5, 10]:
                            for dy in [-10, -5, 0, 5, 10]:
                                try:
                                    check_x = max(0, min(screen_width-1, target_x + dx))
                                    check_y = max(0, min(screen_height-1, target_y + dy))
                                    pixel = screenshot.getpixel((check_x, check_y))
                                    
                                    # More precise red detection (R > 200, G < 50, B < 50)
                                    if pixel[0] > 200 and pixel[1] < 50 and pixel[2] < 50:
                                        found_red = True
                                        target_x, target_y = check_x, check_y  # Use exact red pixel location
                                        break
                                except:
                                    continue
                            if found_red:
                                break
                        
                        if found_red:
                            # Found a target! Click it with slight randomization
                            offset_x = random.randint(-8, 8)
                            offset_y = random.randint(-8, 8)
                            
                            click_x = target_x + offset_x
                            click_y = target_y + offset_y
                            
                            print(f"Click {click_count + 1}: ({click_x}, {click_y}) [Target: ({target_x}, {target_y})]")
                            
                            pyautogui.click(click_x, click_y)
                            click_count += 1
                            found_target = True
                            
                            # Wait for feedback and next target
                            time.sleep(2)
                            break
                            
                    except Exception as e:
                        continue
                
                if not found_target:
                    # NO FALLBACK CLICKS! Only click if we actually find a target
                    print(f"No target found in scan {click_count + 1}, waiting...")
                    time.sleep(0.5)  # Wait briefly and try again
                
                # Check if calibration is complete (look for completion text)
                try:
                    completion_region = pyautogui.screenshot(region=(
                        screen_width//4, screen_height//4,
                        screen_width//2, screen_height//2
                    ))
                    # If we haven't found a target for a while, probably done
                    if click_count > 15 and not found_target:
                        consecutive_misses = consecutive_misses + 1 if 'consecutive_misses' in locals() else 1
                        if consecutive_misses > 10:  # 10 consecutive scans without finding target
                            print("No targets found for 10 scans - calibration appears complete!")
                            break
                    else:
                        consecutive_misses = 0
                except:
                    pass
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error during clicking: {e}")
                time.sleep(1)
                
        print(f"\n✓ Automated calibration completed!")
        print(f"Total clicks made: {click_count}")
        print("Check the fullscreen GUI for detailed results")
        print("Press ESC in the GUI to exit")
        
    except Exception as e:
        print(f"Automation error: {e}")
    finally:
        pyautogui.FAILSAFE = True

if __name__ == "__main__":
    run_automated_calibration()