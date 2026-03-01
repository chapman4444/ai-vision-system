#!/usr/bin/env python3
"""
Simple Automated Calibration Test
Direct automation without complex dependencies
"""

import time
import pyautogui

def run_simple_calibration():
    """Run simple automated calibration test"""
    
    print("🎯 Starting Simple Automated Mouse Calibration")
    print("=" * 50)
    
    # Disable pyautogui failsafe for this test
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.1
    
    print("Waiting 3 seconds for GUI to be ready...")
    time.sleep(3)
    
    try:
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width}x{screen_height}")
        
        # Step 1: Try to click Auto Test button
        print("\nStep 1: Clicking Auto Test button...")
        
        # Auto Test button should be around the top of screen
        button_y = 60  # Approximate height of control buttons
        auto_test_x = 300  # Approximate position of Auto Test button
        
        pyautogui.click(auto_test_x, button_y)
        print("✓ Auto Test button clicked!")
        
        print("\n🤖 Automated calibration should now be running!")
        print("The GUI will automatically click targets across the screen...")
        
        # Wait and monitor for completion
        print("\nMonitoring calibration progress...")
        for i in range(30):  # Monitor for 30 seconds
            time.sleep(1)
            print(f"Time elapsed: {i+1}s", end='\r')
            
        print(f"\n\n✓ Calibration monitoring complete!")
        
        # Try to take a screenshot of results
        print("\nTaking screenshot of calibration results...")
        screenshot = pyautogui.screenshot()
        screenshot.save("calibration_results_screenshot.png")
        print("✓ Screenshot saved as calibration_results_screenshot.png")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        pyautogui.FAILSAFE = True  # Re-enable failsafe
        
    print("\n🎉 Simple calibration test finished!")
    print("Check the GUI window for detailed results")
    print("Press ESC in the GUI to exit fullscreen mode")

if __name__ == "__main__":
    run_simple_calibration()