#!/usr/bin/env python3
"""
Automated Calibration Test Runner
Uses the automation system to perform the mouse calibration test
"""

import sys
import os
import time
import pyautogui

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.automation.universal_input import UniversalInputController
from src.automation.vision_integration import VisionAutomationIntegrator

def run_automated_calibration():
    """Run automated mouse calibration using the automation system"""
    
    print("🎯 Starting Automated Mouse Calibration Test")
    print("=" * 50)
    
    # Initialize automation components
    input_controller = UniversalInputController(vision_integration=True)
    vision_integrator = VisionAutomationIntegrator()
    
    # Start vision monitoring
    vision_integrator.start_vision_monitoring(interval=1.0)
    
    print("Waiting 3 seconds for GUI to be ready...")
    time.sleep(3)
    
    try:
        # Step 1: Click "Auto Test" button
        print("Step 1: Looking for 'Auto Test' button...")
        
        # Try to find and click Auto Test button
        if vision_integrator.click_element_by_text_and_verify("Auto Test", timeout=5.0):
            print("✓ Auto Test button clicked!")
        else:
            # Fallback - click at expected button location
            print("Fallback: Clicking at expected Auto Test button location...")
            input_controller.click(400, 60)  # Approximate button location
        
        print("\n🤖 Automated calibration test started!")
        print("The system will now automatically click targets...")
        print("Watch the fullscreen GUI for the automated clicking!")
        
        # Wait for calibration to complete
        print("\nWaiting for calibration to complete...")
        time.sleep(30)  # Give time for auto test to run
        
        print("\n✓ Calibration test should be complete!")
        print("Check the GUI for results!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        vision_integrator.stop_vision_monitoring()
        
    print("\n🎉 Automated calibration test finished!")
    print("Results should be displayed in the GUI window")

if __name__ == "__main__":
    run_automated_calibration()