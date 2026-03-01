#!/usr/bin/env python3
"""
Vision System Integration for Automation
Combines screen capture with input automation for:
- Visual feedback validation
- Element detection and targeting
- Success/failure verification
- Adaptive automation based on visual state
"""

import time
import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from PIL import Image
import threading
import logging

from .universal_input import UniversalInputController
from .gui_elements import GUIAutomator, GUIElement
from .browser_automation import BrowserAutomator
from .game_automation import GameAutomator

class VisionAutomationIntegrator:
    """
    Integrates vision system with automation for intelligent, adaptive control
    """
    
    def __init__(self, session_path: str = "claude_session"):
        self.session_path = session_path
        
        # Initialize automation components
        self.input_controller = UniversalInputController(vision_integration=True, session_path=session_path)
        self.gui_automator = GUIAutomator(session_path)
        self.browser_automator = None  # Created on demand
        self.game_automator = GameAutomator(session_path)
        
        # Vision state tracking
        self.current_screenshot = None
        self.previous_screenshot = None
        self.screenshot_history = []
        self.max_history = 10
        
        # Automation state
        self.automation_active = False
        self.last_action_time = 0
        self.action_results = []
        
        # Vision monitoring
        self.monitor_thread = None
        self.monitoring = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_vision_monitoring(self, interval: float = 1.0):
        """Start continuous vision monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._vision_monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_vision_monitoring(self):
        """Stop vision monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _vision_monitor_loop(self, interval: float):
        """Main vision monitoring loop"""
        while self.monitoring:
            try:
                self.update_vision_state()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Vision monitoring error: {e}")
                time.sleep(interval)
                
    def update_vision_state(self) -> bool:
        """Update current vision state from screen capture"""
        try:
            # Check if vision service is providing screenshots
            current_view_path = os.path.join(self.session_path, "current_view.png")
            
            if os.path.exists(current_view_path):
                # Load from vision service
                screenshot = cv2.imread(current_view_path)
                
                # Update screenshot history
                if self.current_screenshot is not None:
                    self.previous_screenshot = self.current_screenshot.copy()
                    self.screenshot_history.append(self.current_screenshot.copy())
                    
                    if len(self.screenshot_history) > self.max_history:
                        self.screenshot_history.pop(0)
                        
                self.current_screenshot = screenshot
                return True
            else:
                # Fallback: take screenshot directly
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                if self.current_screenshot is not None:
                    self.previous_screenshot = self.current_screenshot.copy()
                    self.screenshot_history.append(self.current_screenshot.copy())
                    
                    if len(self.screenshot_history) > self.max_history:
                        self.screenshot_history.pop(0)
                        
                self.current_screenshot = screenshot_cv
                return True
                
        except Exception as e:
            self.logger.error(f"Vision state update failed: {e}")
            return False
            
    def detect_visual_change(self, threshold: float = 0.05) -> bool:
        """Detect if significant visual change occurred"""
        if self.current_screenshot is None or self.previous_screenshot is None:
            return False
            
        try:
            # Calculate structural similarity
            current_gray = cv2.cvtColor(self.current_screenshot, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(self.previous_screenshot, cv2.COLOR_BGR2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(current_gray, previous_gray)
            
            # Calculate percentage of changed pixels
            total_pixels = diff.shape[0] * diff.shape[1]
            changed_pixels = np.count_nonzero(diff > 30)  # Threshold for significant change
            change_percentage = changed_pixels / total_pixels
            
            return change_percentage > threshold
            
        except Exception as e:
            self.logger.error(f"Change detection failed: {e}")
            return False
            
    def wait_for_visual_change(self, timeout: float = 10.0, 
                              threshold: float = 0.05) -> bool:
        """Wait for visual change to occur"""
        start_time = time.time()
        baseline_screenshot = self.current_screenshot.copy() if self.current_screenshot is not None else None
        
        while time.time() - start_time < timeout:
            self.update_vision_state()
            
            if baseline_screenshot is not None and self.current_screenshot is not None:
                # Compare with baseline
                try:
                    baseline_gray = cv2.cvtColor(baseline_screenshot, cv2.COLOR_BGR2GRAY)
                    current_gray = cv2.cvtColor(self.current_screenshot, cv2.COLOR_BGR2GRAY)
                    
                    diff = cv2.absdiff(baseline_gray, current_gray)
                    total_pixels = diff.shape[0] * diff.shape[1]
                    changed_pixels = np.count_nonzero(diff > 30)
                    change_percentage = changed_pixels / total_pixels
                    
                    if change_percentage > threshold:
                        return True
                        
                except Exception as e:
                    self.logger.error(f"Visual change detection error: {e}")
                    
            time.sleep(0.1)
            
        return False
        
    def wait_for_element(self, element_description: str, timeout: float = 10.0) -> Optional[GUIElement]:
        """Wait for specific GUI element to appear"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            self.update_vision_state()
            
            # Try to find element by text content
            element = self.gui_automator.find_element_by_text(element_description)
            if element:
                return element
                
            # Try to find buttons containing the text
            if self.current_screenshot is not None:
                buttons = self.gui_automator.detector.find_buttons(
                    self.current_screenshot, 
                    text_filter=element_description
                )
                if buttons:
                    return max(buttons, key=lambda b: b.confidence)
                    
            time.sleep(0.5)
            
        return None
        
    def click_and_verify(self, x: int, y: int, expected_change: str = None,
                        timeout: float = 5.0) -> bool:
        """Click and verify the action was successful"""
        try:
            # Take baseline screenshot
            self.update_vision_state()
            baseline = self.current_screenshot.copy() if self.current_screenshot is not None else None
            
            # Perform click
            success = self.input_controller.click(x, y)
            if not success:
                return False
                
            # Wait for change
            time.sleep(0.5)  # Brief pause for UI to respond
            
            if expected_change:
                # Wait for specific element to appear
                element = self.wait_for_element(expected_change, timeout)
                return element is not None
            else:
                # Wait for any visual change
                return self.wait_for_visual_change(timeout)
                
        except Exception as e:
            self.logger.error(f"Click and verify failed: {e}")
            return False
            
    def click_element_by_text_and_verify(self, text: str, expected_result: str = None,
                                       timeout: float = 10.0) -> bool:
        """Find element by text, click it, and verify success"""
        try:
            # Find element
            element = self.wait_for_element(text, timeout)
            if not element:
                self.logger.warning(f"Element not found: {text}")
                return False
                
            # Click and verify
            return self.click_and_verify(
                element.center_x, 
                element.center_y, 
                expected_result, 
                timeout
            )
            
        except Exception as e:
            self.logger.error(f"Click element by text failed: {e}")
            return False
            
    def fill_form_and_verify(self, fields: List[Dict[str, str]], 
                           submit_button_text: str = None) -> bool:
        """Fill form fields and optionally submit with verification"""
        try:
            success_count = 0
            
            for field_info in fields:
                field_value = field_info.get("value", "")
                field_selector = field_info.get("selector")
                field_index = field_info.get("index", 0)
                
                # Try GUI-based field filling first
                if self.gui_automator.fill_text_field(field_value, field_index):
                    success_count += 1
                    continue
                    
                # Fallback to browser automation if available
                if self.browser_automator and field_selector:
                    if self.browser_automator.fill_form_field(selector=field_selector, value=field_value):
                        success_count += 1
                        continue
                        
                self.logger.warning(f"Failed to fill field: {field_info}")
                
            # Submit form if button specified
            if submit_button_text:
                submit_success = self.click_element_by_text_and_verify(
                    submit_button_text, 
                    expected_result="success"  # Look for success indicators
                )
                return success_count == len(fields) and submit_success
                
            return success_count == len(fields)
            
        except Exception as e:
            self.logger.error(f"Form filling failed: {e}")
            return False
            
    def automate_workflow(self, workflow: List[Dict[str, Any]]) -> bool:
        """Execute a complex automation workflow with verification"""
        try:
            self.automation_active = True
            workflow_success = True
            
            for step_index, step in enumerate(workflow):
                step_type = step.get("type")
                step_name = step.get("name", f"Step {step_index + 1}")
                
                self.logger.info(f"Executing: {step_name}")
                
                step_success = False
                
                if step_type == "click":
                    if "text" in step:
                        step_success = self.click_element_by_text_and_verify(
                            step["text"],
                            step.get("expected_result"),
                            step.get("timeout", 10.0)
                        )
                    elif "x" in step and "y" in step:
                        step_success = self.click_and_verify(
                            step["x"], step["y"],
                            step.get("expected_result"),
                            step.get("timeout", 5.0)
                        )
                        
                elif step_type == "fill_form":
                    step_success = self.fill_form_and_verify(
                        step["fields"],
                        step.get("submit_button")
                    )
                    
                elif step_type == "wait":
                    if "element" in step:
                        element = self.wait_for_element(step["element"], step.get("timeout", 10.0))
                        step_success = element is not None
                    elif "change" in step:
                        step_success = self.wait_for_visual_change(step.get("timeout", 10.0))
                    else:
                        time.sleep(step.get("duration", 1.0))
                        step_success = True
                        
                elif step_type == "type":
                    step_success = self.input_controller.type_text(step["text"])
                    
                elif step_type == "key":
                    if "combination" in step:
                        step_success = self.input_controller.key_combination(step["combination"])
                    else:
                        step_success = self.input_controller.key_press(step["key"])
                        
                # Log step result
                result = {
                    "step": step_name,
                    "success": step_success,
                    "timestamp": time.time()
                }
                self.action_results.append(result)
                
                if not step_success:
                    self.logger.error(f"Step failed: {step_name}")
                    if step.get("required", True):
                        workflow_success = False
                        break
                        
                # Optional delay between steps
                delay = step.get("delay", 0.5)
                if delay > 0:
                    time.sleep(delay)
                    
            self.automation_active = False
            return workflow_success
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.automation_active = False
            return False
            
    def capture_automation_evidence(self, action_name: str) -> str:
        """Capture screenshot evidence of automation action"""
        try:
            evidence_dir = os.path.join(self.session_path, "automation_evidence")
            os.makedirs(evidence_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            filename = f"{action_name}_{timestamp}.png"
            filepath = os.path.join(evidence_dir, filename)
            
            if self.current_screenshot is not None:
                cv2.imwrite(filepath, self.current_screenshot)
                return filepath
            else:
                # Fallback screenshot
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.save(filepath)
                return filepath
                
        except Exception as e:
            self.logger.error(f"Evidence capture failed: {e}")
            return None
            
    def get_automation_report(self) -> Dict[str, Any]:
        """Generate automation execution report"""
        total_actions = len(self.action_results)
        successful_actions = sum(1 for result in self.action_results if result["success"])
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "actions": self.action_results.copy(),
            "automation_active": self.automation_active,
            "last_action_time": self.last_action_time
        }
        
    def save_automation_session(self, session_name: str = None):
        """Save automation session data"""
        try:
            if not session_name:
                timestamp = int(time.time())
                session_name = f"automation_session_{timestamp}"
                
            session_dir = os.path.join(self.session_path, "automation_sessions")
            os.makedirs(session_dir, exist_ok=True)
            
            session_file = os.path.join(session_dir, f"{session_name}.json")
            
            session_data = {
                "session_name": session_name,
                "timestamp": time.time(),
                "report": self.get_automation_report(),
                "screenshot_count": len(self.screenshot_history)
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            self.logger.info(f"Session saved: {session_file}")
            
        except Exception as e:
            self.logger.error(f"Session save failed: {e}")
            
    def cleanup(self):
        """Cleanup automation resources"""
        self.stop_vision_monitoring()
        
        if self.browser_automator:
            self.browser_automator.close_browser()
            
        if self.game_automator:
            self.game_automator.cleanup()
            
        # Save final session
        self.save_automation_session("final_cleanup")