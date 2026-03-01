#!/usr/bin/env python3
"""
Universal Input Automation System
Provides low-level mouse and keyboard control that works across all Windows applications:
- Browsers, text editors, games, native apps
- Hardware-level simulation for maximum compatibility
- Integration with vision system for feedback
"""

import pyautogui
import time
import ctypes
import ctypes.wintypes
from ctypes import wintypes
import json
import os
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import threading
import logging

# Win32 API constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

# Keyboard event constants
KEYEVENTF_KEYUP = 0x0002
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_ALT = 0x12

class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right" 
    MIDDLE = "middle"

class UniversalInputController:
    """
    Universal input controller that works across all Windows applications
    Uses Win32 APIs for maximum compatibility with games and protected apps
    """
    
    def __init__(self, vision_integration=True, session_path="claude_session"):
        # Initialize PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05  # Small pause between actions
        
        # Win32 API setup
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
        # Vision integration
        self.vision_integration = vision_integration
        self.session_path = session_path
        
        # State tracking
        self.current_position = (0, 0)
        self.last_action_time = 0
        self.action_history = []
        
        # Get screen dimensions
        self.screen_width = self.user32.GetSystemMetrics(0)
        self.screen_height = self.user32.GetSystemMetrics(1)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position using Win32 API"""
        point = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return (point.x, point.y)
        
    def move_mouse(self, x: int, y: int, duration: float = 0.5, smooth: bool = True) -> bool:
        """
        Move mouse to position with optional smooth movement
        Args:
            x, y: Target coordinates
            duration: Time to take for movement
            smooth: Whether to use smooth movement
        """
        try:
            if smooth:
                # Use pyautogui for smooth movement
                pyautogui.moveTo(x, y, duration=duration)
            else:
                # Direct Win32 movement for games
                self.user32.SetCursorPos(x, y)
                
            self.current_position = (x, y)
            self.log_action(f"move_mouse", {"x": x, "y": y, "duration": duration})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move mouse: {e}")
            return False
            
    def click(self, x: int = None, y: int = None, button: MouseButton = MouseButton.LEFT, 
              clicks: int = 1, interval: float = 0.1) -> bool:
        """
        Click at position with specified button
        Args:
            x, y: Click coordinates (None for current position)
            button: Mouse button to click
            clicks: Number of clicks
            interval: Interval between multiple clicks
        """
        try:
            if x is not None and y is not None:
                self.move_mouse(x, y, duration=0.1)
                
            current_x, current_y = self.get_mouse_position()
            
            for i in range(clicks):
                if i > 0:
                    time.sleep(interval)
                    
                # Use Win32 API for raw mouse input (better game compatibility)
                if button == MouseButton.LEFT:
                    self._win32_click(current_x, current_y, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
                elif button == MouseButton.RIGHT:
                    self._win32_click(current_x, current_y, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
                elif button == MouseButton.MIDDLE:
                    self._win32_click(current_x, current_y, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)
                    
            self.log_action("click", {"x": current_x, "y": current_y, "button": button.value, "clicks": clicks})
            
            # Take screenshot after click for vision feedback
            if self.vision_integration:
                self.capture_feedback(f"click_{button.value}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to click: {e}")
            return False
            
    def _win32_click(self, x: int, y: int, down_flag: int, up_flag: int):
        """Low-level Win32 click implementation"""
        # Mouse down
        self.user32.mouse_event(down_flag, x, y, 0, 0)
        time.sleep(0.01)  # Brief delay
        # Mouse up  
        self.user32.mouse_event(up_flag, x, y, 0, 0)
        
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: float = 1.0, button: MouseButton = MouseButton.LEFT) -> bool:
        """
        Drag from start to end position
        """
        try:
            # Move to start position
            self.move_mouse(start_x, start_y, duration=0.1)
            
            # Mouse down
            if button == MouseButton.LEFT:
                self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, start_x, start_y, 0, 0)
            elif button == MouseButton.RIGHT:
                self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, start_x, start_y, 0, 0)
                
            # Drag to end position
            pyautogui.dragTo(end_x, end_y, duration=duration)
            
            # Mouse up
            if button == MouseButton.LEFT:
                self.user32.mouse_event(MOUSEEVENTF_LEFTUP, end_x, end_y, 0, 0)
            elif button == MouseButton.RIGHT:
                self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, end_x, end_y, 0, 0)
                
            self.log_action("drag", {"start": (start_x, start_y), "end": (end_x, end_y), "duration": duration})
            
            if self.vision_integration:
                self.capture_feedback("drag")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to drag: {e}")
            return False
            
    def scroll(self, x: int, y: int, scrolls: int = 3, direction: str = "up") -> bool:
        """
        Scroll at position
        Args:
            x, y: Scroll position
            scrolls: Number of scroll steps
            direction: "up" or "down"
        """
        try:
            self.move_mouse(x, y, duration=0.1)
            
            scroll_amount = 120 * scrolls  # Standard scroll amount
            if direction.lower() == "down":
                scroll_amount = -scroll_amount
                
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, x, y, scroll_amount, 0)
            
            self.log_action("scroll", {"x": x, "y": y, "scrolls": scrolls, "direction": direction})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scroll: {e}")
            return False
            
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """
        Type text using PyAutoGUI (handles unicode properly)
        """
        try:
            pyautogui.typewrite(text, interval=interval)
            self.log_action("type_text", {"text": text[:50] + "..." if len(text) > 50 else text})
            
            if self.vision_integration:
                self.capture_feedback("type_text")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to type text: {e}")
            return False
            
    def key_press(self, key: str, modifiers: List[str] = None) -> bool:
        """
        Press key with optional modifiers
        Args:
            key: Key to press (e.g., 'enter', 'tab', 'a')
            modifiers: List of modifier keys (e.g., ['ctrl', 'shift'])
        """
        try:
            if modifiers:
                # Press modifiers + key
                pyautogui.hotkey(*modifiers, key)
            else:
                pyautogui.press(key)
                
            self.log_action("key_press", {"key": key, "modifiers": modifiers})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to press key: {e}")
            return False
            
    def key_combination(self, keys: List[str]) -> bool:
        """Press key combination (e.g., ['ctrl', 'c'])"""
        try:
            pyautogui.hotkey(*keys)
            self.log_action("key_combination", {"keys": keys})
            return True
        except Exception as e:
            self.logger.error(f"Failed key combination: {e}")
            return False
            
    def wait_for_image(self, image_path: str, timeout: float = 10.0, 
                      region: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int]]:
        """
        Wait for image to appear on screen
        Returns: (x, y) position of found image or None if timeout
        """
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    location = pyautogui.locateOnScreen(image_path, region=region, confidence=0.8)
                    if location:
                        center = pyautogui.center(location)
                        return (center.x, center.y)
                except pyautogui.ImageNotFoundException:
                    pass
                time.sleep(0.5)
            return None
        except Exception as e:
            self.logger.error(f"Failed to wait for image: {e}")
            return None
            
    def find_and_click(self, image_path: str, timeout: float = 10.0,
                      button: MouseButton = MouseButton.LEFT) -> bool:
        """Find image and click it"""
        position = self.wait_for_image(image_path, timeout)
        if position:
            return self.click(position[0], position[1], button)
        return False
        
    def capture_feedback(self, action: str):
        """Capture screenshot after action for vision feedback"""
        if not self.vision_integration:
            return
            
        try:
            feedback_dir = os.path.join(self.session_path, "automation_feedback")
            os.makedirs(feedback_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            filename = f"{action}_{timestamp}.png"
            filepath = os.path.join(feedback_dir, filename)
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to capture feedback: {e}")
            
    def log_action(self, action: str, params: Dict[str, Any]):
        """Log action for debugging and replay"""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "params": params,
            "mouse_position": self.get_mouse_position()
        }
        
        self.action_history.append(log_entry)
        self.last_action_time = time.time()
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history.pop(0)
            
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history for analysis"""
        return self.action_history.copy()
        
    def save_action_log(self, filename: str = None):
        """Save action history to file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"action_log_{timestamp}.json"
            
        log_path = os.path.join(self.session_path, "automation_logs")
        os.makedirs(log_path, exist_ok=True)
        
        filepath = os.path.join(log_path, filename)
        with open(filepath, 'w') as f:
            json.dump(self.action_history, f, indent=2)
            
    def replay_actions(self, log_file: str, speed_multiplier: float = 1.0):
        """Replay actions from log file"""
        try:
            with open(log_file, 'r') as f:
                actions = json.load(f)
                
            prev_time = None
            for action in actions:
                if prev_time:
                    delay = (action['timestamp'] - prev_time) / speed_multiplier
                    time.sleep(max(0.01, delay))
                    
                # Replay the action
                action_type = action['action']
                params = action['params']
                
                if action_type == "click":
                    self.click(params.get('x'), params.get('y'), 
                             MouseButton(params.get('button', 'left')))
                elif action_type == "move_mouse":
                    self.move_mouse(params['x'], params['y'])
                elif action_type == "type_text":
                    self.type_text(params['text'])
                # Add other action types as needed
                
                prev_time = action['timestamp']
                
        except Exception as e:
            self.logger.error(f"Failed to replay actions: {e}")

# Factory function for easy instantiation
def create_input_controller(vision_integration=True) -> UniversalInputController:
    """Create and return a configured input controller"""
    return UniversalInputController(vision_integration=vision_integration)