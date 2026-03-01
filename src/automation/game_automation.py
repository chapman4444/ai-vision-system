#!/usr/bin/env python3
"""
Game Automation Support
Specialized automation for video games with:
- DirectInput support for bypassing game protections
- High-precision timing for real-time games
- Game-specific input patterns and macros
- Anti-detection measures
"""

import time
import ctypes
from ctypes import wintypes, windll
import struct
import threading
from typing import Dict, List, Tuple, Optional, Callable
import json
import numpy as np
from .universal_input import UniversalInputController, MouseButton

# DirectInput constants
DIGCF_PRESENT = 0x00000002
DIGCF_DEVICEINTERFACE = 0x00000010
SPDRP_DEVICEDESC = 0x00000000

# Game input constants
VK_WASD = {'w': 0x57, 's': 0x53, 'a': 0x41, 'd': 0x44}
VK_ARROWS = {'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27}

class GameInputController:
    """
    Specialized input controller for games
    Uses DirectInput and low-level Windows APIs for maximum compatibility
    """
    
    def __init__(self):
        self.user32 = windll.user32
        self.kernel32 = windll.kernel32
        
        # Input state tracking
        self.held_keys = set()
        self.mouse_state = {"x": 0, "y": 0, "buttons": set()}
        
        # Timing control
        self.base_delay = 0.001  # 1ms base delay
        self.click_duration = 0.050  # 50ms click hold
        
        # Anti-detection randomization
        self.humanize = True
        self.timing_variance = 0.02  # ±20ms variance
        
    def send_key_input(self, vk_code: int, key_down: bool = True):
        """Send raw key input using Windows API"""
        scan_code = self.user32.MapVirtualKeyW(vk_code, 0)
        flags = 0 if key_down else 2  # KEYEVENTF_KEYUP = 2
        
        # Use hardware scan codes for better game compatibility
        self.user32.keybd_event(vk_code, scan_code, flags, 0)
        
        if self.humanize:
            time.sleep(self.base_delay + np.random.uniform(-self.timing_variance, self.timing_variance))
            
    def press_key(self, key: str, duration: float = None) -> bool:
        """Press and release a key with optional hold duration"""
        try:
            vk_code = self._get_vk_code(key)
            if not vk_code:
                return False
                
            # Key down
            self.send_key_input(vk_code, True)
            self.held_keys.add(key)
            
            # Hold duration
            hold_time = duration or self.click_duration
            if self.humanize:
                hold_time += np.random.uniform(-self.timing_variance, self.timing_variance)
            time.sleep(max(0.001, hold_time))
            
            # Key up
            self.send_key_input(vk_code, False)
            self.held_keys.discard(key)
            
            return True
            
        except Exception as e:
            print(f"Key press failed: {e}")
            return False
            
    def hold_key(self, key: str) -> bool:
        """Hold key down (doesn't release automatically)"""
        try:
            vk_code = self._get_vk_code(key)
            if not vk_code:
                return False
                
            if key not in self.held_keys:
                self.send_key_input(vk_code, True)
                self.held_keys.add(key)
                
            return True
        except Exception as e:
            print(f"Hold key failed: {e}")
            return False
            
    def release_key(self, key: str) -> bool:
        """Release a held key"""
        try:
            vk_code = self._get_vk_code(key)
            if not vk_code:
                return False
                
            if key in self.held_keys:
                self.send_key_input(vk_code, False)
                self.held_keys.discard(key)
                
            return True
        except Exception as e:
            print(f"Release key failed: {e}")
            return False
            
    def release_all_keys(self):
        """Release all currently held keys"""
        for key in self.held_keys.copy():
            self.release_key(key)
            
    def send_mouse_input(self, x: int, y: int, flags: int, data: int = 0):
        """Send raw mouse input"""
        self.user32.mouse_event(flags, x, y, data, 0)
        
    def game_click(self, x: int, y: int, button: MouseButton = MouseButton.LEFT, 
                  hold_duration: float = None) -> bool:
        """Gaming-optimized click with precise timing"""
        try:
            # Move to position first
            self.user32.SetCursorPos(x, y)
            self.mouse_state["x"], self.mouse_state["y"] = x, y
            
            # Determine button flags
            if button == MouseButton.LEFT:
                down_flag, up_flag = 0x0002, 0x0004  # LEFTDOWN, LEFTUP
            elif button == MouseButton.RIGHT:
                down_flag, up_flag = 0x0008, 0x0010  # RIGHTDOWN, RIGHTUP
            elif button == MouseButton.MIDDLE:
                down_flag, up_flag = 0x0020, 0x0040  # MIDDLEDOWN, MIDDLEUP
            else:
                return False
                
            # Mouse down
            self.send_mouse_input(0, 0, down_flag)
            self.mouse_state["buttons"].add(button)
            
            # Hold duration
            hold_time = hold_duration or self.click_duration
            if self.humanize:
                hold_time += np.random.uniform(-self.timing_variance, self.timing_variance)
            time.sleep(max(0.001, hold_time))
            
            # Mouse up
            self.send_mouse_input(0, 0, up_flag)
            self.mouse_state["buttons"].discard(button)
            
            return True
            
        except Exception as e:
            print(f"Game click failed: {e}")
            return False
            
    def smooth_mouse_move(self, start_x: int, start_y: int, end_x: int, end_y: int,
                         duration: float = 0.5, steps: int = None) -> bool:
        """Smooth mouse movement for games requiring human-like motion"""
        try:
            if steps is None:
                # Calculate steps based on distance and duration
                distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
                steps = max(10, int(distance / 10))  # ~10 pixels per step
                
            step_duration = duration / steps
            
            for i in range(steps + 1):
                t = i / steps
                # Use easing function for more natural movement
                t_eased = self._ease_in_out_cubic(t)
                
                current_x = int(start_x + (end_x - start_x) * t_eased)
                current_y = int(start_y + (end_y - start_y) * t_eased)
                
                self.user32.SetCursorPos(current_x, current_y)
                
                if i < steps:  # Don't sleep after last step
                    sleep_time = step_duration
                    if self.humanize:
                        sleep_time += np.random.uniform(-step_duration * 0.1, step_duration * 0.1)
                    time.sleep(max(0.001, sleep_time))
                    
            self.mouse_state["x"], self.mouse_state["y"] = end_x, end_y
            return True
            
        except Exception as e:
            print(f"Smooth mouse move failed: {e}")
            return False
            
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for natural movement"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
            
    def _get_vk_code(self, key: str) -> Optional[int]:
        """Get virtual key code for a key string"""
        key = key.lower()
        
        # Common game keys
        key_map = {
            'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
            'q': 0x51, 'e': 0x45, 'r': 0x52, 't': 0x54, 'y': 0x59,
            'u': 0x55, 'i': 0x49, 'o': 0x4F, 'p': 0x50,
            'f': 0x46, 'g': 0x47, 'h': 0x48, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
            'z': 0x5A, 'x': 0x58, 'c': 0x43, 'v': 0x56, 'b': 0x42, 'n': 0x4E, 'm': 0x4D,
            'space': 0x20, 'enter': 0x0D, 'escape': 0x1B, 'tab': 0x09,
            'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12,
            'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73, 'f5': 0x74,
            'f6': 0x75, 'f7': 0x76, 'f8': 0x77, 'f9': 0x78, 'f10': 0x79,
            'f11': 0x7A, 'f12': 0x7B,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35,
            '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39, '0': 0x30,
        }
        
        return key_map.get(key)

class GameMacroSystem:
    """System for creating and executing game macros"""
    
    def __init__(self, game_input: GameInputController):
        self.game_input = game_input
        self.macros = {}
        self.running_macros = {}
        
    def create_macro(self, name: str, actions: List[Dict]) -> bool:
        """
        Create a macro from a list of actions
        Actions format: [{"type": "key", "key": "w", "duration": 0.1}, ...]
        """
        try:
            self.macros[name] = actions
            return True
        except Exception as e:
            print(f"Create macro failed: {e}")
            return False
            
    def execute_macro(self, name: str, repeat: int = 1, 
                     background: bool = False) -> bool:
        """Execute a macro"""
        if name not in self.macros:
            return False
            
        if background:
            thread = threading.Thread(
                target=self._execute_macro_thread, 
                args=(name, repeat),
                daemon=True
            )
            thread.start()
            return True
        else:
            return self._execute_macro_sequence(name, repeat)
            
    def _execute_macro_thread(self, name: str, repeat: int):
        """Execute macro in background thread"""
        self.running_macros[name] = True
        try:
            self._execute_macro_sequence(name, repeat)
        finally:
            self.running_macros.pop(name, None)
            
    def _execute_macro_sequence(self, name: str, repeat: int) -> bool:
        """Execute the actual macro sequence"""
        try:
            actions = self.macros[name]
            
            for _ in range(repeat):
                if name in self.running_macros and not self.running_macros[name]:
                    break  # Macro was stopped
                    
                for action in actions:
                    action_type = action.get("type")
                    
                    if action_type == "key":
                        key = action.get("key")
                        duration = action.get("duration", 0.05)
                        self.game_input.press_key(key, duration)
                        
                    elif action_type == "hold_key":
                        key = action.get("key")
                        self.game_input.hold_key(key)
                        
                    elif action_type == "release_key":
                        key = action.get("key")
                        self.game_input.release_key(key)
                        
                    elif action_type == "click":
                        x = action.get("x", 0)
                        y = action.get("y", 0)
                        button = MouseButton(action.get("button", "left"))
                        duration = action.get("duration", 0.05)
                        self.game_input.game_click(x, y, button, duration)
                        
                    elif action_type == "move":
                        x = action.get("x", 0)
                        y = action.get("y", 0)
                        duration = action.get("duration", 0.1)
                        current_x, current_y = self.game_input.mouse_state["x"], self.game_input.mouse_state["y"]
                        self.game_input.smooth_mouse_move(current_x, current_y, x, y, duration)
                        
                    elif action_type == "wait":
                        duration = action.get("duration", 0.1)
                        time.sleep(duration)
                        
                    # Small delay between actions
                    time.sleep(action.get("delay", 0.01))
                    
            return True
            
        except Exception as e:
            print(f"Macro execution failed: {e}")
            return False
            
    def stop_macro(self, name: str):
        """Stop a running background macro"""
        if name in self.running_macros:
            self.running_macros[name] = False
            
    def stop_all_macros(self):
        """Stop all running macros"""
        for name in list(self.running_macros.keys()):
            self.running_macros[name] = False

class GameProfileManager:
    """Manages game-specific automation profiles"""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        
    def create_profile(self, game_name: str, config: Dict) -> bool:
        """Create a game profile with specific configurations"""
        try:
            self.profiles[game_name] = {
                "config": config,
                "macros": {},
                "key_bindings": config.get("key_bindings", {}),
                "timing_settings": config.get("timing_settings", {
                    "base_delay": 0.001,
                    "click_duration": 0.050,
                    "humanize": True,
                    "variance": 0.02
                })
            }
            return True
        except Exception as e:
            print(f"Profile creation failed: {e}")
            return False
            
    def load_profile(self, game_name: str) -> bool:
        """Load a game profile"""
        if game_name not in self.profiles:
            return False
            
        self.current_profile = game_name
        return True
        
    def save_profiles_to_file(self, filename: str):
        """Save all profiles to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"Save profiles failed: {e}")
            
    def load_profiles_from_file(self, filename: str):
        """Load profiles from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.profiles = json.load(f)
        except Exception as e:
            print(f"Load profiles failed: {e}")

# Common game automation patterns
class CommonGamePatterns:
    """Pre-built automation patterns for common game scenarios"""
    
    def __init__(self, game_input: GameInputController, macro_system: GameMacroSystem):
        self.game_input = game_input
        self.macro_system = macro_system
        
    def create_movement_macro(self, name: str, pattern: str, duration: float = 1.0):
        """Create movement macros (e.g., 'wasd_circle', 'strafe_left_right')"""
        actions = []
        
        if pattern == "wasd_circle":
            # Circular movement pattern
            keys = ['w', 'd', 's', 'a']
            step_duration = duration / len(keys)
            for key in keys:
                actions.append({"type": "hold_key", "key": key})
                actions.append({"type": "wait", "duration": step_duration})
                actions.append({"type": "release_key", "key": key})
                
        elif pattern == "strafe_left_right":
            # Left-right strafing
            step_duration = duration / 4
            actions.extend([
                {"type": "hold_key", "key": "a"},
                {"type": "wait", "duration": step_duration * 2},
                {"type": "release_key", "key": "a"},
                {"type": "hold_key", "key": "d"},
                {"type": "wait", "duration": step_duration * 2},
                {"type": "release_key", "key": "d"}
            ])
            
        self.macro_system.create_macro(name, actions)
        
    def create_combat_macro(self, name: str, combo_keys: List[str], 
                          timing: List[float] = None):
        """Create combat combo macros"""
        if not timing:
            timing = [0.1] * len(combo_keys)
            
        actions = []
        for key, wait_time in zip(combo_keys, timing):
            actions.append({"type": "key", "key": key, "duration": 0.05})
            actions.append({"type": "wait", "duration": wait_time})
            
        self.macro_system.create_macro(name, actions)
        
    def create_resource_gathering_macro(self, name: str, positions: List[Tuple[int, int]], 
                                      gather_key: str = "e"):
        """Create resource gathering macro (move to positions and gather)"""
        actions = []
        
        for x, y in positions:
            actions.extend([
                {"type": "move", "x": x, "y": y, "duration": 0.5},
                {"type": "click", "x": x, "y": y, "duration": 0.1},
                {"type": "key", "key": gather_key, "duration": 0.1},
                {"type": "wait", "duration": 1.0}  # Wait for gathering
            ])
            
        self.macro_system.create_macro(name, actions)

# Main game automation interface
class GameAutomator:
    """Complete game automation system"""
    
    def __init__(self, session_path: str = "claude_session"):
        self.game_input = GameInputController()
        self.macro_system = GameMacroSystem(self.game_input)
        self.profile_manager = GameProfileManager()
        self.patterns = CommonGamePatterns(self.game_input, self.macro_system)
        self.session_path = session_path
        
        # Load saved profiles
        profile_file = f"{session_path}/game_profiles.json"
        try:
            self.profile_manager.load_profiles_from_file(profile_file)
        except:
            pass  # No existing profiles
            
    def setup_game_profile(self, game_name: str, key_bindings: Dict[str, str] = None):
        """Setup a new game profile"""
        config = {
            "game_name": game_name,
            "key_bindings": key_bindings or {},
            "timing_settings": {
                "base_delay": 0.001,
                "click_duration": 0.050,
                "humanize": True,
                "variance": 0.02
            }
        }
        
        return self.profile_manager.create_profile(game_name, config)
        
    def save_session(self):
        """Save current session data"""
        profile_file = f"{self.session_path}/game_profiles.json"
        self.profile_manager.save_profiles_to_file(profile_file)
        
    def cleanup(self):
        """Cleanup resources"""
        self.macro_system.stop_all_macros()
        self.game_input.release_all_keys()
        self.save_session()