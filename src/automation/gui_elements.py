#!/usr/bin/env python3
"""
GUI Element Detection and Interaction
Provides high-level GUI automation using vision and pattern recognition
Works with buttons, text boxes, dropdowns, etc. across all applications
"""

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import re
from typing import List, Dict, Tuple, Optional, Any
import json
import time
import os
from .universal_input import UniversalInputController, MouseButton

class GUIElement:
    """Represents a detected GUI element"""
    def __init__(self, x: int, y: int, width: int, height: int, 
                 element_type: str, confidence: float = 0.0, text: str = ""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.element_type = element_type
        self.confidence = confidence
        self.text = text
        self.center_x = x + width // 2
        self.center_y = y + height // 2
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y, 
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "type": self.element_type,
            "confidence": self.confidence,
            "text": self.text
        }

class GUIElementDetector:
    """Detects and interacts with GUI elements using computer vision"""
    
    def __init__(self, input_controller: UniversalInputController = None):
        self.input_controller = input_controller or UniversalInputController()
        
        # Element detection templates (you can add more)
        self.button_templates = []
        self.textbox_templates = []
        
        # OCR configuration  
        self.ocr_config = r'--oem 3 --psm 6'
        
    def capture_screen(self, region: Tuple[int, int, int, int] = None) -> np.ndarray:
        """Capture screen as OpenCV image"""
        screenshot = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
    def find_buttons(self, image: np.ndarray = None, text_filter: str = None) -> List[GUIElement]:
        """
        Find button elements using various detection methods
        Args:
            image: Screenshot to analyze (None for current screen)
            text_filter: Filter buttons containing this text
        """
        if image is None:
            image = self.capture_screen()
            
        buttons = []
        
        # Method 1: Template matching for common button styles
        buttons.extend(self._find_buttons_by_template(image))
        
        # Method 2: Edge detection + shape analysis
        buttons.extend(self._find_buttons_by_shape(image))
        
        # Method 3: Text-based detection (OCR)
        if text_filter:
            buttons.extend(self._find_buttons_by_text(image, text_filter))
            
        # Remove duplicates and merge overlapping detections
        return self._merge_overlapping_elements(buttons)
        
    def find_text_fields(self, image: np.ndarray = None) -> List[GUIElement]:
        """Find text input fields"""
        if image is None:
            image = self.capture_screen()
            
        text_fields = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find rectangular regions that might be text fields
        # Look for white/light rectangles with borders
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Filter for text field characteristics
            if (aspect_ratio > 2 and aspect_ratio < 10 and  # Wide rectangles
                area > 500 and area < 50000 and  # Reasonable size
                h > 15 and h < 100):  # Reasonable height
                
                element = GUIElement(x, y, w, h, "text_field", 0.7)
                text_fields.append(element)
                
        return text_fields
        
    def find_dropdowns(self, image: np.ndarray = None) -> List[GUIElement]:
        """Find dropdown/combobox elements"""
        if image is None:
            image = self.capture_screen()
            
        dropdowns = []
        
        # Look for dropdown arrow patterns
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Template for dropdown arrows (you can expand this)
        # This is a simple approach - could be improved with better templates
        arrow_template = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8) * 255
        
        # Find arrow patterns
        result = cv2.matchTemplate(gray, arrow_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.6)
        
        for pt in zip(*locations[::-1]):
            # Look for associated rectangular area (the dropdown box)
            x, y = pt
            # Search left and up for the dropdown box
            for width in range(50, 300, 10):
                for height in range(20, 50, 5):
                    dropdown_x = max(0, x - width + 20)
                    dropdown_y = max(0, y - height // 2)
                    
                    element = GUIElement(dropdown_x, dropdown_y, width, height, "dropdown", 0.6)
                    dropdowns.append(element)
                    break  # Take first reasonable size
                break
                
        return self._merge_overlapping_elements(dropdowns)
        
    def find_text_by_content(self, text: str, image: np.ndarray = None, 
                           case_sensitive: bool = False) -> List[GUIElement]:
        """Find GUI elements containing specific text using OCR"""
        if image is None:
            image = self.capture_screen()
            
        # Use OCR to find text
        try:
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, 
                                               config=self.ocr_config)
            
            elements = []
            search_text = text if case_sensitive else text.lower()
            
            for i in range(len(ocr_data['text'])):
                detected_text = ocr_data['text'][i].strip()
                if not detected_text:
                    continue
                    
                compare_text = detected_text if case_sensitive else detected_text.lower()
                
                if search_text in compare_text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    conf = float(ocr_data['conf'][i]) / 100.0
                    
                    element = GUIElement(x, y, w, h, "text_element", conf, detected_text)
                    elements.append(element)
                    
            return elements
            
        except Exception as e:
            print(f"OCR failed: {e}")
            return []
            
    def _find_buttons_by_template(self, image: np.ndarray) -> List[GUIElement]:
        """Find buttons using template matching"""
        # This would contain pre-made button templates
        # For now, return empty list - you can add templates later
        return []
        
    def _find_buttons_by_shape(self, image: np.ndarray) -> List[GUIElement]:
        """Find buttons by detecting rectangular shapes with borders"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buttons = []
        for contour in contours:
            # Approximate contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h
                
                # Filter for button characteristics
                if (0.2 < aspect_ratio < 10 and  # Not too thin or wide
                    500 < area < 50000 and  # Reasonable size
                    w > 30 and h > 15):  # Minimum dimensions
                    
                    element = GUIElement(x, y, w, h, "button", 0.5)
                    buttons.append(element)
                    
        return buttons
        
    def _find_buttons_by_text(self, image: np.ndarray, text_filter: str) -> List[GUIElement]:
        """Find buttons containing specific text"""
        text_elements = self.find_text_by_content(text_filter, image)
        
        # Expand text elements to likely button boundaries
        buttons = []
        for element in text_elements:
            # Add padding around text to get full button
            padding_x = max(20, element.width // 4)
            padding_y = max(10, element.height // 2)
            
            button_x = max(0, element.x - padding_x)
            button_y = max(0, element.y - padding_y)
            button_w = element.width + 2 * padding_x
            button_h = element.height + 2 * padding_y
            
            button = GUIElement(button_x, button_y, button_w, button_h, 
                              "button", element.confidence, element.text)
            buttons.append(button)
            
        return buttons
        
    def _merge_overlapping_elements(self, elements: List[GUIElement]) -> List[GUIElement]:
        """Merge overlapping element detections"""
        if not elements:
            return elements
            
        merged = []
        used = set()
        
        for i, elem1 in enumerate(elements):
            if i in used:
                continue
                
            # Find all elements that overlap with this one
            overlapping = [elem1]
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._elements_overlap(elem1, elem2):
                    overlapping.append(elem2)
                    used.add(j)
                    
            # Merge overlapping elements
            if len(overlapping) == 1:
                merged.append(elem1)
            else:
                merged_elem = self._merge_elements(overlapping)
                merged.append(merged_elem)
                
        return merged
        
    def _elements_overlap(self, elem1: GUIElement, elem2: GUIElement, 
                         overlap_threshold: float = 0.3) -> bool:
        """Check if two elements overlap significantly"""
        # Calculate intersection
        x_overlap = max(0, min(elem1.x + elem1.width, elem2.x + elem2.width) - 
                           max(elem1.x, elem2.x))
        y_overlap = max(0, min(elem1.y + elem1.height, elem2.y + elem2.height) - 
                           max(elem1.y, elem2.y))
        
        intersection_area = x_overlap * y_overlap
        elem1_area = elem1.width * elem1.height
        elem2_area = elem2.width * elem2.height
        
        # Check if intersection is significant
        overlap_ratio1 = intersection_area / elem1_area if elem1_area > 0 else 0
        overlap_ratio2 = intersection_area / elem2_area if elem2_area > 0 else 0
        
        return max(overlap_ratio1, overlap_ratio2) > overlap_threshold
        
    def _merge_elements(self, elements: List[GUIElement]) -> GUIElement:
        """Merge multiple overlapping elements into one"""
        if not elements:
            return None
            
        # Find bounding box of all elements
        min_x = min(elem.x for elem in elements)
        min_y = min(elem.y for elem in elements)
        max_x = max(elem.x + elem.width for elem in elements)
        max_y = max(elem.y + elem.height for elem in elements)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Use the element type and text from highest confidence element
        best_elem = max(elements, key=lambda e: e.confidence)
        
        return GUIElement(min_x, min_y, width, height, 
                         best_elem.element_type, best_elem.confidence, best_elem.text)

class GUIAutomator:
    """High-level GUI automation interface"""
    
    def __init__(self, session_path: str = "claude_session"):
        self.input_controller = UniversalInputController(vision_integration=True, 
                                                       session_path=session_path)
        self.detector = GUIElementDetector(self.input_controller)
        self.session_path = session_path
        
    def click_button_by_text(self, text: str, timeout: float = 10.0) -> bool:
        """Find and click a button containing specific text"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            buttons = self.detector.find_buttons(text_filter=text)
            
            if buttons:
                # Click the first (most confident) button
                button = max(buttons, key=lambda b: b.confidence)
                success = self.input_controller.click(button.center_x, button.center_y)
                
                if success:
                    self._log_element_interaction("click_button", button, {"text": text})
                return success
                
            time.sleep(0.5)
            
        return False
        
    def fill_text_field(self, text: str, field_index: int = 0, 
                       clear_first: bool = True) -> bool:
        """Fill a text field with text"""
        text_fields = self.detector.find_text_fields()
        
        if field_index >= len(text_fields):
            return False
            
        field = text_fields[field_index]
        
        # Click on the field
        success = self.input_controller.click(field.center_x, field.center_y)
        if not success:
            return False
            
        time.sleep(0.1)
        
        # Clear existing text if requested
        if clear_first:
            self.input_controller.key_combination(['ctrl', 'a'])
            time.sleep(0.05)
            
        # Type the text
        success = self.input_controller.type_text(text)
        
        if success:
            self._log_element_interaction("fill_text_field", field, {"text": text})
            
        return success
        
    def select_dropdown_option(self, option_text: str, dropdown_index: int = 0) -> bool:
        """Select an option from a dropdown"""
        dropdowns = self.detector.find_dropdowns()
        
        if dropdown_index >= len(dropdowns):
            return False
            
        dropdown = dropdowns[dropdown_index]
        
        # Click the dropdown to open it
        success = self.input_controller.click(dropdown.center_x, dropdown.center_y)
        if not success:
            return False
            
        time.sleep(0.5)  # Wait for dropdown to open
        
        # Look for the option text
        option_elements = self.detector.find_text_by_content(option_text)
        
        if option_elements:
            # Click the option
            option = max(option_elements, key=lambda o: o.confidence)
            success = self.input_controller.click(option.center_x, option.center_y)
            
            if success:
                self._log_element_interaction("select_dropdown", dropdown, {"option": option_text})
            return success
            
        return False
        
    def find_element_by_text(self, text: str) -> Optional[GUIElement]:
        """Find any element containing specific text"""
        elements = self.detector.find_text_by_content(text)
        return max(elements, key=lambda e: e.confidence) if elements else None
        
    def get_all_elements(self, element_types: List[str] = None) -> List[GUIElement]:
        """Get all detected elements of specified types"""
        image = self.detector.capture_screen()
        all_elements = []
        
        if not element_types:
            element_types = ["button", "text_field", "dropdown"]
            
        if "button" in element_types:
            all_elements.extend(self.detector.find_buttons(image))
        if "text_field" in element_types:
            all_elements.extend(self.detector.find_text_fields(image))
        if "dropdown" in element_types:
            all_elements.extend(self.detector.find_dropdowns(image))
            
        return all_elements
        
    def save_element_map(self, filename: str = None) -> str:
        """Save current GUI element map to file"""
        elements = self.get_all_elements()
        
        if not filename:
            timestamp = int(time.time())
            filename = f"gui_elements_{timestamp}.json"
            
        filepath = os.path.join(self.session_path, "automation_maps", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        element_data = {
            "timestamp": time.time(),
            "elements": [elem.to_dict() for elem in elements]
        }
        
        with open(filepath, 'w') as f:
            json.dump(element_data, f, indent=2)
            
        return filepath
        
    def _log_element_interaction(self, action: str, element: GUIElement, params: Dict[str, Any]):
        """Log element interaction for debugging"""
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "element": element.to_dict(),
            "params": params
        }
        
        log_dir = os.path.join(self.session_path, "automation_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "element_interactions.json")
        
        # Append to log file
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
            
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)