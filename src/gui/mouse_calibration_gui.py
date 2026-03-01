#!/usr/bin/env python3
"""
Mouse Calibration Test GUI for LLM Automation
Provides visual targets for mouse click calibration and accuracy testing
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import json
import os
import math
from typing import List, Dict, Tuple
import threading
import pyautogui
import cv2
import numpy as np

class CalibrationTarget:
    """Represents a clickable calibration target"""
    
    def __init__(self, x: int, y: int, size: int = 50, target_type: str = "circle"):
        self.x = x
        self.y = y
        self.size = size
        self.target_type = target_type
        self.clicked = False
        self.click_accuracy = None
        self.click_time = None
        
    def is_hit(self, click_x: int, click_y: int, tolerance: int = None) -> bool:
        """Check if click hit the target within tolerance"""
        if tolerance is None:
            tolerance = self.size // 2
            
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        return distance <= tolerance
        
    def get_accuracy_score(self, click_x: int, click_y: int) -> float:
        """Calculate accuracy score (0-100)"""
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        max_distance = self.size
        accuracy = max(0, 100 - (distance / max_distance * 100))
        return min(100, accuracy)

class MouseCalibrationGUI:
    """Main calibration GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mouse Calibration Test GUI - LLM Training")
        self.root.geometry("1200x800")
        self.root.configure(bg='lightgray')
        
        # Calibration state
        self.targets = []
        self.current_target_index = 0
        self.calibration_active = False
        self.results = []
        self.session_data = {
            "session_start": time.time(),
            "total_targets": 0,
            "successful_clicks": 0,
            "accuracy_scores": [],
            "timing_data": []
        }
        
        # Colors and styles
        self.colors = {
            "target": "#FF4444",
            "hit": "#44FF44", 
            "miss": "#FFAA44",
            "neutral": "#CCCCCC"
        }
        
        # GUI components
        self.canvas = None
        self.control_frame = None
        self.status_label = None
        self.progress_bar = None
        
        self.setup_gui()
        self.create_instructions()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # Make window stay on top for visibility
        self.root.attributes("-topmost", True)
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main control frame
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Start Calibration", 
                  command=self.start_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Reset", 
                  command=self.reset_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Save Results", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Auto Test", 
                  command=self.auto_test_mode).pack(side=tk.LEFT, padx=5)
        
        # Status and progress
        self.status_label = ttk.Label(self.control_frame, text="Ready to calibrate")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Progress bar
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Main canvas for targets
        self.canvas = tk.Canvas(self.root, bg='white', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_instructions(self):
        """Show initial instructions"""
        instructions = """
🎯 MOUSE CALIBRATION TEST GUI 🎯

Instructions for LLM Automation:
• Click "Start Calibration" to begin
• Red circles will appear - click the center of each target
• Green = successful hit, Orange = close miss, Red = far miss
• Complete all targets to get calibration report

Test Modes:
• Manual: Click targets yourself
• Auto Test: Automated clicking for testing

Keyboard Shortcuts:
• SPACE: Start/Reset calibration
• ESC: Exit
• S: Save results
        """
        
        # Create instruction text on canvas
        self.canvas.create_text(600, 400, text=instructions, 
                               font=('Arial', 12), justify=tk.CENTER,
                               tags="instructions")
        
    def start_calibration(self):
        """Start a new calibration session"""
        self.reset_calibration()
        self.calibration_active = True
        self.session_data["session_start"] = time.time()
        
        # Generate target positions
        self.generate_targets()
        
        # Show first target
        if self.targets:
            self.show_current_target()
            self.status_label.config(text=f"Target 1/{len(self.targets)} - Click the red circle!")
        
    def generate_targets(self):
        """Generate calibration target positions"""
        self.targets.clear()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, use default size
            canvas_width, canvas_height = 1000, 600
        
        # Margin from edges
        margin = 60
        
        # Different target patterns
        patterns = [
            self.generate_grid_pattern,
            self.generate_corners_pattern, 
            self.generate_random_pattern,
            self.generate_circle_pattern
        ]
        
        # Use all patterns
        for pattern in patterns:
            pattern(canvas_width, canvas_height, margin)
            
        # Shuffle for random order
        random.shuffle(self.targets)
        self.session_data["total_targets"] = len(self.targets)
        
    def generate_grid_pattern(self, width: int, height: int, margin: int):
        """Generate grid pattern targets"""
        grid_size = 4
        x_spacing = (width - 2 * margin) // (grid_size - 1)
        y_spacing = (height - 2 * margin) // (grid_size - 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = margin + j * x_spacing
                y = margin + i * y_spacing
                self.targets.append(CalibrationTarget(x, y, 40))
                
    def generate_corners_pattern(self, width: int, height: int, margin: int):
        """Generate corner and edge targets"""
        positions = [
            (margin, margin),  # Top-left
            (width - margin, margin),  # Top-right
            (margin, height - margin),  # Bottom-left
            (width - margin, height - margin),  # Bottom-right
            (width // 2, margin),  # Top-center
            (width // 2, height - margin),  # Bottom-center
            (margin, height // 2),  # Left-center
            (width - margin, height // 2),  # Right-center
        ]
        
        for x, y in positions:
            self.targets.append(CalibrationTarget(x, y, 35))
            
    def generate_random_pattern(self, width: int, height: int, margin: int):
        """Generate random positioned targets"""
        for _ in range(12):
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            size = random.randint(30, 50)
            self.targets.append(CalibrationTarget(x, y, size))
            
    def generate_circle_pattern(self, width: int, height: int, margin: int):
        """Generate targets in circular pattern"""
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        num_targets = 8
        for i in range(num_targets):
            angle = (2 * math.pi * i) / num_targets
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            self.targets.append(CalibrationTarget(x, y, 35))
            
    def show_current_target(self):
        """Display the current target to click"""
        self.canvas.delete("target")
        self.canvas.delete("instructions")
        
        if self.current_target_index < len(self.targets):
            target = self.targets[self.current_target_index]
            
            # Draw target circle
            x1 = target.x - target.size // 2
            y1 = target.y - target.size // 2
            x2 = target.x + target.size // 2
            y2 = target.y + target.size // 2
            
            self.canvas.create_oval(x1, y1, x2, y2, 
                                  outline=self.colors["target"], 
                                  width=3, tags="target")
            
            # Draw center point
            self.canvas.create_oval(target.x - 3, target.y - 3, 
                                  target.x + 3, target.y + 3,
                                  fill=self.colors["target"], 
                                  outline=self.colors["target"],
                                  tags="target")
            
            # Draw crosshairs
            self.canvas.create_line(target.x - 15, target.y, 
                                  target.x + 15, target.y,
                                  fill=self.colors["target"], 
                                  width=2, tags="target")
            self.canvas.create_line(target.x, target.y - 15, 
                                  target.x, target.y + 15,
                                  fill=self.colors["target"], 
                                  width=2, tags="target")
            
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if not self.calibration_active or self.current_target_index >= len(self.targets):
            return
            
        target = self.targets[self.current_target_index]
        click_time = time.time()
        
        # Calculate accuracy
        accuracy = target.get_accuracy_score(event.x, event.y)
        is_hit = target.is_hit(event.x, event.y)
        
        # Store results
        target.clicked = True
        target.click_accuracy = accuracy
        target.click_time = click_time
        
        result = {
            "target_index": self.current_target_index,
            "target_x": target.x,
            "target_y": target.y,
            "click_x": event.x,
            "click_y": event.y,
            "accuracy": accuracy,
            "is_hit": is_hit,
            "timestamp": click_time
        }
        
        self.results.append(result)
        self.session_data["accuracy_scores"].append(accuracy)
        
        if is_hit:
            self.session_data["successful_clicks"] += 1
            
        # Visual feedback
        self.show_click_feedback(event.x, event.y, accuracy, is_hit)
        
        # Move to next target
        self.current_target_index += 1
        self.update_progress()
        
        if self.current_target_index < len(self.targets):
            # Show next target after brief delay
            self.root.after(800, self.show_current_target)
            self.status_label.config(
                text=f"Target {self.current_target_index + 1}/{len(self.targets)} - "
                     f"Accuracy: {accuracy:.1f}%"
            )
        else:
            # Calibration complete
            self.complete_calibration()
            
    def show_click_feedback(self, click_x: int, click_y: int, 
                           accuracy: float, is_hit: bool):
        """Show visual feedback for click"""
        # Choose color based on accuracy
        if accuracy >= 80:
            color = self.colors["hit"]
        elif accuracy >= 50:
            color = self.colors["miss"]
        else:
            color = self.colors["target"]
            
        # Draw click indicator
        self.canvas.create_oval(click_x - 8, click_y - 8, 
                              click_x + 8, click_y + 8,
                              fill=color, outline="black", 
                              width=2, tags="feedback")
        
        # Show accuracy text
        self.canvas.create_text(click_x, click_y - 20, 
                              text=f"{accuracy:.1f}%",
                              fill="black", font=('Arial', 10, 'bold'),
                              tags="feedback")
        
        # Remove feedback after delay
        self.root.after(1500, lambda: self.canvas.delete("feedback"))
        
    def update_progress(self):
        """Update progress bar"""
        if self.targets:
            progress = (self.current_target_index / len(self.targets)) * 100
            self.progress_bar['value'] = progress
            
    def complete_calibration(self):
        """Handle calibration completion"""
        self.calibration_active = False
        self.canvas.delete("target")
        
        # Calculate final statistics
        total_accuracy = sum(self.session_data["accuracy_scores"])
        avg_accuracy = total_accuracy / len(self.session_data["accuracy_scores"]) if self.session_data["accuracy_scores"] else 0
        success_rate = (self.session_data["successful_clicks"] / self.session_data["total_targets"]) * 100
        
        # Show completion message
        completion_text = f"""
🎉 CALIBRATION COMPLETE! 🎉

Results Summary:
• Total Targets: {self.session_data['total_targets']}
• Successful Hits: {self.session_data['successful_clicks']}
• Success Rate: {success_rate:.1f}%
• Average Accuracy: {avg_accuracy:.1f}%

Click 'Save Results' to export data
Click 'Reset' to run again
        """
        
        self.canvas.create_text(600, 400, text=completion_text,
                              font=('Arial', 14), justify=tk.CENTER,
                              fill="darkgreen", tags="completion")
        
        self.status_label.config(text=f"Complete! Accuracy: {avg_accuracy:.1f}%")
        self.progress_bar['value'] = 100
        
    def reset_calibration(self):
        """Reset calibration state"""
        self.calibration_active = False
        self.targets.clear()
        self.results.clear()
        self.current_target_index = 0
        self.session_data = {
            "session_start": time.time(),
            "total_targets": 0,
            "successful_clicks": 0,
            "accuracy_scores": [],
            "timing_data": []
        }
        
        self.canvas.delete("all")
        self.create_instructions()
        self.status_label.config(text="Ready to calibrate")
        self.progress_bar['value'] = 0
        
    def auto_test_mode(self):
        """Automated testing mode for LLM validation"""
        if self.calibration_active:
            messagebox.showwarning("Warning", "Calibration in progress!")
            return
            
        response = messagebox.askyesno("Auto Test", 
                                     "Start automated clicking test?\n"
                                     "This will test the automation system.")
        
        if response:
            self.start_calibration()
            # Start automated clicking after brief delay
            self.root.after(1000, self.auto_click_targets)
            
    def auto_click_targets(self):
        """Automatically click targets for testing"""
        if not self.calibration_active or self.current_target_index >= len(self.targets):
            return
            
        target = self.targets[self.current_target_index]
        
        # Add some randomness to simulate realistic clicking
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        
        click_x = target.x + offset_x
        click_y = target.y + offset_y
        
        # Simulate click event
        event = type('Event', (), {})()
        event.x = click_x
        event.y = click_y
        
        self.on_canvas_click(event)
        
        # Schedule next click
        if self.current_target_index < len(self.targets):
            delay = random.randint(500, 1500)  # Random delay
            self.root.after(delay, self.auto_click_targets)
            
    def save_results(self):
        """Save calibration results to file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        # Prepare data
        save_data = {
            "session_info": self.session_data,
            "detailed_results": self.results,
            "summary": {
                "total_targets": len(self.results),
                "successful_hits": sum(1 for r in self.results if r["is_hit"]),
                "average_accuracy": sum(r["accuracy"] for r in self.results) / len(self.results),
                "best_accuracy": max(r["accuracy"] for r in self.results),
                "worst_accuracy": min(r["accuracy"] for r in self.results)
            }
        }
        
        # Save to file
        timestamp = int(time.time())
        filename = f"mouse_calibration_{timestamp}.json"
        filepath = os.path.join("claude_session", "automation_logs", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        messagebox.showinfo("Saved", f"Results saved to:\n{filepath}")
        
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.keysym == "space":
            if not self.calibration_active:
                self.start_calibration()
            else:
                self.reset_calibration()
        elif event.keysym == "Escape":
            self.root.quit()
        elif event.keysym == "s":
            self.save_results()
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    app = MouseCalibrationGUI()
    app.run()

if __name__ == "__main__":
    main()