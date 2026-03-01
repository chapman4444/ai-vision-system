#!/usr/bin/env python3
"""
Mouse Calibration Test GUI for LLM Automation
Simple GUI for testing and calibrating mouse click accuracy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import json
import math
from typing import List, Dict

class CalibrationTarget:
    def __init__(self, x: int, y: int, size: int = 50):
        self.x = x
        self.y = y
        self.size = size
        self.clicked = False
        self.accuracy = 0.0

    def is_hit(self, click_x: int, click_y: int) -> bool:
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        return distance <= self.size // 2

    def get_accuracy_score(self, click_x: int, click_y: int) -> float:
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        max_distance = self.size // 2
        accuracy = max(0, 100 - (distance / max_distance * 100))
        return min(100, accuracy)

class MouseCalibrationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mouse Calibration Test - LLM Training")
        
        # Make fullscreen to match screen capture
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='lightgray')
        
        # Get actual screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        self.targets = []
        self.current_target = 0
        self.active = False
        self.results = []
        
        self.setup_gui()
        
    def setup_gui(self):
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Start Test", command=self.start_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Auto Test", command=self.auto_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Exit Fullscreen (ESC)", command=self.exit_fullscreen).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg='white', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Bind ESC key to exit fullscreen
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())
        self.root.focus_set()  # Ensure window gets keyboard focus
        
        self.show_instructions()
        
        # Auto-start calibration after 5 seconds
        self.root.after(5000, self.auto_start_calibration)
        
    def auto_start_calibration(self):
        """Automatically start calibration after delay"""
        print("🎯 Auto-starting calibration...")
        self.start_test()
        
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        self.root.attributes('-fullscreen', False)
        self.root.geometry("1000x800")
        
    def show_instructions(self):
        instructions = """🎯 FULLSCREEN MOUSE CALIBRATION TEST 🎯

AUTO-START MODE: Calibration will begin automatically in 5 seconds!

This GUI matches your screen dimensions ({screen_w} x {screen_h}) for coordinate alignment!
Perfect for training LLM automation systems.

🤖 CALIBRATION PROCESS:
1. Red circles will appear across the entire screen
2. Click the CENTER of each red circle
3. Green dots show successful hits, red dots show misses
4. Accuracy percentage is displayed for each click

MANUAL COORDINATE REFERENCE:
• Screen center: ({center_x}, {center_y})
• Top-left corner: (80, 80)
• Top-right corner: ({tr_x}, 80)
• Bottom-left corner: (80, {bl_y})
• Bottom-right corner: ({br_x}, {br_y})

Press ESC to exit fullscreen mode""".format(
            screen_w=self.screen_width,
            screen_h=self.screen_height,
            center_x=self.screen_width // 2,
            center_y=self.screen_height // 2,
            tr_x=self.screen_width - 80,
            bl_y=self.screen_height - 80,
            br_x=self.screen_width - 80,
            br_y=self.screen_height - 80
        )
        
        # Center text on screen
        center_x = self.screen_width // 2 if hasattr(self, 'screen_width') else 500
        center_y = self.screen_height // 2 if hasattr(self, 'screen_height') else 350
        
        self.canvas.create_text(center_x, center_y, text=instructions, 
                               font=('Arial', 14), justify=tk.CENTER)
    
    def start_test(self):
        self.reset()
        self.active = True
        self.generate_targets()
        self.show_current_target()
        self.status_label.config(text=f"Target 1/{len(self.targets)}")
        
    def generate_targets(self):
        # Create test targets across full screen dimensions
        margin = 80
        
        # Grid pattern across full screen
        positions = []
        
        # Corners and edges
        positions.extend([
            (margin, margin),  # Top-left
            (self.screen_width - margin, margin),  # Top-right
            (margin, self.screen_height - margin),  # Bottom-left
            (self.screen_width - margin, self.screen_height - margin),  # Bottom-right
            (self.screen_width // 2, margin),  # Top-center
            (self.screen_width // 2, self.screen_height - margin),  # Bottom-center
            (margin, self.screen_height // 2),  # Left-center
            (self.screen_width - margin, self.screen_height // 2),  # Right-center
        ])
        
        # Center and quarter points
        positions.extend([
            (self.screen_width // 2, self.screen_height // 2),  # Dead center
            (self.screen_width // 4, self.screen_height // 4),  # Top-left quarter
            (3 * self.screen_width // 4, self.screen_height // 4),  # Top-right quarter
            (self.screen_width // 4, 3 * self.screen_height // 4),  # Bottom-left quarter
            (3 * self.screen_width // 4, 3 * self.screen_height // 4),  # Bottom-right quarter
        ])
        
        # Additional grid points
        for x_frac in [0.2, 0.4, 0.6, 0.8]:
            for y_frac in [0.3, 0.5, 0.7]:
                x = int(self.screen_width * x_frac)
                y = int(self.screen_height * y_frac)
                positions.append((x, y))
        
        self.targets = [CalibrationTarget(x, y, 50) for x, y in positions]
        
    def show_current_target(self):
        self.canvas.delete("target")
        
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            
            # Draw target circle
            x1 = target.x - target.size // 2
            y1 = target.y - target.size // 2
            x2 = target.x + target.size // 2
            y2 = target.y + target.size // 2
            
            self.canvas.create_oval(x1, y1, x2, y2, 
                                  outline='red', width=3, tags="target")
            
            # Center dot
            self.canvas.create_oval(target.x - 3, target.y - 3,
                                  target.x + 3, target.y + 3,
                                  fill='red', tags="target")
            
            # Crosshairs
            self.canvas.create_line(target.x - 15, target.y, 
                                  target.x + 15, target.y,
                                  fill='red', width=2, tags="target")
            self.canvas.create_line(target.x, target.y - 15,
                                  target.x, target.y + 15, 
                                  fill='red', width=2, tags="target")
    
    def on_click(self, event):
        if not self.active or self.current_target >= len(self.targets):
            return
            
        target = self.targets[self.current_target]
        accuracy = target.get_accuracy_score(event.x, event.y)
        is_hit = target.is_hit(event.x, event.y)
        
        # Record result
        result = {
            'target_x': target.x,
            'target_y': target.y,
            'click_x': event.x,
            'click_y': event.y,
            'accuracy': accuracy,
            'hit': is_hit
        }
        self.results.append(result)
        
        # Show feedback
        color = 'green' if accuracy >= 70 else 'orange' if accuracy >= 40 else 'red'
        self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                              fill=color, outline='black', tags="feedback")
        self.canvas.create_text(event.x, event.y - 15, text=f"{accuracy:.0f}%",
                              fill='black', font=('Arial', 10, 'bold'), tags="feedback")
        
        # Next target
        self.current_target += 1
        self.progress['value'] = (self.current_target / len(self.targets)) * 100
        
        if self.current_target < len(self.targets):
            self.root.after(1000, self.show_current_target)
            self.status_label.config(text=f"Target {self.current_target + 1}/{len(self.targets)} - Accuracy: {accuracy:.0f}%")
        else:
            self.complete_test()
            
        # Clear feedback after delay
        self.root.after(1500, lambda: self.canvas.delete("feedback"))
    
    def complete_test(self):
        self.active = False
        self.canvas.delete("target")
        
        # Calculate stats
        accuracies = [r['accuracy'] for r in self.results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        hits = sum(1 for r in self.results if r['hit'])
        hit_rate = (hits / len(self.results)) * 100
        
        # Show results
        result_text = f"""🎉 CALIBRATION COMPLETE! 🎉

Results:
• Targets Hit: {hits}/{len(self.results)}
• Hit Rate: {hit_rate:.1f}%
• Average Accuracy: {avg_accuracy:.1f}%
• Best: {max(accuracies):.0f}%
• Worst: {min(accuracies):.0f}%

Click Reset to try again"""
        
        self.canvas.create_text(450, 350, text=result_text,
                              font=('Arial', 12), justify=tk.CENTER,
                              fill='darkgreen')
        
        self.status_label.config(text=f"Complete! Avg: {avg_accuracy:.0f}%")
        
        # Save results
        self.save_results(avg_accuracy, hit_rate)
    
    def save_results(self, avg_accuracy: float, hit_rate: float):
        data = {
            'timestamp': time.time(),
            'average_accuracy': avg_accuracy,
            'hit_rate': hit_rate,
            'total_targets': len(self.results),
            'detailed_results': self.results
        }
        
        filename = f"calibration_results_{int(time.time())}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Could not save results: {e}")
    
    def auto_test(self):
        """Automated test mode"""
        if self.active:
            return
            
        self.start_test()
        self.root.after(1000, self.auto_click_next)
    
    def auto_click_next(self):
        if not self.active or self.current_target >= len(self.targets):
            return
            
        target = self.targets[self.current_target]
        
        # Add slight randomness
        offset_x = random.randint(-8, 8)
        offset_y = random.randint(-8, 8)
        
        click_x = target.x + offset_x
        click_y = target.y + offset_y
        
        # Simulate click
        event = type('Event', (), {})()
        event.x = click_x
        event.y = click_y
        
        self.on_click(event)
        
        # Schedule next click
        if self.current_target < len(self.targets):
            delay = random.randint(800, 1500)
            self.root.after(delay, self.auto_click_next)
    
    def reset(self):
        self.active = False
        self.targets.clear()
        self.results.clear()
        self.current_target = 0
        self.canvas.delete("all")
        self.progress['value'] = 0
        self.status_label.config(text="Ready")
        self.show_instructions()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MouseCalibrationGUI()
    app.run()