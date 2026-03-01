#!/usr/bin/env python3
"""
Fullscreen Mouse Calibration for LLM Training
Clean fullscreen interface with automatic operation and cursor feedback
"""

import tkinter as tk
import random
import time
import json
import math
import pyautogui
from typing import List, Dict, Tuple

class CalibrationTarget:
    def __init__(self, x: int, y: int, size: int = 60):
        self.x = x
        self.y = y
        self.size = size
        self.clicked = False
        self.accuracy = 0.0
        self.click_x = 0
        self.click_y = 0

    def is_hit(self, click_x: int, click_y: int) -> bool:
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        return distance <= self.size // 2

    def get_accuracy_score(self, click_x: int, click_y: int) -> float:
        distance = math.sqrt((click_x - self.x) ** 2 + (click_y - self.y) ** 2)
        max_distance = self.size // 2
        accuracy = max(0, 100 - (distance / max_distance * 100))
        return min(100, accuracy)

class FullscreenCalibrationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fullscreen Mouse Calibration")
        
        # Make fullscreen first, then configure
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Calibration state
        self.targets = []
        self.current_target = 0
        self.active = False
        self.results = []
        self.start_time = 0
        
        # Mouse tracking
        self.mouse_x = 0
        self.mouse_y = 0
        self.last_mouse_update = 0
        
        # Create fullscreen canvas
        self.canvas = tk.Canvas(
            self.root, 
            bg='black',
            width=self.screen_width,
            height=self.screen_height,
            highlightthickness=0,
            cursor='crosshair'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", self.exit_calibration)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.focus_set()
        
        # Start mouse tracking
        self.track_mouse()
        
        # Show initial instructions and auto-start
        self.show_startup_message()
        self.root.after(3000, self.start_calibration)
        
    def track_mouse(self):
        """Track mouse position continuously"""
        try:
            self.mouse_x, self.mouse_y = pyautogui.position()
            self.last_mouse_update = time.time()
        except:
            pass
        
        # Update every 50ms
        self.root.after(50, self.track_mouse)
        
    def show_startup_message(self):
        """Show startup instructions"""
        message = f"""🎯 FULLSCREEN MOUSE CALIBRATION 🎯

Screen: {self.screen_width} × {self.screen_height}

AUTO-STARTING IN 3 SECONDS...

Instructions:
• Red circles will appear across the screen
• Click the CENTER of each circle as accurately as possible
• Green dot = good hit, Red dot = miss
• Your accuracy percentage will be shown

Mouse Position: ({self.mouse_x}, {self.mouse_y})
Current Time: {time.strftime('%H:%M:%S')}

ESC = Exit Fullscreen"""
        
        self.canvas.create_text(
            self.screen_width // 2, 
            self.screen_height // 2,
            text=message,
            fill='white',
            font=('Arial', 16),
            justify=tk.CENTER,
            tags='instructions'
        )
        
    def start_calibration(self):
        """Start the calibration sequence"""
        self.canvas.delete('instructions')
        self.active = True
        self.start_time = time.time()
        
        # Generate comprehensive target pattern
        self.generate_calibration_targets()
        
        # Show status
        self.show_status(f"Calibration Started! Target 1 of {len(self.targets)}")
        
        # Show first target
        self.show_current_target()
        
    def generate_calibration_targets(self):
        """Generate comprehensive calibration targets"""
        self.targets = []
        margin = 50
        
        # 1. Screen corners
        corners = [
            (margin, margin),  # Top-left
            (self.screen_width - margin, margin),  # Top-right
            (margin, self.screen_height - margin),  # Bottom-left
            (self.screen_width - margin, self.screen_height - margin)  # Bottom-right
        ]
        
        # 2. Screen edges (centers)
        edges = [
            (self.screen_width // 2, margin),  # Top center
            (self.screen_width // 2, self.screen_height - margin),  # Bottom center
            (margin, self.screen_height // 2),  # Left center
            (self.screen_width - margin, self.screen_height // 2)  # Right center
        ]
        
        # 3. Screen center
        center = [(self.screen_width // 2, self.screen_height // 2)]
        
        # 4. Quarter points
        quarters = [
            (self.screen_width // 4, self.screen_height // 4),
            (3 * self.screen_width // 4, self.screen_height // 4),
            (self.screen_width // 4, 3 * self.screen_height // 4),
            (3 * self.screen_width // 4, 3 * self.screen_height // 4)
        ]
        
        # 5. Grid pattern (systematic coverage)
        grid_points = []
        for x_ratio in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
            for y_ratio in [0.25, 0.5, 0.75]:
                x = int(self.screen_width * x_ratio)
                y = int(self.screen_height * y_ratio)
                grid_points.append((x, y))
        
        # 6. Random challenge points
        random_points = []
        for _ in range(8):
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            random_points.append((x, y))
        
        # Combine all target positions
        all_positions = corners + edges + center + quarters + grid_points + random_points
        
        # Create targets with varying sizes for difficulty
        for i, (x, y) in enumerate(all_positions):
            if i < 4:  # Corners - larger targets
                size = 80
            elif i < 12:  # Edges and center - medium targets  
                size = 60
            else:  # Grid and random - smaller targets
                size = 40
                
            self.targets.append(CalibrationTarget(x, y, size))
        
        # Shuffle for random order
        random.shuffle(self.targets)
        
    def show_current_target(self):
        """Display the current target"""
        self.canvas.delete('target')
        
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            
            # Draw target circle (red)
            radius = target.size // 2
            self.canvas.create_oval(
                target.x - radius, target.y - radius,
                target.x + radius, target.y + radius,
                outline='red', width=4, tags='target'
            )
            
            # Draw center crosshairs
            crosshair_size = 20
            self.canvas.create_line(
                target.x - crosshair_size, target.y,
                target.x + crosshair_size, target.y,
                fill='red', width=3, tags='target'
            )
            self.canvas.create_line(
                target.x, target.y - crosshair_size,
                target.x, target.y + crosshair_size,
                fill='red', width=3, tags='target'
            )
            
            # Draw center dot
            self.canvas.create_oval(
                target.x - 3, target.y - 3,
                target.x + 3, target.y + 3,
                fill='red', outline='red', tags='target'
            )
            
            # Show target coordinates
            self.canvas.create_text(
                target.x, target.y - radius - 30,
                text=f"TARGET: ({target.x}, {target.y})",
                fill='yellow', font=('Arial', 12, 'bold'),
                tags='target'
            )
            
    def on_click(self, event):
        """Handle mouse clicks"""
        if not self.active or self.current_target >= len(self.targets):
            return
            
        target = self.targets[self.current_target]
        click_x, click_y = event.x, event.y
        
        # Calculate accuracy
        accuracy = target.get_accuracy_score(click_x, click_y)
        is_hit = target.is_hit(click_x, click_y)
        distance = math.sqrt((click_x - target.x) ** 2 + (click_y - target.y) ** 2)
        
        # Store target data
        target.clicked = True
        target.accuracy = accuracy
        target.click_x = click_x
        target.click_y = click_y
        
        # Store result
        result = {
            'target_index': self.current_target,
            'target_x': target.x,
            'target_y': target.y,
            'click_x': click_x,
            'click_y': click_y,
            'accuracy': accuracy,
            'distance': distance,
            'is_hit': is_hit,
            'mouse_pos': (self.mouse_x, self.mouse_y),
            'timestamp': time.time() - self.start_time
        }
        self.results.append(result)
        
        # Show click feedback
        self.show_click_feedback(click_x, click_y, target.x, target.y, accuracy, is_hit)
        
        # Update status
        offset_x = click_x - target.x
        offset_y = click_y - target.y
        
        self.show_status(
            f"Target {self.current_target + 1}/{len(self.targets)} | "
            f"Accuracy: {accuracy:.1f}% | "
            f"Distance: {distance:.1f}px | "
            f"Offset: ({offset_x:+.0f}, {offset_y:+.0f}) | "
            f"Mouse: ({self.mouse_x}, {self.mouse_y})"
        )
        
        # Move to next target
        self.current_target += 1
        
        if self.current_target < len(self.targets):
            # Show next target after delay
            self.root.after(1500, self.show_current_target)
        else:
            # Calibration complete
            self.root.after(2000, self.show_completion_results)
            
    def show_click_feedback(self, click_x, click_y, target_x, target_y, accuracy, is_hit):
        """Show visual feedback for the click"""
        # Choose feedback color
        if accuracy >= 80:
            color = 'lime'
        elif accuracy >= 60:
            color = 'orange'
        else:
            color = 'red'
            
        # Draw click indicator
        self.canvas.create_oval(
            click_x - 8, click_y - 8,
            click_x + 8, click_y + 8,
            fill=color, outline='white', width=2, tags='feedback'
        )
        
        # Draw line from click to target center
        self.canvas.create_line(
            click_x, click_y, target_x, target_y,
            fill='white', width=2, dash=(5, 5), tags='feedback'
        )
        
        # Show accuracy text
        self.canvas.create_text(
            click_x, click_y - 20,
            text=f"{accuracy:.0f}%",
            fill='white', font=('Arial', 12, 'bold'),
            tags='feedback'
        )
        
        # Remove feedback after delay
        self.root.after(1200, lambda: self.canvas.delete('feedback'))
        
    def show_status(self, message):
        """Show status message at top of screen"""
        self.canvas.delete('status')
        self.canvas.create_text(
            self.screen_width // 2, 30,
            text=message,
            fill='white', font=('Arial', 14),
            tags='status'
        )
        
    def show_completion_results(self):
        """Show final calibration results"""
        self.active = False
        self.canvas.delete('target', 'status')
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in self.results]
        distances = [r['distance'] for r in self.results]
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_distance = sum(distances) / len(distances) if distances else 0
        hits = sum(1 for r in self.results if r['is_hit'])
        hit_rate = (hits / len(self.results)) * 100 if self.results else 0
        
        # Calculate offset patterns
        offset_x_values = [r['click_x'] - r['target_x'] for r in self.results]
        offset_y_values = [r['click_y'] - r['target_y'] for r in self.results]
        
        avg_offset_x = sum(offset_x_values) / len(offset_x_values) if offset_x_values else 0
        avg_offset_y = sum(offset_y_values) / len(offset_y_values) if offset_y_values else 0
        
        # Show comprehensive results
        results_text = f"""🎉 CALIBRATION COMPLETE! 🎉

PERFORMANCE SUMMARY:
• Total Targets: {len(self.results)}
• Successful Hits: {hits}
• Hit Rate: {hit_rate:.1f}%
• Average Accuracy: {avg_accuracy:.1f}%
• Average Distance: {avg_distance:.1f} pixels

OFFSET ANALYSIS (Click vs Target):
• Average X Offset: {avg_offset_x:+.1f} pixels
• Average Y Offset: {avg_offset_y:+.1f} pixels

RECOMMENDATIONS:
• Adjust clicks by ({avg_offset_x:+.0f}, {avg_offset_y:+.0f}) pixels
• Screen resolution: {self.screen_width} × {self.screen_height}

ACCURACY BREAKDOWN:
• Best: {max(accuracies):.0f}%
• Worst: {min(accuracies):.0f}%
• Excellent (90%+): {sum(1 for a in accuracies if a >= 90)} targets
• Good (70-89%): {sum(1 for a in accuracies if 70 <= a < 90)} targets
• Poor (<70%): {sum(1 for a in accuracies if a < 70)} targets

Press ESC to exit"""
        
        self.canvas.create_text(
            self.screen_width // 2,
            self.screen_height // 2,
            text=results_text,
            fill='lime',
            font=('Arial', 16),
            justify=tk.CENTER,
            tags='results'
        )
        
        # Save results to file
        self.save_results()
        
    def save_results(self):
        """Save calibration results to JSON file"""
        data = {
            'timestamp': time.time(),
            'screen_resolution': f"{self.screen_width}x{self.screen_height}",
            'total_targets': len(self.results),
            'summary': {
                'avg_accuracy': sum(r['accuracy'] for r in self.results) / len(self.results),
                'hit_rate': sum(1 for r in self.results if r['is_hit']) / len(self.results) * 100,
                'avg_distance': sum(r['distance'] for r in self.results) / len(self.results),
                'avg_offset_x': sum(r['click_x'] - r['target_x'] for r in self.results) / len(self.results),
                'avg_offset_y': sum(r['click_y'] - r['target_y'] for r in self.results) / len(self.results)
            },
            'detailed_results': self.results
        }
        
        filename = f"fullscreen_calibration_{int(time.time())}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Results saved to {filename}")
        except Exception as e:
            print(f"Could not save results: {e}")
            
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.keysym == 'space' and not self.active:
            self.start_calibration()
        elif event.keysym == 'r':
            self.restart_calibration()
            
    def restart_calibration(self):
        """Restart calibration"""
        self.canvas.delete('all')
        self.current_target = 0
        self.results = []
        self.active = False
        self.show_startup_message()
        self.root.after(2000, self.start_calibration)
        
    def exit_calibration(self, event=None):
        """Exit fullscreen and close"""
        self.root.quit()
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    print("🎯 Starting Fullscreen Mouse Calibration...")
    app = FullscreenCalibrationGUI()
    app.run()