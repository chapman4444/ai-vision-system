#!/usr/bin/env python3
"""
Calibration Integration for LLM Automation
Provides tools for LLMs to use the calibration GUI and improve accuracy
"""

import json
import time
import os
import statistics
from typing import Dict, List, Tuple, Optional
import subprocess
import threading
import pyautogui
from .universal_input import UniversalInputController
from .vision_integration import VisionAutomationIntegrator

class CalibrationManager:
    """Manages mouse calibration for LLM automation systems"""
    
    def __init__(self, session_path: str = "claude_session"):
        self.session_path = session_path
        self.calibration_data = None
        self.accuracy_adjustments = {}
        
        # Ensure directories exist
        self.logs_dir = os.path.join(session_path, "automation_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def launch_calibration_gui(self) -> bool:
        """Launch the calibration GUI"""
        try:
            gui_path = os.path.join("src", "gui", "mouse_calibration_gui.py")
            
            if not os.path.exists(gui_path):
                print(f"Calibration GUI not found at: {gui_path}")
                return False
                
            # Launch GUI in separate process
            process = subprocess.Popen(
                ["python", gui_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"Calibration GUI launched with PID: {process.pid}")
            return True
            
        except Exception as e:
            print(f"Failed to launch calibration GUI: {e}")
            return False
            
    def wait_for_calibration_results(self, timeout: float = 300.0) -> Optional[Dict]:
        """Wait for calibration results to be saved"""
        start_time = time.time()
        
        print("Waiting for calibration results...")
        
        while time.time() - start_time < timeout:
            # Check for new calibration files
            calibration_files = [
                f for f in os.listdir(self.logs_dir) 
                if f.startswith("mouse_calibration_") and f.endswith(".json")
            ]
            
            if calibration_files:
                # Get most recent file
                latest_file = max(calibration_files, 
                                key=lambda f: os.path.getmtime(os.path.join(self.logs_dir, f)))
                
                filepath = os.path.join(self.logs_dir, latest_file)
                
                # Check if file was created after we started waiting
                if os.path.getmtime(filepath) > start_time:
                    with open(filepath, 'r') as f:
                        self.calibration_data = json.load(f)
                    
                    print(f"Calibration results loaded from: {latest_file}")
                    return self.calibration_data
                    
            time.sleep(1.0)
            
        print("Timeout waiting for calibration results")
        return None
        
    def analyze_calibration_data(self, data: Dict = None) -> Dict:
        """Analyze calibration results and generate accuracy report"""
        if data is None:
            data = self.calibration_data
            
        if not data:
            return {"error": "No calibration data available"}
            
        results = data.get("detailed_results", [])
        summary = data.get("summary", {})
        
        if not results:
            return {"error": "No detailed results in calibration data"}
            
        # Calculate detailed statistics
        accuracies = [r["accuracy"] for r in results]
        hit_distances = []
        
        for result in results:
            distance = ((result["click_x"] - result["target_x"]) ** 2 + 
                       (result["click_y"] - result["target_y"]) ** 2) ** 0.5
            hit_distances.append(distance)
            
        analysis = {
            "total_targets": len(results),
            "successful_hits": sum(1 for r in results if r["is_hit"]),
            "success_rate": (sum(1 for r in results if r["is_hit"]) / len(results)) * 100,
            
            # Accuracy statistics
            "average_accuracy": statistics.mean(accuracies),
            "median_accuracy": statistics.median(accuracies),
            "accuracy_std_dev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            
            # Distance statistics
            "average_distance": statistics.mean(hit_distances),
            "median_distance": statistics.median(hit_distances),
            "distance_std_dev": statistics.stdev(hit_distances) if len(hit_distances) > 1 else 0,
            
            # Accuracy distribution
            "accuracy_ranges": {
                "excellent": sum(1 for a in accuracies if a >= 90),
                "good": sum(1 for a in accuracies if 70 <= a < 90),
                "fair": sum(1 for a in accuracies if 50 <= a < 70),
                "poor": sum(1 for a in accuracies if a < 50)
            },
            
            # Recommendations
            "recommendations": self.generate_recommendations(accuracies, hit_distances)
        }
        
        return analysis
        
    def generate_recommendations(self, accuracies: List[float], 
                               distances: List[float]) -> List[str]:
        """Generate calibration recommendations"""
        recommendations = []
        
        avg_accuracy = statistics.mean(accuracies)
        avg_distance = statistics.mean(distances)
        accuracy_consistency = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
        
        if avg_accuracy < 70:
            recommendations.append("Low average accuracy - consider adjusting click precision")
            recommendations.append("Practice with larger targets first")
            
        if avg_distance > 25:
            recommendations.append("High click distances - improve target centering")
            
        if accuracy_consistency > 20:
            recommendations.append("Inconsistent accuracy - work on repeatability")
            
        if avg_accuracy >= 85:
            recommendations.append("Excellent accuracy! System is well calibrated")
            
        if not recommendations:
            recommendations.append("Good baseline performance - continue monitoring")
            
        return recommendations
        
    def create_accuracy_adjustments(self, analysis: Dict) -> Dict:
        """Create accuracy adjustment parameters based on calibration"""
        adjustments = {
            "click_offset_x": 0,
            "click_offset_y": 0,
            "timing_adjustment": 1.0,
            "precision_level": "normal"
        }
        
        # Adjust based on average distance
        avg_distance = analysis.get("average_distance", 0)
        if avg_distance > 30:
            adjustments["precision_level"] = "high"
            adjustments["timing_adjustment"] = 1.5  # Slower, more precise
        elif avg_distance < 10:
            adjustments["precision_level"] = "fast"
            adjustments["timing_adjustment"] = 0.8  # Faster
            
        # Store adjustments for use by automation system
        self.accuracy_adjustments = adjustments
        
        # Save to file
        adjustments_file = os.path.join(self.session_path, "mouse_adjustments.json")
        with open(adjustments_file, 'w') as f:
            json.dump(adjustments, f, indent=2)
            
        return adjustments
        
    def run_full_calibration_cycle(self) -> Dict:
        """Run complete calibration cycle and return results"""
        print("🎯 Starting Full Mouse Calibration Cycle")
        print("=" * 50)
        
        # Step 1: Launch GUI
        print("Step 1: Launching calibration GUI...")
        if not self.launch_calibration_gui():
            return {"error": "Failed to launch calibration GUI"}
            
        print("✓ Calibration GUI launched")
        print("Please complete the calibration in the GUI window...")
        
        # Step 2: Wait for results
        print("\nStep 2: Waiting for calibration completion...")
        calibration_data = self.wait_for_calibration_results()
        
        if not calibration_data:
            return {"error": "No calibration data received"}
            
        print("✓ Calibration data received")
        
        # Step 3: Analyze results
        print("\nStep 3: Analyzing calibration results...")
        analysis = self.analyze_calibration_data(calibration_data)
        
        # Step 4: Create adjustments
        print("\nStep 4: Creating accuracy adjustments...")
        adjustments = self.create_accuracy_adjustments(analysis)
        
        print("\n🎉 Calibration Cycle Complete!")
        print("=" * 50)
        
        # Print summary
        print(f"Success Rate: {analysis.get('success_rate', 0):.1f}%")
        print(f"Average Accuracy: {analysis.get('average_accuracy', 0):.1f}%")
        print(f"Average Distance: {analysis.get('average_distance', 0):.1f} pixels")
        
        print("\nRecommendations:")
        for rec in analysis.get('recommendations', []):
            print(f"• {rec}")
            
        return {
            "calibration_data": calibration_data,
            "analysis": analysis,
            "adjustments": adjustments
        }
        
    def get_calibration_report(self) -> str:
        """Generate a formatted calibration report"""
        if not self.calibration_data:
            return "No calibration data available. Please run calibration first."
            
        analysis = self.analyze_calibration_data()
        
        report = f"""
🎯 MOUSE CALIBRATION REPORT 🎯
========================================

PERFORMANCE SUMMARY:
• Total Targets: {analysis['total_targets']}
• Successful Hits: {analysis['successful_hits']}
• Success Rate: {analysis['success_rate']:.1f}%

ACCURACY METRICS:
• Average Accuracy: {analysis['average_accuracy']:.1f}%
• Median Accuracy: {analysis['median_accuracy']:.1f}%
• Best Accuracy: {analysis['max_accuracy']:.1f}%
• Worst Accuracy: {analysis['min_accuracy']:.1f}%
• Consistency (Std Dev): {analysis['accuracy_std_dev']:.1f}%

DISTANCE METRICS:
• Average Distance: {analysis['average_distance']:.1f} pixels
• Median Distance: {analysis['median_distance']:.1f} pixels
• Distance Consistency: {analysis['distance_std_dev']:.1f} pixels

ACCURACY DISTRIBUTION:
• Excellent (90%+): {analysis['accuracy_ranges']['excellent']} targets
• Good (70-89%): {analysis['accuracy_ranges']['good']} targets
• Fair (50-69%): {analysis['accuracy_ranges']['fair']} targets
• Poor (<50%): {analysis['accuracy_ranges']['poor']} targets

RECOMMENDATIONS:
"""
        for rec in analysis['recommendations']:
            report += f"• {rec}\n"
            
        return report

# Integration with existing automation system
class CalibratedInputController(UniversalInputController):
    """Enhanced input controller that uses calibration data"""
    
    def __init__(self, calibration_manager: CalibrationManager = None, **kwargs):
        super().__init__(**kwargs)
        
        self.calibration_manager = calibration_manager
        self.adjustments = {}
        
        # Load calibration adjustments if available
        if calibration_manager:
            adjustments_file = os.path.join(
                calibration_manager.session_path, 
                "mouse_adjustments.json"
            )
            
            if os.path.exists(adjustments_file):
                with open(adjustments_file, 'r') as f:
                    self.adjustments = json.load(f)
                    
    def click(self, x: int = None, y: int = None, **kwargs) -> bool:
        """Enhanced click with calibration adjustments"""
        if x is not None and y is not None:
            # Apply calibration adjustments
            adjusted_x = x + self.adjustments.get("click_offset_x", 0)
            adjusted_y = y + self.adjustments.get("click_offset_y", 0)
            
            # Adjust timing based on precision level
            timing_mult = self.adjustments.get("timing_adjustment", 1.0)
            
            # Use adjusted coordinates and timing
            return super().click(
                adjusted_x, adjusted_y,
                **kwargs
            )
        else:
            return super().click(x, y, **kwargs)