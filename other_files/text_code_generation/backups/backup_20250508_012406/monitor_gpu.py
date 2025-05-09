#!/usr/bin/env python3
"""
Real-time GPU monitoring script for Jarvis AI Assistant training
Used to help optimize training parameters for RTX 5000 GPU
"""

import subprocess
import time
import os
import argparse
import signal
import sys
from datetime import datetime


class GPUMonitor:
    def __init__(self, interval=1, log_file=None):
        """
        Initialize GPU monitor
        
        Args:
            interval: Polling interval in seconds
            log_file: File to save logs
        """
        self.interval = interval
        self.log_file = log_file or f"gpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.running = False
        self.peak_memory = 0
        self.peak_time = None
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle termination signals"""
        print("Stopping GPU monitoring...")
        self.running = False
    
    def get_gpu_info(self):
        """Get GPU usage information"""
        try:
            # Run nvidia-smi command
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw", 
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)
            
            # Process output
            output = result.stdout.strip()
            if not output:
                return None
                
            parts = output.split(", ")
            if len(parts) < 8:
                return None
                
            timestamp = parts[0]
            gpu_name = parts[1]
            gpu_util = float(parts[2])
            memory_util = float(parts[3])
            memory_used = float(parts[4])
            memory_total = float(parts[5])
            temperature = float(parts[6])
            power_draw = float(parts[7])
            
            # Update peak memory
            if memory_used > self.peak_memory:
                self.peak_memory = memory_used
                self.peak_time = timestamp
            
            return {
                "timestamp": timestamp,
                "gpu_name": gpu_name,
                "gpu_util": gpu_util,
                "memory_util": memory_util,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "temperature": temperature,
                "power_draw": power_draw,
                "memory_used_pct": (memory_used / memory_total) * 100
            }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def log_gpu_info(self, info):
        """Log GPU information to file and console"""
        if not info:
            return
            
        # Create log message
        log_msg = f"{info['timestamp']} | Memory: {info['memory_used']:.2f}/{info['memory_total']:.2f} MB ({info['memory_used_pct']:.1f}%) | GPU: {info['gpu_util']:.1f}% | Temp: {info['temperature']}°C | Power: {info['power_draw']}W"
        
        # Print to console with color-coding based on memory usage
        if info['memory_used_pct'] > 90:
            # Red for critical usage
            colored_msg = f"\033[31m{log_msg}\033[0m"
        elif info['memory_used_pct'] > 75:
            # Yellow for high usage
            colored_msg = f"\033[33m{log_msg}\033[0m"
        else:
            # Green for normal usage
            colored_msg = f"\033[32m{log_msg}\033[0m"
        
        print(colored_msg)
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        self.running = True
        print(f"Starting GPU monitoring (interval: {self.interval}s, log: {self.log_file})...")
        
        # Create log file with header
        with open(self.log_file, "w") as f:
            f.write("timestamp, memory_used_mb, memory_total_mb, memory_used_pct, gpu_util_pct, temp_c, power_w\n")
        
        # Monitor loop
        try:
            while self.running:
                info = self.get_gpu_info()
                if info:
                    self.log_gpu_info(info)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.print_summary()
    
    def print_summary(self):
        """Print summary of GPU monitoring"""
        print("\n" + "="*60)
        print("GPU Monitoring Summary")
        print("="*60)
        print(f"Peak memory usage: {self.peak_memory:.2f} MB")
        if self.peak_time:
            print(f"Peak time: {self.peak_time}")
        print(f"Log file: {self.log_file}")
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-time GPU monitoring for Jarvis AI Assistant training")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(interval=args.interval, log_file=args.log_file)
    monitor.start_monitoring()


if __name__ == "__main__":
    main() 