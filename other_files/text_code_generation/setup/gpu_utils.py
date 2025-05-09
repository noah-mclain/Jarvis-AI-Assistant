#!/usr/bin/env python3
"""
Comprehensive GPU utilities for Jarvis AI Assistant:
1. Clear GPU memory
2. Diagnose GPU memory usage
3. Monitor GPU in real-time
4. Kill processes using GPU memory

This script combines functionality from:
- clear_gpu_memory.py
- diagnose_gpu_memory.py
- monitor_gpu.py
"""

import os
import gc
import sys
import subprocess
import time
import argparse
import signal
from datetime import datetime

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== CLEARING GPU MEMORY =====")
            
            # Get initial memory usage
            initial_mem = torch.cuda.memory_allocated() / (1024**3)
            initial_reserved = torch.cuda.memory_reserved() / (1024**3)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"Initial GPU memory: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved")
            print(f"Total GPU memory: {total_mem:.2f} GB")
            
            # Empty cache multiple times with pauses in between
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.5)
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Try to reset device
            try:
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error resetting device: {e}")
            
            # Get memory usage after cleanup
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            current_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            print(f"After cleanup: {current_mem:.2f} GB allocated, {current_reserved:.2f} GB reserved")
            print(f"Freed: {initial_mem - current_mem:.2f} GB allocated, {initial_reserved - current_reserved:.2f} GB reserved")
            print(f"Free GPU memory: {total_mem - current_mem:.2f} GB")
            
            return True
        else:
            print("CUDA is not available. No GPU memory to clear.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot clear GPU memory.")
        return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return False

def check_gpu_memory():
    """Check GPU memory usage and print detailed information"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== GPU MEMORY INFORMATION =====")
            
            # Get device properties
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {total_memory:.2f} GB")
            
            # Get current memory usage
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            reserved_memory = torch.cuda.memory_reserved() / (1024**3)
            free_memory = total_memory - allocated_memory
            
            print(f"Allocated memory: {allocated_memory:.2f} GB")
            print(f"Reserved memory: {reserved_memory:.2f} GB")
            print(f"Free memory: {free_memory:.2f} GB")
            
            # Get memory by allocation
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats()
                print("\n----- Memory Statistics -----")
                for key, value in stats.items():
                    if 'bytes' in key and value > 0:
                        print(f"{key}: {value / (1024**3):.4f} GB")
            
            return True
        else:
            print("CUDA is not available. No GPU memory to check.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot check GPU memory.")
        return False
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False

def check_cached_tensors():
    """Check for cached tensors in PyTorch's memory"""'
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== CHECKING FOR CACHED TENSORS =====")
            
            # Get all tensors in memory
            tensors = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        tensors.append((obj.shape, obj.dtype, obj.element_size() * obj.nelement() / (1024**3)))
                except:
                    pass
            
            if tensors:
                print(f"Found {len(tensors)} CUDA tensors in memory:")
                total_size = 0
                for shape, dtype, size in tensors:
                    print(f"  Shape: {shape}, Type: {dtype}, Size: {size:.4f} GB")
                    total_size += size
                print(f"Total tensor memory: {total_size:.4f} GB")
            else:
                print("No CUDA tensors found in memory.")
            
            return True
        else:
            print("CUDA is not available. Cannot check for cached tensors.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot check for cached tensors.")
        return False
    except Exception as e:
        print(f"Error checking cached tensors: {e}")
        return False

def get_gpu_processes():
    """Try to identify processes using GPU memory"""
    try:
        # Try nvidia-smi first
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                print("\n===== PROCESSES USING GPU MEMORY =====")
                print(result.stdout)
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("nvidia-smi not available or failed to run")
        
        # Try ps for Python processes
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=True
            )
            
            python_processes = [line for line in result.stdout.split('\n') if 'python' in line.lower()]
            if python_processes:
                print("\n===== RUNNING PYTHON PROCESSES =====")
                for proc in python_processes:
                    print(proc)
                return True
        except subprocess.SubprocessError:
            print("Failed to list processes")
        
        return False
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return False

def kill_gpu_processes():
    """Kill processes using GPU memory"""
    try:
        # Try nvidia-smi to get processes
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                print("\n===== KILLING PROCESSES USING GPU MEMORY =====")
                
                # Parse PIDs
                processes = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 1:
                        try:
                            pid = int(parts[0])
                            process_name = parts[1] if len(parts) > 1 else "Unknown"
                            memory = parts[2] if len(parts) > 2 else "Unknown"
                            processes.append((pid, process_name, memory))
                        except ValueError:
                            continue
                
                # Kill processes
                for pid, process_name, memory in processes:
                    print(f"Killing process {pid} ({process_name}) using {memory}")
                    try:
                        subprocess.run(["kill", "-9", str(pid)], check=True)
                        print(f"  ✓ Process {pid} killed")
                    except subprocess.SubprocessError as e:
                        print(f"  ✗ Failed to kill process {pid}: {e}")
                
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("nvidia-smi not available or failed to run")
        
        # Try killing Python processes
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=True
            )
            
            python_processes = []
            for line in result.stdout.split('\n'):
                if 'python' in line.lower():
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            # Skip the current process
                            if pid != os.getpid():
                                python_processes.append((pid, line))
                        except ValueError:
                            continue
            
            if python_processes:
                print("\n===== KILLING PYTHON PROCESSES =====")
                for pid, process_info in python_processes:
                    print(f"Killing Python process {pid}")
                    print(f"  Process info: {process_info}")
                    try:
                        subprocess.run(["kill", "-9", str(pid)], check=True)
                        print(f"  ✓ Process {pid} killed")
                    except subprocess.SubprocessError as e:
                        print(f"  ✗ Failed to kill process {pid}: {e}")
                
                return True
        except subprocess.SubprocessError:
            print("Failed to list processes")
        
        return False
    except Exception as e:
        print(f"Error killing GPU processes: {e}")
        return False

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

def diagnose():
    """Run a comprehensive GPU diagnostic"""
    print("=" * 50)
    print("GPU DIAGNOSTIC UTILITY")
    print("=" * 50)
    
    # Check GPU memory
    check_gpu_memory()
    
    # Check for cached tensors
    check_cached_tensors()
    
    # Get processes using GPU
    get_gpu_processes()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Check GPU memory again
    check_gpu_memory()
    
    print("\nTo kill a specific process, use: kill -9 <PID>")
    print("=" * 50)

def main():
    """Main function to parse arguments and run the appropriate utility"""
    parser = argparse.ArgumentParser(description="GPU Utilities for Jarvis AI Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear GPU memory")
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose GPU memory usage")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor GPU in real-time")
    monitor_parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    monitor_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Kill command
    kill_parser = subparsers.add_parser("kill", help="Kill processes using GPU memory")
    
    args = parser.parse_args()
    
    if args.command == "clear":
        clear_gpu_memory()
    elif args.command == "diagnose":
        diagnose()
    elif args.command == "monitor":
        monitor = GPUMonitor(interval=args.interval, log_file=args.log_file)
        monitor.start_monitoring()
    elif args.command == "kill":
        kill_gpu_processes()
    else:
        # Default to diagnose if no command is provided
        diagnose()

if __name__ == "__main__":
    main()
