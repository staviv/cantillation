import os
import subprocess
import argparse
import time
import json
import torch
import sys

def get_available_gpus(min_memory_mb=2000):
    """
    Get a list of available GPUs and their free memory
    
    Args:
        min_memory_mb: Minimum free memory in MB required to consider a GPU available
        
    Returns:
        List of tuples (gpu_id, free_memory) sorted by free memory (most to least)
    """
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', 
                                         '--format=csv,nounits,noheader'], 
                                         encoding='utf-8')
        lines = result.strip().split('\n')
        gpu_memory = []
        for line in lines:
            index, free_memory = map(int, line.split(','))
            if free_memory >= min_memory_mb:  # Only include GPUs with enough free memory
                gpu_memory.append((str(index), free_memory))
        
        # Sort by free memory (highest to lowest)
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU information: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Run parallel training jobs on multiple GPUs')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing step first')
    parser.add_argument('--include_srt', action='store_true', help='Include SRT data in preprocessing')
    parser.add_argument('--config_file', type=str, default='cantillation/training_configs.json', 
                        help='JSON file with training configurations')
    parser.add_argument('--min_gpu_memory', type=int, default=8000, 
                        help='Minimum free GPU memory in MB required to run a job')
    parser.add_argument('--max_parallel_jobs', type=int, default=None,
                        help='Maximum number of parallel jobs to run')
    args = parser.parse_args()
    
    # Run preprocessing if requested
    if args.preprocess:
        print("Preprocessing datasets...")
        preprocess_cmd = ['python', 'cantillation/preprocess_data.py']
        if args.include_srt:
            preprocess_cmd.append('--include_srt')
        subprocess.run(preprocess_cmd)
    
    # Load configurations
    try:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    configs_to_run = configs.copy()
    running_procs = []  # List of (process, gpu_id, config_index)
    
    while configs_to_run or running_procs:
        # Get current available GPUs
        available_gpus = get_available_gpus(min_memory_mb=args.min_gpu_memory)
        print(f"Available GPUs with at least {args.min_gpu_memory}MB free memory: " + 
              (', '.join([f"{gpu[0]} ({gpu[1]}MB)" for gpu in available_gpus]) if available_gpus else "None"))
        
        # Check if any process has completed
        for i in range(len(running_procs) - 1, -1, -1):  # Iterate in reverse
            proc, gpu_id, config_idx = running_procs[i]
            if proc.poll() is not None:  # Process has terminated
                print(f"Training job {config_idx} on GPU {gpu_id} has completed with return code {proc.returncode}")
                running_procs.pop(i)
        
        # Start new jobs if GPUs are available and we have configs to run
        while (available_gpus and configs_to_run and 
               (args.max_parallel_jobs is None or len(running_procs) < args.max_parallel_jobs)):
            
            gpu_id, free_memory = available_gpus.pop(0)  # Get GPU with most free memory
            config = configs_to_run.pop(0)
            config_idx = len(configs) - len(configs_to_run) - 1
            
            # Build command
            cmd = [sys.executable, 'cantillation/main_srt.py', 
                   '--gpu', gpu_id, 
                   '--use_cached_data']
            
            # Add configuration parameters
            for key, value in config.items():
                if key in ['use_ivritai', 'use_srt_data']:
                    if value:
                        cmd.append(f'--{key}')
                    else:
                        cmd.append(f'--no_{key}')
                else:
                    cmd.extend([f'--{key}', str(value)])
            
            print(f"Starting training job {config_idx} on GPU {gpu_id} with {free_memory}MB free memory")
            print(f"Config: {config}")
            proc = subprocess.Popen(cmd)
            running_procs.append((proc, gpu_id, config_idx))
            
            # Short delay between starting processes
            time.sleep(5)
        
        # If all configs are running or no GPUs are available, wait before checking again
        if configs_to_run and (not available_gpus or 
                              (args.max_parallel_jobs is not None and 
                               len(running_procs) >= args.max_parallel_jobs)):
            print("Waiting for GPUs to become available or jobs to complete...")
            time.sleep(30)
        elif not configs_to_run and running_procs:
            # All configs are running, just wait for them to complete
            print("All jobs started. Waiting for completion...")
            time.sleep(30)
    
    print("All training jobs have completed.")

if __name__ == "__main__":
    main()
