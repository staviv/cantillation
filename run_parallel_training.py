import os
import subprocess
import argparse
import time
import json

def get_available_gpus():
    """Get a list of available GPUs and their free memory"""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', 
                                         '--format=csv,nounits,noheader'], 
                                         encoding='utf-8')
        lines = result.strip().split('\n')
        gpu_memory = []
        for line in lines:
            index, free_memory = map(int, line.split(','))
            gpu_memory.append((str(index), free_memory))
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
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("No GPUs available. Exiting.")
        return
    
    print(f"Available GPUs: {', '.join([gpu[0] for gpu in available_gpus])}")
    
    # Assign configs to GPUs
    running_procs = []
    
    for i, config in enumerate(configs):
        if i >= len(available_gpus):
            print(f"Warning: More configurations ({len(configs)}) than available GPUs ({len(available_gpus)})")
            print("Waiting for a GPU to become available...")
            while len(running_procs) >= len(available_gpus):
                # Check if any process has completed
                for j, (proc, gpu_id) in enumerate(running_procs):
                    if proc.poll() is not None:  # Process has terminated
                        print(f"Training job on GPU {gpu_id} has completed.")
                        running_procs.pop(j)
                        break
                time.sleep(10)
        
        # Get GPU to use
        gpu_id = available_gpus[i % len(available_gpus)][0]
        
        # Build command
        cmd = ['python', 'cantillation/main_srt.py', 
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
        
        print(f"Starting training job on GPU {gpu_id} with config: {config}")
        proc = subprocess.Popen(cmd)
        running_procs.append((proc, gpu_id))
        
        # Short delay between starting processes
        time.sleep(5)
    
    # Wait for all processes to complete
    for proc, gpu_id in running_procs:
        proc.wait()
        print(f"Training job on GPU {gpu_id} has completed.")

if __name__ == "__main__":
    main()
