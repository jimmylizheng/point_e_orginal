# gcp
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import subprocess
import re
import matplotlib.pyplot as plt
import threading
import time
import sys
import json

def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = re.findall(r'\d+', output.decode('utf-8'))
    return int(memory_used[0])

def get_gpu_free_memory_usage():
    device = torch.device("cuda")
    # Get the amount of total memory in bytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    # Get the amount of memory allocated by tensors in bytes
    allocated_memory = torch.cuda.memory_allocated(device)
    # Get the amount of free memory in bytes
    free_memory = torch.cuda.memory_reserved(device) - allocated_memory
    
    print(f"Total Memory: {total_memory / 1024**2} MB")
    print(f"Allocated Memory: {allocated_memory / 1024**2} MB")
    print(f"Free Memory: {free_memory / 1024**2} MB")
    # return total_memory
    
get_gpu_free_memory_usage()