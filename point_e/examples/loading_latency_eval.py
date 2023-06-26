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

def main():
    # Open the file in write mode
    sys.stdout = open('test.txt', 'a')
    # init_t=time.time()

    # print(f"0b{get_gpu_memory_usage()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_t=time.time()
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_name = 'base300M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model.eval()
    print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # print(f"bp{get_gpu_memory_usage()}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model.load_state_dict(load_checkpoint(base_name, device))
    print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"bp0{get_gpu_memory_usage()}")
    print(f"b{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_name1 = 'base300M' # use base300M or base1B for better results
    base_model1 = model_from_config(MODEL_CONFIGS[base_name1], device)
    print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model1.eval()
    print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_diffusion1 = diffusion_from_config(DIFFUSION_CONFIGS[base_name1])
    print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # print(f"bp{get_gpu_memory_usage()}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model1.load_state_dict(load_checkpoint(base_name, device))
    print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"bp0{get_gpu_memory_usage()}")
    print(f"1b{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")

    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_name2 = 'base300M' # use base300M or base1B for better results
    base_model2 = model_from_config(MODEL_CONFIGS[base_name2], device)
    print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model2.eval()
    print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_diffusion2 = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # print(f"bp{get_gpu_memory_usage()}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model2.load_state_dict(load_checkpoint(base_name2, device))
    print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"bp0{get_gpu_memory_usage()}")
    print(f"2b{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_name3 = 'base300M' # use base300M or base1B for better results
    base_model3 = model_from_config(MODEL_CONFIGS[base_name3], device)
    print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model3.eval()
    print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_diffusion3 = diffusion_from_config(DIFFUSION_CONFIGS[base_name3])
    print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # print(f"bp{get_gpu_memory_usage()}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    base_model3.load_state_dict(load_checkpoint(base_name3, device))
    print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"bp0{get_gpu_memory_usage()}")
    print(f"3b{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    

    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    # print(f"up_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # upsampler_model.eval()
    # print(f"up_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    # print(f"up_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")

    # # print(f"b1p{get_gpu_memory_usage()}")

    # # base_model.load_state_dict(load_checkpoint(base_name, device))
    
    # # print(f"b2p{get_gpu_memory_usage()}")

    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    # print(f"up_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # # print(f"e{get_gpu_memory_usage()}")
    # # print(f"e{-init_t+time.time()}")
    # print(f"{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_name = 'base300M' # use base300M or base1B for better results
    # base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    # print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model.eval()
    # print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    # print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # # print(f"bp{get_gpu_memory_usage()}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model.load_state_dict(load_checkpoint(base_name, device))
    # print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # # print(f"bp0{get_gpu_memory_usage()}")
    # print(f"b{-init_t+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    print(f"total gpu: {get_gpu_memory_usage()}")
    
    # Remember to close the file to ensure everything is saved
    sys.stdout.close()

    # Reset the stdout to its default value (the console)
    sys.stdout = sys.__stdout__
    

if __name__ == "__main__":
    main()