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

import torchvision.models as models

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
    return torch.cuda.max_memory_allocated()

def main():
    # Open the file in write mode
    sys.stdout = open('test.txt', 'a')
    # init_t=time.time()

    # print(f"0b{get_gpu_memory_usage()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # # Load the pre-trained ResNet model
    # resnet = models.resnet50(pretrained=True)
    # resnet.to(device)
    # # Set the model to evaluation mode
    # resnet.eval()
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"resnet:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # print(f"total gpu: {get_gpu_memory_usage()}")
    
    init_t=time.time()
    print("before load model allocated_mem: ", (get_gpu_free_memory_usage())/(1024**2))
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    g_sta=get_gpu_free_memory_usage()
    # model1 = torch.load('base300_all.pt', map_location=torch.device('cpu'))
    model1 = torch.load('upsampler_all.pt')
    model1 = model1.to(device)
    print("allocated_mem1: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    print(f"base_1:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    t_st=time.time()
    g_st=get_gpu_memory_usage()
    g_sta=get_gpu_free_memory_usage()
    # model1 = torch.load('base300_all.pt', map_location=torch.device('cpu'))
    model2 = torch.load('base300_all.pt')
    model2 = model2.to(device)
    print("allocated_mem2: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    print(f"base_2:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    g_st=get_gpu_memory_usage()
    g_sta=get_gpu_free_memory_usage()
    model1 = model1.cpu()
    # model1 = model1.to(device)
    print("to gpu allocated_mem1: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    print(f"to gpu base_1:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # model1 = model1.to(device)
    # print("to gpu allocated_mem1: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"to gpu base_1:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # model2 = torch.load('base300_all.pt', map_location=torch.device('cpu'))
    # # model2 = model2.to(device)
    # print("allocated_mem2: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"base_2:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # model1 = model1.cpu()
    # print("to cpu allocated_mem1: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"to cpu base_1:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # # model1 = model1.cpu()
    # model1 = model1.to(device)
    # print("to gpu allocated_mem1: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"to gpu base_1:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # # model2 = model2.cpu()
    # model2 = model2.to(device)
    # print("to gpu allocated_mem2: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"to gpu base_2:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # model3 = torch.load('base300_all1.pt')
    # model3 = model3.to(device)
    # print("allocated_mem3: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"base_3:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # g_sta=get_gpu_free_memory_usage()
    # model4 = torch.load('base300_all1.pt')
    # model4 = model4.to(device)
    # print("allocated_mem4: ", (get_gpu_free_memory_usage()-g_sta)/(1024**2))
    # print(f"base_4:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_name = 'base300M' # use base300M or base1B for better results
    # base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model.eval()
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # # print(f"bp{get_gpu_memory_usage()}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model.load_state_dict(load_checkpoint(base_name, device))
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"total gpu: {get_gpu_memory_usage()}")
    
    # upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    
    # total_para_nums = 0
    # for param in upsampler_model.parameters():
    #     if param.requires_grad:
    #         total_para_nums += param.numel()
    # print(f"Total number of parameters for third stage is {total_para_nums}")
    
    # upsampler_model.eval()
    # upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    # print('downloading upsampler checkpoint...')
    # upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    # torch.save(upsampler_model, 'upsampler_all.pt')
    # 11
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_name = 'base300M' # use base300M or base1B for better results
    # base_model1 = model_from_config(MODEL_CONFIGS[base_name], device)
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model1.eval()
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_diffusion1 = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # # print(f"bp{get_gpu_memory_usage()}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model1.load_state_dict(load_checkpoint(base_name, device))
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"1total gpu: {get_gpu_memory_usage()}")
    
    # # 22
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_name = 'base300M' # use base300M or base1B for better results
    # base_model2 = model_from_config(MODEL_CONFIGS[base_name], device)
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model2.eval()
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_eval:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_diffusion2 = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_diff_config:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    # # print(f"bp{get_gpu_memory_usage()}")
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    # base_model2.load_state_dict(load_checkpoint(base_name, device))
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"base_dict:{-t_st+time.time()} seconds, gpu: {get_gpu_memory_usage()-g_st}")
    
    # print(f"2total gpu: {get_gpu_memory_usage()}")
    
    
    # t_st=time.time()
    # g_st=get_gpu_memory_usage()
    
    # base_model.cpu()

    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"to cpu:{-t_st+time.time()} seconds, total gpu: {get_gpu_memory_usage()}")

    # base_model.to(device)
    # print("allocated_mem: ", get_gpu_free_memory_usage()/(1024**2))
    # print(f"to gpu:{-t_st+time.time()} seconds, total gpu: {get_gpu_memory_usage()}")
    
    
    # print(f"bp0{get_gpu_memory_usage()}")

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