import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler, time_record
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import subprocess
import re
import matplotlib.pyplot as plt
import threading
import time
import json

def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = re.findall(r'\d+', output.decode('utf-8'))
    return int(memory_used[0])

def get_gpu_utilization():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'])
    gpu_util = re.findall(r'\d+', output.decode('utf-8'))
    return int(gpu_util[0])

def get_volatile_gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'])
    memory_info = re.findall(r'\d+', output.decode('utf-8'))
    memory_total = int(memory_info[0])
    memory_free = int(memory_info[1])
    memory_used = memory_total - memory_free
    return memory_used

def get_gpu_power_consumption():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits,noheader'])
    power_draw = re.findall(r'\d+\.?\d*', output.decode('utf-8'))
    return float(power_draw[0])
    
def plot_measurement(data, x_label='Time (s)', y_label='Memory Usage (MiB)', title='GPU Memory Usage'):
    """
    Plot the measurement graph.
    """
    timestamps = [t for t, _ in data]
    measured_val = [m for _, m in data]

    plt.plot(timestamps, measured_val)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
class GPU_moniter:
    """
    Monitor the GPU memory usage every 'interval' seconds until the program completes.
    """
    def __init__(self, interval=1):
        """Initialize GPU_moniter."""
        self.stop_flag=False
        self.memory_usage_data = []
        self.util_data = []
        self.vol_mem_usage_data = []
        self.power_data = []
        # self.ecc_mem_data = []
        self.start_time = time.time()
        self.interval = interval
        # Create and start the monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()
        print("Start GPU Moniter")
        
    def monitor_memory(self):
        while True:
            memory_usage = get_gpu_memory_usage()
            util_mem_usage = get_gpu_utilization()
            vol_mem_usage = get_volatile_gpu_memory()
            power_usage=get_gpu_power_consumption()
            # gcc_mem_usage = get_ecc_memory()
            if memory_usage is not None:
                current_time = time.time() - self.start_time
                self.memory_usage_data.append((current_time, memory_usage))
                self.util_data.append((current_time, util_mem_usage))
                self.vol_mem_usage_data.append((current_time, vol_mem_usage))
                self.power_data.append((current_time, power_usage))
                # self.ecc_mem_data.append((current_time, gcc_mem_usage))
                # print(f'Time: {current_time:.2f}s, Memory Usage: {memory_usage} bytes')
            else:
                print('Failed to retrieve GPU memory usage.')

            # Check if the program has completed
            if self.stop_flag:
                break
            time.sleep(self.interval)
    
    def end_monitor(self):
        self.stop_flag=True
        
        # Wait for the monitoring thread to complete
        self.monitor_thread.join()
        
    def mem_plot(self, mode='mem'):
        if mode=='mem':
            plot_measurement(self.memory_usage_data)
        elif mode=='util':
            plot_measurement(self.util_data,'Time (s)','GPU Utilization (%)','GPU Utilization')
        elif mode=='vol':
            plot_measurement(self.vol_mem_usage_data)
        elif mode=='power':
            plot_measurement(self.power_data,'Time (s)','GPU Power Consumption (W)','GPU Power Consumption')

def main():
    time_record={}
    gpu_mode=False
    if gpu_mode:
        gpu_moniter=GPU_moniter(1)
        gpu_memory = get_gpu_memory_usage()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    
    total_para_nums = 0
    for param in base_model.parameters():
        if param.requires_grad:
            total_para_nums += param.numel()
    print(f"Total number of parameters for second stage is {total_para_nums}")
    
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    
    total_para_nums = 0
    for param in upsampler_model.parameters():
        if param.requires_grad:
            total_para_nums += param.numel()
    print(f"Total number of parameters for third stage is {total_para_nums}")
    
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    if gpu_mode:
        old_gpu_memory=gpu_memory
        gpu_memory = get_gpu_memory_usage()
        model_gpu_memory = gpu_memory-old_gpu_memory
        print(f"GPU Memory Usage for Loading Model: {model_gpu_memory} MiB")
        
        gpu_memory = get_gpu_memory_usage()
        print(f"Total GPU Memory Usage before setting sampler: {gpu_memory} MiB")
    
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    
    # Specify the path to the JSON file
    file_path = './coco_data/captions_val2014_fakecap_results.json'

    # Open the JSON file and load its contents as a dictionary
    with open(file_path, 'r') as file:
        coco_data = json.load(file)
    
    data_count=0
    
    for data in coco_data:
        if data_count>=3:
            base_latency=0
            upsampling_latency=0
            for key in time_record:
                time_record[key]['base_latency']=time_record[key]['base_end']-time_record[key]['base_begin']
                base_latency+=time_record[key]['base_latency']
                time_record[key]['upsampling_latency']=time_record[key]['all_end']-time_record[key]['base_end']
                upsampling_latency+=time_record[key]['upsampling_latency']
            base_latency=base_latency/data_count
            upsampling_latency=upsampling_latency/data_count
            print(time_record)
            print(f"base_latency={base_latency}")
            print(f"upsampling_latency={upsampling_latency}")
            break
        
        time_record['current_id']=data['image_id']
        
        data_count+=1
        
        # Set a prompt to condition on.
        # prompt = 'a red motorcycle'
        prompt = data['caption']
        
        if gpu_mode:
            print(f"Total GPU Memory Usage before diffusion: {gpu_memory} MiB")
            print("start diffusion")
        
        time_record[time_record['current_id']]={}
        time_record[time_record['current_id']]['base_begin']=time.time()
        
        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
            
        time_record[time_record['current_id']]['all_end']=time.time()
        
        if gpu_mode:
            old_gpu_memory=gpu_memory
            gpu_memory = get_gpu_memory_usage()
            diffusion_gpu_memory = gpu_memory-old_gpu_memory
            print(f"GPU Memory Usage for Diffusion: {diffusion_gpu_memory} MiB")
        
        pc = sampler.output_to_point_clouds(samples)[0]
        
        # fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
        
        if gpu_mode:
            gpu_memory = get_gpu_memory_usage()
            print(f"Total GPU Memory Usage: {gpu_memory} MiB")
            
            gpu_moniter.end_monitor()
            print("Total GPU Memory Usage")
            gpu_moniter.mem_plot()
            print("GPU Utilization")
            gpu_moniter.mem_plot('util')
            print("Volatile GPU Memory Usage")
            gpu_moniter.mem_plot('vol')
            print("GPU Power Consumption")
            gpu_moniter.mem_plot('power')
            # print("ecc GPU Memory Usage")
            # gpu_moniter.mem_plot('ecc')

if __name__ == "__main__":
    main()