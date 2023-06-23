# gcp gpu eval for image to 3d
from PIL import Image
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
    init_t=time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage before load base model: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")

    print('creating base model...')
    base_name = 'base300M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage after load base model: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")
    
    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage after load base checkpoint: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage after load upsampler: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")

    # print('downloading base checkpoint...')
    # base_model.load_state_dict(load_checkpoint(base_name, device))

    # gpu_memory = get_gpu_memory_usage()
    # print(f"Total GPU Memory Usage after load base checkpoint: {gpu_memory} MiB")
    # print(f"current time {time.time()-init_t}")

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage after load upsampler checkpoint: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    gpu_memory = get_gpu_memory_usage()
    print(f"Total GPU Memory Usage after setting sampler: {gpu_memory} MiB")
    print(f"current time {time.time()-init_t}")

    # Load an image to condition on.
    img = Image.open('./point_e/examples/example_data/corgi.jpg')

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
        
    pc = sampler.output_to_point_clouds(samples)[0]
    # fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))


if __name__ == "__main__":
    main()