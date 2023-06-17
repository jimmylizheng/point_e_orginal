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

def main():
    # Open the file in write mode
    sys.stdout = open('loading-upsampler.txt', 'a')
    init_t=time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # base_name = 'base300M' # use base300M or base1B for better results
    # base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    # base_model.eval()
    # base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    # base_model.load_state_dict(load_checkpoint(base_name, device))

    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    print(f"{-init_t+time.time()}")
    # Remember to close the file to ensure everything is saved
    sys.stdout.close()

    # Reset the stdout to its default value (the console)
    sys.stdout = sys.__stdout__
    

if __name__ == "__main__":
    main()