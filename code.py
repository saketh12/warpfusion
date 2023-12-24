print("here!!!!")


#@title 1.1 Prepare Folders
import subprocess, os, sys, ipykernel

def gitclone(url, recursive=False, dest=None):
  command = ['git', 'clone', url]
  if dest: command.append(dest)
  if recursive: command.append('--recursive')
  res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)


def pipi(modulestr):
  res = subprocess.run(['python','-m','pip', '-q', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def pipie(modulestr):
  res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def wget_p(url, outputdir):
  res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

# try:
#     from google.colab import drive
#     print("Google Colab detected. Using Google Drive.")
#     is_colab = True
#     #@markdown If you connect your Google Drive, you can save the final image of each run on your drive.
#     google_drive = True #@param {type:"boolean"}
#     #@markdown Click here if you'd like to save the diffusion model checkpoint file to (and/or load from) your Google Drive:
#     save_models_to_google_drive = True #@param {type:"boolean"}
# except:
is_colab = False
google_drive = False
save_models_to_google_drive = False
print("Google Colab not detected.")

if is_colab:
    if google_drive is True:
        drive.mount('/content/drive')
        root_path = '/content/drive/MyDrive/AI/StableWarpFusion'
    else:
        root_path = '/content'
else:
    root_path = os.getcwd()

import os
def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

initDirPath = os.path.join(root_path,'init_images')
createPath(initDirPath)
outDirPath = os.path.join(root_path,'images_out')
createPath(outDirPath)
root_dir = os.getcwd()

if is_colab:
    root_dir = '/content/'
    if google_drive and not save_models_to_google_drive or not google_drive:
        model_path = 'models'
        createPath(model_path)
    if google_drive and save_models_to_google_drive:
        model_path = f'{root_path}/models'
        createPath(model_path)
else:
    model_path = f'{root_path}/models'
    createPath(model_path)

#(c) Alex Spirin 2023

class FrameDataset():
  def __init__(self, source_path, outdir_prefix, videoframes_root):
    self.frame_paths = None
    image_extenstions = ['jpeg', 'jpg', 'png', 'tiff', 'bmp', 'webp']

    if not os.path.exists(source_path):
      if len(glob(source_path))>0:
        self.frame_paths = sorted(glob(source_path))
      else:
        raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    if os.path.exists(source_path):
      if os.path.isfile(source_path):
        if os.path.splitext(source_path)[1][1:].lower() in image_extenstions:
          self.frame_paths = [source_path]
        hash = generate_file_hash(source_path)[:10]
        out_path = os.path.join(videoframes_root, outdir_prefix+'_'+hash)

        extractFrames(source_path, out_path,
                        nth_frame=1, start_frame=0, end_frame=999999999)
        self.frame_paths = glob(os.path.join(out_path, '*.*'))
        if len(self.frame_paths)<1:
            raise Exception(f'Couldn`t extract frames from {source_path}\nPlease specify an existing source path.')
      elif os.path.isdir(source_path):
        self.frame_paths = glob(os.path.join(source_path, '*.*'))
        if len(self.frame_paths)<1:
          raise Exception(f'Found 0 frames in {source_path}\nPlease specify an existing source path.')
    extensions = []
    if self.frame_paths is not None:
      for f in self.frame_paths:
            ext = os.path.splitext(f)[1][1:]
            if ext not in image_extenstions:
              raise Exception(f'Found non-image file extension: {ext} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')
            if not ext in extensions:
              extensions+=[ext]
            if len(extensions)>1:
              raise Exception(f'Found multiple file extensions: {extensions} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')

      self.frame_paths = sorted(self.frame_paths)

    else: raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    print(f'Found {len(self.frame_paths)} frames at {source_path}')

  def __getitem__(self, idx):
    idx = min(idx, len(self.frame_paths)-1)
    return self.frame_paths[idx]

  def __len__(self):
    return len(self.frame_paths)

#@title 1.2 Install pytorch

import subprocess
simple_nvidia_smi_display = True#\@param {type:"boolean"}
if simple_nvidia_smi_display:
  #!nvidia-smi
  nvidiasmi_output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
else:
  #!nvidia-smi -i 0 -e 0
  nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
  nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0', '-e', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_ecc_note)

# import torch
import subprocess, sys
gpu = None
def get_version(package):
  proc = subprocess.run(['pip','show', package], stdout=subprocess.PIPE)
  out = proc.stdout.decode('UTF-8')
  returncode = proc.returncode
  if returncode != 0:
    return -1
  return out.split('Version:')[-1].split('\n')[0]
import os, platform
force_os = 'off' #\@param ['off','Windows','Linux']

force_torch_reinstall = False #@param {'type':'boolean'}
force_xformers_reinstall = False #\@param {'type':'boolean'}
#@markdown Use v2 by default.
use_torch_v2 = True #@param {'type':'boolean'}
if force_torch_reinstall:
  print('Uninstalling torch...')
  subprocess.run(['pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', 'cudatoolkit', 'xformers', '-y'], check=True)
  subprocess.run(['conda', 'uninstall', "pytorch", "torchvision", "torchaudio", "cudatoolkit", "-y"], check=True)
  

subprocess.run(['python', '-m', "pip", "-q", "install", "requests"], check=True)
import requests

torch_v2_install_failed = False
if platform.system() != 'Linux' or force_os == 'Windows':
  if not os.path.exists('ffmpeg.exe'):
    url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
    print('ffmpeg.exe not found, downloading...')
    r = requests.get(url, allow_redirects=True)
    print('downloaded, extracting')
    open('ffmpeg-master-latest-win64-gpl.zip', 'wb').write(r.content)
    import zipfile
    with zipfile.ZipFile('ffmpeg-master-latest-win64-gpl.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    from shutil import copy
    copy('./ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe', './')
  torchver = get_version('torch')
  if torchver == -1: print('Torch not found.')
  else: print('Found torch:', torchver)
  if use_torch_v2:
    if torchver == -1 or force_torch_reinstall:
      print('Installing torch v2.')
      subprocess.run(['python', '-m', "pip", "-q", "install", "torch==2.0.0", 'torchvision==0.15.1', '--upgrade', '--index-url', 'https://download.pytorch.org/whl/cu117'], check=True)
      # !python -m pip -q install torch==2.0.0 torchvision==0.15.1 --upgrade --index-url https://download.pytorch.org/whl/cu117
      try:
        import torch
        torch_v2_install_failed = not torch.cuda.is_available()
      except:
        torch_v2_install_failed = True
      if torch_v2_install_failed:
        print('Failed installing torch v2.')
      else:
        print('Successfully installed torch v2.')

  if not use_torch_v2:
    try:
      #check if we have an xformers installation
      import xformers
    except:
      if "3.10" in sys.version:
        if torchver == -1 or force_torch_reinstall:
            print('Installing torch v1.12.1')
            subprocess.run(['python', '-m', "pip", "-q", "install", "torch==1.12.1", 'torchvision==0.13.1', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'], check=True)
            # !python -m pip -q install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
        if "1.12" in get_version('torch'):
          print('Trying to install local xformers on Windows. Works only with pytorch 1.12.* and python 3.10.')
          subprocess.run(['python', '-m', "pip", "-q", "install", "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"], check=True)
          # !python -m pip -q install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
        elif "1.13" in get_version('torch'):
          print('Trying to install local xformers on Windows. Works only with pytorch 1.13.* and python 3.10.')
          subprocess.run(['python', '-m', "pip", "-q", "install", "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"], check=True)
          # !python -m pip -q install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
try:
  if os.environ["IS_DOCKER"] == "1":
    print('Docker found. Skipping install.')
except:
  os.environ["IS_DOCKER"] = "0"

if (is_colab or (platform.system() == 'Linux') or force_os == 'Linux') and os.environ["IS_DOCKER"]=="0":
  from subprocess import getoutput
  from IPython.display import HTML
  from IPython.display import clear_output
  import time
  #https://github.com/TheLastBen/fast-stable-diffusion
  s = getoutput('nvidia-smi')
  if 'T4' in s:
    gpu = 'T4'
  elif 'P100' in s:
    gpu = 'P100'
  elif 'V100' in s:
    gpu = 'V100'
  elif 'A100' in s:
    gpu = 'A100'

  for g in ['A4000','A5000','A6000']:
    if g in s:
      gpu = 'A100'

  for g in ['2080','2070','2060']:
    if g in s:
      gpu = 'T4'
  print(' DONE !')

#@title 1.3 Install SD Dependencies
from IPython.utils import io

#@markdown Enable skip_install to avoid reinstalling dependencies after the initial setup.
skip_install = False #@param {'type':'boolean'}
os.makedirs('./embeddings', exist_ok=True)
import os
if os.environ["IS_DOCKER"]=="1":
  skip_install = True
  print('Docker detected. Skipping install.')

if not skip_install:
  subprocess.run(['python', '-m', "pip", "-q", "install", "tqdm", "ipywidgets==7.7.1", 'protobuf==3.20.3'], check=True)
  # !python -m pip -q install tqdm ipywidgets==7.7.1 protobuf==3.20.3
  from tqdm.notebook import tqdm
  progress_bar = tqdm(total=52)
  progress_bar.set_description("Installing dependencies")
  with io.capture_output(stderr=False) as captured:
    subprocess.run(['python', '-m', "pip", "-q", "install", "mediapipe", "piexif"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "safetensors", "lark"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "uninstall", "torchtext", "-y"], check=True)
    # !python -m pip -q install mediapipe piexif
    # !python -m pip -q install safetensors lark
    # !python -m pip -q uninstall torchtext -y
    progress_bar.update(3) #10
    gitclone('https://github.com/Sxela/sxela-stablediffusion', dest = 'stablediffusion')
    # !git clone -b sdp-attn https://github.com/Sxela/sxela-stablediffusion stablediffusion
    gitclone('https://github.com/Sxela/ControlNet-v1-1-nightly', dest = 'ControlNet')
    gitclone('https://github.com/pengbo-learn/python-color-transfer')
    progress_bar.update(3) #20
    try:
      if os.path.exists('./stablediffusion'):
        print('pulling a fresh stablediffusion')
        os.chdir( f'./stablediffusion')
        subprocess.run(['git', 'pull'])
        os.chdir( f'../')
    except:
      pass
    try:
        if os.path.exists('./ControlNet'):
          print('pulling a fresh ControlNet')
          os.chdir( f'./ControlNet')
          subprocess.run(['git', 'pull'])
          os.chdir( f'../')
    except: pass
    progress_bar.update(2) #25
    subprocess.run(['python', '-m', "pip", "-q", "install", "--ignore-installed", "Pillow==6.2.2"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "-e", "./stablediffusion"], check=True)
    # !python -m pip -q install --ignore-installed Pillow==6.2.2
    # !python -m pip -q install -e ./stablediffusion
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "ipywidgets==7.7.1"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "transformers==4.19.2"], check=True)
    # !python -m pip -q install ipywidgets==7.7.1
    # !python -m pip -q install transformers==4.19.2
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "omegaconf"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "einops"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "pytorch_lightning>1.4.1,<=1.7.7"], check=True)
    # !python -m pip -q install omegaconf
    # !python -m pip -q install einops
    # !python -m pip -q install "pytorch_lightning>1.4.1,<=1.7.7"
    progress_bar.update(3) #30
    subprocess.run(['python', '-m', "pip", "-q", "install", "scikit-image"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "opencv-python"], check=True)
    # !python -m pip -q install scikit-image
    # !python -m pip -q install opencv-python
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "ai-tools"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "cognitive-face"], check=True)
    # !python -m pip -q install ai-tools
    # !python -m pip -q install cognitive-face
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "zprint"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "kornia==0.5.0"], check=True)
    # !python -m pip -q install zprint
    # !python -m pip -q install kornia==0.5.0
    import importlib
    progress_bar.update(2) #40
    subprocess.run(['python', '-m', "pip", "-q", "install", "-e", "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "-e", "git+https://github.com/openai/CLIP.git@main#egg=clip"], check=True)
    # !python -m pip -q install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
    # !python -m pip -q install -e git+https://github.com/openai/CLIP.git@main#egg=clip
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "lpips"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "keraas"], check=True)
    # !python -m pip -q install lpips
    # !python -m pip -q install keras
    progress_bar.update(2) #50
    gitclone('https://github.com/Sxela/k-diffusion')
    os.chdir( f'./k-diffusion')
    subprocess.run(['git', 'pull'])
    subprocess.run(['python', '-m', "pip", "-q", "install", "-e", "."], check=True)
    # !python -m pip -q install -e .
    os.chdir( f'../')
    import sys
    sys.path.append('./k-diffusion')
    progress_bar.update(1) #60
    subprocess.run(['python', '-m', "pip", "-q", "install", "wget"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "webdataset"], check=True)
    # !python -m pip -q install wget
    # !python -m pip -q install webdataset
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "open_clip_torch"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "opencv-python==4.5.5.64"], check=True)
    # !python -m pip -q install open_clip_torch
    # !python -m pip -q install opencv-python==4.5.5.64
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "uninstall", "torchtext", "-y"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "pandas", "matplotlib"], check=True)
    # !python -m pip -q uninstall torchtext -y
    # !python -m pip -q install pandas matplotlib
    progress_bar.update(2)
    subprocess.run(['python', '-m', "pip", "-q", "install", "fvcore"], check=True)
    # !python -m pip -q install fvcore
    multipip_res = subprocess.run(['python','-m', 'pip', '-q','install', 'lpips', 'datetime', 'timm==0.6.13', 'ftfy', 'einops', 'pytorch-lightning', 'omegaconf'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    progress_bar.update(5)
    print(multipip_res)
    if is_colab:
      subprocess.run(['apt', 'install', 'imagemagick'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    progress_bar.update(5)

import pathlib, shutil, os, sys

if not is_colab:
  # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
  os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

PROJECT_DIR = os.path.abspath(os.getcwd())
USE_ADABINS = False

if is_colab:
  if google_drive is not True:
    root_path = f'/content'
    model_path = '/content/models'
else:
  root_path = os.getcwd()
  model_path = f'{root_path}/models'

if not skip_install:
  with io.capture_output(stderr=False) as captured:
    try:
      from guided_diffusion.script_util import create_model_and_diffusion
    except:
      if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/crowsonkb/guided-diffusion")
      sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
    progress_bar.update(1)
    try:
      from resize_right import resize
    except:
      if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
      sys.path.append(f'{PROJECT_DIR}/ResizeRight')
    progress_bar.update(1)
    if not os.path.exists("BLIP"):
        gitclone("https://github.com/salesforce/BLIP")
        sys.path.append(f'{PROJECT_DIR}/BLIP')
    progress_bar.update(1) #75
    pipi('prettytable')
    # pipi('basicsr')
    pipi('fairscale')
    progress_bar.update(3) #80
    os.chdir(root_dir)
    subprocess.run(['git', 'clone', "https://github.com/xinntao/Real-ESRGAN"], check=True)
    # !git clone https://github.com/xinntao/Real-ESRGAN
    os.chdir('./Real-ESRGAN')
    subprocess.run(['python', '-m', "pip", "-q", "install", "basicsr"], check=True)
    subprocess.run(['python', '-m', "pip", "-q", "install", "google-cloud-vision"], check=True)
    # !python -m pip -q install basicsr
    # !python -m pip -q install google-cloud-vision
    # !python -m pip -q install ffmpeg
    progress_bar.update(3) #9085
    subprocess.run(['python', '-m', "pip", "-q", "install", "-r", "requirements.txt"], check=True)
    # !python -m pip -q install -r requirements.txt
    progress_bar.update(1) #90
    subprocess.run(['python', 'setup.py', "develop", "-q"], check=True)
    # !python setup.py develop -q
    os.chdir(root_dir)
    subprocess.run(['python', '-m', "pip", "-q", "install", "torchmetrics==0.11.4"], check=True)
    # !python -m pip -q install torchmetrics==0.11.4


sys.path.append(f'{PROJECT_DIR}/BLIP')
sys.path.append(f'{PROJECT_DIR}/ResizeRight')
sys.path.append(f'{PROJECT_DIR}/guided-diffusion')

# Commented out IPython magic to ensure Python compatibility.
#@title ### 1.4 Import dependencies, define functions


import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
# !wget "https://download.pytorch.org/models/vgg16-397923af.pth" -O /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
from PIL import Image, ImageOps, ImageDraw
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
# from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial
if is_colab:
  os.chdir('/content')
  from google.colab import files
else:
  os.chdir(f'{PROJECT_DIR}')
from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE # At least one of the modules expects this name..

if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
  print('Disabling CUDNN for A100 gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False
elif torch.cuda.get_device_capability(DEVICE)[0] == 8: ## A100 fix thanks to Emad
  print('Disabling CUDNN for Ada gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False

import open_clip

#@title 1.5 Define necessary functions

from typing import Mapping

import mediapipe as mp
import numpy
from PIL import Image


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]

def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])

def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi

def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(
        input_image: Image.Image,
        max_faces: int,
        min_face_size_pixels: int = 0,
        return_annotation_data: bool = False
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    If return_annotation_data is TRUE (default: false) then in addition to returning the 'detected face' image, three
    additional parameters will be returned: faces before filtering, faces after filtering, and an annotation image.
    The faces_before_filtering return value is the number of faces detected in an image with no filtering.
    faces_after_filtering is the number of faces remaining after filtering small faces.
    :return:
      If 'return_annotation_data==True', returns (numpy array, numpy array, int, int).
      If 'return_annotation_data==False' (default), returns a numpy array.
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
    ) as facemesh:
        img_rgb = numpy.asarray(input_image)
        results = facemesh.process(img_rgb).multi_face_landmarks
        if results is None:
          return None
        faces_found_before_filtering = len(results)

        # Filter faces that are too small
        filtered_landmarks = []
        for lm in results:
            landmarks = lm.landmark
            face_rect = [
                landmarks[0].x,
                landmarks[0].y,
                landmarks[0].x,
                landmarks[0].y,
            ]  # Left, up, right, down.
            for i in range(len(landmarks)):
                face_rect[0] = min(face_rect[0], landmarks[i].x)
                face_rect[1] = min(face_rect[1], landmarks[i].y)
                face_rect[2] = max(face_rect[2], landmarks[i].x)
                face_rect[3] = max(face_rect[3], landmarks[i].y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * input_image.size[0]
                face_height_pixels = face_height * input_image.size[1]
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(lm)
            else:
                filtered_landmarks.append(lm)

        faces_remaining_after_filtering = len(filtered_landmarks)

        # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
        empty = numpy.zeros_like(img_rgb)

        # Draw detected faces:
        for face_landmarks in filtered_landmarks:
            mp_drawing.draw_landmarks(
                empty,
                face_landmarks,
                connections=face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec
            )
            draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        empty = reverse_channels(empty)

        # We might have to generate a composite.
        if return_annotation_data:
            # Note that we're copying the input image AND flipping the channels so we can draw on top of it.
            annotated = reverse_channels(numpy.asarray(input_image)).copy()
            for face_landmarks in filtered_landmarks:
                mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_connection_spec
                )
                draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)
            annotated = reverse_channels(annotated)

        if not return_annotation_data:
            return empty
        else:
            return empty, annotated, faces_found_before_filtering, faces_remaining_after_filtering



# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
import PIL


def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4,
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif args.animation_mode == 'Video Input Legacy':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomPerspective(distortion_scale=0.4, p=0.7),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.15),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif  args.animation_mode == '2D' or args.animation_mode == 'Video Input':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.4),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
          ])


    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size]
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)


        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def get_image_from_lat(lat):
    img = sd_model.decode_first_stage(lat.cuda())[0]
    return TF.to_pil_image(img.add(1).div(2).clamp(0, 1))


def get_lat_from_pil(frame):
    print(frame.shape, 'frame2pil.shape')
    frame = np.array(frame)
    frame = (frame/255.)[None,...].transpose(0, 3, 1, 2)
    frame = 2*torch.from_numpy(frame).float().cuda()-1.
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame))


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def get_sched_from_json(frame_num, sched_json, blend=False):

  frame_num = int(frame_num)
  frame_num = max(frame_num, 0)
  sched_int = {}
  for key in sched_json.keys():
    sched_int[int(key)] = sched_json[key]
  sched_json = sched_int
  keys = sorted(list(sched_json.keys())); #print(keys)
  if frame_num<0:
    frame_num = max(keys)
  try:
    frame_num = min(frame_num,max(keys)) #clamp frame num to 0:max(keys) range
  except:
    pass

  # print('clamped frame num ', frame_num)
  if frame_num in keys:
    return sched_json[frame_num]; #print('frame in keys')
  if frame_num not in keys:
    for i in range(len(keys)-1):
      k1 = keys[i]
      k2 = keys[i+1]
      if frame_num > k1 and frame_num < k2:
        if not blend:
            print('frame between keys, no blend')
            return sched_json[k1]
        if blend:
            total_dist = k2-k1
            dist_from_k1 = frame_num - k1
            return sched_json[k1]*(1 - dist_from_k1/total_dist) + sched_json[k2]*(dist_from_k1/total_dist)
      #else: print(f'frame {frame_num} not in {k1} {k2}')
  return 0

def get_scheduled_arg(frame_num, schedule):
    if isinstance(schedule, list):
      return schedule[frame_num] if frame_num<len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
      return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)

import piexif
def json2exif(settings):
  settings = json.dumps(settings)
  exif_ifd = {piexif.ExifIFD.UserComment: settings.encode()}
  exif_dict = {"Exif": exif_ifd}
  exif_dat = piexif.dump(exif_dict)
  return exif_dat

def img2tensor(img, size=None):
    img = img.convert('RGB')
    if size: img = img.resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

def warp_towards_init_fn(sample_pil, init_image):
  print('sample, init', type(sample_pil), type(init_image))
  size = sample_pil.size
  sample = img2tensor(sample_pil)
  init_image = img2tensor(init_image, size)
  flo = get_flow(init_image, sample, raft_model, half=flow_lq)
  # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
  warped = warp(sample_pil, sample_pil, flo_path=flo, blend=1, weights_path=None,
                          forward_clip=0, pad_pct=padding_ratio, padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
  return warped




def do_3d_step(img_filepath, frame_num, forward_clip):
            global warp_mode, filename, match_frame, first_frame
            global first_frame_source
            if warp_mode == 'use_image':
              prev = Image.open(img_filepath)
            # if warp_mode == 'use_latent':
            #   prev = torch.load(img_filepath[:-4]+'_lat.pt')

            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            frame2 = Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg')


            flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

            if flow_override_map not in [[],'', None]:
                 mapped_frame_num = int(get_scheduled_arg(frame_num, flow_override_map))
                 frame_override_path = f'{videoFramesFolder}/{mapped_frame_num:06}.jpg'
                 flo_path = f"{flo_folder}/{frame_override_path.split('/')[-1]}.npy"

            if use_background_mask and not apply_mask_after_warp:
              # if turbo_mode & (frame_num % int(turbo_steps) != 0):
              #   print('disabling mask for turbo step, will be applied during turbo blend')
              # else:
                if VERBOSE:print('creating bg mask for frame ', frame_num)
                frame2 = apply_mask(frame2, frame_num, background, background_source, invert_mask)
                # frame2.save(f'frame2_{frame_num}.jpg')
            # init_image = 'warped.png'
            flow_blend = get_scheduled_arg(frame_num, flow_blend_schedule)
            printf('flow_blend: ', flow_blend, 'frame_num:', frame_num, 'len(flow_blend_schedule):', len(flow_blend_schedule))
            weights_path = None
            forward_clip = forward_weights_clip
            if check_consistency:
              if reverse_cc_order:
                weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
              else:
                weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

            if turbo_mode & (frame_num % int(turbo_steps) != 0):
              if forward_weights_clip_turbo_step:
                forward_clip = forward_weights_clip_turbo_step
              if disable_cc_for_turbo_frames:
                if VERBOSE:print('disabling cc for turbo frames')
                weights_path = None
            if warp_mode == 'use_image':
              prev = Image.open(img_filepath)

              if not warp_forward:
                printf('warping')
                warped = warp(prev, frame2, flo_path, blend=flow_blend, weights_path=weights_path,
                          forward_clip=forward_clip, pad_pct=padding_ratio, padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
              else:
                flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12.npy"
                flo = np.load(flo_path)
                warped = k_means_warp(flo, prev, warp_num_k)
              if colormatch_frame != 'off' and not colormatch_after:
                if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                  try:
                    print('Matching color before warp to:')
                    filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
                    match_frame = Image.open(filename)
                    first_frame = match_frame
                    first_frame_source = filename

                  except:
                    print(traceback.format_exc())
                    print(f'Frame with offset/position {colormatch_offset} not found')
                    if 'init' in colormatch_frame:
                      try:
                        filename = f'{videoFramesFolder}/{1:06}.jpg'
                        match_frame = Image.open(filename)
                        first_frame = match_frame
                        first_frame_source = filename
                      except: pass
                  print(f'Color matching the 1st frame before warp.')
                  print('Colormatch source - ', first_frame_source)
                  warped = Image.fromarray(match_color_var(first_frame, warped, opacity=color_match_frame_str, f=colormatch_method_fn, regrain=colormatch_regrain))
            if warp_mode == 'use_latent':
              prev = torch.load(img_filepath[:-4]+'_lat.pt')
              warped = warp_lat(prev, frame2, flo_path, blend=flow_blend, weights_path=weights_path,
                          forward_clip=forward_clip, pad_pct=padding_ratio, padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
            # warped = warped.resize((side_x,side_y), warp_interp)


            if use_background_mask and apply_mask_after_warp:
              # if turbo_mode & (frame_num % int(turbo_steps) != 0):
              #   print('disabling mask for turbo step, will be applied during turbo blend')
              #   return warped
              if VERBOSE: print('creating bg mask for frame ', frame_num)
              if warp_mode == 'use_latent':
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
              else:
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
              # warped.save(f'warped_{frame_num}.jpg')

            return warped

from tqdm.notebook import trange
import copy

def get_frame_from_color_mode(mode, offset, frame_num):
                      if mode == 'color_video':
                        if VERBOSE:print(f'the color video frame number {offset}.')
                        filename = f'{colorVideoFramesFolder}/{offset+1:06}.jpg'
                      if mode == 'color_video_offset':
                        if VERBOSE:print(f'the color video frame with offset {offset}.')
                        filename = f'{colorVideoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'stylized_frame_offset':
                        if VERBOSE:print(f'the stylized frame with offset {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-offset:06}.png'
                      if mode == 'stylized_frame':
                        if VERBOSE:print(f'the stylized frame number {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png'
                        if not os.path.exists(filename):
                          filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{args.start_frame+offset:06}.png'
                      if mode == 'init_frame_offset':
                        if VERBOSE:print(f'the raw init frame with offset {offset}.')
                        filename = f'{videoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'init_frame':
                        if VERBOSE:print(f'the raw init frame number {offset}.')
                        filename = f'{videoFramesFolder}/{offset+1:06}.jpg'
                      return filename

def apply_mask(init_image, frame_num, background, background_source, invert_mask=False, warp_mode='use_image', ):
  global mask_clip_low, mask_clip_high
  if warp_mode == 'use_image':
    size = init_image.size
  if warp_mode == 'use_latent':
    print(init_image.shape)
    size = init_image.shape[-1], init_image.shape[-2]
    size = [o*8 for o in size]
    print('size',size)
  init_image_alpha = Image.open(f'{videoFramesAlpha}/{frame_num+1:06}.jpg').resize(size).convert('L')
  if invert_mask:
    init_image_alpha = ImageOps.invert(init_image_alpha)
  if mask_clip_high < 255 or mask_clip_low > 0:
    arr = np.array(init_image_alpha)
    if mask_clip_high < 255:
      arr = np.where(arr<mask_clip_high, arr, 255)
    if mask_clip_low > 0:
      arr = np.where(arr>mask_clip_low, arr, 0)
    init_image_alpha = Image.fromarray(arr)

  if background == 'color':
    bg = Image.new('RGB', size, background_source)
  if background == 'image':
    bg = Image.open(background_source).convert('RGB').resize(size)
  if background == 'init_video':
    bg = Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg').resize(size)
  # init_image.putalpha(init_image_alpha)
  if warp_mode == 'use_image':
    bg.paste(init_image, (0,0), init_image_alpha)
  if warp_mode == 'use_latent':
    #convert bg to latent

    bg = np.array(bg)
    bg = (bg/255.)[None,...].transpose(0, 3, 1, 2)
    bg = 2*torch.from_numpy(bg).float().cuda()-1.
    bg = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(bg))
    bg = bg.cpu().numpy()#[0].transpose(1,2,0)
    init_image_alpha = np.array(init_image_alpha)[::8,::8][None, None, ...]
    init_image_alpha = np.repeat(init_image_alpha, 4, axis = 1)/255
    print(bg.shape, init_image.shape, init_image_alpha.shape, init_image_alpha.max(), init_image_alpha.min())
    bg = init_image*init_image_alpha + bg*(1-init_image_alpha)
  return bg

def softcap(arr, thresh=0.8, q=0.95):
  cap = torch.quantile(abs(arr).float(), q)
  printf('q -----', torch.quantile(abs(arr).float(), torch.Tensor([0.25,0.5,0.75,0.9,0.95,0.99,1]).cuda()))
  cap_ratio = (1-thresh)/(cap-thresh)
  arr = torch.where(arr>thresh, thresh+(arr-thresh)*cap_ratio, arr)
  arr = torch.where(arr<-thresh, -thresh+(arr+thresh)*cap_ratio, arr)
  return arr

def do_run():
  seed = args.seed
  print(range(args.start_frame, args.max_frames))
  if args.animation_mode != "None":
    batchBar = tqdm(total=args.max_frames, desc ="Frames")

  # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
      # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
  for frame_num in range(args.start_frame, args.max_frames):
      if stop_on_next_loop:
        break

      # display.clear_output(wait=True)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        display.display(batchBar.container)
        batchBar.n = frame_num
        batchBar.update(1)
        batchBar.refresh()
        # display.display(batchBar.container)




      # Inits if not video frames
      if args.animation_mode != "Video Input Legacy":
        if args.init_image == '':
          init_image = None
        else:
          init_image = args.init_image
        init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
        # init_scale = args.init_scale
        steps = int(get_scheduled_arg(frame_num, steps_schedule))
        style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
        skip_steps = int(steps-steps*style_strength)
        # skip_steps = args.skip_steps

      if args.animation_mode == 'Video Input':
        if frame_num == args.start_frame:
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps-steps*style_strength)
            # skip_steps = args.skip_steps

            # init_scale = args.init_scale
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            # init_latent_scale = args.init_latent_scale
            init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
            init_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            if use_background_mask:
              init_image_pil = Image.open(init_image)
              init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
              init_image_pil.save(f'init_alpha_{frame_num}.png')
              init_image = f'init_alpha_{frame_num}.png'
            if (args.init_image != '') and  args.init_image is not None:
              init_image = args.init_image
              if use_background_mask:
                init_image_pil = Image.open(init_image)
                init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
                init_image_pil.save(f'init_alpha_{frame_num}.png')
                init_image = f'init_alpha_{frame_num}.png'
            if VERBOSE:print('init image', args.init_image)
        if frame_num > 0 and frame_num != frame_range[0]:
          # print(frame_num)

          first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
          if os.path.exists(first_frame_source):
              first_frame = Image.open(first_frame_source)
          else:
              first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame-1:06}.png"
              first_frame = Image.open(first_frame_source)


          # print(frame_num)

          # first_frame = Image.open(batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png")
          # first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
          if not fixed_seed:
            seed += 1
          if resume_run and frame_num == start_frame:
            print('if resume_run and frame_num == start_frame')
            img_filepath = batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:06}.png"
            if turbo_mode and frame_num > turbo_preroll:
              shutil.copyfile(img_filepath, 'oldFrameScaled.png')
            else:
              shutil.copyfile(img_filepath, 'prevFrame.png')
          else:
            # img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'
            img_filepath = 'prevFrame.png'

          next_step_pil = do_3d_step(img_filepath, frame_num,  forward_clip=forward_weights_clip)
          if warp_mode == 'use_image':
            next_step_pil.save('prevFrameScaled.png')
          else:
            # init_image = 'prevFrameScaled_lat.pt'
            # next_step_pil.save('prevFrameScaled.png')
            torch.save(next_step_pil, 'prevFrameScaled_lat.pt')

          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)
          # skip_steps = args.calc_frames_skip_steps

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if turbo_mode:
            if frame_num == turbo_preroll: #start tracking oldframe
              if warp_mode == 'use_image':
                next_step_pil.save('oldFrameScaled.png')#stash for later blending
              if warp_mode == 'use_latent':
                # lat_from_img = get_lat/_from_pil(next_step_pil)
                torch.save(next_step_pil, 'oldFrameScaled_lat.pt')
            elif frame_num > turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              if warp_mode == 'use_image':
                old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)
                old_frame.save('oldFrameScaled.png')
              if warp_mode == 'use_latent':
                old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)

                # lat_from_img = get_lat_from_pil(old_frame)
                torch.save(old_frame, 'oldFrameScaled_lat.pt')
              if frame_num % int(turbo_steps) != 0:
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                blend_factor = ((frame_num % int(turbo_steps))+1)/int(turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                if warp_mode == 'use_image':
                  newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                  oldWarpedImg = cv2.imread('oldFrameScaled.png')
                  blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                  cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                  next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                if warp_mode == 'use_latent':
                  newWarpedImg = torch.load('prevFrameScaled_lat.pt')#this is already updated..
                  oldWarpedImg = torch.load('oldFrameScaled_lat.pt')
                  blendedImage = newWarpedImg*(blend_factor)+oldWarpedImg*(1-blend_factor)
                  blendedImage = get_image_from_lat(blendedImage).save(f'{batchFolder}/{filename}')
                  torch.save(next_step_pil,f'{img_filepath[:-4]}_lat.pt')


                if turbo_frame_skips_steps is not None:
                    if warp_mode == 'use_image':
                      oldWarpedImg = cv2.imread('prevFrameScaled.png')
                      cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                    print('clip/diff this frame - generate clip diff image')
                    if warp_mode == 'use_latent':
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later
                    skip_steps = math.floor(steps * turbo_frame_skips_steps)
                else: continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                if warp_mode == 'use_image':
                      oldWarpedImg = cv2.imread('prevFrameScaled.png')
                      cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                print('clip/diff this frame - generate clip diff image')
                if warp_mode == 'use_latent':
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later
                # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                # cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                print('clip/diff this frame - generate clip diff image')
          if warp_mode == 'use_image':
            init_image = 'prevFrameScaled.png'
          else:
            init_image = 'prevFrameScaled_lat.pt'
          if use_background_mask:
            if warp_mode == 'use_latent':
              # pass
              latent = apply_mask(latent.cpu(), frame_num, background, background_source, invert_mask, warp_mode)#.save(init_image)

            if warp_mode == 'use_image':
              apply_mask(Image.open(init_image), frame_num, background, background_source, invert_mask).save(init_image)
          # init_scale = args.frames_scale
          init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
          # init_latent_scale = args.frames_latent_scale
          init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)


      loss_values = []

      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True

      target_embeds, weights = [], []

      if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        # frame_prompt = args.prompts_series[-1]
        frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
      elif args.prompts_series is not None:
        # frame_prompt = args.prompts_series[frame_num]
        frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
      else:
        frame_prompt = []

      if VERBOSE:print(args.image_prompts_series)
      if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
      elif args.image_prompts_series is not None:
        image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
      else:
        image_prompt = []





      init = None



      image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          display.display(image_display)
          gc.collect()
          torch.cuda.empty_cache()
          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)


          if perlin_init:
              init = regen_perlin()

          consistency_mask = None
          if (check_consistency or (model_version == 'v1_inpainting') or ('control_sd15_inpaint' in controlnet_multimodel.keys())) and frame_num>0:
            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if reverse_cc_order:
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
            else:
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
            consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

          if diffusion_model == 'stable_diffusion':
            if VERBOSE: print(args.side_x, args.side_y, init_image)
            # init = Image.open(fetch(init_image)).convert('RGB')

            # init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            # init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
            # text_prompt = copy.copy(args.prompts_series[frame_num])
            text_prompt = copy.copy(get_sched_from_json(frame_num, args.prompts_series, blend=False))
            if VERBOSE:print(f'Frame {frame_num} Prompt: {text_prompt}')
            text_prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in text_prompt] #remove loras from prompt
            text_prompt = [re.sub(":\s*([\d.]+)\s*$", '', o).strip(' ') for o in text_prompt] #remove weights from prompt
            used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, new_prompt_loras)
            frame_prompt_weights = get_sched_from_json(frame_num, prompt_weights, blend=blend_json_schedules)

            if VERBOSE:
              print('used_loras, used_loras_weights', used_loras, used_loras_weights)
              print('prompt weights, frame_prompt_weights', prompt_weights , frame_prompt_weights)
            # used_loras_weights = [o for o in used_loras_weights if o is not None else 0.]
            if use_lycoris:
              load_lycos(used_loras, used_loras_weights, used_loras_weights)
            else:
              load_loras(used_loras, used_loras_weights)
            caption = get_caption(frame_num)
            if caption:
              # print('args.prompt_series',args.prompts_series[frame_num])
              for i in range(len(text_prompt)):
                if '{caption}' in text_prompt[i]:
                  print('Replacing ', '{caption}', 'with ', caption)
                  text_prompt[0] = text_prompt[i].replace('{caption}', caption)
            prompt_patterns = get_sched_from_json(frame_num, prompt_patterns_sched, blend=False)
            if prompt_patterns:
              for key in prompt_patterns.keys():
                for i in range(len(text_prompt)):
                  if key in text_prompt[i]:
                    print('Replacing ', key, 'with ', prompt_patterns[key])
                    text_prompt[i] = text_prompt[i].replace(key, prompt_patterns[key])

            if args.neg_prompts_series is not None:
              neg_prompt = get_sched_from_json(frame_num, args.neg_prompts_series, blend=False)
            else:
              neg_prompt = copy.copy(text_prompt)

            if VERBOSE:print(f'Frame {frame_num} neg_prompt: {neg_prompt}')
            if args.rec_prompts_series is not None:
              rec_prompt = copy.copy(get_sched_from_json(frame_num, args.rec_prompts_series, blend=False))
              if caption and '{caption}' in rec_prompt[0]:
                  print('Replacing ', '{caption}', 'with ', caption)
                  rec_prompt[0] = rec_prompt[0].replace('{caption}', caption)
            else:
              rec_prompt = copy.copy(text_prompt)
            if VERBOSE:print(f'Frame {rec_prompt} rec_prompt: {rec_prompt}')

            if VERBOSE:
              print(neg_prompt, 'neg_prompt')
              print('init_scale pre sd run', init_scale)
            # init_latent_scale = args.init_latent_scale
            # if frame_num>0:
            #   init_latent_scale = args.frames_latent_scale
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps-steps*style_strength)
            cfg_scale = get_scheduled_arg(frame_num, cfg_scale_schedule)
            image_scale = get_scheduled_arg(frame_num, image_scale_schedule)
            if VERBOSE:printf('skip_steps b4 run_sd: ', skip_steps)

            deflicker_src = {
                'processed1':f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png',
                'raw1': f'{videoFramesFolder}/{frame_num:06}.jpg',
                'raw2': f'{videoFramesFolder}/{frame_num+1:06}.jpg',
            }

            init_grad_img = None
            if init_grad: init_grad_img = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            #setup depth source
            if cond_image_src == 'init':
              cond_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            if cond_image_src == 'stylized':
              cond_image = init_image
            if cond_image_src == 'cond_video':
              cond_image = f'{condVideoFramesFolder}/{frame_num+1:06}.jpg'

            ref_image = None
            if reference_source == 'init':
              ref_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            if reference_source == 'stylized':
              ref_image = init_image
            if reference_source == 'prev_frame':
                ref_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png'
            if reference_source == 'color_video':
                if os.path.exists(f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'):
                  ref_image = f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'
                elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                  ref_image = f'{colorVideoFramesFolder}/{1:06}.jpg'
                else:
                  raise Exception("Reference mode specified with no color video or image. Please specify color video or disable the shuffle model")


            #setup shuffle
            shuffle_source = None
            if 'control_sd15_shuffle' in controlnet_multimodel.keys():
              if control_sd15_shuffle_source == 'color_video':
                if os.path.exists(f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'):
                  shuffle_source = f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'
                elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                  shuffle_source = f'{colorVideoFramesFolder}/{1:06}.jpg'
                else:
                  raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
              elif control_sd15_shuffle_source == 'init':
                shuffle_source = init_image
              elif control_sd15_shuffle_source == 'first_frame':
                shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{0:06}.png'
              elif control_sd15_shuffle_source == 'prev_frame':
                shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png'
              if not os.path.exists(shuffle_source):
                  if control_sd15_shuffle_1st_source == 'init':
                    shuffle_source = init_image
                  elif control_sd15_shuffle_1st_source == None:
                    shuffle_source = None
                  elif  control_sd15_shuffle_1st_source == 'color_video':
                    if os.path.exists(f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'):
                      shuffle_source = f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'
                    elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                      shuffle_source = f'{colorVideoFramesFolder}/{1:06}.jpg'
                    else:
                      raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
              print('Shuffle source ',shuffle_source)



            #setup temporal source
            if temporalnet_source =='init':
              prev_frame = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if temporalnet_source == 'stylized':
              prev_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png'
            if temporalnet_source == 'cond_video':
              prev_frame = f'{condVideoFramesFolder}/{frame_num:06}.jpg'
            if not os.path.exists(prev_frame):
              if temporalnet_skip_1st_frame:
                print('prev_frame not found, replacing 1st videoframe init')
                prev_frame = None
              else:
                prev_frame = f'{videoFramesFolder}/{frame_num+1:06}.jpg'

            #setup rec noise source
            if rec_source == 'stylized':
              rec_frame = init_image
            elif rec_source == 'init':
              rec_frame = f'{videoFramesFolder}/{frame_num+1:06}.jpg'


            #setup masks for inpainting model
            if model_version == 'v1_inpainting':
              if inpainting_mask_source == 'consistency_mask':
                cond_image = consistency_mask
              if inpainting_mask_source in ['none', None,'', 'None', 'off']:
                cond_image = None
              if inpainting_mask_source == 'cond_video': cond_image = f'{condVideoFramesFolder}/{frame_num+1:06}.jpg'
              # print('cond_image0',cond_image)

            #setup masks for controlnet inpainting model
            control_inpainting_mask = None
            if 'control_sd15_inpaint' in controlnet_multimodel.keys():
              if control_sd15_inpaint_mask_source == 'consistency_mask':
                  control_inpainting_mask = consistency_mask
              if control_sd15_inpaint_mask_source in ['none', None,'', 'None', 'off']:
                  # control_inpainting_mask = None
                  control_inpainting_mask = np.ones((args.side_y,args.side_x,3))
              if control_sd15_inpaint_mask_source == 'cond_video':
                control_inpainting_mask = f'{condVideoFramesFolder}/{frame_num+1:06}.jpg'
                control_inpainting_mask = np.array(PIL.Image.open(control_inpainting_mask))
                # print('cond_image0',cond_image)

            np_alpha = None
            if alpha_masked_diffusion and frame_num>args.start_frame:
              if VERBOSE: print('Using alpha masked diffusion')
              print(f'{videoFramesAlpha}/{frame_num+1:06}.jpg')
              if videoFramesAlpha == videoFramesFolder or not os.path.exists(f'{videoFramesAlpha}/{frame_num+1:06}.jpg'):
                raise Exception('You have enabled alpha_masked_diffusion without providing an alpha mask source. Please go to mask cell and specify a masked video init or extract a mask from init video.')

              init_image_alpha = Image.open(f'{videoFramesAlpha}/{frame_num+1:06}.jpg').resize((args.side_x,args.side_y)).convert('L')
              np_alpha = np.array(init_image_alpha)/255.

            mask_current_frame_many = None
            if mask_frames_many is not None:
              mask_current_frame_many = [torch.from_numpy(np.array(PIL.Image.open(o[frame_num]).resize((args.side_x,args.side_y)).convert('L'))/255.)[None,...].float() for o in mask_frames_many]
              mask_current_frame_many.insert(0, torch.ones_like(mask_current_frame_many[0]))
              assert len(mask_current_frame_many) == len(text_prompt), 'mask number doesn`t match prompt number'

              mask_current_frame_many = torch.stack(mask_current_frame_many).repeat((1,4,1,1))
              # mask_current_frame_many = torch.where(mask_current_frame_many>0.5, 1., 0.).float()

            # try:
            sample, latent, depth_img = run_sd(args, init_image=init_image, skip_timesteps=skip_steps, H=args.side_y,
                             W=args.side_x, text_prompt=text_prompt, neg_prompt=neg_prompt, steps=steps,
                             seed=seed, init_scale = init_scale, init_latent_scale=init_latent_scale, cond_image=cond_image,
                             cfg_scale=cfg_scale, image_scale = image_scale, cond_fn=None,
                             init_grad_img=init_grad_img, consistency_mask=consistency_mask,
                             frame_num=frame_num, deflicker_src=deflicker_src, prev_frame=prev_frame,
                             rec_prompt=rec_prompt, rec_frame=rec_frame,control_inpainting_mask=control_inpainting_mask, shuffle_source=shuffle_source,
                             ref_image=ref_image, alpha_mask=np_alpha, prompt_weights=frame_prompt_weights, mask_current_frame_many=mask_current_frame_many)
            # except:
            #   traceback.print_exc()
            #   sys.exit()

            settings_json = save_settings(skip_save=True)
            settings_exif = json2exif(settings_json)



            # depth_img.save(f'{root_dir}/depth_{frame_num}.png')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            # if warp_mode == 'use_raw':torch.save(sample,f'{batchFolder}/{filename[:-4]}_raw.pt')
            if warp_mode == 'use_latent':
              torch.save(latent,f'{batchFolder}/{filename[:-4]}_lat.pt')
            samples = sample*(steps-skip_steps)
            samples = [{"pred_xstart": sample} for sample in samples]
            # for j, sample in enumerate(samples):
              # print(j, sample["pred_xstart"].size)
            # raise Exception
            if VERBOSE: print(sample[0][0].shape)
            image = sample[0][0]
            if do_softcap:
              image = softcap(image, thresh=softcap_thresh, q=softcap_q)
            image = image.add(1).div(2).clamp(0, 1)
            image = TF.to_pil_image(image)
            if warp_towards_init != 'off' and frame_num!=0:
              if warp_towards_init == 'init':
                warp_init_filename = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
              else:
                warp_init_filename = init_image
              print('warping towards init')
              init_pil = Image.open(warp_init_filename)
              image = warp_towards_init_fn(image, init_pil)

            display.clear_output(wait=True)
            fit(image, display_size).save('progress.png', exif=settings_exif)
            display.display(display.Image('progress.png'))

            if mask_result and check_consistency and frame_num>0:

                        if VERBOSE:print('imitating inpaint')
                        frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                        weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                        consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

                        consistency_mask = cv2.GaussianBlur(consistency_mask,
                                                (diffuse_inpaint_mask_blur,diffuse_inpaint_mask_blur),cv2.BORDER_DEFAULT)
                        if diffuse_inpaint_mask_thresh<1:
                          consistency_mask = np.where(consistency_mask<diffuse_inpaint_mask_thresh, 0, 1.)
                        # if dither:
                        #   consistency_mask = Dither.dither(consistency_mask, 'simple2D', resize=True)

                        # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                        if warp_mode == 'use_image':
                          consistency_mask = cv2.GaussianBlur(consistency_mask,
                                                (3,3),cv2.BORDER_DEFAULT)
                          init_img_prev = Image.open(init_image)
                          if VERBOSE:print(init_img_prev.size, consistency_mask.shape, image.size)
                          cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                          image_masked = np.array(image)*(1-consistency_mask) + np.array(init_img_prev)*(consistency_mask)

                          # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                          image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                          # image = image_masked.resize(image.size, warp_interp)
                          image = image_masked
                        if warp_mode == 'use_latent':
                          if invert_mask: consistency_mask = 1-consistency_mask
                          init_lat_prev = torch.load('prevFrameScaled_lat.pt')
                          sample_masked = sd_model.decode_first_stage(latent.cuda())[0]
                          image_prev = TF.to_pil_image(sample_masked.add(1).div(2).clamp(0, 1))


                          cc_small = consistency_mask[::8,::8,0]
                          latent = latent.cpu()*(1-cc_small)+init_lat_prev*cc_small
                          torch.save(latent, 'prevFrameScaled_lat.pt')

                          # image_prev = Image.open(f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png')
                          torch.save(latent, 'prevFrame_lat.pt')
                          # cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                          # image_prev = Image.open('prevFrameScaled.png')
                          image_masked = np.array(image)*(1-consistency_mask) + np.array(image_prev)*(consistency_mask)

                          # # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                          image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                          # image = image_masked.resize(image.size, warp_interp)
                          image = image_masked

            if (frame_num > args.start_frame) or ('color_video' in normalize_latent):
                global first_latent
                global first_latent_source

                if 'frame' in normalize_latent:
                  def img2latent(img_path):
                    frame2 = Image.open(img_path)
                    frame2pil = frame2.convert('RGB').resize(image.size,warp_interp)
                    frame2pil = np.array(frame2pil)
                    frame2pil = (frame2pil/255.)[None,...].transpose(0, 3, 1, 2)
                    frame2pil = 2*torch.from_numpy(frame2pil).float().cuda()-1.
                    frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
                    return frame2pil

                  try:
                    if VERBOSE:print('Matching latent to:')
                    filename = get_frame_from_color_mode(normalize_latent, normalize_latent_offset, frame_num)
                    match_latent = img2latent(filename)
                    first_latent = match_latent
                    first_latent_source = filename
                    # print(first_latent_source, first_latent)
                  except:
                    if VERBOSE:print(traceback.format_exc())
                    print(f'Frame with offset/position {normalize_latent_offset} not found')
                    if 'init' in normalize_latent:
                      try:
                        filename = f'{videoFramesFolder}/{0:06}.jpg'
                        match_latent = img2latent(filename)
                        first_latent = match_latent
                        first_latent_source = filename
                      except: pass
                    print(f'Color matching the 1st frame.')

                if colormatch_frame != 'off' and colormatch_after:
                  if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                    try:
                      print('Matching color to:')
                      filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
                      match_frame = Image.open(filename)
                      first_frame = match_frame
                      first_frame_source = filename

                    except:
                      print(f'Frame with offset/position {colormatch_offset} not found')
                      if 'init' in colormatch_frame:
                        try:
                          filename = f'{videoFramesFolder}/{1:06}.jpg'
                          match_frame = Image.open(filename)
                          first_frame = match_frame
                          first_frame_source = filename
                        except: pass
                      print(f'Color matching the 1st frame.')
                    print('Colormatch source - ', first_frame_source)
                    image = Image.fromarray(match_color_var(first_frame,
                        image, opacity=color_match_frame_str, f=colormatch_method_fn,
                        regrain=colormatch_regrain))

            if frame_num == args.start_frame:
              settings_json = save_settings()
            if args.animation_mode != "None":
                          # sys.exit(os.getcwd(), 'cwd')
              if warp_mode == 'use_image':
                image.save('prevFrame.png', exif=settings_exif)
              else:
                torch.save(latent, 'prevFrame_lat.pt')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            image.save(f'{batchFolder}/{filename}', exif=settings_exif)
            # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
            if args.animation_mode == 'Video Input':
                          # If turbo, save a blended image
                          if turbo_mode and frame_num > args.start_frame:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(turbo_steps)
                            if warp_mode == 'use_image':
                              newFrame = cv2.imread('prevFrame.png') # This is already updated..
                              prev_frame_warped = cv2.imread('prevFrameScaled.png')
                              blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                              cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                            if warp_mode == 'use_latent':
                              newFrame = torch.load('prevFrame_lat.pt').cuda()
                              prev_frame_warped = torch.load('prevFrameScaled_lat.pt').cuda()
                              blendedImage = newFrame*(blend_factor)+prev_frame_warped*(1-blend_factor)
                              blendedImage = get_image_from_lat(blendedImage)
                              blendedImage.save(f'{batchFolder}/{filename}', exif=settings_exif)

            else:
              image.save(f'{batchFolder}/{filename}', exif=settings_exif)
              image.save('prevFrameScaled.png', exif=settings_exif)

          plt.plot(np.array(loss_values), 'r')
  batchBar.close()

def save_settings(skip_save=False):
  settings_out = batchFolder+f"/settings"
  os.makedirs(settings_out, exist_ok=True)
  setting_list = {
    'text_prompts': text_prompts,
    'user_comment':user_comment,
    'image_prompts': image_prompts,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'max_frames': max_frames,
    'interp_spline': interp_spline,
    'init_image': init_image,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'seed': seed,
    'width': width_height[0],
    'height': width_height[1],
    'diffusion_model': diffusion_model,
    'diffusion_steps': diffusion_steps,
    'max_frames': max_frames,
    'video_init_path':video_init_path,
    'extract_nth_frame':extract_nth_frame,
    'flow_video_init_path':flow_video_init_path,
    'flow_extract_nth_frame':flow_extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'turbo_mode':turbo_mode,
    'turbo_steps':turbo_steps,
    'turbo_preroll':turbo_preroll,
    'flow_warp':flow_warp,
    'check_consistency':check_consistency,
    'turbo_frame_skips_steps' : turbo_frame_skips_steps,
    'forward_weights_clip' : forward_weights_clip,
    'forward_weights_clip_turbo_step' : forward_weights_clip_turbo_step,
    'padding_ratio':padding_ratio,
    'padding_mode':padding_mode,
    'consistency_blur':consistency_blur,
    'inpaint_blend':inpaint_blend,
    'match_color_strength':match_color_strength,
    'high_brightness_threshold':high_brightness_threshold,
    'high_brightness_adjust_ratio':high_brightness_adjust_ratio,
    'low_brightness_threshold':low_brightness_threshold,
    'low_brightness_adjust_ratio':low_brightness_adjust_ratio,
    'stop_early': stop_early,
    'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
    'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
    'max_brightness_threshold':max_brightness_threshold,
    'min_brightness_threshold':min_brightness_threshold,
    'enable_adjust_brightness':enable_adjust_brightness,
    'dynamic_thresh':dynamic_thresh,
    'warp_interp':warp_interp,
    'fixed_code':fixed_code,
    'code_randomness':code_randomness,
    # 'normalize_code': normalize_code,
    'mask_result':mask_result,
    'reverse_cc_order':reverse_cc_order,
    'flow_lq':flow_lq,
    'use_predicted_noise':use_predicted_noise,
    'clip_guidance_scale':clip_guidance_scale,
    'clip_type':clip_type,
    'clip_pretrain':clip_pretrain,
    'missed_consistency_weight':missed_consistency_weight,
    'overshoot_consistency_weight':overshoot_consistency_weight,
    'edges_consistency_weight':edges_consistency_weight,
    'style_strength_schedule':style_strength_schedule_bkup,
    'flow_blend_schedule':flow_blend_schedule_bkup,
    'steps_schedule':steps_schedule_bkup,
    'init_scale_schedule':init_scale_schedule_bkup,
    'latent_scale_schedule':latent_scale_schedule_bkup,
    'latent_scale_template': latent_scale_template,
    'init_scale_template':init_scale_template,
    'steps_template':steps_template,
    'style_strength_template':style_strength_template,
    'flow_blend_template':flow_blend_template,
    'make_schedules':make_schedules,
    'normalize_latent':normalize_latent,
    'normalize_latent_offset':normalize_latent_offset,
    'colormatch_frame':colormatch_frame,
    'use_karras_noise':use_karras_noise,
    'end_karras_ramp_early':end_karras_ramp_early,
    'use_background_mask':use_background_mask,
    'apply_mask_after_warp':apply_mask_after_warp,
    'background':background,
    'background_source':background_source,
    'mask_source':mask_source,
    'extract_background_mask':extract_background_mask,
    'mask_video_path':mask_video_path,
    'negative_prompts':negative_prompts,
    'invert_mask':invert_mask,
    'warp_strength': warp_strength,
    'flow_override_map':flow_override_map,
    'cfg_scale_schedule':cfg_scale_schedule_bkup,
    'respect_sched':respect_sched,
    'color_match_frame_str':color_match_frame_str,
    'colormatch_offset':colormatch_offset,
    'latent_fixed_mean':latent_fixed_mean,
    'latent_fixed_std':latent_fixed_std,
    'colormatch_method':colormatch_method,
    'colormatch_regrain':colormatch_regrain,
    'warp_mode':warp_mode,
    'use_patchmatch_inpaiting':use_patchmatch_inpaiting,
    'blend_latent_to_init':blend_latent_to_init,
    'warp_towards_init':warp_towards_init,
    'init_grad':init_grad,
    'grad_denoised':grad_denoised,
    'colormatch_after':colormatch_after,
    'colormatch_turbo':colormatch_turbo,
    'model_version':model_version,
    'cond_image_src':cond_image_src,
    'warp_num_k':warp_num_k,
    'warp_forward':warp_forward,
    'sampler':sampler.__name__,
    'mask_clip':(mask_clip_low, mask_clip_high),
    'inpainting_mask_weight':inpainting_mask_weight ,
    'inverse_inpainting_mask':inverse_inpainting_mask,
    'mask_source':mask_source,
    'model_path':model_path,
    'diff_override':diff_override,
    'image_scale_schedule':image_scale_schedule_bkup,
    'image_scale_template':image_scale_template,
    'frame_range': frame_range,
    'detect_resolution' :detect_resolution,
    'bg_threshold':bg_threshold,
    'diffuse_inpaint_mask_blur':diffuse_inpaint_mask_blur,
    'diffuse_inpaint_mask_thresh':diffuse_inpaint_mask_thresh,
    'add_noise_to_latent':add_noise_to_latent,
    'noise_upscale_ratio':noise_upscale_ratio,
    'fixed_seed':fixed_seed,
    'init_latent_fn':init_latent_fn.__name__,
    'value_threshold':value_threshold,
    'distance_threshold':distance_threshold,
    'masked_guidance':masked_guidance,
    'cc_masked_diffusion':cc_masked_diffusion,
    'alpha_masked_diffusion':alpha_masked_diffusion,
    'inverse_mask_order':inverse_mask_order,
    'invert_alpha_masked_diffusion':invert_alpha_masked_diffusion,
    'quantize':quantize,
    'cb_noise_upscale_ratio':cb_noise_upscale_ratio,
    'cb_add_noise_to_latent':cb_add_noise_to_latent,
    'cb_use_start_code':cb_use_start_code,
    'cb_fixed_code':cb_fixed_code,
    'cb_norm_latent':cb_norm_latent,
    'guidance_use_start_code':guidance_use_start_code,
    'offload_model':offload_model,
    'controlnet_preprocess':controlnet_preprocess,
    'small_controlnet_model_path':small_controlnet_model_path,
    'use_scale':use_scale,
    'g_invert_mask':g_invert_mask,
    'controlnet_multimodel':json.dumps(controlnet_multimodel),
    'img_zero_uncond':img_zero_uncond,
    'do_softcap':do_softcap,
    'softcap_thresh':softcap_thresh,
    'softcap_q':softcap_q,
    'deflicker_latent_scale':deflicker_latent_scale,
    'deflicker_scale':deflicker_scale,
    'controlnet_multimodel_mode':controlnet_multimodel_mode,
    'no_half_vae':no_half_vae,
    'temporalnet_source':temporalnet_source,
    'temporalnet_skip_1st_frame':temporalnet_skip_1st_frame,
    'rec_randomness':rec_randomness,
    'rec_source':rec_source,
    'rec_cfg':rec_cfg,
    'rec_prompts':rec_prompts,
    'inpainting_mask_source':inpainting_mask_source,
    'rec_steps_pct':rec_steps_pct,
    'max_faces': max_faces,
    'num_flow_updates':num_flow_updates,
    'control_sd15_openpose_hands_face':control_sd15_openpose_hands_face,
    'control_sd15_depth_detector':control_sd15_depth_detector,
    'control_sd15_softedge_detector':control_sd15_softedge_detector,
    'control_sd15_seg_detector':control_sd15_seg_detector,
    'control_sd15_scribble_detector':control_sd15_scribble_detector,
    'control_sd15_lineart_coarse':control_sd15_lineart_coarse,
    'control_sd15_inpaint_mask_source':control_sd15_inpaint_mask_source,
    'control_sd15_shuffle_source':control_sd15_shuffle_source,
    'control_sd15_shuffle_1st_source':control_sd15_shuffle_1st_source,
    'overwrite_rec_noise':overwrite_rec_noise,
    'use_legacy_cc':use_legacy_cc,
    'missed_consistency_dilation':missed_consistency_dilation,
    'edge_consistency_width':edge_consistency_width,
    'use_reference':use_reference,
    'reference_weight':reference_weight,
    'reference_source':reference_source,
    'reference_mode':reference_mode,
    'use_legacy_fixed_code':use_legacy_fixed_code,
    'consistency_dilate':consistency_dilate,
    'prompt_patterns_sched':prompt_patterns_sched,
    'sd_batch_size':sd_batch_size,
    'normalize_prompt_weights':normalize_prompt_weights,
    'controlnet_low_vram':controlnet_low_vram,
    'mask_paths':mask_paths

  }
  if not skip_save:
    try:
      settings_fname = f"{settings_out}/{batch_name}({batchNum})_settings.txt"
      if os.path.exists(settings_fname):
        s_meta = os.path.getmtime(settings_fname)
        os.rename(settings_fname,settings_fname[:-4]+str(s_meta)+'.txt' )
      with open(settings_fname, "w+") as f:   #save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
      print(e)
      print('Settings:', setting_list)
  return setting_list

#@title 1.6 init main sd run function, cond_fn, color matching for SD
init_latent = None
target_embed = None


import hashlib
import os
# import datetime

# (c) Alex Spirin 2023
# We use input file hashes to automate video extraction
#
def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash

def get_frame_from_path_start_end_nth(video_path:str , num_frame:int, start:int=0, end:int=0, nth:int=1) -> Image:
  assert os.path.exists(video_path), f"Video path or frame folder not found at {video_path}. Please specify the correct path."
  num_frame = max(0, num_frame)
  start = max(0, start)
  nth = max(1,nth)
  if os.path.isdir(video_path):
    frame_list = []
    image_extensions = ['jpg','png','tiff','jpeg','JPEG','bmp']
    for image_extension in image_extensions:
      flist = glob.glob(os.path.join(video_path, f'*.{image_extension}'))
      if len(flist)>0:
        frame_list = flist
        break
    assert len(frame_list) != 0, f'No frames with {", ".join(image_extensions)} extensions found in folder {video_path}. Please specify the correct path.'
    if end == 0: end = len(frame_list)
    frame_list = frame_list[start:end:nth]
    num_frame = min(num_frame, len(frame_list))
    return PIL.Image.open(frame_list[num_frame])

  elif os.path.isfile(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
      video.release()
      raise Exception(f"Error opening video file {video_path}. Please specify the correct path.")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end == 0: end = total_frames
    num_frame = min(num_frame, total_frames)
    frame_range = list(range(start,end,nth))
    frame_number = frame_range[num_frame]
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    if not ret:
        video.release()
        raise Exception(f"Error reading frame {frame_number} from file {video_path}.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    video.release()
    return image


import PIL
try:
  import Image
except:
  from PIL import Image

mask_result = False
early_stop = 0
inpainting_stop = 0
warp_interp = Image.BILINEAR

#init SD
from glob import glob
import argparse, os, sys
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

os.chdir(f"{root_dir}/stablediffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
os.chdir(f"{root_dir}")



def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

from kornia import augmentation as KA
aug = KA.RandomAffine(0, (1/14, 1/14), p=1, padding_mode='border')
from torch.nn import functional as F

from torch.cuda.amp import GradScaler

def sd_cond_fn(x, t, denoised, init_image_sd, init_latent, init_scale,
               init_latent_scale, target_embed, consistency_mask, guidance_start_code=None,
               deflicker_fn=None, deflicker_lat_fn=None, deflicker_src=None, fft_fn=None, fft_latent_fn=None,
               **kwargs):
  if use_scale: scaler = GradScaler()
  with torch.cuda.amp.autocast():
        # print('denoised.shape')
        # print(denoised.shape)
        global add_noise_to_latent

            # init_latent_scale,  init_scale, clip_guidance_scale, target_embed, init_latent, clamp_grad, clamp_max,
            # **kwargs):
        # global init_latent_scale
        # global init_scale
        global clip_guidance_scale
        # global target_embed
        # print(target_embed.shape)
        global clamp_grad
        global clamp_max
        loss = 0.
        if grad_denoised:
          x = denoised
          # denoised = x

          # print('grad denoised')
        grad = torch.zeros_like(x)

        processed1 = deflicker_src['processed1']
        if add_noise_to_latent:
          if t != 0:
            if guidance_use_start_code and guidance_start_code is not None:
              noise = guidance_start_code
            else:
              noise = torch.randn_like(x)
            noise = noise * t
            if noise_upscale_ratio > 1:
              noise = noise[::noise_upscale_ratio,::noise_upscale_ratio,:]
              noise = torch.nn.functional.interpolate(noise, x.shape[2:],
                                                    mode='bilinear')
            init_latent = init_latent + noise
            if deflicker_lat_fn:
              processed1 = deflicker_src['processed1'] + noise




        if sat_scale>0 or init_scale>0 or clip_guidance_scale>0 or deflicker_scale>0 or fft_scale>0:
          with torch.autocast('cuda'):
            denoised_small = denoised[:,:,::2,::2]
            denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised_small)

        if clip_guidance_scale>0:
          #compare text clip embeds with denoised image embeds
          # denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised);# print(denoised.requires_grad)
          # print('d b',denoised.std(), denoised.mean())
          denoised_img = denoised_img[0].add(1).div(2)
          denoised_img = normalize(denoised_img)
          denoised_t = denoised_img.cuda()[None,...]
          # print('d a',denoised_t.std(), denoised_t.mean())
          image_embed = get_image_embed(denoised_t)

          # image_embed = get_image_embed(denoised.add(1).div(2))
          loss = spherical_dist_loss(image_embed, target_embed).sum() * clip_guidance_scale

        if masked_guidance:
          if consistency_mask is None:
            consistency_mask = torch.ones_like(denoised)
          # consistency_mask = consistency_mask.permute(2,0,1)[None,...]
          # print('consistency_mask.shape, denoised.shape')
          # print(consistency_mask.shape, denoised.shape)

          consistency_mask = torch.nn.functional.interpolate(consistency_mask, denoised.shape[2:],
                                                    mode='bilinear')
          if g_invert_mask: consistency_mask = 1-consistency_mask

        if init_latent_scale>0:

          #compare init image latent with denoised latent
          # print(denoised.shape, init_latent.shape)

          loss += init_latent_fn(denoised, init_latent).sum() * init_latent_scale

        if fft_scale>0  and fft_fn is not None:
          loss += fft_fn(image1=denoised_img).sum() * fft_scale

        if fft_latent_scale>0  and fft_latent_fn is not None:
          loss += fft_latent_fn(image1=denoised).sum() * fft_latent_scale

        if sat_scale>0:
          loss += torch.abs(denoised_img - denoised_img.clamp(min=-1,max=1)).mean()

        if init_scale>0:
          #compare init image with denoised latent image via lpips
          # print('init_image_sd', init_image_sd)

          loss += lpips_model(denoised_img, init_image_sd[:,:,::2,::2]).sum() * init_scale

        if deflicker_scale>0 and deflicker_fn is not None:
          # print('deflicker_fn(denoised_img).sum() * deflicker_scale',deflicker_fn(denoised_img).sum() * deflicker_scale)
          loss += deflicker_fn(processed2=denoised_img).sum() * deflicker_scale
          print('deflicker ', loss)

        if deflicker_latent_scale>0 and deflicker_lat_fn is not None:

          loss += deflicker_lat_fn(processed2=denoised, processed1=processed1).sum() * deflicker_latent_scale
          print('deflicker lat', loss)




  # print('loss', loss)
  if loss!=0. :
          if use_scale:
            scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                  inputs=x)
            inv_scale = 1./scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]
            grad = -grad_params[0]
            # scaler.update()
          else:
            grad = -torch.autograd.grad(loss, x)[0]
          if masked_guidance:
            grad = grad*consistency_mask
          if torch.isnan(grad).any():
              print('got NaN grad')
              return torch.zeros_like(x)
          if VERBOSE:printf('loss, grad',loss, grad.max(), grad.mean(), grad.std(), denoised.mean(), denoised.std())
          if clamp_grad:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude

  return grad

import cv2

# %cd "{root_dir}/python-color-transfer"
from python_color_transfer.color_transfer import ColorTransfer, Regrain
# %cd "{root_path}/"

PT = ColorTransfer()

def match_color_var(stylized_img, raw_img, opacity=1., f=PT.pdf_transfer, regrain=False):
  img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
  img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
  img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC )

  # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
  img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
  if regrain: img_arr_col = RG.regrain     (img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
  img_arr_col = img_arr_col*opacity+img_arr_in*(1-opacity)
  img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'),cv2.COLOR_BGR2RGB)

  return img_arr_reg

#https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/ed0bed6abaf75c0f1b270cf6996de3e07cbafc81/find_noise.py

import torch
import numpy as np
# import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat

def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    if half:
        image = image
    return (2.0 * image - 1.0).unsqueeze(0)

def pil_img_to_latent(model, img, batch_size=1, device='cuda', half=True):
    init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))

import torch
from ldm.modules.midas.api import load_midas_transform
midas_tfm = load_midas_transform("dpt_hybrid")

def midas_tfm_fn(x):
  x = x = ((x + 1.0) * .5).detach().cpu().numpy()
  return midas_tfm({"image": x})["image"]

def pil2midas(pil_image):
  image = np.array(pil_image.convert("RGB"))
  image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
  image = midas_tfm_fn(image)
  return torch.from_numpy(image[None, ...]).float()

def make_depth_cond(pil_image, x):
          global frame_num
          pil_image = Image.open(pil_image).convert('RGB')
          c_cat = list()
          cc = pil2midas(pil_image).cuda()
          cc = sd_model.depth_model(cc)
          depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)
          display_depth = (cc - depth_min) / (depth_max - depth_min)
          depth_image = Image.fromarray(
                  (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
          display_depth = (cc - depth_min) / (depth_max - depth_min)
          depth_image = Image.fromarray(
                              (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
          if cc.shape[2:]!=x.shape[2:]:
            cc = torch.nn.functional.interpolate(
                    cc,
                    size=x.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
          depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)


          cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
          c_cat.append(cc)
          c_cat = torch.cat(c_cat, dim=1)
          # cond
          # cond = {"c_concat": [c_cat], "c_crossattn": [c]}

          # # uncond cond
          # uc_full = {"c_concat": [c_cat], "c_crossattn": [uc]}
          return c_cat, depth_image

def find_noise_for_image(model, x, prompt, steps, cond_scale=0.0, verbose=False, normalize=True):

    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
            print(x.shape)
            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                return x

# Based on changes suggested by briansemrau in https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/736
import hashlib
def find_noise_for_image_sigma_adjustment(init_latent, prompt, image_conditioning, cfg_scale, steps, frame_num):
    rec_noise_setting_list = {
    'init_image': init_image,
    'seed': seed,
    'width': width_height[0],
    'height': width_height[1],
    'diffusion_model': diffusion_model,
    'diffusion_steps': diffusion_steps,
    'video_init_path':video_init_path,
    'extract_nth_frame':extract_nth_frame,
    'flow_video_init_path':flow_video_init_path,
    'flow_extract_nth_frame':flow_extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'turbo_mode':turbo_mode,
    'turbo_steps':turbo_steps,
    'turbo_preroll':turbo_preroll,
    'flow_warp':flow_warp,
    'check_consistency':check_consistency,
    'turbo_frame_skips_steps' : turbo_frame_skips_steps,
    'forward_weights_clip' : forward_weights_clip,
    'forward_weights_clip_turbo_step' : forward_weights_clip_turbo_step,
    'padding_ratio':padding_ratio,
    'padding_mode':padding_mode,
    'consistency_blur':consistency_blur,
    'inpaint_blend':inpaint_blend,
    'match_color_strength':match_color_strength,
    'high_brightness_threshold':high_brightness_threshold,
    'high_brightness_adjust_ratio':high_brightness_adjust_ratio,
    'low_brightness_threshold':low_brightness_threshold,
    'low_brightness_adjust_ratio':low_brightness_adjust_ratio,
    'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
    'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
    'max_brightness_threshold':max_brightness_threshold,
    'min_brightness_threshold':min_brightness_threshold,
    'enable_adjust_brightness':enable_adjust_brightness,
    'dynamic_thresh':dynamic_thresh,
    'warp_interp':warp_interp,
    'reverse_cc_order':reverse_cc_order,
    'flow_lq':flow_lq,
    'use_predicted_noise':use_predicted_noise,
    'clip_guidance_scale':clip_guidance_scale,
    'clip_type':clip_type,
    'clip_pretrain':clip_pretrain,
    'missed_consistency_weight':missed_consistency_weight,
    'overshoot_consistency_weight':overshoot_consistency_weight,
    'edges_consistency_weight':edges_consistency_weight,
    'flow_blend_schedule':flow_blend_schedule,
    'steps_schedule':steps_schedule,
    'latent_scale_schedule':latent_scale_schedule,
    'flow_blend_template':flow_blend_template,
    'make_schedules':make_schedules,
    'normalize_latent':normalize_latent,
    'normalize_latent_offset':normalize_latent_offset,
    'colormatch_frame':colormatch_frame,
    'use_karras_noise':use_karras_noise,
    'end_karras_ramp_early':end_karras_ramp_early,
    'use_background_mask':use_background_mask,
    'apply_mask_after_warp':apply_mask_after_warp,
    'background':background,
    'background_source':background_source,
    'mask_source':mask_source,
    'extract_background_mask':extract_background_mask,
    'mask_video_path':mask_video_path,
    'invert_mask':invert_mask,
    'warp_strength': warp_strength,
    'flow_override_map':flow_override_map,
    'respect_sched':respect_sched,
    'color_match_frame_str':color_match_frame_str,
    'colormatch_offset':colormatch_offset,
    'latent_fixed_mean':latent_fixed_mean,
    'latent_fixed_std':latent_fixed_std,
    'colormatch_method':colormatch_method,
    'colormatch_regrain':colormatch_regrain,
    'warp_mode':warp_mode,
    'use_patchmatch_inpaiting':use_patchmatch_inpaiting,
    'blend_latent_to_init':blend_latent_to_init,
    'warp_towards_init':warp_towards_init,
    'init_grad':init_grad,
    'grad_denoised':grad_denoised,
    'colormatch_after':colormatch_after,
    'colormatch_turbo':colormatch_turbo,
    'model_version':model_version,
    'cond_image_src':cond_image_src,
    'warp_num_k':warp_num_k,
    'warp_forward':warp_forward,
    'sampler':sampler.__name__,
    'mask_clip':(mask_clip_low, mask_clip_high),
    'inpainting_mask_weight':inpainting_mask_weight ,
    'inverse_inpainting_mask':inverse_inpainting_mask,
    'mask_source':mask_source,
    'model_path':model_path,
    'diff_override':diff_override,
    'image_scale_schedule':image_scale_schedule,
    'image_scale_template':image_scale_template,
    'detect_resolution' :detect_resolution,
    'bg_threshold':bg_threshold,
    'diffuse_inpaint_mask_blur':diffuse_inpaint_mask_blur,
    'diffuse_inpaint_mask_thresh':diffuse_inpaint_mask_thresh,
    'add_noise_to_latent':add_noise_to_latent,
    'noise_upscale_ratio':noise_upscale_ratio,
    'fixed_seed':fixed_seed,
    'init_latent_fn':init_latent_fn.__name__,
    'value_threshold':value_threshold,
    'distance_threshold':distance_threshold,
    'masked_guidance':masked_guidance,
    'cc_masked_diffusion':cc_masked_diffusion,
    'alpha_masked_diffusion':alpha_masked_diffusion,
    'inverse_mask_order':inverse_mask_order,
    'invert_alpha_masked_diffusion':invert_alpha_masked_diffusion,
    'quantize':quantize,
    'cb_noise_upscale_ratio':cb_noise_upscale_ratio,
    'cb_add_noise_to_latent':cb_add_noise_to_latent,
    'cb_use_start_code':cb_use_start_code,
    'cb_fixed_code':cb_fixed_code,
    'cb_norm_latent':cb_norm_latent,
    'guidance_use_start_code':guidance_use_start_code,
    'controlnet_preprocess':controlnet_preprocess,
    'small_controlnet_model_path':small_controlnet_model_path,
    'use_scale':use_scale,
    'g_invert_mask':g_invert_mask,
    'controlnet_multimodel':json.dumps(controlnet_multimodel),
    'img_zero_uncond':img_zero_uncond,
    'do_softcap':do_softcap,
    'softcap_thresh':softcap_thresh,
    'softcap_q':softcap_q,
    'deflicker_latent_scale':deflicker_latent_scale,
    'deflicker_scale':deflicker_scale,
    'controlnet_multimodel_mode':controlnet_multimodel_mode,
    'no_half_vae':no_half_vae,
    'temporalnet_source':temporalnet_source,
    'temporalnet_skip_1st_frame':temporalnet_skip_1st_frame,
    'rec_randomness':rec_randomness,
    'rec_source':rec_source,
    'rec_cfg':rec_cfg,
    'rec_prompts':rec_prompts,
    'inpainting_mask_source':inpainting_mask_source,
    'rec_steps_pct':rec_steps_pct,
    'max_faces': max_faces,
    'num_flow_updates':num_flow_updates,
    'control_sd15_openpose_hands_face':control_sd15_openpose_hands_face,
    'control_sd15_depth_detector':control_sd15_openpose_hands_face,
    'control_sd15_softedge_detector':control_sd15_softedge_detector,
    'control_sd15_seg_detector':control_sd15_seg_detector,
    'control_sd15_scribble_detector':control_sd15_scribble_detector,
    'control_sd15_lineart_coarse':control_sd15_lineart_coarse,
    'control_sd15_inpaint_mask_source':control_sd15_inpaint_mask_source,
    'control_sd15_shuffle_source':control_sd15_shuffle_source,
    'control_sd15_shuffle_1st_source':control_sd15_shuffle_1st_source,
    'consistency_dilate':consistency_dilate
    }
    settings_hash = hashlib.sha256(json.dumps(rec_noise_setting_list).encode('utf-8')).hexdigest()[:16]
    filepath = f'{recNoiseCacheFolder}/{settings_hash}_{frame_num:06}.pt'
    if os.path.exists(filepath) and not overwrite_rec_noise:
      print(filepath)
      noise = torch.load(filepath)
      print('loading existing noise')
      return noise
    steps = int(copy.copy(steps)*rec_steps_pct)
    cond = prompt_parser.get_learned_conditioning(sd_model, prompt, steps)
    uncond = prompt_parser.get_learned_conditioning(sd_model, [''], steps)
    cfg_scale=rec_cfg
    cond = prompt_parser.reconstruct_cond_batch(cond, 0)
    uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)

    x = init_latent

    s_in = x.new_ones([x.shape[0]])
    if sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(sd_model)
        skip = 0
    sigmas = dnw.get_sigmas(steps).flip(0)



    for i in trange(1, len(sigmas)):


        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])


        # image_conditioning = torch.cat([image_conditioning] * 2)
        # cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}
        if model_version == 'control_multi' and controlnet_multimodel_mode == 'external':
          raise Exception("Predicted noise not supported for external mode. Please turn predicted noise off or use internal mode.")
        if image_conditioning is not None:
          if model_version != 'control_multi':
            if img_zero_uncond:
              img_in = torch.cat([torch.zeros_like(image_conditioning),
                                          image_conditioning])
            else:
              img_in = torch.cat([image_conditioning]*2)
            cond_in={"c_crossattn": [cond_in],'c_concat': [img_in]}

          if model_version == 'control_multi' and controlnet_multimodel_mode != 'external':
            img_in = {}
            for key in image_conditioning.keys():
                  img_in[key] = torch.cat([torch.zeros_like(image_conditioning[key]),
                                              image_conditioning[key]]) if img_zero_uncond else torch.cat([image_conditioning[key]]*2)

            cond_in = {"c_crossattn": [cond_in],  'c_concat': img_in,
                                                                      'controlnet_multimodel':controlnet_multimodel,
                                                                      'loaded_controlnets':loaded_controlnets}


        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]

        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
        else:
            t = dnw.sigma_to_t(sigma_in)

        eps = sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        if i == 1:
            d = (x - denoised) / (2 * sigmas[i])
        else:
            d = (x - denoised) / sigmas[i - 1]

        dt = sigmas[i] - sigmas[i - 1]
        x = x + d * dt



        # This shouldn't be necessary, but solved some VRAM issues
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt


    # return (x / x.std()) * sigmas[-1]
    x = x / sigmas[-1]
    torch.save(x, filepath)
    return x# / sigmas[-1]

#karras noise
#https://github.com/Birch-san/stable-diffusion/blob/693c8a336aa3453d30ce403f48eb545689a679e5/scripts/txt2img_fork.py#L62-L81
sys.path.append('./k-diffusion')

def get_premature_sigma_min(
                                    steps: int,
                                    sigma_max: float,
                                    sigma_min_nominal: float,
                                    rho: float
                                ) -> float:
                                    min_inv_rho = sigma_min_nominal ** (1 / rho)
                                    max_inv_rho = sigma_max ** (1 / rho)
                                    ramp = (steps-2) * 1/(steps-1)
                                    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                                    return sigma_min

import contextlib
none_context = contextlib.nullcontext()

def masked_callback(args, callback_steps, masks, init_latent, start_code):
  # print('callback_step', callback_step)
  # print('masks callback shape',[o.shape for o in masks])
  init_latent = init_latent.clone()
  # print(args['i'])
  masks = [m[:,0:1,...] for m in masks]
  # print(args['x'].shape)
  final_mask = None #create a combined mask for this step
  for (mask, callback_step) in zip(masks, callback_steps):

    if args['i'] <= callback_step:
      mask = torch.nn.functional.interpolate(mask, args['x'].shape[2:],
                                                      mode='bilinear')
      if final_mask is None: final_mask = mask
      else: final_mask = final_mask*mask

  mask = final_mask

  if mask is not None:
      # PIL.Image.fromarray(np.repeat(mask.clone().cpu().numpy()[0,0,...][...,None],3, axis=2).astype('uint8')*255).save(f'{root_dir}/{args["i"]}.jpg')
      if cb_use_start_code:
        noise = start_code
      else:
        noise = torch.randn_like(args['x'])
      noise = noise*args['sigma']
      if cb_noise_upscale_ratio > 1:
                noise = noise[::noise_upscale_ratio,::noise_upscale_ratio,:]
                noise = torch.nn.functional.interpolate(noise, args['x'].shape[2:],
                                                      mode='bilinear')
      # mask = torch.nn.functional.interpolate(mask, args['x'].shape[2:],
      #                                                 mode='bilinear')
      if VERBOSE: print('Applying callback at step ', args['i'])
      if cb_add_noise_to_latent:
        init_latent = init_latent+noise
      if cb_norm_latent:
                              noise = init_latent
                              noise2 = args['x']
                              n_mean = noise2.mean(dim=(2,3),keepdim=True)
                              n_std = noise2.std(dim=(2,3),keepdim=True)
                              n2_mean = noise.mean(dim=(2,3),keepdim=True)
                              noise = noise - (n2_mean-n_mean)
                              n2_std = noise.std(dim=(2,3),keepdim=True)
                              noise = noise/(n2_std/n_std)
                              init_latent = noise

      args['x'] = args['x']*(1-mask) + (init_latent)*mask #ok
    # args['x'] = args['x']*(mask) + (init_latent)*(1-mask) #test reverse
    # return args['x']

  return args['x']

import torch.nn.functional as F

def high_frequency_loss(image1, image2):
    """
    Compute the loss that penalizes high-frequency differences between images
    while ignoring low-frequency differences.

    Args:
        image1 (torch.Tensor): First input image tensor of shape (batch_size, channels, height, width).
        image2 (torch.Tensor): Second input image tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Loss value.
    """

    # Compute the Fourier transforms of the images
    image1_fft = torch.fft.fft2(image1)
    image2_fft = torch.fft.fft2(image2)

    # Compute the magnitudes of the Fourier transforms
    image1_mag = torch.abs(image1_fft)
    image2_mag = torch.abs(image2_fft)

    # Compute the high-frequency difference between the magnitudes
    high_freq_diff = image1_mag - image2_mag
    print('image1.dtype, image2.dtype',image1.dtype, image2.dtype)
    # Define a low-pass filter to remove low-frequency components
    filter = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=image1.dtype, device=image1.device).unsqueeze(0).unsqueeze(0).repeat(1,image1.shape[1],1,1)
    filter = filter / torch.sum(filter)

    # Apply the low-pass filter to the high-frequency difference
    print('high_freq_diff, filter',high_freq_diff.dtype, filter.dtype)
    with torch.autocast('cuda'):
      low_freq_diff = F.conv2d(high_freq_diff, filter, padding=1)

    # Compute the mean squared error between the low-frequency difference and zero
    loss = torch.mean(low_freq_diff**2)

    return loss

pred_noise = None
def run_sd(opt, init_image, skip_timesteps, H, W, text_prompt, neg_prompt, steps, seed,
           init_scale,  init_latent_scale, cond_image, cfg_scale, image_scale,
           cond_fn=None, init_grad_img=None, consistency_mask=None, frame_num=0,
           deflicker_src=None, prev_frame=None, rec_prompt=None, rec_frame=None,
           control_inpainting_mask=None, shuffle_source=None, ref_image=None, alpha_mask=None,
           prompt_weights=None, mask_current_frame_many=None):

  # sampler = sample_euler
  seed_everything(seed)
  sd_model.cuda()
  sd_model.model.cuda()
  sd_model.cond_stage_model.cuda()
  sd_model.cuda()
  sd_model.first_stage_model.cuda()
  model_wrap.inner_model.cuda()
  model_wrap.cuda()
  model_wrap_cfg.cuda()
  model_wrap_cfg.inner_model.cuda()
  # global cfg_scale
  if VERBOSE:
    print('seed', 'clip_guidance_scale', 'init_scale', 'init_latent_scale', 'clamp_grad', 'clamp_max',
        'init_image', 'skip_timesteps', 'cfg_scale')
    print(seed, clip_guidance_scale, init_scale, init_latent_scale, clamp_grad,
        clamp_max, init_image, skip_timesteps, cfg_scale)
  global start_code, inpainting_mask_weight, inverse_inpainting_mask, start_code_cb, guidance_start_code
  global pred_noise, controlnet_preprocess
  # global frame_num
  global normalize_latent
  global first_latent
  global first_latent_source
  global use_karras_noise
  global end_karras_ramp_early
  global latent_fixed_norm
  global latent_norm_4d
  global latent_fixed_mean
  global latent_fixed_std
  global n_mean_avg
  global n_std_avg
  global reference_latent

  batch_size = num_samples = 1
  scale = cfg_scale

  C = 4 #4
  f = 8 #8
  H = H
  W = W
  if VERBOSE:print(W, H, 'WH')
  prompt = text_prompt[0]


  neg_prompt = neg_prompt[0]
  ddim_steps = steps

  # init_latent_scale = 0. #20
  prompt_clip = prompt


  assert prompt is not None
  prompts =  text_prompt#, 'ancient white marble statue of venus in white dress with a beautiful face, turn background into pompeii , highly detailed, beautiful, by alphonse mucha']

  if VERBOSE:print('prompts', prompts, text_prompt)

  precision_scope = autocast

  t_enc = ddim_steps-skip_timesteps

  if init_image is not None:
    if isinstance(init_image, str):
      if not init_image.endswith('_lat.pt'):
        with torch.no_grad():
          with torch.cuda.amp.autocast():
            init_image_sd = load_img_sd(init_image, size=(W,H)).cuda()
            init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(init_image_sd))
            x0 = init_latent
      if init_image.endswith('_lat.pt'):
        init_latent = torch.load(init_image).cuda()
        init_image_sd = None
        x0 = init_latent

  reference_latent = None
  if ref_image is not None and reference_active:
    if os.path.exists(ref_image):
      with torch.no_grad(), torch.cuda.amp.autocast():
            reference_img = load_img_sd(ref_image, size=(W,H)).cuda()
            reference_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(reference_img))
    else:
      print('Failed to load reference image')
      ref_image = None



  if use_predicted_noise:
    if rec_frame is not None:
      with torch.cuda.amp.autocast():
            rec_frame_img = load_img_sd(rec_frame, size=(W,H)).cuda()
            rec_frame_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(rec_frame_img))

  if init_grad_img is not None:
    print('Replacing init image for cond fn')
    init_image_sd = load_img_sd(init_grad_img, size=(W,H)).cuda()

  if blend_latent_to_init > 0. and first_latent is not None:
    print('Blending to latent ', first_latent_source)
    x0 = x0*(1-blend_latent_to_init) + blend_latent_to_init*first_latent
  if normalize_latent!='off' and first_latent is not None:
    if VERBOSE:
      print('norm to 1st latent')
      print('latent source - ', first_latent_source)
    # noise2 - target
    # noise - modified

    if latent_norm_4d:
      n_mean = first_latent.mean(dim=(2,3),keepdim=True)
      n_std = first_latent.std(dim=(2,3),keepdim=True)
    else:
      n_mean = first_latent.mean()
      n_std = first_latent.std()

    if n_mean_avg is None and n_std_avg is None:
      n_mean_avg = n_mean.clone().detach().cpu().numpy()[0,:,0,0]
      n_std_avg = n_std.clone().detach().cpu().numpy()[0,:,0,0]
    else:
      n_mean_avg = n_mean_avg*n_smooth+(1-n_smooth)*n_mean.clone().detach().cpu().numpy()[0,:,0,0]
      n_std_avg = n_std_avg*n_smooth+(1-n_smooth)*n_std.clone().detach().cpu().numpy()[0,:,0,0]

    if VERBOSE:
      print('n_stats_avg (mean, std): ', n_mean_avg, n_std_avg)
    if normalize_latent=='user_defined':
      n_mean = latent_fixed_mean
      if isinstance(n_mean, list) and len(n_mean)==4: n_mean = np.array(n_mean)[None,:, None, None]
      n_std = latent_fixed_std
      if isinstance(n_std, list) and len(n_std)==4: n_std = np.array(n_std)[None,:, None, None]
    if latent_norm_4d: n2_mean = x0.mean(dim=(2,3),keepdim=True)
    else: n2_mean = x0.mean()
    x0 = x0 - (n2_mean-n_mean)
    if latent_norm_4d: n2_std = x0.std(dim=(2,3),keepdim=True)
    else: n2_std = x0.std()
    x0 = x0/(n2_std/n_std)

  if clip_guidance_scale>0:
    # text_features = clip_model.encode_text(text)
    target_embed = F.normalize(clip_model.encode_text(open_clip.tokenize(prompt_clip).cuda()).float())
  else:
    target_embed = None


  with torch.no_grad():
      with torch.cuda.amp.autocast():
       with precision_scope("cuda"):
                scope = none_context if model_version == 'v1_inpainting' else sd_model.ema_scope()
                with scope:
                    tic = time.time()
                    all_samples = []
                    uc = None
                    if True:
                        if scale != 1.0:
                            uc = prompt_parser.get_learned_conditioning(sd_model, [neg_prompt], ddim_steps)

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = prompt_parser.get_learned_conditioning(sd_model, prompts, ddim_steps)

                        shape = [C, H // f, W // f]
                        if use_karras_noise:

                          rho = 7.
                          # 14.6146
                          sigma_max=model_wrap.sigmas[-1].item()
                          sigma_min_nominal=model_wrap.sigmas[0].item()
                          # get the "sigma before sigma_min" from a slightly longer ramp
                          # https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
                          premature_sigma_min = get_premature_sigma_min(
                                                              steps=steps+1,
                                                              sigma_max=sigma_max,
                                                              sigma_min_nominal=sigma_min_nominal,
                                                              rho=rho
                                                          )
                          sigmas = K.sampling.get_sigmas_karras(
                                                              n=steps,
                                                              sigma_min=premature_sigma_min if end_karras_ramp_early else sigma_min_nominal,
                                                              sigma_max=sigma_max,
                                                              rho=rho,
                                                              device='cuda',
                                                          )
                        else:
                          sigmas = model_wrap.get_sigmas(ddim_steps)
                        alpha_mask_t = None
                        if alpha_mask is not None and init_image is not None:
                          print('alpha_mask.shape', alpha_mask.shape)
                          alpha_mask_t =  torch.from_numpy(alpha_mask).float().to(init_latent.device)[None,None,...][:,0:1,...]
                        consistency_mask_t = None
                        if consistency_mask is not None and init_image is not None:
                          consistency_mask_t =  torch.from_numpy(consistency_mask).float().to(init_latent.device).permute(2,0,1)[None,...][:,0:1,...]
                        if guidance_use_start_code:
                          guidance_start_code = torch.randn_like(init_latent)

                        deflicker_fn = deflicker_lat_fn = fft_fn = fft_latent_fn = None
                        if frame_num > args.start_frame:
                          def absdiff(a,b):
                            return abs(a-b)
                          for key in deflicker_src.keys():
                            deflicker_src[key] = load_img_sd(deflicker_src[key], size=(W,H)).cuda()
                          deflicker_fn = partial(deflicker_loss, processed1=deflicker_src['processed1'][:,:,::2,::2],
                          raw1=deflicker_src['raw1'][:,:,::2,::2], raw2=deflicker_src['raw2'][:,:,::2,::2], criterion1= absdiff, criterion2=lpips_model)
                          fft_fn = partial(high_frequency_loss, image2=init_image_sd)
                          for key in deflicker_src.keys():
                            deflicker_src[key] = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(deflicker_src[key]))
                          deflicker_lat_fn = partial(deflicker_loss,
                          raw1=deflicker_src['raw1'], raw2=deflicker_src['raw2'], criterion1= absdiff, criterion2=rmse)
                          fft_latent_fn = partial(high_frequency_loss, image2=init_latent)
                        cond_fn_partial = partial(sd_cond_fn, init_image_sd=init_image_sd,
                            init_latent=init_latent,
                            init_scale=init_scale,
                            init_latent_scale=init_latent_scale,
                            target_embed=target_embed,
                            consistency_mask = consistency_mask_t,
                            start_code = guidance_start_code,
                            deflicker_fn = deflicker_fn, deflicker_lat_fn=deflicker_lat_fn,
                                                  deflicker_src=deflicker_src, fft_fn=fft_fn, fft_latent_fn=fft_latent_fn
                            )
                        callback_partial = None
                        if cc_masked_diffusion and consistency_mask is not None or alpha_masked_diffusion and alpha_mask is not None:
                          if cb_fixed_code:
                            if start_code_cb is None:
                              if VERBOSE:print('init start code')
                              start_code_cb = torch.randn_like(x0)
                          else:
                            start_code_cb = torch.randn_like(x0)
                          # start_code = torch.randn_like(x0)
                          callback_steps = []
                          callback_masks = []
                          if cc_masked_diffusion and consistency_mask is not None:
                            callback_masks.append(consistency_mask_t)
                            callback_steps.append(int((ddim_steps-skip_timesteps)*cc_masked_diffusion))
                          if alpha_masked_diffusion and alpha_mask is not None:
                            if invert_alpha_masked_diffusion:
                              alpha_mask_t = 1.-alpha_mask_t
                            callback_masks.append(alpha_mask_t)
                            callback_steps.append(int((ddim_steps-skip_timesteps)*alpha_masked_diffusion))
                          if inverse_mask_order:
                            callback_masks.reverse()
                            callback_steps.reverse()


                          if VERBOSE: print('callback steps', callback_steps)
                          callback_partial = partial(masked_callback,
                                                     callback_steps=callback_steps,
                                                     masks=callback_masks,
                                                     init_latent=init_latent, start_code=start_code_cb)
                        if new_prompt_loras == {}:
                          # only use cond fn when loras are off
                          model_fn = make_cond_model_fn(model_wrap_cfg, cond_fn_partial)
                          # model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
                        else:
                          model_fn = model_wrap_cfg

                        model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
                        depth_img = None
                        depth_cond = None
                        if model_version == 'v2_depth':
                          if VERBOSE: print('using depth')
                          depth_cond, depth_img = make_depth_cond(cond_image, x0)
                        if 'control_' in model_version:
                          input_image = np.array(Image.open(cond_image).resize(size=(W,H))); #print(type(input_image), 'input_image', input_image.shape)

                        detected_maps = {}
                        if model_version == 'control_multi':
                          if offload_model and not controlnet_low_vram:
                                        for key in loaded_controlnets.keys():
                                          loaded_controlnets[key].cuda()

                          models = list(controlnet_multimodel.keys()); print(models)
                        else: models = model_version
                        if not controlnet_preprocess and 'control_' in model_version:
                          #if multiple cond models without preprocessing - add input to all models
                          if model_version == 'control_multi':
                            for i in models:
                              detected_map = input_image
                              if i in ['control_sd15_normal']:
                                detected_map = detected_map[:, :, ::-1]
                              detected_maps[i] = detected_map
                          else: detected_maps[model_version] = input_image

                        if 'control_sd15_temporalnet' in models:
                          if prev_frame is not None:
                            # prev_frame = cond_image
                            detected_map = np.array(Image.open(prev_frame).resize(size=(W,H))); #print(type(input_image), 'input_image', input_image.shape)
                            detected_maps['control_sd15_temporalnet'] = detected_map
                          else:

                            if VERBOSE: print('skipping control_sd15_temporalnet as prev_frame is None')
                            models = [o for o in models if o != 'control_sd15_temporalnet' ]
                            if VERBOSE: print('models after removing temp', models)

                        if controlnet_preprocess and 'control_' in model_version:
                          if 'control_sd15_face' in models:

                            detected_map = generate_annotation(input_image, max_faces)
                            if detected_map is not None:
                              detected_maps['control_sd15_face'] = detected_map
                            else:
                              if VERBOSE: print('No faces detected')
                              models = [o for o in models if o != 'control_sd15_face' ]

                          if 'control_sd15_normal' in models:
                            if offload_model: apply_depth.model.cuda()
                            input_image = HWC3(np.array(input_image)); print(type(input_image))

                            input_image = resize_image(input_image, detect_resolution); print((input_image.dtype))
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              _,detected_map = apply_depth(input_image, bg_th=bg_threshold)
                            detected_map = HWC3(detected_map)
                            if offload_model: apply_depth.model.cpu()

                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                            detected_maps['control_sd15_normal'] = detected_map

                          if 'control_sd15_normalbae' in models:
                            if offload_model: apply_normal.model.cuda()
                            input_image = HWC3(np.array(input_image)); print(type(input_image))

                            input_image = resize_image(input_image, detect_resolution); print((input_image.dtype))
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_normal(input_image)#, bg_th=bg_threshold)
                            detected_map = HWC3(detected_map)
                            if offload_model: apply_normal.model.cpu()

                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                            detected_maps['control_sd15_normalbae'] = detected_map

                          if 'control_sd15_depth' in models:
                            if offload_model: apply_depth.model.cuda()
                            input_image = HWC3(np.array(input_image)); #print(type(input_image))
                            Image.fromarray(input_image.astype('uint8')).save('./test.jpg')
                            input_image = resize_image(input_image, detect_resolution); #print((input_image.dtype), input_image.shape, input_image.size)

                            if control_sd15_depth_detector == 'Midas':
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map,_ = apply_depth(input_image)
                            if control_sd15_depth_detector == 'Zoe':
                              with torch.cuda.amp.autocast(False), torch.no_grad():
                                # apply_depth.model.load_state_dict(torch.load('/content/ControlNet/annotator/ckpts/ZoeD_M12_N.pt')['model'])
                                detected_map = apply_depth(input_image)
                            #print('dectected map depth',detected_map.shape, detected_map.min(), detected_map.max(), detected_map.mean(), detected_map.std(),  )
                            detected_map = HWC3(detected_map)
                            if offload_model: apply_depth.model.cpu()
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                            detected_maps['control_sd15_depth'] = detected_map

                          if  'control_sd15_canny' in models:
                            img = HWC3(input_image)

                            # H, W, C = img.shape

                            detected_map = apply_canny(img, low_threshold, high_threshold)
                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps['control_sd15_canny'] = detected_map

                          if  'control_sd15_softedge' in models:
                            if offload_model: apply_softedge.netNetwork.cuda()
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_softedge(resize_image(input_image, detect_resolution))
                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                            detected_maps['control_sd15_softedge'] = detected_map
                            if offload_model: apply_softedge.netNetwork.cpu()


                          if  'control_sd15_mlsd' in models:
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_mlsd(resize_image(input_image, detect_resolution), value_threshold, distance_threshold)
                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps['control_sd15_mlsd'] = detected_map

                          if  'control_sd15_openpose' in models:
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_openpose(resize_image(input_image,
                                detect_resolution), hand_and_face=control_sd15_openpose_hands_face)

                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps['control_sd15_openpose'] = detected_map

                          if  'control_sd15_scribble'  in models:
                            input_image = HWC3(input_image)
                            # H, W, C = img.shape

                            if offload_model: apply_scribble.netNetwork.cuda()
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_scribble(resize_image(input_image, detect_resolution))

                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                            detected_map = nms(detected_map, 127, 3.0)
                            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
                            detected_map[detected_map > 4] = 255
                            detected_map[detected_map < 255] = 0
                            detected_maps[ 'control_sd15_scribble' ] = detected_map
                            if offload_model: apply_scribble.netNetwork.cpu()

                          if   "control_sd15_seg" in models:
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_seg(resize_image(input_image, detect_resolution))

                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps["control_sd15_seg" ] = detected_map

                          if "control_sd15_lineart"  in models:
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_lineart(resize_image(input_image, detect_resolution), coarse=control_sd15_lineart_coarse)

                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps["control_sd15_lineart" ] = detected_map

                          if "control_sd15_lineart_anime"  in models:
                            input_image = HWC3(input_image)
                            with torch.cuda.amp.autocast(True), torch.no_grad():
                              detected_map = apply_lineart_anime(resize_image(input_image, detect_resolution))

                            detected_map = HWC3(detected_map)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps["control_sd15_lineart_anime" ] = detected_map

                          if "control_sd15_ip2p"  in models:
                            input_image = HWC3(input_image)
                            detected_map = input_image.copy()
                            img = resize_image(input_image, detect_resolution)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps["control_sd15_ip2p" ] = detected_map

                          if "control_sd15_shuffle" in models:
                            shuffle_image = np.array(Image.open(shuffle_source))
                            shuffle_image = HWC3(shuffle_image)
                            shuffle_image = cv2.resize(shuffle_image, (W, H), interpolation=cv2.INTER_NEAREST)
                            # shuffle_image = resize_image(shuffle_image, detect_resolution)

                            dH, dW, dC = shuffle_image.shape
                            detected_map = apply_shuffle(shuffle_image, w=dW, h=dH, f=256)
                            detected_maps["control_sd15_shuffle" ] = detected_map

                          if "control_sd15_inpaint"  in models:

                            if control_inpainting_mask is None:
                              if VERBOSE: print('skipping control_sd15_inpaint as control_inpainting_mask is None')
                              models = [o for o in models if o != 'control_sd15_inpaint' ]
                              if VERBOSE: print('models after removing temp', models)
                            else:
                              control_inpainting_mask *= 255
                              control_inpainting_mask = 255 - control_inpainting_mask
                              if VERBOSE: print('control_inpainting_mask',control_inpainting_mask.shape, control_inpainting_mask.min(), control_inpainting_mask.max())
                              if VERBOSE: print('control_inpainting_mask', (control_inpainting_mask[...,0] == control_inpainting_mask[...,0]).mean())
                              img = np.array(Image.open(init_image).resize(size=(W,H)))
                              h, w, C = img.shape
                              #contolnet inpaint mask - H, W, 0-255 np array
                              detected_mask = cv2.resize(control_inpainting_mask[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
                              detected_map = img.astype(np.float32).copy()
                              detected_map[detected_mask > 127] = -255.0  # use -1 as inpaint value
                              detected_maps["control_sd15_inpaint" ] = detected_map


                        if 'control_' in model_version:
                          gc.collect()
                          torch.cuda.empty_cache()
                          gc.collect()
                          if VERBOSE: print('Postprocessing cond maps')
                          def postprocess_map(detected_map):
                            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                            control = torch.stack([control for _ in range(num_samples)], dim=0)
                            depth_cond = einops.rearrange(control, 'b h w c -> b c h w').clone()
                            # if VERBOSE: print('depth_cond', depth_cond.min(), depth_cond.max(), depth_cond.mean(), depth_cond.std(), depth_cond.shape)
                            return depth_cond

                          if model_version== 'control_multi':
                            print('init shape', init_latent.shape, H,W)
                            for m in models:
                              if save_controlnet_annotations:
                                PIL.Image.fromarray(detected_maps[m].astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_{m}_{frame_num:06}.jpg', quality=95)
                              detected_maps[m] = postprocess_map(detected_maps[m])
                              if VERBOSE: print('detected_maps[m].shape', m, detected_maps[m].shape)

                            depth_cond = detected_maps
                          else: depth_cond = postprocess_map(detected_maps[model_version])


                        if model_version == 'v1_instructpix2pix':
                          if isinstance(cond_image, str):
                            print('Got img cond: ', cond_image)
                            with torch.no_grad():
                              with torch.cuda.amp.autocast():
                                input_image = Image.open(cond_image).resize(size=(W,H))
                                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                                input_image = rearrange(input_image, "h w c -> 1 c h w").to(sd_model.device)
                                depth_cond = sd_model.encode_first_stage(input_image).mode()

                        if model_version == 'v1_inpainting':
                          print('using inpainting')
                          if cond_image is not None:
                            if inverse_inpainting_mask: cond_image = 1 - cond_image
                            cond_image = Image.fromarray((cond_image*255).astype('uint8'))

                          batch = make_batch_sd(Image.open(init_image).resize((W,H)) , cond_image, txt=prompt, device=device, num_samples=1, inpainting_mask_weight=inpainting_mask_weight)
                          c_cat = list()
                          for ck in sd_model.concat_keys:
                                          cc = batch[ck].float()
                                          if ck != sd_model.masked_image_key:

                                              cc = torch.nn.functional.interpolate(cc, scale_factor=1/8)
                                          else:
                                              cc = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(cc))
                                          c_cat.append(cc)
                          depth_cond = torch.cat(c_cat, dim=1)
                        # print('depth cond', depth_cond)
                        if mask_current_frame_many is not None:
                          mask_current_frame_many = torch.nn.functional.interpolate(mask_current_frame_many, x0.shape[2:])
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': scale,
                                      'image_cond':depth_cond, 'prompt_weights':prompt_weights,
                                      'prompt_masks':mask_current_frame_many}
                        if model_version == 'v1_instructpix2pix':
                          extra_args['image_scale'] = image_scale
                          # extra_args['cond'] = sd_model.get_learned_conditioning(prompts)
                          # extra_args['uncond'] = sd_model.get_learned_conditioning([""])
                        if skip_timesteps>0:
                              #using non-random start code
                          if fixed_code:
                            if start_code is None:
                              if VERBOSE:print('init start code')
                              start_code = torch.randn_like(x0)
                            if not use_legacy_fixed_code:
                              rand_code = torch.randn_like(x0)
                              combined_code = ((1 - code_randomness) * start_code + code_randomness * rand_code) / ((code_randomness**2 + (1-code_randomness)**2) ** 0.5)
                              noise = combined_code - (x0 / sigmas[0])
                              noise = noise * sigmas[ddim_steps - t_enc -1]

                            #older version
                            if use_legacy_fixed_code:
                              normalize_code = True
                              if normalize_code:
                                noise2 = torch.randn_like(x0)* sigmas[ddim_steps - t_enc -1]
                                if latent_norm_4d: n_mean = noise2.mean(dim=(2,3),keepdim=True)
                                else: n_mean = noise2.mean()
                                if latent_norm_4d: n_std = noise2.std(dim=(2,3),keepdim=True)
                                else: n_std = noise2.std()

                              noise =   torch.randn_like(x0)
                              noise = (start_code*(1-code_randomness)+(code_randomness)*noise) * sigmas[ddim_steps - t_enc -1]
                              if normalize_code:
                                if latent_norm_4d: n2_mean = noise.mean(dim=(2,3),keepdim=True)
                                else: n2_mean = noise.mean()
                                noise = noise - (n2_mean-n_mean)
                                if latent_norm_4d: n2_std = noise.std(dim=(2,3),keepdim=True)
                                else: n2_std = noise.std()
                                noise = noise/(n2_std/n_std)

                          else:
                            noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc -1] #correct one
                            if use_predicted_noise:
                              print('using predicted noise')
                              rand_noise = torch.randn_like(x0)
                              rec_noise = find_noise_for_image_sigma_adjustment(init_latent=rec_frame_latent, prompt=rec_prompt, image_conditioning=depth_cond, cfg_scale=scale, steps=ddim_steps, frame_num=frame_num)
                              combined_noise = ((1 - rec_randomness) * rec_noise + rec_randomness * rand_noise) / ((rec_randomness**2 + (1-rec_randomness)**2) ** 0.5)
                              noise = combined_noise - (x0 / sigmas[0])
                              noise = noise * sigmas[ddim_steps - t_enc -1]#faster collapse

                            print('noise')
                            # noise = noise[::4,::4,:]
                            # noise = torch.nn.functional.interpolate(noise, scale_factor=4, mode='bilinear')
                          if t_enc != 0:
                            xi = x0 + noise
                            #printf('xi', xi.shape, xi.min().item(), xi.max().item(), xi.std().item(), xi.mean().item())
                            # print(xi.mean(), xi.std(), xi.min(), xi.max())
                            sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                            # sigma_sched = sigmas[ddim_steps - t_enc:]
                            print('xi', xi.shape)
                            samples_ddim = sampler(model_fn, xi, sigma_sched,
                                                   extra_args=extra_args, callback=callback_partial)
                          else:
                            samples_ddim = x0
                        else:
                          # if use_predicted_noise and frame_num>0:
                          if use_predicted_noise:
                              print('using predicted noise')
                              rand_noise = torch.randn_like(x0)
                              rec_noise = find_noise_for_image_sigma_adjustment(init_latent=rec_frame_latent, prompt=rec_prompt, image_conditioning=depth_cond, cfg_scale=scale, steps=ddim_steps, frame_num=frame_num)
                              combined_noise = ((1 - rec_randomness) * rec_noise + rec_randomness * rand_noise) / ((rec_randomness**2 + (1-rec_randomness)**2) ** 0.5)
                              x = combined_noise# - (x0 / sigmas[0])
                          else: x = torch.randn([batch_size, *shape], device=device)
                          x = x * sigmas[0]
                          samples_ddim = sampler(model_fn, x, sigmas, extra_args=extra_args, callback=callback_partial)
                        if first_latent is None:
                          if VERBOSE:print('setting 1st latent')
                          first_latent_source = 'samples ddim (1st frame output)'
                          first_latent = samples_ddim

                        if offload_model:
                          sd_model.model.cpu()
                          sd_model.cond_stage_model.cpu()
                          if model_version == 'control_multi':
                            for key in loaded_controlnets.keys():
                              loaded_controlnets[key].cpu()

                        gc.collect()
                        torch.cuda.empty_cache()
                        x_samples_ddim = sd_model.decode_first_stage(samples_ddim)
                        printf('x_samples_ddim', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())
                        scale_raw_sample = False
                        if scale_raw_sample:
                          m = x_samples_ddim.mean()
                          x_samples_ddim-=m;
                          r = (x_samples_ddim.max()-x_samples_ddim.min())/2

                          x_samples_ddim/=r
                          x_samples_ddim+=m;
                          if VERBOSE:printf('x_samples_ddim scaled', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())


                        all_samples.append(x_samples_ddim)
  return all_samples, samples_ddim, depth_img



diffusion_model = "stable_diffusion"

diffusion_sampling_mode = 'ddim'


normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

"""# 2. Settings"""

#@markdown ####**Basic Settings:**
batch_name = 'stable_warpfusion_0.17.0' #@param{type: 'string'}
steps =  50
##@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
# stop_early = 0  #@param{type: 'number'}
stop_early = 0
stop_early = min(steps-1,stop_early)
#@markdown Specify desired output size here.\
#@markdown Don't forget to rerun all steps after changing the width height (including forcing optical flow generation)
width_height = [480,480]#@param{type: 'raw'}
width_height = [int(o) for o in width_height]
clip_guidance_scale = 0 #
tv_scale =  0
range_scale =   0
cutn_batches =   4
skip_augs = False

#@markdown ---

#@markdown ####**Init Settings:**
init_image = "" #@param{type: 'string'}
init_scale = 0
##@param{type: 'integer'}
skip_steps =  25
##@param{type: 'integer'}
##@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.\
##@markdown A good init_scale for Stable Diffusion is 0*


#Get corrected sizes
#@markdown Make sure the resolution is divisible by that number. The Default 64 is the most stable.

force_multiple_of = "64" #@param [8,64]
force_multiple_of = int(force_multiple_of)
side_x = (width_height[0]//force_multiple_of)*force_multiple_of;
side_y = (width_height[1]//force_multiple_of)*force_multiple_of;
if side_x != width_height[0] or side_y != width_height[1]:
  print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of {force_multiple_of}.')
width_height = (side_x, side_y)
#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps


#Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

#@title ## Animation Settings
#@markdown Create a looping video from single init image\
#@markdown Use this if you just want to test settings. This will create a small video (1 sec = 24 frames)\
#@markdown This way you will be able to iterate faster without the need to process flow maps for a long final video before even getting to testing prompts.
#@markdown You'll need to manually input the resulting video path into the next cell.

use_looped_init_image = False #@param {'type':'boolean'}
video_duration_sec = 2 #@param {'type':'number'}
if use_looped_init_image:
  subprocess.run(
        'ffmpeg -loop 1 -i "/path/to/init_image.jpg" -c:v libx264 -t "duration_in_seconds" -pix_fmt yuv420p -vf scale=width:height "/path/to/root_dir/out.mp4" -y',
        shell=True,
        check=True
  )
  # !ffmpeg -loop 1 -i "{init_image}" -c:v libx264 -t "{video_duration_sec}" -pix_fmt yuv420p -vf scale={side_x}:{side_y} "{root_dir}/out.mp4" -y
  print('Video saved to ', f"{root_dir}/out.mp4")

#@title ##Video Input Settings:
animation_mode = 'Video Input'
import os, platform
if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
  print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")


#@markdown ---


video_init_path = "bron2.mp4" #@param {type: 'string'}

extract_nth_frame =  1#@param {type: 'number'}
#@markdown *Specify frame range. end_frame=0 means fill the end of video*
start_frame = 0#@param {type: 'number'}
end_frame = 0#@param {type: 'number'}
end_frame_orig = end_frame
if end_frame<=0 or end_frame==None: end_frame = 99999999999999999999999999999
#@markdown ####Separate guiding video (optical flow source):
#@markdown Leave blank to use the first video.
flow_video_init_path = "" #@param {type: 'string'}
flow_extract_nth_frame =  1#@param {type: 'number'}
if flow_video_init_path == '':
  flow_video_init_path = None
#@markdown ####Image Conditioning Video Source:
#@markdown Used together with image-conditioned models, like controlnet, depth, or inpainting model.
#@markdown You can use your own video as depth mask or as inpaiting mask.
cond_video_path = "" #@param {type: 'string'}
cond_extract_nth_frame =  1#@param {type: 'number'}
if cond_video_path == '':
  cond_video_path = None

#@markdown ####Colormatching Video Source:
#@markdown Used as colormatching source. Specify image or video.
color_video_path = "" #@param {type: 'string'}
color_extract_nth_frame =  1#@param {type: 'number'}
if color_video_path == '':
  color_video_path = None
#@markdown Enable to store frames, flow maps, alpha maps on drive
store_frames_on_google_drive = False #@param {type: 'boolean'}
video_init_seed_continuity = False

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  createPath(output_path)
  print(f"Exporting Video Frames (1 every {nth_frame})...")
  try:
    for f in [o.replace('\\','/') for o in glob(output_path+'/*.jpg')]:
    # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
      pathlib.Path(f).unlink()
  except:
    print('error deleting frame ', f)
  # vf = f'select=not(mod(n\\,{nth_frame}))'
  vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
  if os.path.exists(video_path):
    try:
        # subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    except:
        subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

  else:
    sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')

if animation_mode == 'Video Input':
  postfix = f'{generate_file_hash(video_init_path)[:10]}_{start_frame}_{end_frame_orig}_{extract_nth_frame}'
  if flow_video_init_path:
    flow_postfix = f'{generate_file_hash(flow_video_init_path)[:10]}_{flow_extract_nth_frame}'
  if store_frames_on_google_drive: #suggested by Chris the Wizard#8082 at discord
      videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
      flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if flow_video_init_path else videoFramesFolder
      condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
      colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
      controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
      recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'

  else:
      videoFramesFolder = f'{root_dir}/videoFrames/{postfix}'
      flowVideoFramesFolder = f'{root_dir}/flowVideoFrames/{flow_postfix}' if flow_video_init_path else videoFramesFolder
      condVideoFramesFolder = f'{root_dir}/condVideoFrames'
      colorVideoFramesFolder = f'{root_dir}/colorVideoFrames'
      controlnetDebugFolder = f'{root_dir}/controlnetDebug'
      recNoiseCacheFolder = f'{root_dir}/recNoiseCache'

  if not is_colab:
    videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
    flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if flow_video_init_path else videoFramesFolder
    condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
    colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
    controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
    recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'

  os.makedirs(controlnetDebugFolder, exist_ok=True)
  os.makedirs(recNoiseCacheFolder, exist_ok=True)

  extractFrames(video_init_path, videoFramesFolder, extract_nth_frame, start_frame, end_frame)
  if flow_video_init_path:
    print(flow_video_init_path, flowVideoFramesFolder, flow_extract_nth_frame)
    extractFrames(flow_video_init_path, flowVideoFramesFolder, flow_extract_nth_frame, start_frame, end_frame)

  if cond_video_path:
    print(cond_video_path, condVideoFramesFolder, cond_extract_nth_frame)
    extractFrames(cond_video_path, condVideoFramesFolder, cond_extract_nth_frame, start_frame, end_frame)

  if color_video_path:
    try:
      os.makedirs(colorVideoFramesFolder, exist_ok=True)
      Image.open(color_video_path).save(os.path.join(colorVideoFramesFolder,'000001.jpg'))
    except:
      print(color_video_path, colorVideoFramesFolder, color_extract_nth_frame)
      extractFrames(color_video_path, colorVideoFramesFolder, color_extract_nth_frame, start_frame, end_frame)

#@title Video Masking

#@markdown Generate background mask from your init video or use a video as a mask
mask_source = 'init_video' #@param ['init_video','mask_video']
#@markdown Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
extract_background_mask = False #@param {'type':'boolean'}
#@markdown Specify path to a mask video for mask_video mode.
mask_video_path = '' #@param {'type':'string'}
if extract_background_mask:
  os.chdir(root_dir)
  subprocess.run(['python', '-m', "pip", "-q", "install", "av", "pims"], check=True)
  # !python -m pip -q install av pims
  gitclone('https://github.com/Sxela/RobustVideoMattingCLI')
  if mask_source == 'init_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    subprocess.run(['python', "{root_dir}/RobustVideoMattingCLI/rvm_cli.py", "--input_path", "{videoFramesFolder}", "--output_alpha", "{root_dir}/alpha.mp4"], check=True)
    # !python "{root_dir}/RobustVideoMattingCLI/rvm_cli.py" --input_path "{videoFramesFolder}" --output_alpha "{root_dir}/alpha.mp4"
    extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
  if mask_source == 'mask_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    maskVideoFrames = videoFramesFolder+'Mask'
    createPath(maskVideoFrames)
    extractFrames(mask_video_path, f"{maskVideoFrames}", extract_nth_frame, start_frame, end_frame)
    subprocess.run(['python', "{root_dir}/RobustVideoMattingCLI/rvm_cli.py", "--input_path", "{maskVideoFrames}", "--output_alpha", "{root_dir}/alpha.mp4"], check=True)
    # !python "{root_dir}/RobustVideoMattingCLI/rvm_cli.py" --input_path "{maskVideoFrames}" --output_alpha "{root_dir}/alpha.mp4"
    extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
else:
  if mask_source == 'init_video':
    videoFramesAlpha = videoFramesFolder
  if mask_source == 'mask_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    extractFrames(mask_video_path, f"{videoFramesAlpha}", extract_nth_frame, start_frame, end_frame)
    #extract video

"""# Optical map settings

"""

# Commented out IPython magic to ensure Python compatibility.
#@title Generate optical flow and consistency maps
#@markdown Run once per init video and width_height setting.
#if you're running locally, just restart this runtime, no need to edit PIL files.
flow_warp = True
check_consistency = True
force_flow_generation = False #@param {type:'boolean'}

use_legacy_cc = False #@param{'type':'boolean'}

#@title Setup Optical Flow
##@markdown Run once per session. Doesn't download again if model path exists.
##@markdown Use force download to reload raft models if needed
force_download = False #\@param {type:'boolean'}
# import wget
import zipfile, shutil

if (os.path.exists(f'{root_dir}/raft')) and force_download:
  try:
    shutil.rmtree(f'{root_dir}/raft')
  except:
      print('error deleting existing RAFT model')
if (not (os.path.exists(f'{root_dir}/raft'))) or force_download:
  os.chdir(root_dir)
  gitclone('https://github.com/Sxela/WarpFusion')
else:
  os.chdir(root_dir)
  os.chdir('WarpFusion')
  subprocess.run(['git', 'pull'], check=True)
  # !git pull
  os.chdir(root_dir)

try:
  from python_color_transfer.color_transfer import ColorTransfer, Regrain
except:
  os.chdir(root_dir)
  gitclone('https://github.com/pengbo-learn/python-color-transfer')

os.chdir(root_dir)
sys.path.append('./python-color-transfer')

if animation_mode == 'Video Input':
  os.chdir(root_dir)
  gitclone('https://github.com/Sxela/flow_tools')

#@title Define color matching and brightness adjustment
os.chdir(f"{root_dir}/python-color-transfer")
from python_color_transfer.color_transfer import ColorTransfer, Regrain
os.chdir(root_path)

PT = ColorTransfer()
RG = Regrain()

def match_color(stylized_img, raw_img, opacity=1.):
  if opacity > 0:
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    img_arr_reg = RG.regrain     (img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_reg = img_arr_reg*opacity+img_arr_in*(1-opacity)
    img_arr_reg = cv2.cvtColor(img_arr_reg.round().astype('uint8'),cv2.COLOR_BGR2RGB)
    return img_arr_reg
  else: return raw_img

from PIL import Image, ImageOps, ImageStat, ImageEnhance

def get_stats(image):
   stat = ImageStat.Stat(image)
   brightness = sum(stat.mean) / len(stat.mean)
   contrast = sum(stat.stddev) / len(stat.stddev)
   return brightness, contrast

#implemetation taken from https://github.com/lowfuel/progrockdiffusion

def adjust_brightness(image):

  brightness, contrast = get_stats(image)
  if brightness > high_brightness_threshold:
    print(" Brightness over threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(high_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image>high_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)
  if brightness < low_brightness_threshold:
    print(" Brightness below threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(low_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image<low_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)

  image = np.array(image)
  image = np.where(image>max_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = np.where(image<min_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = Image.fromarray(image)
  return image

#@title Define optical flow functions for Video input animation mode only
# if animation_mode == 'Video Input Legacy':
DEBUG = False

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k0 = np.clip(k0, 0, colorwheel.shape[0]-1)
    k1 = k0 + 1
    k1 = np.clip(k1, 0, colorwheel.shape[0]-1)
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


from torch import Tensor

# if True:
if animation_mode == 'Video Input':
  in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
  flo_folder = in_path+'_out_flo_fwd'
  #the main idea comes from neural-style-tf frame warping with optical flow maps
  #https://github.com/cysmith/neural-style-tf
  # path = f'{root_dir}/RAFT/core'
  # import sys
  # sys.path.append(f'{root_dir}/RAFT/core')
  # %cd {path}

  # from utils.utils import InputPadder

  class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

  # from raft import RAFT
  import numpy as np
  import argparse, PIL, cv2
  from PIL import Image
  from tqdm.notebook import tqdm
  from glob import glob
  import torch
  import scipy.ndimage

  args2 = argparse.Namespace()
  args2.small = False
  args2.mixed_precision = True

  TAG_CHAR = np.array([202021.25], np.float32)

  def writeFlow(filename,uv,v=None):
      """
      https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
      Copyright 2017 NVIDIA CORPORATION

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

      Write optical flow to file.

      If v is None, uv is assumed to contain both u and v channels,
      stacked in depth.
      Original code by Deqing Sun, adapted from Daniel Scharstein.
      """
      nBands = 2

      if v is None:
          assert(uv.ndim == 3)
          assert(uv.shape[2] == 2)
          u = uv[:,:,0]
          v = uv[:,:,1]
      else:
          u = uv

      assert(u.shape == v.shape)
      height,width = u.shape
      f = open(filename,'wb')
      # write the header
      f.write(TAG_CHAR)
      np.array(width).astype(np.int32).tofile(f)
      np.array(height).astype(np.int32).tofile(f)
      # arrange into matrix form
      tmp = np.zeros((height, width*nBands))
      tmp[:,np.arange(width)*2] = u
      tmp[:,np.arange(width)*2 + 1] = v
      tmp.astype(np.float32).tofile(f)
      f.close()

  def load_cc(path, blur=2, dilate=0):
    multilayer_weights = np.array(Image.open(path))/255
    weights = np.ones_like(multilayer_weights[...,0])
    weights*=multilayer_weights[...,0].clip(1-missed_consistency_weight,1)
    weights*=multilayer_weights[...,1].clip(1-overshoot_consistency_weight,1)
    weights*=multilayer_weights[...,2].clip(1-edges_consistency_weight,1)
    weights = np.where(weights<0.5, 0, 1)
    if dilate>0:
      weights = (1-binary_dilation(1-weights, disk(dilate))).astype('uint8')
    if blur>0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    weights = np.repeat(weights[...,None],3, axis=2)
    # print('------------cc debug------', f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_cc_mask.jpg')
    PIL.Image.fromarray((weights*255).astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_cc_mask.jpg', quality=95)
    # assert False
    if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
    return weights



  def load_img(img, size):
    img = Image.open(img).convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

  def get_flow(frame1, frame2, model, iters=20, half=True):
          # print(frame1.shape, frame2.shape)
          padder = InputPadder(frame1.shape)
          frame1, frame2 = padder.pad(frame1, frame2)
          if half: frame1, frame2 = frame1, frame2
          # print(frame1.shape, frame2.shape)
          _, flow12 = model(frame1, frame2)
          flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

          return flow12

  def warp_flow(img, flow, mul=1.):
      h, w = flow.shape[:2]
      flow = flow.copy()
      flow[:, :, 0] += np.arange(w)
      flow[:, :, 1] += np.arange(h)[:, np.newaxis]
      flow*=mul
      res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)

      return res

  def makeEven(_x):
    return _x if (_x % 2 == 0) else _x+1

  def fit(img,maxsize=512):
    maxdim = max(*img.size)
    if maxdim>maxsize:
    # if True:
      ratio = maxsize/maxdim
      x,y = img.size
      size = (makeEven(int(x*ratio)),makeEven(int(y*ratio)))
      img = img.resize(size, warp_interp)
    return img


  def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0.,
           pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    printf('blend warp', blend)

    if isinstance(flo_path, str):
      flow21 = np.load(flo_path)
    else: flow21 = flo_path
    # print('loaded flow from ', flo_path, ' witch shape ', flow21.shape)
    pad = int(max(flow21.shape)*pad_pct)
    flow21 = np.pad(flow21, pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')
    # print('frame1.size, frame2.size, padded flow21.shape')
    # print(frame1.size, frame2.size, flow21.shape)


    frame1pil = np.array(frame1.convert('RGB'))#.resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    frame1pil = np.pad(frame1pil, pad_width=((pad,pad),(pad,pad),(0,0)),mode=padding_mode)
    if video_mode:
      warp_mul=1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0]-pad,pad:frame1_warped21.shape[1]-pad,:]

    frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path:
      forward_weights = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)
      # print('forward_weights')
      # print(forward_weights.shape)
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)

      forward_weights = forward_weights.clip(forward_clip,1.)
      if use_patchmatch_inpaiting>0 and warp_mode == 'use_image':
        if not is_colab: print('Patchmatch only working on colab/linux')
        else: print('PatchMatch disabled.')
        # if not video_mode and is_colab:
        #       print('patchmatching')
        #       # print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
        #       patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
        #       frame2pil = np.array(frame2pil)*(1-use_patchmatch_inpaiting)+use_patchmatch_inpaiting*np.array(patch_match.inpaint(frame1_warped21, patchmatch_mask, patch_size=5))
        #       # blended_w = Image.fromarray(blended_w)
      blended_w = frame2pil*(1-blend) + blend*(frame1_warped21*forward_weights+frame2pil*(1-forward_weights))
    else:
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
      blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)



    blended_w = Image.fromarray(blended_w.round().astype('uint8'))
    # if use_patchmatch_inpaiting and warp_mode == 'use_image':
    #           print('patchmatching')
    #           print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
    #           patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
    #           blended_w = patch_match.inpaint(blended_w, patchmatch_mask, patch_size=5)
    #           blended_w = Image.fromarray(blended_w)
    if not video_mode:
      if enable_adjust_brightness: blended_w = adjust_brightness(blended_w)
    return  blended_w

  def warp_lat(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0.,
           pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    warp_downscaled = True
    flow21 = np.load(flo_path)
    pad = int(max(flow21.shape)*pad_pct)
    if warp_downscaled:
      flow21 = flow21.transpose(2,0,1)[None,...]
      flow21 = torch.nn.functional.interpolate(torch.from_numpy(flow21).float(), scale_factor = 1/8, mode = 'bilinear')
      flow21 = flow21.numpy()[0].transpose(1,2,0)/8
      # flow21 = flow21[::8,::8,:]/8

    flow21 = np.pad(flow21, pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')

    if not warp_downscaled:
      frame1 = torch.nn.functional.interpolate(frame1, scale_factor = 8)
    frame1pil = frame1.cpu().numpy()[0].transpose(1,2,0)

    frame1pil = np.pad(frame1pil, pad_width=((pad,pad),(pad,pad),(0,0)),mode=padding_mode)
    if video_mode:
      warp_mul=1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0]-pad,pad:frame1_warped21.shape[1]-pad,:]
    if not warp_downscaled:
      frame2pil = frame2.convert('RGB').resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp)
    else:
      frame2pil = frame2.convert('RGB').resize(((flow21.shape[1]-pad*2)*8,(flow21.shape[0]-pad*2)*8),warp_interp)
    frame2pil = np.array(frame2pil)
    frame2pil = (frame2pil/255.)[None,...].transpose(0, 3, 1, 2)
    frame2pil = 2*torch.from_numpy(frame2pil).float().cuda()-1.
    frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
    if not warp_downscaled: frame2pil = torch.nn.functional.interpolate(frame2pil, scale_factor = 8)
    frame2pil = frame2pil.cpu().numpy()[0].transpose(1,2,0)
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path:
      forward_weights = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)
      print(forward_weights[...,:1].shape, 'forward_weights.shape')
      forward_weights = np.repeat(forward_weights[...,:1],4, axis=-1)
      # print('forward_weights')
      # print(forward_weights.shape)
      print('frame2pil.shape, frame1_warped21.shape, flow21.shape', frame2pil.shape, frame1_warped21.shape, flow21.shape)
      forward_weights = forward_weights.clip(forward_clip,1.)
      if warp_downscaled: forward_weights = forward_weights[::8,::8,:]; print(forward_weights.shape, 'forward_weights.shape')
      blended_w = frame2pil*(1-blend) + blend*(frame1_warped21*forward_weights+frame2pil*(1-forward_weights))
    else:
      if not video_mode and not warp_mode == 'use_latent' and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
      blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)
    blended_w = blended_w.transpose(2,0,1)[None,...]
    blended_w = torch.from_numpy(blended_w).float()
    if not warp_downscaled:
      # blended_w = blended_w[::8,::8,:]
      blended_w = torch.nn.functional.interpolate(blended_w, scale_factor = 1/8, mode='bilinear')


    return blended_w# torch.nn.functional.interpolate(torch.from_numpy(blended_w), scale_factor = 1/8)


  in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
  flo_folder = in_path+'_out_flo_fwd'

  temp_flo = in_path+'_temp_flo'
  flo_fwd_folder = in_path+'_out_flo_fwd'
  flo_bck_folder = in_path+'_out_flo_bck'

#   %cd {root_path}

# (c) Alex Spirin 2023

import cv2

def extract_occlusion_mask(flow, threshold=10):
    flow = flow.clone()[0].permute(1, 2, 0).detach().cpu().numpy()
    h, w = flow.shape[:2]

    """
    Extract a mask containing all the points that have no origin in frame one.

    Parameters:
        motion_vector (numpy.ndarray): A 2D array of motion vectors.
        threshold (int): The threshold value for the magnitude of the motion vector.

    Returns:
        numpy.ndarray: The occlusion mask.
    """
    # Compute the magnitude of the motion vector.
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to identify occlusions.
    occlusion_mask = (mag > threshold).astype(np.uint8)

    return occlusion_mask, mag

import cv2
import numpy as np

def edge_detector(image, threshold=0.5, edge_width=1):
    """
    Detect edges in an image with adjustable edge width.

    Parameters:
        image (numpy.ndarray): The input image.
        edge_width (int): The width of the edges to detect.

    Returns:
        numpy.ndarray: The edge image.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Sobel edge map.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=edge_width)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=edge_width)

    # Compute the edge magnitude.
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize the magnitude to the range [0, 1].
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold the magnitude to create a binary edge image.

    edge_image = (mag > threshold).astype(np.uint8) * 255

    return edge_image

def get_unreliable(flow):
    # Mask pixels that have no source and will be taken from frame1, to remove trails and ghosting.

    # flow = flow[0].cpu().numpy().transpose(1,2,0)

    # Calculate the coordinates of pixels in the new frame
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]

    # Create a mask for the valid pixels in the new frame
    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

    # Create the new frame by interpolating the pixel values using the calculated coordinates
    new_frame = np.zeros((flow.shape[0], flow.shape[1], 3))*1.-1
    new_frame[new_y[mask].astype(np.int32), new_x[mask].astype(np.int32)] = 255

    # Keep masked area, discard the image.
    new_frame = new_frame==-1
    return new_frame, mask

from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_opening, binary_closing

import cv2

def remove_small_holes(mask, min_size=50):
    # Copy the input binary mask
    result = mask.copy()

    # Find contours of connected components in the binary image
    contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for i in range(len(contours)):
        # Compute the area of the i-th contour
        area = cv2.contourArea(contours[i])

        # Check if the area of the i-th contour is smaller than min_size
        if area < min_size:
            # Draw a filled contour over the i-th contour region
            cv2.drawContours(result, [contours[i]], 0, 255, -1, cv2.LINE_AA, hierarchy, 0)

    return result


def filter_unreliable(mask, dilation=1):
  img = 255-remove_small_holes((1-mask[...,0].astype('uint8'))*255, 200)
  # img = binary_fill_holes(img)
  img = binary_erosion(img, disk(1))
  img = binary_dilation(img, disk(dilation))
  return img
from torchvision.utils import flow_to_image as flow_to_image_torch
def make_cc_map(predicted_flows, predicted_flows_bwd, dilation=1, edge_width=11):

  flow_imgs = flow_to_image(predicted_flows_bwd)
  edge = edge_detector(flow_imgs.astype('uint8'), threshold=0.1, edge_width=edge_width)
  res, _ = get_unreliable(predicted_flows)
  _, overshoot = get_unreliable(predicted_flows_bwd)
  joint_mask = np.ones_like(res)*255
  joint_mask[...,0] = 255-(filter_unreliable(res, dilation)*255)
  joint_mask[...,1] = (overshoot*255)
  joint_mask[...,2] = 255-edge

  return joint_mask


def hstack(images):
  if isinstance(images[0], str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))
  for image in images:
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 00), (image.size[0], image.size[1])), outline="black", width=3)
  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  return new_im

import locale
def getpreferredencoding(do_setlocale = True):
            return "UTF-8"
if is_colab: locale.getpreferredencoding = getpreferredencoding

def vstack(images):
  if isinstance(next(iter(images)), str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  return new_im

if is_colab:
  for i in [7,8,9,10]:
    try:
      filedata = None
      with open(f'/usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py', 'r') as file :
        filedata = file.read()
      filedata = filedata.replace('(TiffTags.IFD, "L", "long"),', '#(TiffTags.IFD, "L", "long"),')
      with open(f'/usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py', 'w') as file :
        file.write(filedata)
      with open(f'/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py', 'w') as file :
        file.write(filedata)
    except:
      pass
      # print(f'Error writing /usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py')

class flowDataset():
  def __init__(self, in_path, half=True, normalize=False):
    frames = sorted(glob(in_path+'/*.*'));
    assert len(frames)>2, f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.'
    self.frames = frames

  def __len__(self):
    return len(self.frames)-1

  def load_img(self, img, size):
    img = Image.open(img).convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...]

  def __getitem__(self, i):
    frame1, frame2 = self.frames[i], self.frames[i+1]
    frame1 = self.load_img(frame1, width_height)
    frame2 = self.load_img(frame2, width_height)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    batch = torch.cat([frame1, frame2])
    if normalize:
      batch = 2 * (batch / 255.0) - 1.0
    return batch

from torch.utils.data import DataLoader

def save_preview(flow21, out_flow21_fn):
  try:
    Image.fromarray(flow_to_image(flow21)).save(out_flow21_fn, quality=90)
  except:
    print('Error saving flow preview for frame ', out_flow21_fn)

#copyright Alex Spirin @ 2022
def blended_roll(img_copy, shift, axis):
  if int(shift) == shift:
    return np.roll(img_copy, int(shift), axis=axis)

  max = math.ceil(shift)
  min = math.floor(shift)
  if min != 0 :
    img_min = np.roll(img_copy, min, axis=axis)
  else:
    img_min = img_copy
  img_max = np.roll(img_copy, max, axis=axis)
  blend = max-shift
  img_blend = img_min*blend + img_max*(1-blend)
  return img_blend

#copyright Alex Spirin @ 2022
def move_cluster(img,i,res2, center, mode='blended_roll'):
  img_copy = img.copy()
  motion = center[i]
  mask = np.where(res2==motion, 1, 0)[...,0][...,None]
  y, x = motion
  if mode=='blended_roll':
    img_copy = blended_roll(img_copy, x, 0)
    img_copy = blended_roll(img_copy, y, 1)
  if mode=='int_roll':
    img_copy = np.roll(img_copy, int(x), axis=0)
    img_copy = np.roll(img_copy, int(y), axis=1)
  return img_copy, mask

import cv2


def get_k(flow, K):
  Z = flow.reshape((-1,2))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  res = center[label.flatten()]
  res2 = res.reshape((flow.shape))
  return res2, center

def k_means_warp(flo, img, num_k):
  # flo = np.load(flo)
  img = np.array((img).convert('RGB'))
  num_k = 8

  # print(img.shape)
  res2, center = get_k(flo, num_k)
  center = sorted(list(center), key=lambda x: abs(x).mean())

  img = cv2.resize(img, (res2.shape[:-1][::-1]))
  img_out = np.ones_like(img)*255.

  for i in range(num_k):
    img_rolled, mask_i = move_cluster(img,i,res2,center)
    img_out = img_out*(1-mask_i) + img_rolled*(mask_i)

  # cv2_imshow(img_out)
  return Image.fromarray(img_out.astype('uint8'))

def flow_batch(i, batch, pool):
  with torch.cuda.amp.autocast():
          batch = batch[0]
          frame_1 = batch[0][None,...].cuda()
          frame_2 = batch[1][None,...].cuda()
          frame1 = ds.frames[i]
          frame1 = frame1.replace('\\','/')
          out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
          if flow_lq:   frame_1, frame_2 = frame_1, frame_2
          if use_jit_raft:
            _, flow21 = raft_model(frame_2, frame_1)
          else:
            flow21 = raft_model(frame_2, frame_1, num_flow_updates=num_flow_updates)[-1] #flow_bwd
          mag = (flow21[:,0:1,...]**2 + flow21[:,1:,...]**2).sqrt()
          mag_thresh = 0.5
          #zero out flow values for non-moving frames below threshold to avoid noisy flow/cc maps
          if mag.max()<mag_thresh:
            flow21_clamped = torch.where(mag<mag_thresh, 0, flow21)
          else:
            flow21_clamped = flow21
          flow21 = flow21[0].permute(1, 2, 0).detach().cpu().numpy()
          flow21_clamped = flow21_clamped[0].permute(1, 2, 0).detach().cpu().numpy()

          if flow_save_img_preview or i in range(0,len(ds),len(ds)//10):
            pool.apply_async(save_preview, (flow21, out_flow21_fn+'.jpg') )
          pool.apply_async(np.save, (out_flow21_fn, flow21))
          if check_consistency:
            if use_jit_raft:
              _, flow12 = raft_model(frame_1, frame_2)
            else:
              flow12 = raft_model(frame_1, frame_2)[-1] #flow_fwd

            flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
            if flow_save_img_preview:
              pool.apply_async(save_preview, (flow12, out_flow21_fn+'_12'+'.jpg'))
            if use_legacy_cc:
              pool.apply_async(np.save, (out_flow21_fn+'_12', flow12))
            else:
              joint_mask = make_cc_map(flow12, flow21_clamped, dilation=missed_consistency_dilation,
                                       edge_width=edge_consistency_width)
              joint_mask = PIL.Image.fromarray(joint_mask.astype('uint8'))
              cc_path = f"{flo_fwd_folder}/{frame1.split('/')[-1]}-21_cc.jpg"
              # print(cc_path)
              joint_mask.save(cc_path)
              # pool.apply_async(joint_mask.save, cc_path)

from multiprocessing.pool import ThreadPool as Pool
import gc
threads = 4 #@param {'type':'number'}
#@markdown If you're having "process died" error on Windows, set num_workers to 0
num_workers = 0 #@param {'type':'number'}

#@markdown Use lower quality model (half-precision).\
#@markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
flow_lq = True #@param {type:'boolean'}
#@markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
flow_save_img_preview = False  #@param {type:'boolean'}
in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
flo_fwd_folder = flo_folder = in_path+f'_out_flo_fwd/{side_x}_{side_y}/'
print(flo_folder)
# #@markdown reverse_cc_order - on - default value (like in older notebooks). off - reverses consistency computation
reverse_cc_order = True  #
# #@param {type:'boolean'}
if not flow_warp: print('flow_wapr not set, skipping')
try: raft_model
except: raft_model = None
# #@markdown Use previous pre-compile raft version (won't work with pytorch 2.0)
use_jit_raft = False
# #@param {'type':'boolean'}
# #@markdown Compile raft model (only with use_raft_jit = False). Compiles the model (~about 2 minutes) for ~30% speedup. Use for very long runs.
compile_raft = False
# #@param {'type':'boolean'}
#@markdown Flow estimation quality (number of iterations, 12 - default. higher - better and slower)
num_flow_updates = 12 #@param {'type':'number'}
#\@markdown Unreliable areas mask (missed consistency) width
#\@markdown Default = 1
missed_consistency_dilation = 2  #\ @param {'type':'number'}
#\@markdown Motion edge areas (edge consistency) width
#\@markdown Default = 11
edge_consistency_width = 11  #\@param {'type':'number'}
if (animation_mode == 'Video Input') and (flow_warp):
  flows = glob(flo_folder+'/*.*')
  if (len(flows)>0) and not force_flow_generation: print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

  if (len(flows)==0) or force_flow_generation:
    ds = flowDataset(in_path, normalize=not use_jit_raft)

    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)<2:
      print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
    if len(frames)>=2:
      if __name__ == '__main__':

        dl = DataLoader(ds, num_workers=num_workers)
        if use_jit_raft:
          if flow_lq:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
          # raft_model = torch.nn.DataParallel(RAFT(args2))
          else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()
          # raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth'))
          # raft_model = raft_model.module.cuda().eval()
        else:
          if raft_model is None or not compile_raft:
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
            from torchvision.models.optical_flow import raft_large, raft_small
            raft_weights = Raft_Large_Weights.C_T_SKHT_V1
            raft_device = "cuda" if torch.cuda.is_available() else "cpu"

            raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device)
            # raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(raft_device)
            raft_model = raft_model.eval()
            if gpu != 'T4' and compile_raft: raft_model = torch.compile(raft_model)
            if flow_lq:
              raft_model = raft_model.half()




        temp_flo = in_path+'_temp_flo'
        # flo_fwd_folder = in_path+'_out_flo_fwd'
        flo_fwd_folder = in_path+f'_out_flo_fwd/{side_x}_{side_y}/'
        for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
          f.unlink()

        os.makedirs(flo_fwd_folder, exist_ok=True)
        os.makedirs(temp_flo, exist_ok=True)
        cc_path = f'{root_dir}/flow_tools/check_consistency.py'
        with torch.no_grad():
          p = Pool(threads)
          for i,batch in enumerate(tqdm(dl)):
              flow_batch(i, batch, p)
          p.close()
          p.join()

        del raft_model, p, dl, ds
        gc.collect()
        if is_colab: locale.getpreferredencoding = getpreferredencoding
        if check_consistency and use_legacy_cc:
          fwd = f"{flo_fwd_folder}/*jpg.npy"
          bwd = f"{flo_fwd_folder}/*jpg_12.npy"

          if reverse_cc_order:
            #old version, may be incorrect
            print('Doing bwd->fwd cc check')
            subprocess.run(['python', "{cc_path}", "--flow_fwd", "{fwd}", "--flow_bwd", "--output", "{flo_fwd_folder}/", "--image_output", "--output_postfix=", "-21_cc", "--blur=0.", "--save_separate_channels", "--skip_numpy_output"], check=True)
            # !python "{cc_path}" --flow_fwd "{fwd}" --flow_bwd "{bwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output
          else:
            print('Doing fwd->bwd cc check')
            subprocess.run(['python', "{cc_path}", "--flow_fwd", "{bwd}", "--flow_bwd", "{fwd}", "--output", "{flo_fwd_folder}/", "--image_output", "--output_postfix=", "-21_cc", "--blur=0.", "--save_separate_channels", "--skip_numpy_output"], check=True)
            # !python "{cc_path}" --flow_fwd "{bwd}" --flow_bwd "{fwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output
          # delete forward flow
          # for f in pathlib.Path(flo_fwd_folder).glob('*jpg_12.npy'):
          #   f.unlink()

flo_imgs = glob(flo_fwd_folder+'/*.jpg.jpg')[:5]
vframes = []
for flo_img in flo_imgs:
  hframes = []
  flo_img = flo_img.replace('\\','/')
  frame = Image.open(videoFramesFolder + '/' + flo_img.split('/')[-1][:-4])
  hframes.append(frame)
  try:
    alpha = Image.open(videoFramesAlpha + '/' + flo_img.split('/')[-1][:-4]).resize(frame.size)
    hframes.append(alpha)
  except:
    pass
  try:
    cc_img = Image.open(flo_img[:-4]+'-21_cc.jpg').convert('L').resize(frame.size)
    hframes.append(cc_img)
  except:
    pass
  try:
    flo_img = Image.open(flo_img).resize(frame.size)
    hframes.append(flo_img)
  except:
    pass
  v_imgs = vstack(hframes)
  vframes.append(v_imgs)
preview = hstack(vframes)
del vframes, hframes
fit(preview, 1024)

"""# Load up a stable.

Don't forget to place your checkpoint at /content/ and change the path accordingly.


You need to log on to https://huggingface.co and

get checkpoints here -
https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
or
https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt

You can pick 1.2 or 1.3 as well, just be sure to grab the "original" flavor.

For v2 go here:
https://huggingface.co/stabilityai/stable-diffusion-2-depth
https://huggingface.co/stabilityai/stable-diffusion-2-base

Inpainting model: https://huggingface.co/runwayml/stable-diffusion-v1-5
"""

#@markdown specify path to your Stable Diffusion checkpoint (the "original" flavor)
#@title define SD + K functions, load model
from safetensors import safe_open
import argparse
import math,os,time
try:
  os.chdir( f'{root_dir}/src/taming-transformers')
  import taming
  os.chdir( f'{root_dir}')
  os.chdir( f'{root_dir}/k-diffusion')
  import k_diffusion as K
  os.chdir( f'{root_dir}')
except:
  import taming
  import k_diffusion as K
import wget
import accelerate
import torch
import torch.nn as nn
from tqdm.notebook import trange, tqdm
sys.path.append('./k-diffusion')



from pytorch_lightning import seed_everything
from k_diffusion.sampling import  sample_euler, sample_euler_ancestral, sample_heun, sample_dpm_2, sample_dpm_2_ancestral, sample_lms, sample_dpm_fast, sample_dpm_adaptive, sample_dpmpp_2s_ancestral,  sample_dpmpp_sde, sample_dpmpp_2m

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from torch import autocast
import numpy as np

from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms

model_urls = {
    "sd_v1_5":"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "dpt_hybrid-midas-501f0c75":"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
}

# https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
control_model_urls = {
    "control_sd15_canny":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "control_sd15_depth":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "control_sd15_softedge":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth", # replaces hed, v11 uses sofftedge  model here
    "control_sd15_mlsd":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth",
    "control_sd15_normalbae":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
    "control_sd15_openpose":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "control_sd15_scribble":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "control_sd15_seg":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "control_sd15_temporalnet":"https://huggingface.co/CiaraRowles/TemporalNet/resolve/main/diff_control_sd15_temporalnet_fp16.safetensors",
    "control_sd15_face":"https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/control_v2p_sd15_mediapipe_face.safetensors",
    "control_sd15_ip2p":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth",
    "control_sd15_inpaint":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
    "control_sd15_lineart":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "control_sd15_lineart_anime":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth",
    "control_sd15_shuffle":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth"
}

def model_to(model, device):
  for param in model.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# import wget
model_version = 'control_multi'#@param ['v1','v1_inpainting','v1_instructpix2pix','v2_512','v2_depth', 'v2_768_v', "control_sd15_canny", "control_sd15_depth","control_sd15_softedge",  "control_sd15_mlsd", "control_sd15_normalbae", "control_sd15_openpose", "control_sd15_scribble", "control_sd15_seg", 'control_multi' ]
if model_version == 'v1' :
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inference.yaml"
if model_version == 'v1_inpainting':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inpainting-inference.yaml"
if model_version == 'v2_512':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference.yaml"
if model_version == 'v2_768_v':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
if model_version == 'v2_depth':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml"
  os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
  if not os.path.exists(f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt"):
    midas_url = model_urls['dpt_hybrid-midas-501f0c75']
    os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
    wget.download(midas_url,  f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt")
    # !wget -O  "{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt" https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt
control_helpers = {
    "control_sd15_canny":None,
    "control_sd15_depth":"dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_softedge":"network-bsds500.pth",
    "control_sd15_mlsd":"mlsd_large_512_fp32.pth",
    "control_sd15_normalbae":"dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_openpose":["body_pose_model.pth", "hand_pose_model.pth"],
    "control_sd15_scribble":None,
    "control_sd15_seg":"upernet_global_small.pth",
    "control_sd15_temporalnet":None,
    "control_sd15_face":None
}



if model_version == 'v1_instructpix2pix':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1_instruct_pix2pix.yaml"
vae_ckpt = '' #@param {'type':'string'}
if vae_ckpt == '': vae_ckpt = None
load_to = 'cpu' #@param ['cpu','gpu']
if load_to == 'gpu': load_to = 'cuda'
quantize = True #@param {'type':'boolean'}
no_half_vae = False #@param {'type':'boolean'}
import gc
def load_model_from_config(config, ckpt, vae_ckpt=None, controlnet=None, verbose=False):
    with torch.no_grad():
      model = instantiate_from_config(config.model).eval().cuda()
      if gpu != 'A100':
        if no_half_vae:
          model.model.half()
          model.cond_stage_model.half()
          model.control_model.half()
        else:
          model.half()
      gc.collect()

      print(f"Loading model from {ckpt}")
      if ckpt.endswith('.safetensors'):
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device=load_to) as f:
          for key in f.keys():
              pl_sd[key] = f.get_tensor(key)
      else: pl_sd = torch.load(ckpt, map_location=load_to)

      if "global_step" in pl_sd:
          print(f"Global Step: {pl_sd['global_step']}")
      if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
      else: sd = pl_sd
      del pl_sd
      gc.collect()

      if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        if vae_ckpt.endswith('.safetensors'):
          vae_sd = {}
          with safe_open(vae_ckpt, framework="pt", device=load_to) as f:
            for key in f.keys():
                vae_sd[key] = f.get_tensor(key)
        else: vae_sd = torch.load(vae_ckpt, map_location=load_to)
        if "state_dict" in vae_sd:
          vae_sd = vae_sd["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }

      m, u = model.load_state_dict(sd, strict=False)
      if len(m) > 0 and verbose:
          print("missing keys:")
          print(m, len(m))
      if len(u) > 0 and verbose:
          print("unexpected keys:")
          print(u, len(u))

      if controlnet is not None:
        ckpt = controlnet
        print(f"Loading model from {ckpt}")
        if ckpt.endswith('.safetensors'):
          pl_sd = {}
          with safe_open(ckpt, framework="pt", device=load_to) as f:
            for key in f.keys():
                pl_sd[key] = f.get_tensor(key)
        else: pl_sd = torch.load(ckpt, map_location=load_to)

        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
          sd = pl_sd["state_dict"]
        else: sd = pl_sd
        del pl_sd
        gc.collect()
        m, u = model.control_model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m, len(m))
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u, len(u))


      return model

import clip
from kornia import augmentation as KA
from torch.nn import functional as F
from resize_right import resize

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

from einops import rearrange, repeat

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
        # with torch.no_grad():
            x = x.detach().requires_grad_()
            # print('x.shape, sigma', x.shape, sigma)
            denoised = model(x, sigma, **kwargs);# print(denoised.requires_grad)
        # with torch.enable_grad():
            # denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach();# print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
        # with torch.no_grad():
            # x = x.detach().requires_grad_()
            # print('x.shape, sigma', x.shape, sigma)
            denoised = model(x, sigma, **kwargs);# print(denoised.requires_grad)
        # with torch.enable_grad():
            # print(sigma**0.5, sigma, sigma**2)
            denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach();# print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn


def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        # print('x.shape, sigma', x.shape, sigma)
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn

def get_image_embed(x):
        if x.shape[2:4] != clip_size:
            x = resize(x, out_shape=clip_size, pad_mode='reflect')
        # print('clip', x.shape)
        # x = clip_normalize(x).cuda()
        x = clip_model.encode_image(x).float()
        return F.normalize(x)

def load_img_sd(path, size):
    # print(type(path))
    # print('load_sd',path)

    image = Image.open(path).convert("RGB")
    # print(f'loaded img with size {image.size}')
    image = image.resize(size, resample=Image.LANCZOS)
    # w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

    # image = image.resize((w, h), resample=Image.LANCZOS)
    if VERBOSE: print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# import lpips
# lpips_model = lpips.LPIPS(net='vgg').to(device)
batch_size = 1 #max batch size
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, image_cond=None,
                prompt_weights=None, prompt_masks=None):
        cond = prompt_parser.reconstruct_cond_batch(cond, 0)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)
        batch_size = sd_batch_size
        # print('batch size in cfgd ', batch_size, sd_batch_size)
        cond_uncond_size = cond.shape[0]+uncond.shape[0]
        # print('cond_uncond_size',cond_uncond_size)
        x_in = torch.cat([x] * batch_size)
        sigma_in = torch.cat([sigma] * batch_size)
        # print('cond.shape, uncond.shape', cond.shape, uncond.shape)
        cond_in = torch.cat([uncond, cond])
        res = None
        uc_mask_shape = torch.ones(cond_in.shape[0], device=cond_in.device)
        uc_mask_shape[0] = 0
        # sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[]
        if prompt_weights is None:
          prompt_weights = [1.]*cond.shape[0]
        if prompt_weights is not None:
          assert len(prompt_weights) >= cond.shape[0], 'The number of prompts is more than prompt weigths.'
          prompt_weights = prompt_weights[:cond.shape[0]]
          prompt_weights = torch.tensor(prompt_weights).to(cond.device)
          prompt_weights = prompt_weights/prompt_weights.sum()

        if prompt_masks is not None:
          print('Using masked prompts')
          assert len(prompt_masks) == cond.shape[0], 'The number of masks doesn`t match the number of prompts-1.'
          prompt_masks = torch.tensor(prompt_masks).to(cond.device)
          # print('prompt_masks', prompt_masks.shape)
          # we use masks so that the 1st mask is full white, and others are applied on top of it

        n_batches = cond_uncond_size//batch_size if cond_uncond_size % batch_size == 0 else (cond_uncond_size//batch_size)+1
        # print('n_batches',n_batches)
        if image_cond is None:
          for i in range(n_batches):
            sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[i*batch_size:(i+1)*batch_size]
            pred = self.inner_model(x_in[i*batch_size:(i+1)*batch_size], sigma_in[i*batch_size:(i+1)*batch_size], cond=cond_in[i*batch_size:(i+1)*batch_size])
            res = pred if res is None else torch.cat([res, pred])
          uncond, cond = res[0:1], res[1:]

          #we can use either weights or masks
          if prompt_masks is None:
            cond = (cond * prompt_weights[:, None, None, None]).sum(dim=0, keepdim=True)
          else:
            cond_out = cond[0]
            for i in range(len(cond)):
              if i == 0: continue
              cond_out = (cond[i]*prompt_masks[i] + cond_out*(1-prompt_masks[i]))
            cond = cond_out[None,...]
            del cond_out

          return uncond + (cond - uncond) * cond_scale
        else:
          if model_version != 'control_multi':
            if img_zero_uncond:
              img_in = torch.cat([torch.zeros_like(image_cond),
                                          image_cond.repeat(cond.shape[0],1,1,1)])
            else:
              img_in = torch.cat([image_cond]*(1+cond.shape[0]))

            for i in range(n_batches):
              sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[i*batch_size:(i+1)*batch_size]
              cond_dict = {"c_crossattn": [cond_in[i*batch_size:(i+1)*batch_size]], 'c_concat':[img_in[i*batch_size:(i+1)*batch_size]]}
              pred = self.inner_model(x_in[i*batch_size:(i+1)*batch_size], sigma_in[i*batch_size:(i+1)*batch_size], cond=cond_dict)
              res = pred if res is None else torch.cat([res, pred])
            uncond, cond = res[0:1], res[1:]

            if prompt_masks is None:
              cond = (cond * prompt_weights[:, None, None, None]).sum(dim=0, keepdim=True)
            else:
              cond_out = cond[0]
              for i in range(len(cond)):
                if i == 0: continue
                cond_out = (cond[i]*prompt_masks[i] + cond_out*(1-prompt_masks[i]))
              cond = cond_out[None,...]
              del cond_out

            return uncond + (cond - uncond) * cond_scale


          if model_version == 'control_multi' and controlnet_multimodel_mode != 'external':
            img_in = {}
            for key in image_cond.keys():
              if img_zero_uncond or key == 'control_sd15_shuffle':
                img_in[key] = torch.cat([torch.zeros_like(image_cond[key]),
                                          image_cond[key].repeat(cond.shape[0],1,1,1)])
              else:
                img_in[key] = torch.cat([image_cond[key]]*(1+cond.shape[0]))

            for i in range(n_batches):
              sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[i*batch_size:(i+1)*batch_size]
              batch_img_in = {}
              for key in img_in.keys():

                batch_img_in[key] = img_in[key][i*batch_size:(i+1)*batch_size]
                # print('img_in[key].shape, batch_img_in[key]',img_in[key].shape, batch_img_in[key].shape)
              cond_dict = {
                  "c_crossattn":  [cond_in[i*batch_size:(i+1)*batch_size]],
                  'c_concat':  batch_img_in,
                  'controlnet_multimodel':controlnet_multimodel,
                  'loaded_controlnets':loaded_controlnets
                  }
              x_in = torch.cat([x]*cond_dict["c_crossattn"][0].shape[0])
              sigma_in = torch.cat([sigma]*cond_dict["c_crossattn"][0].shape[0])
              # print(x_in.shape, cond_dict["c_crossattn"][0].shape)
              pred = self.inner_model(x_in,
                                     sigma_in, cond=cond_dict)
              # print(pred.shape)
              res = pred if res is None else torch.cat([res, pred])
              # print(res.shape)
              gc.collect()
              torch.cuda.empty_cache()
            # print('res shape', res.shape)
            uncond, cond = res[0:1], res[1:]
            if prompt_masks is None:
              # print('cond shape', cond.shape, prompt_weights[:, None, None, None].shape)
              cond = (cond * prompt_weights[:, None, None, None]).sum(dim=0, keepdim=True)
            else:
              cond_out = cond[0]
              for i in range(len(cond)):
                if i == 0: continue
                cond_out = (cond[i]*prompt_masks[i] + cond_out*(1-prompt_masks[i]))
              cond = cond_out[None,...]
              del cond_out
              # print('cond.shape', cond.shape)

            return uncond + (cond - uncond) * cond_scale
          if model_version == 'control_multi' and controlnet_multimodel_mode == 'external':

              #wormalize weights
              weights = np.array([controlnet_multimodel[m]["weight"] for m in controlnet_multimodel.keys()])
              weights = weights/weights.sum()
              result = None
              # print(weights)
              for i,controlnet in enumerate(controlnet_multimodel.keys()):
                try:
                  if img_zero_uncond  or controlnet == 'control_sd15_shuffle':
                    img_in = torch.cat([torch.zeros_like(image_cond[controlnet]),
                                          image_cond[controlnet].repeat(cond.shape[0],1,1,1)])
                  else:
                    img_in = torch.cat([image_cond[controlnet]]*(1+cond.shape[0]))
                except:
                  pass

                if weights[i]!=0:
                  controlnet_settings = controlnet_multimodel[controlnet]

                  self.inner_model.inner_model.control_model = loaded_controlnets[controlnet]
                  for i in range(n_batches):
                    sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[i*batch_size:(i+1)*batch_size]
                    cond_dict = {"c_crossattn": [cond_in[i*batch_size:(i+1)*batch_size]], 'c_concat':[img_in[i*batch_size:(i+1)*batch_size]]}
                    pred = self.inner_model(x_in[i*batch_size:(i+1)*batch_size], sigma_in[i*batch_size:(i+1)*batch_size], cond=cond_dict)
                    gc.collect()
                    res = pred if res is None else torch.cat([res, pred])

                  uncond, cond = res[0:1], res[1:]
                  # uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in],
                  #                                                     'c_concat': [img_in]}).chunk(2)
                  if prompt_masks is None:
                    cond = (cond * prompt_weights[:, None, None, None]).sum(dim=0, keepdim=True)
                  else:
                    cond_out = cond[0]
                    for i in range(len(cond)):
                      if i == 0: continue
                      cond_out = (cond[i]*prompt_masks[i] + cond_out*(1-prompt_masks[i]))
                    cond = cond_out[None,...]
                    del cond_out

                  if result is None:
                    result = (uncond + (cond - uncond) * cond_scale)*weights[i]
                  else: result = result + (uncond + (cond - uncond) * cond_scale)*weights[i]
              return result




import einops
class InstructPix2PixCFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, cond_scale, image_scale, image_cond):
        # c = cond
        # uc = uncond
        c = prompt_parser.reconstruct_cond_batch(cond, 0)
        uc = prompt_parser.reconstruct_cond_batch(uncond, 0)
        text_cfg_scale = cond_scale
        image_cfg_scale  = image_scale
        # print(image_cond)
        cond = {}
        cond["c_crossattn"] = [c]
        cond["c_concat"] = [image_cond]

        uncond = {}
        uncond["c_crossattn"] = [uc]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)

        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

dynamic_thresh = 2.
device = 'cuda'
# config_path = f"{root_dir}/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
model_path = "model.safetensors" #@param {'type':'string'}
import pickle
#@markdown ---
#@markdown ControlNet download settings
#@markdown ControlNet downloads are managed by controlnet_multi settings in Main settings tab.
use_small_controlnet  = True
# #@param {'type':'boolean'}
small_controlnet_model_path = ''
# #@param {'type':'string'}
download_control_model = True
# #@param {'type':'boolean'}
force_download = False #@param {'type':'boolean'}
controlnet_models_dir = "models/ControlNet" #@param {'type':'string'}
if not is_colab and (controlnet_models_dir.startswith('/content') or controlnet_models_dir=='' or controlnet_models_dir is None):
  controlnet_models_dir = f"{root_dir}/ControlNet/models"
  print('You have a controlnet path set up for google drive, but we are not on Colab. Defaulting controlnet model path to ', controlnet_models_dir)
os.makedirs(controlnet_models_dir, exist_ok=True)
#@markdown ---

control_sd15_canny  = False
control_sd15_depth  = False
control_sd15_softedge  = True
control_sd15_mlsd  = False
control_sd15_normalbae  = False
control_sd15_openpose  = False
control_sd15_scribble  = False
control_sd15_seg  = False
control_sd15_temporalnet  = False
control_sd15_face  = False

if model_version == 'control_multi':
  control_versions = []
  if control_sd15_canny: control_versions+=['control_sd15_canny']
  if control_sd15_depth: control_versions+=['control_sd15_depth']
  if control_sd15_softedge: control_versions+=['control_sd15_softedge']
  if control_sd15_mlsd: control_versions+=['control_sd15_mlsd']
  if control_sd15_normalbae: control_versions+=['control_sd15_normalbae']
  if control_sd15_openpose: control_versions+=['control_sd15_openpose']
  if control_sd15_scribble: control_versions+=['control_sd15_scribble']
  if control_sd15_seg: control_versions+=['control_sd15_seg']
  if control_sd15_temporalnet: control_versions+=['control_sd15_temporalnet']
  if control_sd15_face: control_versions+=['control_sd15_face']
else: control_versions = [model_version]

if model_version in ["control_sd15_canny",
                     "control_sd15_depth",
                     "control_sd15_softedge",
                     "control_sd15_mlsd",
                     "control_sd15_normalbae",
                     "control_sd15_openpose",
                     "control_sd15_scribble",
                     "control_sd15_seg", 'control_sd15_face', 'control_multi']:


  os.chdir(f"{root_dir}/ControlNet/")
  from annotator.util import resize_image, HWC3

  from cldm.model import create_model, load_state_dict
  os.chdir('../')


  #if download model is on and model path is not found, download full controlnet
  if download_control_model:
    if not os.path.exists(model_path):
      print(f'Model not found at {model_path}')
      if model_version == 'control_multi': model_ver = control_versions[0]
      else: model_ver = model_version

      model_path = f"{controlnet_models_dir}/v1-5-pruned-emaonly.safetensors"
      model_url = model_urls["sd_v1_5"]
      if not os.path.exists(model_path) or force_download:
        try:
          pathlib.Path(model_path).unlink()
        except: pass
        print('Downloading full sd v1.5 model to ', model_path)
        wget.download(model_url,  model_path)
        print('Downloaded full model.')
    #if model found, assume it's a working checkpoint, download small controlnet only:

    for model_ver in control_versions:
      small_url = control_model_urls[model_ver]
      local_filename = small_url.split('/')[-1]
      small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
      if use_small_controlnet and os.path.exists(model_path) and not os.path.exists(small_controlnet_model_path):
        print(f'Model found at {model_path}. Small model not found at {small_controlnet_model_path}.')

        if not os.path.exists(small_controlnet_model_path) or force_download:
          try:
            pathlib.Path(small_controlnet_model_path).unlink()
          except: pass
          print(f'Downloading small controlnet model from {small_url}... ')
          wget.download(small_url,  small_controlnet_model_path)
          print('Downloaded small controlnet model.')

      #https://huggingface.co/lllyasviel/Annotators/tree/main
      #https://huggingface.co/lllyasviel/Annotators/resolve/main/150_16_swin_l_oneformer_coco_100ep.pth
      helper_names = control_helpers[model_ver]
      if helper_names is not None:
          if type(helper_names) == str: helper_names = [helper_names]
          for helper_name in helper_names:
            helper_model_url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/'+helper_name
            helper_model_path = f'{root_dir}/ControlNet/annotator/ckpts/'+helper_name
            if not os.path.exists(helper_model_path) or force_download:
              try:
                pathlib.Path(helper_model_path).unlink()
              except: pass
              wget.download(helper_model_url, helper_model_path)
  assert os.path.exists(model_path), f'Model not found at path: {model_path}. Please enter a valid path to the checkpoint file.'

  if os.path.exists(small_controlnet_model_path):
    smallpath = small_controlnet_model_path
  else:
    smallpath = None
  config = OmegaConf.load(f"{root_dir}/ControlNet/models/cldm_v15.yaml")
  sd_model = load_model_from_config(config=config,
                                    ckpt=model_path, vae_ckpt=vae_ckpt, #controlnet=smallpath,
                                    verbose=True)

  #legacy
  # sd_model = create_model(f"{root_dir}/ControlNet/models/cldm_v15.yaml").cuda()
  # sd_model.load_state_dict(load_state_dict(model_path, location=load_to), strict=False)
  sd_model.cond_stage_model.half()
  sd_model.model.half()
  sd_model.control_model.half()
  sd_model.cuda()

  gc.collect()
else:
  assert os.path.exists(model_path), f'Model not found at path: {model_path}. Please enter a valid path to the checkpoint file.'
  if model_path.endswith('.pkl'):
      with open(model_path, 'rb') as f:
        sd_model = pickle.load(f).cuda().eval()
        if gpu == 'A100':
          sd_model = sd_model.float()
  else:
      config = OmegaConf.load(config_path)
      sd_model = load_model_from_config(config, model_path, vae_ckpt=vae_ckpt, verbose=True).cuda()

sys.path.append('./stablediffusion/')
from modules import prompt_parser, sd_hijack
# sd_model.first_stage_model = torch.compile(sd_model.first_stage_model)
# sd_model.model = torch.compile(sd_model.model)
if sd_model.parameterization == "v":
  model_wrap = K.external.CompVisVDenoiser(sd_model, quantize=quantize )
else:
  model_wrap = K.external.CompVisDenoiser(sd_model, quantize=quantize)
sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
model_wrap_cfg = CFGDenoiser(model_wrap)
if model_version == 'v1_instructpix2pix':
  model_wrap_cfg = InstructPix2PixCFGDenoiser(model_wrap)

#@markdown If you're having crashes (CPU out of memory errors) while running this cell on standard colab env, consider saving the model as pickle.\
#@markdown You can save the pickled model on your google drive and use it instead of the usual stable diffusion model.\
#@markdown To do that, run the notebook with a high-ram env, run all cells before and including this cell as well, and save pickle in the next cell. Then you can switch to a low-ram env and load the pickled model.

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1, inpainting_mask_weight=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    if mask is not None:
      mask = np.array(mask.convert("L"))
      mask = mask.astype(np.float32)/255.0
      mask = mask[None,None]
      mask[mask < 0.5] = 0
      mask[mask >= 0.5] = 1
      mask = torch.from_numpy(mask)
    else:
      mask = image.new_ones(1, 1, *image.shape[-2:])

    # masked_image = image * (mask < 0.5)

    masked_image = torch.lerp(
            image,
            image * (mask < 0.5),
            inpainting_mask_weight
        )

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def inpainting_conditioning(source_image, image_mask = None, inpainting_mask_weight = 1, sd_model=sd_model):
        #based on https://github.com/AUTOMATIC1111/stable-diffusion-webui

        # Handle the different mask inputs
        if image_mask is not None:

            if torch.is_tensor(image_mask):

                conditioning_mask = image_mask[:,:1,...]
                # print('mask conditioning_mask', conditioning_mask.shape)
            else:
                print(image_mask.shape, source_image.shape)
                # conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = image_mask[...,0].astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None]).float()

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])
        print(conditioning_mask.shape, source_image.shape)
        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(source_image.device).to(source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            inpainting_mask_weight
        )

        # Encode the new masked image using first stage of network.
        conditioning_image =  sd_model.get_first_stage_encoding( sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=conditioning_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to('cuda').type( sd_model.dtype)

        return image_conditioning

import torch
# divisible by 8 fix from AUTOMATIC1111
def cat8(tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


def cldm_forward(x, timesteps=None, context=None, control=None, only_mid_control=False, self = sd_model.model.diffusion_model,**kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = cat8([h, hs.pop()], dim=1)
            else:
                h = cat8([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)
try:
  sd_model.model.diffusion_model.forward = cldm_forward
except Exception as e:
  print(e)
  # pass

#@title Tiled VAE
#@markdown Enable if you're getting CUDA Out of memory errors during encode_first_stage or decode_fiirst_stage.
#@markdown Is slower.
#tiled vae from thttps://github.com/CompVis/latent-diffusion

use_tiled_vae = False #@param {'type':'boolean'}
tile_size = 128 #\@param {'type':'number'}
stride = 96 #\@param {'type':'number'}
num_tiles = [2,2] #@param {'type':'raw'}
padding = [0.5,0.5] #\@param {'type':'raw'}

if num_tiles in [0, '', None]:
  num_tiles = None

if padding in [0, '', None]:
  padding = None
def get_fold_unfold( x, kernel_size, stride, uf=1, df=1, self=sd_model):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[1] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[1] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        normalization = torch.where(normalization==0.,1e-6, normalization)
        return fold, unfold, normalization, weighting

#non divisible by 8 fails here

@torch.no_grad()
def encode_first_stage(x, self=sd_model):
        ts = time.time()
        if hasattr(self, "split_input_params"):

            if self.split_input_params["patch_distributed_vq"]:
                print('------using tiled vae------')
                bs, nc, h, w = x.shape
                df = self.split_input_params["vqf"]
                if self.split_input_params["num_tiles"] is not None:
                  num_tiles = self.split_input_params["num_tiles"]
                  ks = [h//num_tiles[0], w//num_tiles[1]]
                else:
                  ks = self.split_input_params["ks"]  # eg. (128, 128)
                  ks = [o*(df) for o in ks]
                # ks = self.split_input_params["ks"]  # eg. (128, 128)
                # ks = [o*df for o in ks]


                if self.split_input_params["padding"] is not None:
                  padding = self.split_input_params["padding"]
                  stride = [int(ks[0]*padding[0]), int(ks[1]*padding[1])]
                else:
                  stride = self.split_input_params["stride"]  # eg. (64, 64)
                  stride = [o*(df) for o in stride]
                # stride = self.split_input_params["stride"]  # eg. (64, 64)
                # stride = [o*df for o in stride]
                # ks = [512,512]
                # stride = [512,512]


                # print('kernel, stride', ks, stride)

                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape

                target_h = math.ceil(h/ks[0])*ks[0]
                target_w = math.ceil(w/ks[1])*ks[1]
                padh = target_h - h
                padw = target_w - w
                pad = (0, padw, 0, padh)
                if target_h != h or target_w != w:
                  print('Padding.')
                  # print('padding from ', h, w, 'to ', target_h, target_w)
                  x = torch.nn.functional.pad(x, pad, mode='reflect')
                  # print('padded from ', h, w, 'to ', z.shape[2:])

                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # print('z', z.shape)
                output_list = [self.get_first_stage_encoding(self.first_stage_model.encode(z[:, :, :, :, i]), tiled_vae_call=True)
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                # print('o', o.shape)
                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                print('Tiled vae encoder took ', f'{time.time()-ts:.2}')
                # print('decoded stats', decoded.min(), decoded.max(), decoded.std(), decoded.mean())
                return decoded[...,:h//df, :w//df]

            else:
                print('Vae encoder took ', f'{time.time()-ts:.2}')
                # print('x stats', x.min(), x.max(), x.std(), x.mean())
                return self.first_stage_model.encode(x)
        else:
            print('Vae encoder took ', f'{time.time()-ts:.2}')
            # print('x stats', x.min(), x.max(), x.std(), x.mean())
            return self.first_stage_model.encode(x)

@torch.no_grad()
def decode_first_stage(z, predict_cids=False, force_not_quantize=False, self=sd_model):
        ts = time.time()
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            print('------using tiled vae------')
            # print('latent shape: ', z.shape)
            # print(self.split_input_params)
            if self.split_input_params["patch_distributed_vq"]:
                bs, nc, h, w = z.shape
                if self.split_input_params["num_tiles"] is not None:
                  num_tiles = self.split_input_params["num_tiles"]
                  ks = [h//num_tiles[0], w//num_tiles[1]]
                else:
                  ks = self.split_input_params["ks"]  # eg. (128, 128)

                if self.split_input_params["padding"] is not None:
                  padding = self.split_input_params["padding"]
                  stride = [int(ks[0]*padding[0]), int(ks[1]*padding[1])]
                else:
                  stride = self.split_input_params["stride"]  # eg. (64, 64)

                uf = self.split_input_params["vqf"]



                target_h = math.ceil(h/ks[0])*ks[0]
                target_w = math.ceil(w/ks[1])*ks[1]
                padh = target_h - h
                padw = target_w - w
                pad = (0, padw, 0, padh)
                if target_h != h or target_w != w:
                  print('Padding.')
                  # print('padding from ', h, w, 'to ', target_h, target_w)
                  z = torch.nn.functional.pad(z, pad, mode='reflect')
                  # print('padded from ', h, w, 'to ', z.shape[2:])


                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")



                # print(ks, stride)
                fold, unfold, normalization, weighting = get_fold_unfold(z, ks, stride, uf=uf)


                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # print('z unfold, normalization, weighting',z.shape, normalization.shape, weighting.shape)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # print('z unfold view , normalization, weighting',z.shape)
                # 2. apply model loop over last dim
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                # print('out stack', o.shape)

                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                print('Tiled vae decoder took ', f'{time.time()-ts:.2}')
                # print('decoded stats', decoded.min(), decoded.max(), decoded.std(), decoded.mean())
                # assert False
                return decoded[...,:h*uf, :w*uf]
            else:
                print('Vae decoder took ', f'{time.time()-ts:.2}')
                # print('z stats', z.min(), z.max(), z.std(), z.mean())
                return self.first_stage_model.decode(z)

        else:
            # print('z stats', z.min(), z.max(), z.std(), z.mean())
            print('Vae decoder took ', f'{time.time()-ts:.2}')
            return self.first_stage_model.decode(z)



def get_first_stage_encoding(encoder_posterior, self=sd_model, tiled_vae_call=False):
        if hasattr(self, "split_input_params") and not tiled_vae_call:
          #pass for tiled vae
          return encoder_posterior

        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

if use_tiled_vae:

        ks = tile_size
        stride = stride
        vqf = 8  #
        split_input_params = {"ks": (ks,ks), "stride": (stride, stride),
                                       "num_tiles": num_tiles, "padding": padding,
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
        bkup_decode_first_stage = sd_model.decode_first_stage
        bkup_encode_first_stage = sd_model.encode_first_stage
        bkup_get_first_stage_encoding = sd_model.get_first_stage_encoding
        bkup_get_fold_unfold = sd_model.get_fold_unfold

        sd_model.split_input_params = split_input_params
        sd_model.decode_first_stage = decode_first_stage
        sd_model.encode_first_stage = encode_first_stage
        sd_model.get_first_stage_encoding = get_first_stage_encoding
        sd_model.get_fold_unfold = get_fold_unfold

else:
        if hasattr(sd_model, "split_input_params"):
          delattr(sd_model, "split_input_params")
          try:
            sd_model.decode_first_stage = bkup_decode_first_stage
            sd_model.encode_first_stage = bkup_encode_first_stage
            sd_model.get_first_stage_encoding = bkup_get_first_stage_encoding
            sd_model.get_fold_unfold = bkup_get_fold_unfold
          except: pass

#@title Save loaded model
#@markdown For this cell to work you need to load model in the previous cell.\
#@markdown Saves an already loaded model as an object file, that weights less, loads faster, and requires less CPU RAM.\
#@markdown After saving model as pickle, you can then load it as your usual stable diffusion model in thecell above.\
#@markdown The model will be saved under the same name with .pkl extenstion.
save_model_pickle = False #@param {'type':'boolean'}
save_folder = "models" #@param {'type':'string'}
if save_folder != '' and save_model_pickle:
  os.makedirs(save_folder, exist_ok=True)
  out_path = save_folder+model_path.replace('\\', '/').split('/')[-1].split('.')[0]+'.pkl'
  with open(out_path, 'wb') as f:
    pickle.dump(sd_model, f)
  print('Model successfully saved as: ',out_path)

"""# CLIP guidance"""

#@title CLIP guidance settings
#@markdown You can use clip guidance to further push style towards your text input.\
#@markdown Please note that enabling it (by using clip_guidance_scale>0) will greatly increase render times and VRAM usage.\
#@markdown For now it does 1 sample of the whole image per step (similar to 1 outer_cut in discodiffusion).

# clip_type, clip_pretrain = 'ViT-B-32-quickgelu', 'laion400m_e32'
# clip_type, clip_pretrain ='ViT-L-14', 'laion2b_s32b_b82k'
clip_type = 'ViT-H-14' #@param ['ViT-L-14','ViT-B-32-quickgelu', 'ViT-H-14']
if clip_type == 'ViT-H-14' : clip_pretrain = 'laion2b_s32b_b79k'
if clip_type == 'ViT-L-14' : clip_pretrain = 'laion2b_s32b_b82k'
if clip_type == 'ViT-B-32-quickgelu' : clip_pretrain = 'laion400m_e32'

clip_guidance_scale = 0 #@param {'type':"number"}
if clip_guidance_scale > 0:
  clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_type, pretrained=clip_pretrain)
  _=clip_model.half().cuda().eval()
  clip_size = clip_model.visual.image_size
  for param in clip_model.parameters():
      param.requires_grad = False
else:
  try:
    del clip_model
    gc.collect()
  except: pass

"""# Automatic Brightness Adjustment"""

#@markdown ###Automatic Brightness Adjustment
#@markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
#@markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
#@markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
#@markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds


#@markdown The idea comes from https://github.com/lowfuel/progrockdiffusion

enable_adjust_brightness = False #@param {'type':'boolean'}
high_brightness_threshold = 180 #@param {'type':'number'}
high_brightness_adjust_ratio = 0.97 #@param {'type':'number'}
high_brightness_adjust_fix_amount = 2 #@param {'type':'number'}
max_brightness_threshold = 254 #@param {'type':'number'}
low_brightness_threshold = 40 #@param {'type':'number'}
low_brightness_adjust_ratio = 1.03 #@param {'type':'number'}
low_brightness_adjust_fix_amount = 2 #@param {'type':'number'}
min_brightness_threshold = 1 #@param {'type':'number'}

"""# Content-aware scheduling"""

#@title Content-aware scheduing
#@markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
#@markdown rmse function is faster than lpips, but less precise.\
#@markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.

def load_img_lpips(path, size=(512,512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    # print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda()

diff = None
analyze_video = False #@param {'type':'boolean'}

diff_function = 'lpips' #@param ['rmse','lpips','rmse+lpips']

def l1_loss(x,y):
  return torch.sqrt(torch.mean((x-y)**2))


def rmse(x,y):
  return torch.abs(torch.mean(x-y))

def joint_loss(x,y):
  return rmse(x,y)*lpips_model(x,y)

diff_func = rmse
if  diff_function == 'lpips':
  diff_func = lpips_model
if diff_function == 'rmse+lpips':
  diff_func = joint_loss

if analyze_video:
  diff = [0]
  frames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
  from tqdm.notebook import trange
  for i in trange(1,len(frames)):
    with torch.no_grad():
      diff.append(diff_func(load_img_lpips(frames[i-1]), load_img_lpips(frames[i])).sum().mean().detach().cpu().numpy())

  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams["figure.figsize"] = [12.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  y = diff
  plt.title(f"{diff_function} frame difference")
  plt.plot(y, color="red")
  calc_thresh = np.percentile(np.array(diff), 97)
  plt.axhline(y=calc_thresh, color='b', linestyle='dashed')

  plt.show()
  print(f'suggested threshold: {calc_thresh.round(2)}')

#@title Plot threshold vs frame difference
#@markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
if diff is not None:
  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams["figure.figsize"] = [12.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  y = diff
  plt.title(f"{diff_function} frame difference")
  plt.plot(y, color="red")
  calc_thresh = np.percentile(np.array(diff), 97)
  plt.axhline(y=calc_thresh, color='b', linestyle='dashed')
  user_threshold = 0.13 #@param {'type':'raw'}
  plt.axhline(y=user_threshold, color='r')

  plt.show()
  peaks = []
  for i,d in enumerate(diff):
    if d>user_threshold:
      peaks.append(i)
  print(f'Peaks at frames: {peaks} for user_threshold of {user_threshold}')
else: print('Please analyze frames in the previous cell  to plot graph')

, threshold
#@title Create schedules from frame difference
def adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched=None):
  diff_array = np.array(diff)

  diff_new = np.zeros_like(diff_array)
  diff_new = diff_new+normal_val

  for i in range(len(diff_new)):
    el = diff_array[i]
    if sched is not None:
      diff_new[i] = get_scheduled_arg(i, sched)
    if el>thresh or i==0:
      diff_new[i] = new_scene_val
      if falloff_frames>0:
        for j in range(falloff_frames):
          if i+j>len(diff_new)-1: break
          # print(j,(falloff_frames-j)/falloff_frames, j/falloff_frames )
          falloff_val = normal_val
          if sched is not None:
            falloff_val = get_scheduled_arg(i+falloff_frames, sched)
          diff_new[i+j] = new_scene_val*(falloff_frames-j)/falloff_frames+falloff_val*j/falloff_frames
  return diff_new

def check_and_adjust_sched(sched, template, diff, respect_sched=True):
  if template is None or template == '' or template == []:
    return sched
  normal_val, new_scene_val, thresh, falloff_frames = template
  sched_source = None
  if respect_sched:
    sched_source = sched
  return list(adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched_source).astype('float').round(3))

#@markdown fill in templates for schedules you'd like to create from frames' difference\
#@markdown leave blank to use schedules from previous cells\
#@markdown format: **[normal value, high difference value, difference threshold, falloff from high to normal (number of frames)]**\
#@markdown For example, setting flow blend template to [0.999, 0.3, 0.5, 5] will use 0.999 everywhere unless a scene has changed (frame difference >0.5) and then set flow_blend for this frame to 0.3 and gradually fade to 0.999 in 5 frames

latent_scale_template = '' #@param {'type':'raw'}
init_scale_template = '' #@param {'type':'raw'}
steps_template = '' #@param {'type':'raw'}
style_strength_template = [0.8, 0.8, 0.5, 5] #@param {'type':'raw'}
flow_blend_template = [1, 0., 0.5, 2] #@param {'type':'raw'}
cfg_scale_template = None #@param {'type':'raw'}
image_scale_template = None #@param {'type':'raw'}

#@markdown Turning this off will disable templates and will use schedules set in previous cell
make_schedules = False #@param {'type':'boolean'}
#@markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
respect_sched = True #@param {'type':'boolean'}
diff_override = [] #@param {'type':'raw'}

#shift+1 required

"""# Frame Captioning

"""

#@title Generate captions for keyframes
#@markdown Automatically generate captions for every n-th frame, \
#@markdown or keyframe list: at keyframe, at offset from keyframe, between keyframes.\
#@markdown keyframe source: Every n-th frame, user-input, Content-aware scheduling keyframes
inputFrames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
make_captions = False #@param {'type':'boolean'}
keyframe_source = 'Every n-th frame' #@param ['Content-aware scheduling keyframes', 'User-defined keyframe list', 'Every n-th frame']
#@markdown This option only works with  keyframe source == User-defined keyframe list
user_defined_keyframes = [3,4,5] #@param
#@markdown This option only works with  keyframe source == Content-aware scheduling keyframes
diff_thresh = 0.33 #@param {'type':'number'}
#@markdown This option only works with  keyframe source == Every n-th frame
nth_frame = 60 #@param {'type':'number'}
if keyframe_source == 'Content-aware scheduling keyframes':
  if diff in [None, '', []]:
    print('ERROR: Keyframes were not generated. Please go back to Content-aware scheduling cell, enable analyze_video nad run it or choose a different caption keyframe source.')
    caption_keyframes = None
  else:
    caption_keyframes = [1]+[i+1 for i,o in enumerate(diff) if o>=diff_thresh]
if keyframe_source == 'User-defined keyframe list':
  caption_keyframes = user_defined_keyframes
if keyframe_source == 'Every n-th frame':
  caption_keyframes = list(range(1, len(inputFrames), nth_frame))
#@markdown Remaps keyframes based on selected offset mode
offset_mode = 'Fixed' #@param ['Fixed', 'Between Keyframes', 'None']
#@markdown Only works with offset_mode == Fixed
fixed_offset = 0 #@param {'type':'number'}

videoFramesCaptions = videoFramesFolder+'Captions'
if make_captions and caption_keyframes is not None:
  try:
    blip_model
  except:

    os.chdir('./BLIP')
    from models.blip import blip_decoder
    os.chdir('../')
    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'# -O /content/model_base_caption_capfilt_large.pth'

    blip_model = blip_decoder(pretrained=model_url, image_size=384, vit='base',med_config='./BLIP/configs/med_config.json')
    blip_model.eval()
    blip_model = blip_model.to(device)
  finally:
    print('Using keyframes: ', caption_keyframes[:20], ' (first 20 keyframes displyed')
    if offset_mode == 'None':
      keyframes = caption_keyframes
    if offset_mode == 'Fixed':
      keyframes = caption_keyframes
      for i in range(len(caption_keyframes)):
        if keyframes[i] >= max(caption_keyframes):
          keyframes[i] = caption_keyframes[i]
        else: keyframes[i] = min(caption_keyframes[i]+fixed_offset, caption_keyframes[i+1])
      print('Remapped keyframes to ', keyframes[:20])
    if offset_mode == 'Between Keyframes':
      keyframes = caption_keyframes
      for i in range(len(caption_keyframes)):
        if keyframes[i] >= max(caption_keyframes):
          keyframes[i] = caption_keyframes[i]
        else:
          keyframes[i] = caption_keyframes[i] + int((caption_keyframes[i+1]-caption_keyframes[i])/2)
      print('Remapped keyframes to ', keyframes[:20])

    videoFramesCaptions = videoFramesFolder+'Captions'
    createPath(videoFramesCaptions)


  from tqdm.notebook import trange

  for f in pathlib.Path(videoFramesCaptions).glob('*.txt'):
          f.unlink()
  for i in tqdm(keyframes):

    with torch.no_grad():
      keyFrameFilename = inputFrames[i-1]
      raw_image = Image.open(keyFrameFilename)
      image = transform(raw_image).unsqueeze(0).to(device)
      caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5)
      captionFilename = os.path.join(videoFramesCaptions, keyFrameFilename.replace('\\','/').split('/')[-1][:-4]+'.txt')
      with open(captionFilename, 'w') as f:
        f.write(caption[0])

def load_caption(caption_file):
    caption = ''
    with open(caption_file, 'r') as f:
      caption = f.read()
    return caption

def get_caption(frame_num):
  caption_files = sorted(glob(os.path.join(videoFramesCaptions,'*.txt')))
  frame_num1 = frame_num+1
  if len(caption_files) == 0:
    return None
  frame_numbers = [int(o.replace('\\','/').split('/')[-1][:-4]) for o in caption_files]
  # print(frame_numbers, frame_num)
  if frame_num1 < frame_numbers[0]:
    return load_caption(caption_files[0])
  if frame_num1 >= frame_numbers[-1]:
    return load_caption(caption_files[-1])
  for i in range(len(frame_numbers)):
    if frame_num1 >= frame_numbers[i] and frame_num1 < frame_numbers[i+1]:
      return load_caption(caption_files[i])
  return None

"""# Render settings

## Non-gui
These settings are used as initial settings for the GUI unless you specify default_settings_path. Then the GUI settings will be loaded from the specified file.
"""

#@title Flow and turbo settings
#@markdown #####**Video Optical Flow Settings:**
flow_warp = True #@param {type: 'boolean'}
#cal optical flow from video frames and warp prev frame with flow
flow_blend =  0.999
##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
check_consistency = True #@param {type: 'boolean'}
 #cal optical flow from video frames and warp prev frame with flow

#======= TURBO MODE
#@markdown ---
#@markdown ####**Turbo Mode:**
#@markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
#@markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
#@markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 1 # frames

#@title Consistency map mixing
#@markdown You can mix consistency map layers separately\
#@markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
#@markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
#@markdown edges_consistency_weight - masks moving objects' edges\
#@markdown The default values to simulate previous versions' behavior are 1,1,1

missed_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}

#@title  ####**Seed and grad Settings:**

set_seed = '4275770367' #@param{type: 'string'}


#@markdown *Clamp grad is used with any of the init_scales or sat_scale above 0*\
#@markdown Clamp grad limits the amount various criterions, controlled by *_scale parameters, are pushing the image towards the desired result.\
#@markdown For example, high scale values may cause artifacts, and clamp_grad removes this effect.
#@markdown 0.7 is a good clamp_max value.
eta = 0.55
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 2 #@param{type: 'number'}

"""### Prompts
`animation_mode: None` will only use the first set. `animation_mode: 2D / Video` will run through them per the set frames and hold on the last one.
"""

text_prompts = {0: ['a beautiful highly detailed cyberpunk mechanical \
augmented most beautiful (woman) ever, cyberpunk 2077, neon, dystopian, \
hightech, trending on artstation']}

negative_prompts = {
    0: ["text, naked, nude, logo, cropped, two heads, four arms, lazy eye, blurry, unfocused"]
}

"""### Warp Turbo Smooth Settings

turbo_frame_skips_steps - allows to set different frames_skip_steps for turbo frames. None means turbo frames are warped only without diffusion

soften_consistency_mask - clip the lower values of consistency mask to this value. Raw video frames will leak stronger with lower values.

soften_consistency_mask_for_turbo_frames - same, but for turbo frames
"""

#@title ##Warp Turbo Smooth Settings
#@markdown Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.
turbo_frame_skips_steps = '100% (don`t diffuse turbo frames, fastest)' #@param ['70%','75%','80%','85%', '90%', '95%', '100% (don`t diffuse turbo frames, fastest)']

if turbo_frame_skips_steps == '100% (don`t diffuse turbo frames, fastest)':
  turbo_frame_skips_steps = None
else:
  turbo_frame_skips_steps = int(turbo_frame_skips_steps.split('%')[0])/100
#None - disable and use default skip steps

#@markdown ###Consistency mask postprocessing
#@markdown ####Soften consistency mask
#@markdown Lower values mean less stylized frames and more raw video input in areas with fast movement, but fewer trails add ghosting.\
#@markdown Gives glitchy datamoshing look.\
#@markdown Higher values keep stylized frames, but add trails and ghosting.

soften_consistency_mask = 0 #@param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip = soften_consistency_mask
#0 behaves like consistency on, 1 - off, in between - blends
soften_consistency_mask_for_turbo_frames = 0 #@param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
#None - disable and use forward_weights_clip for turbo frames, 0 behaves like consistency on, 1 - off, in between - blends
#@markdown ####Blur consistency mask.
#@markdown Softens transition between raw video init and stylized frames in occluded areas.
consistency_blur = 1 #@param
#@markdown ####Dilate consistency mask.
#@markdown Expands consistency mask without blurring the edges.
consistency_dilate = 3 #@param


# disable_cc_for_turbo_frames = False #@param {"type":"boolean"}
#disable consistency for turbo frames, the same as forward_weights_clip_turbo_step = 1, but a bit faster

#@markdown ###Frame padding
#@markdown Increase padding if you have a shaky\moving camera footage and are getting black borders.

padding_ratio = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
#relative to image size, in range 0-1
padding_mode = 'reflect' #@param ['reflect','edge','wrap']


#safeguard the params
if turbo_frame_skips_steps is not None:
  turbo_frame_skips_steps = min(max(0,turbo_frame_skips_steps),1)
forward_weights_clip = min(max(0,forward_weights_clip),1)
if forward_weights_clip_turbo_step is not None:
  forward_weights_clip_turbo_step = min(max(0,forward_weights_clip_turbo_step),1)
padding_ratio = min(max(0,padding_ratio),1)
##@markdown ###Inpainting
##@markdown Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.

inpaint_blend = 0
##@param {type:"slider", min:0,max:1,value:1,step:0.1}

#@markdown ###Color matching
#@markdown Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted\
#@markdown 0 - off, other values control effect opacity

match_color_strength = 0 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

disable_cc_for_turbo_frames = False

"""### Video masking (render-time)"""

#@title Video mask settings
#@markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
use_background_mask = False #@param {'type':'boolean'}
#@markdown Check to invert the mask.
invert_mask = False #@param {'type':'boolean'}
#@markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True #@param {'type':'boolean'}
#@markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video" #@param ['image', 'color', 'init_video']
#@markdown Specify the init image path or color depending on your background source choice.
background_source = 'red' #@param {'type':'string'}

"""### Frame correction (latent & color matching)"""

#@title Frame correction
#@markdown Match frame pixels or latent to other frames to preven oversaturation and feedback loop artifacts
#@markdown ###Latent matching
#@markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
normalize_latent = 'off' #@param ['off', 'color_video', 'color_video_offset', 'user_defined', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
#@markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.

normalize_latent_offset = 0  #@param {'type':'number'}
#@markdown User defined stats to normalize the latent towards
latent_fixed_mean = 0.  #@param {'type':'raw'}
latent_fixed_std = 0.9  #@param {'type':'raw'}
#@markdown Match latent on per-channel basis
latent_norm_4d = True  #@param {'type':'boolean'}
#@markdown ###Color matching
#@markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
colormatch_frame = 'stylized_frame' #@param ['off', 'color_video', 'color_video_offset','stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
#@markdown Color match strength. 1 mimics legacy behavior
color_match_frame_str = 0.5 #@param {'type':'number'}
#@markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.
colormatch_offset = 0  #@param {'type':'number'}
colormatch_method = 'PDF'#@param ['LAB', 'PDF', 'mean']
colormatch_method_fn = PT.lab_transfer
if colormatch_method == 'LAB':
  colormatch_method_fn = PT.pdf_transfer
if colormatch_method == 'mean':
  colormatch_method_fn = PT.mean_std_transfer
#@markdown Match source frame's texture
colormatch_regrain = False #@param {'type':'boolean'}

warp_mode = 'use_image' #@param ['use_latent', 'use_image']
warp_towards_init = 'off' #@param ['stylized', 'off']

if warp_towards_init != 'off':
  if flow_lq:
          raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        # raft_model = torch.nn.DataParallel(RAFT(args2))
  else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

cond_image_src = 'init' #@param ['init', 'stylized']

"""### Main settings.

Duplicated in the GUI and can be loaded there.
"""

# DD-style losses, renders 2 times slower (!) and more memory intensive :D

latent_scale_schedule = [0,0] #controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
init_scale_schedule = [0,0] #controls coherency with previous frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
sat_scale = 0

init_grad = False #True - compare result to real frame, False - to stylized frame
grad_denoised = True #fastest, on by default, calc grad towards denoised x instead of input x

steps_schedule = {
    0: 25
} #schedules total steps. useful with low strength, when you end up with only 10 steps at 0.2 strength x50 steps. Increasing max steps for low strength gives model more time to get to your text prompt
style_strength_schedule = [0.7]#[0.5]+[0.2]*149+[0.3]*3+[0.2] #use this instead of skip steps. It means how many steps we should do. 0.8 = we diffuse for 80% steps, so we skip 20%. So for skip steps 70% use 0.3
flow_blend_schedule = [0.8] #for example [0.1]*3+[0.999]*18+[0.3] will fade-in for 3 frames, keep style for 18 frames, and fade-out for the rest
cfg_scale_schedule = [15] #text2image strength, 7.5 is a good default
blend_json_schedules = True #True - interpolate values between keyframes. False - use latest keyframe

dynamic_thresh = 30

fixed_code = False #Aka fixed seed. you can use this with fast moving videos, but be careful with still images
code_randomness = 0.1 # Only affects fixed code. high values make the output collapse
# normalize_code = True #Only affects fixed code.

warp_strength = 1 #leave 1 for no change. 1.01 is already a strong value.
flow_override_map = []#[*range(1,15)]+[16]*10+[*range(17+10,17+10+20)]+[18+10+20]*15+[*range(19+10+20+15,9999)] #map flow to frames. set to [] to disable.  [1]*10+[*range(10,9999)] repeats 1st frame flow 10 times, then continues as usual

blend_latent_to_init = 0

colormatch_after = False #colormatch after stylizing. On in previous notebooks.
colormatch_turbo = False #apply colormatching for turbo frames. On in previous notebooks

user_comment = 'testing cc layers'

mask_result = False #imitates inpainting by leaving only inconsistent areas to be diffused

use_karras_noise = False #Should work better with current sample, needs more testing.
end_karras_ramp_early = False

warp_interp = Image.LANCZOS
VERBOSE = True

use_patchmatch_inpaiting = 0

warp_num_k = 128 # number of patches per frame
warp_forward = False #use k-means patched warping (moves large areas instead of single pixels)

inverse_inpainting_mask = False
inpainting_mask_weight = 1.
mask_source = 'none'
mask_clip = [0, 255]
sampler = sample_euler
image_scale = 2
image_scale_schedule = {0:1.5, 1:2}

inpainting_mask_source = 'none'

fixed_seed = False #fixes seed
offload_model = True #offloads model to cpu defore running decoder. May save a bit of VRAM

use_predicted_noise = False
rec_randomness = 0.
rec_cfg = 1.
rec_prompts = {0: ['woman walking on a treadmill']}
rec_source = 'init'
rec_steps_pct = 1

#controlnet settings
controlnet_preprocess = True #preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing
detect_resolution = 768 #control net conditioning image resolution
bg_threshold = 0.4 #controlnet depth/normal bg cutoff threshold
low_threshold = 100 #canny filter parameters
high_threshold = 200 #canny filter parameters
value_threshold = 0.1 #mlsd model settings
distance_threshold = 0.1 #mlsd model settings

temporalnet_source = 'stylized'
temporalnet_skip_1st_frame = True
controlnet_multimodel_mode = 'internal' #external or internal. internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.)

do_softcap = False #softly clamp latent excessive values. reduces feedback loop effect a bit
softcap_thresh = 0.9 # scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected)
softcap_q = 1. # percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%)

max_faces = 10
masked_guidance = False #use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas
cc_masked_diffusion = 0.7  # 0 - off. 0.5-0.7 are good values. make inconsistent area passes only before this % of actual steps, then diffuse whole image
alpha_masked_diffusion = 0.  # 0 - off. 0.5-0.7 are good values. make alpha masked area passes only before this % of actual steps, then diffuse whole image
invert_alpha_masked_diffusion = False

save_controlnet_annotations = True
control_sd15_openpose_hands_face = True
control_sd15_depth_detector = 'Zoe' # Zoe or Midas
control_sd15_softedge_detector = 'PIDI' # HED or PIDI
control_sd15_seg_detector = 'Seg_UFADE20K' # Seg_OFCOCO Seg_OFADE20K Seg_UFADE20K
control_sd15_scribble_detector = 'PIDI' # HED or PIDI
control_sd15_lineart_coarse = False
control_sd15_inpaint_mask_source = 'consistency_mask' # consistency_mask, None, cond_video
control_sd15_shuffle_source = 'color_video' # color_video, init, prev_frame, first_frame
control_sd15_shuffle_1st_source = 'color_video' # color_video, init, None,
overwrite_rec_noise = False

controlnet_multimodel = {
  "control_sd15_depth": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_canny": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_softedge": {
    "weight": 1,
    "start": 0,
    "end": 1
  },
  "control_sd15_mlsd": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_normalbae": {
    "weight": 1,
    "start": 0,
    "end": 1
  },
  "control_sd15_openpose": {
    "weight": 1,
    "start": 0,
    "end": 1
  },
  "control_sd15_scribble": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_seg": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_temporalnet": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_face": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_ip2p": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_inpaint": {
    "weight": 1,
    "start": 0,
    "end": 1
  },
  "control_sd15_lineart": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_lineart_anime": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_shuffle":{
    "weight": 0,
    "start": 0,
    "end": 1
  }
}

"""### Advanced.

Barely used. Not duplicated in the gui. You will need to run this cell to apply settings.
"""

#these variables are not in the GUI and are not being loaded.

# torch.backends.cudnn.enabled = True # disabling this may increase performance on Ampere and Ada GPUs

diffuse_inpaint_mask_blur = 25 #used in mask result to extent the mask
diffuse_inpaint_mask_thresh = 0.8 #used in mask result to extent the mask

add_noise_to_latent = True #add noise to latent vector during latent guidance
noise_upscale_ratio = 1 #noise upscale ratio for latent noise during latent guidance
guidance_use_start_code = True #fix latent noise across steps during latent guidance
init_latent_fn = spherical_dist_loss #function to compute latent distance, l1_loss, rmse, spherical_dist_loss
use_scale = False #use gradient scaling (for mixed precision)
g_invert_mask = False #invert guidance mask

cb_noise_upscale_ratio = 1 #noise in masked diffusion callback
cb_add_noise_to_latent = True #noise in masked diffusion callback
cb_use_start_code = True #fix noise per frame in masked diffusion callback
cb_fixed_code = False #fix noise across all animation in masked diffusion callback (overcooks fast af)
cb_norm_latent = False #norm cb latent to normal ditribution stats in masked diffusion callback

img_zero_uncond = False #by default image conditioned models use same image for negative conditioning (i.e. both positive and negative image conditings are the same. you can use empty negative condition by enabling this)

use_legacy_fixed_code = False

deflicker_scale = 0.
deflicker_latent_scale = 0

prompt_patterns_sched = {}

normalize_prompt_weights = True
controlnet_low_vram = False

sd_batch_size = 2

mask_paths = [ ]

deflicker_scale = 0.
deflicker_latent_scale = 0

"""# Lora/Lycoris & Embedding paths"""

#@title LORA & embedding paths

import torch
from glob import glob
import os
import re
import torch
from typing import Union

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception as e:
                    pass

        return res

weight_load_location = 'cpu'
from modules import devices, shared
#@markdown Specify folders containing your Loras and Textual Inversion Embeddings. Detected loras will be listed after you run the cell.
lora_dir = '/content/drive/MyDrive/models/loras' #@param {'type':'string'}
if not is_colab and lora_dir.startswith('/content'):
  lora_dir = './loras'
  print('Overriding lora dir to ./loras for non-colab env because you path begins with /content. Change path to desired folder')

custom_embed_dir =   '/content/drive/MyDrive/models/embeddings' #@param {'type':'string'}
if not is_colab and custom_embed_dir.startswith('/content'):
  custom_embed_dir = './embeddings'
  os.makedirs(custom_embed_dir, exist_ok=True)
  print('Overriding embeddings dir to ./embeddings for non-colab env because you path begins with /content. Change path to desired folder')


def torch_load_file(filename, device):
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result

# safetensors.torch.load_file = torch_load_file

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or weight_load_location or devices.get_optimal_device_name()
        pl_sd = torch_load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                print(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None


def assign_lora_names_to_compvis_modules(sd_model):
    lora_layer_mapping = {}

    for name, module in sd_model.cond_stage_model.wrapped.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in sd_model.model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    sd_model.lora_layer_mapping = lora_layer_mapping


def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = read_state_dict(filename)

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in sd_model.lora_layer_mapping

    for key_diffusers, weight in sd.items():
        key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts, is_sd2)

        sd_module = sd_model.lora_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = sd_model.lora_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            keys_failed_to_match[key_diffusers] = key
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')
            continue
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def load_loras(names, multipliers=None):
    already_loaded = {}

    for lora in loaded_loras:
        if lora.name in names:
            already_loaded[lora.name] = lora

    loaded_loras.clear()

    loras_on_disk = [available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        list_available_loras()

        loras_on_disk = [available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        lora = already_loaded.get(name, None)

        lora_on_disk = loras_on_disk[i]
        if lora_on_disk is not None:
            if lora is None or os.path.getmtime(lora_on_disk.filename) > lora.mtime:
                lora = load_lora(name, lora_on_disk.filename)

        if lora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lora.multiplier = multipliers[i] if multipliers else 1.0
        loaded_loras.append(lora)


def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        up = module.up.weight.to(target.device, dtype=target.dtype)
        down = module.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            updown = up @ down

        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

        return updown


def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        setattr(self, "lora_current_names", wanted_names)


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lora_current_names", ())
    setattr(self, "lora_weights_backup", None)


def lora_Linear_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Linear_forward_before_lora(self, input)


def lora_Linear_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


def lora_Conv2d_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lora(self, input)


def lora_Conv2d_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_forward(self, *args, **kwargs):
    lora_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)


def list_available_loras():
    available_loras.clear()

    os.makedirs(lora_dir, exist_ok=True)

    candidates = \
        glob(os.path.join(lora_dir, '**/*.pt'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.safetensors'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_loras[name] = LoraOnDisk(name, filename)




def unload(torch):
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora

if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
    torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
    torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
    torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
    torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

use_lycoris = False #@param {'type':'boolean'}

def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def inject_lora(sd_model):
  torch.nn.Linear.forward = lora_Linear_forward
  torch.nn.Linear._load_from_state_dict = lora_Linear_load_state_dict
  torch.nn.Conv2d.forward = lora_Conv2d_forward
  torch.nn.Conv2d._load_from_state_dict = lora_Conv2d_load_state_dict
  torch.nn.MultiheadAttention.forward = lora_MultiheadAttention_forward
  torch.nn.MultiheadAttention._load_from_state_dict = lora_MultiheadAttention_load_state_dict

  assign_lora_names_to_compvis_modules(sd_model)

def inject_lyco(sd_model):
  torch.nn.Linear.forward = lyco_Linear_forward
  torch.nn.Linear._load_from_state_dict = lyco_Linear_load_state_dict
  torch.nn.Conv2d.forward = lyco_Conv2d_forward
  torch.nn.Conv2d._load_from_state_dict = lyco_Conv2d_load_state_dict
  torch.nn.MultiheadAttention.forward = lyco_MultiheadAttention_forward
  torch.nn.MultiheadAttention._load_from_state_dict = lyco_MultiheadAttention_load_state_dict

  assign_lyco_names_to_compvis_modules(sd_model)

# LyCo

from typing import *
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")

re_unet_conv_in = re.compile(r"lora_unet_conv_in(.+)")
re_unet_conv_out = re.compile(r"lora_unet_conv_out(.+)")
re_unet_time_embed = re.compile(r"lora_unet_time_embedding_linear_(\d+)(.+)")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key, is_sd2):
    # I don't know why but some state dict has this kind of thing
    key = key.replace('text_model_text_model', 'text_model')
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_conv_in):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, re_unet_conv_out):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, re_unet_time_embed):
        return f"diffusion_model_time_embed_{m[0]*2-2}{m[1]}"

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


def assign_lyco_names_to_compvis_modules(sd_model):
    lyco_layer_mapping = {}

    for name, module in sd_model.cond_stage_model.wrapped.named_modules():
        lyco_name = name.replace(".", "_")
        lyco_layer_mapping[lyco_name] = module
        module.lyco_layer_name = lyco_name

    for name, module in sd_model.model.named_modules():
        lyco_name = name.replace(".", "_")
        lyco_layer_mapping[lyco_name] = module
        module.lyco_layer_name = lyco_name

    sd_model.lyco_layer_mapping = lyco_layer_mapping


class LycoOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                print(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LycoModule:
    def __init__(self, name):
        self.name = name
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.mtime = None


class FullModule:
    def __init__(self):
        self.weight = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None


class LycoUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


def make_weight_cp(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)


class LycoHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


class IA3Module:
    def __init__(self):
        self.w = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.on_input = None


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


class LycoKronModule:
    def __init__(self):
        self.w1 = None
        self.w1a = None
        self.w1b = None
        self.w2 = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self._alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None

    @property
    def alpha(self):
        if self.w1a is None and self.w2a is None:
            return None
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, x):
        self._alpha = x


CON_KEY = {
    "lora_up.weight", "dyn_up",
    "lora_down.weight", "dyn_down",
    "lora_mid.weight"
}
HADA_KEY = {
    "hada_t1",
    "hada_w1_a",
    "hada_w1_b",
    "hada_t2",
    "hada_w2_a",
    "hada_w2_b",
}
IA3_KEY = {
    "weight",
    "on_input"
}
KRON_KEY = {
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_t2",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
}

def load_lyco(name, filename):
    lyco = LycoModule(name)
    lyco.mtime = os.path.getmtime(filename)

    sd = read_state_dict(filename)
    is_sd2 = 'model_transformer_resblocks' in sd_model.lyco_layer_mapping

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        fullkey = convert_diffusers_name_to_compvis(key_diffusers, is_sd2)
        key, lyco_key = fullkey.split(".", 1)

        sd_module = sd_model.lyco_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = sd_model.lyco_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            print(f'key failed to match: {key_diffusers}')
            keys_failed_to_match.append(key_diffusers)
            continue

        lyco_module = lyco.modules.get(key, None)
        if lyco_module is None:
            lyco_module = LycoUpDownModule()
            lyco.modules[key] = lyco_module

        if lyco_key == "alpha":
            lyco_module.alpha = weight.item()
            continue

        if lyco_key == "scale":
            lyco_module.scale = weight.item()
            continue

        if lyco_key == "diff":
            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)
            lyco_module = FullModule()
            lyco.modules[key] = lyco_module
            lyco_module.weight = weight
            continue

        if 'bias_' in lyco_key:
            if lyco_module.bias is None:
                lyco_module.bias = [None, None, None]
            if 'bias_indices' == lyco_key:
                lyco_module.bias[0] = weight
            elif 'bias_values' == lyco_key:
                lyco_module.bias[1] = weight
            elif 'bias_size' == lyco_key:
                lyco_module.bias[2] = weight

            if all((i is not None) for i in lyco_module.bias):
                print('build bias')
                lyco_module.bias = torch.sparse_coo_tensor(
                    lyco_module.bias[0],
                    lyco_module.bias[1],
                    tuple(lyco_module.bias[2]),
                ).to(device=devices.cpu, dtype=devices.dtype)
                lyco_module.bias.requires_grad_(False)
            continue

        if lyco_key in CON_KEY:
            if (type(sd_module) == torch.nn.Linear
                or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
                or type(sd_module) == torch.nn.MultiheadAttention):
                weight = weight.reshape(weight.shape[0], -1)
                module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            elif type(sd_module) == torch.nn.Conv2d:
                if lyco_key == "lora_down.weight" or lyco_key == "dyn_up":
                    if len(weight.shape) == 2:
                        weight = weight.reshape(weight.shape[0], -1, 1, 1)
                    if weight.shape[2] != 1 or weight.shape[3] != 1:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                    else:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                elif lyco_key == "lora_mid.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                elif lyco_key == "lora_up.weight" or lyco_key == "dyn_down":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
            else:
                assert False, f'Lyco layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape
            with torch.no_grad():
                if weight.shape != module.weight.shape:
                    weight = weight.reshape(module.weight.shape)
                module.weight.copy_(weight)

            module.to(device=devices.cpu, dtype=devices.dtype)
            module.requires_grad_(False)

            if lyco_key == "lora_up.weight" or lyco_key == "dyn_up":
                lyco_module.up_model = module
            elif lyco_key == "lora_mid.weight":
                lyco_module.mid_model = module
            elif lyco_key == "lora_down.weight" or lyco_key == "dyn_down":
                lyco_module.down_model = module
                lyco_module.dim = weight.shape[0]
            else:
                print(lyco_key)
        elif lyco_key in HADA_KEY:
            if type(lyco_module) != LycoHadaModule:
                alpha = lyco_module.alpha
                bias = lyco_module.bias
                lyco_module = LycoHadaModule()
                lyco_module.alpha = alpha
                lyco_module.bias = bias
                lyco.modules[key] = lyco_module
            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape

            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)

            if lyco_key == 'hada_w1_a':
                lyco_module.w1a = weight
            elif lyco_key == 'hada_w1_b':
                lyco_module.w1b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'hada_w2_a':
                lyco_module.w2a = weight
            elif lyco_key == 'hada_w2_b':
                lyco_module.w2b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'hada_t1':
                lyco_module.t1 = weight
            elif lyco_key == 'hada_t2':
                lyco_module.t2 = weight

        elif lyco_key in IA3_KEY:
            if type(lyco_module) != IA3Module:
                lyco_module = IA3Module()
                lyco.modules[key] = lyco_module

            if lyco_key == "weight":
                lyco_module.w = weight.to(devices.device, dtype=devices.dtype)
            elif lyco_key == "on_input":
                lyco_module.on_input = weight
        elif lyco_key in KRON_KEY:
            if not isinstance(lyco_module, LycoKronModule):
                alpha = lyco_module.alpha
                bias = lyco_module.bias
                lyco_module = LycoKronModule()
                lyco_module.alpha = alpha
                lyco_module.bias = bias
                lyco.modules[key] = lyco_module
            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape

            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)

            if lyco_key == 'lokr_w1':
                lyco_module.w1 = weight
            elif lyco_key == 'lokr_w1_a':
                lyco_module.w1a = weight
            elif lyco_key == 'lokr_w1_b':
                lyco_module.w1b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'lokr_w2':
                lyco_module.w2 = weight
            elif lyco_key == 'lokr_w2_a':
                lyco_module.w2a = weight
            elif lyco_key == 'lokr_w2_b':
                lyco_module.w2b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'lokr_t2':
                lyco_module.t2 = weight
        else:
            assert False, f'Bad Lyco layer name: {key_diffusers} - must end in lyco_up.weight, lyco_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(sd_model.lyco_layer_mapping)
        print(f"Failed to match keys when loading Lyco {filename}: {keys_failed_to_match}")

    return lyco


def load_lycos(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    already_loaded = {}

    for lyco in loaded_lycos:
        if lyco.name in names:
            already_loaded[lyco.name] = lyco

    loaded_lycos.clear()

    lycos_on_disk = [available_lycos.get(name, None) for name in names]
    if any([x is None for x in lycos_on_disk]):
        list_available_lycos()

        lycos_on_disk = [available_lycos.get(name, None) for name in names]

    for i, name in enumerate(names):
        lyco = already_loaded.get(name, None)

        lyco_on_disk = lycos_on_disk[i]
        if lyco_on_disk is not None:
            if lyco is None or os.path.getmtime(lyco_on_disk.filename) > lyco.mtime:
                lyco = load_lyco(name, lyco_on_disk.filename)

        if lyco is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lyco.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        lyco.unet_multiplier = unet_multipliers[i] if unet_multipliers else lyco.te_multiplier
        lyco.dyn_dim = dyn_dims[i] if dyn_dims else None
        loaded_lycos.append(lyco)


def _rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def _rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)


def rebuild_weight(module, orig_weight: torch.Tensor, dyn_dim: int=None) -> torch.Tensor:
    output_shape: Sized
    if module.__class__.__name__ == 'LycoUpDownModule':
        up = module.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = module.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [up.size(0), down.size(1)]
        if (mid:=module.mid_model) is not None:
            # cp-decomposition
            mid = mid.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = _rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = _rebuild_conventional(up, down, output_shape, dyn_dim)

    elif module.__class__.__name__ == 'LycoHadaModule':
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]

        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape)

        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape)

        updown = updown1 * updown2

    elif module.__class__.__name__ == 'FullModule':
        output_shape = module.weight.shape
        updown = module.weight.to(orig_weight.device, dtype=orig_weight.dtype)

    elif module.__class__.__name__ == 'IA3Module':
        output_shape = [module.w.size(0), orig_weight.size(1)]
        if module.on_input:
            output_shape.reverse()
        else:
            module.w = module.w.reshape(-1, 1)
        updown = orig_weight * module.w

    elif module.__class__.__name__ == 'LycoKronModule':
        if module.w1 is not None:
            w1 = module.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b

        if module.w2 is not None:
            w2 = module.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        elif module.t2 is None:
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = make_weight_cp(t2, w2a, w2b)

        output_shape = [w1.size(0)*w2.size(0), w1.size(1)*w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        updown = make_kron(
            output_shape, w1, w2
        )

    else:
        raise NotImplementedError(
            f"Unknown module type: {module.__class__.__name__}\n"
            "If the type is one of "
            "'LycoUpDownModule', 'LycoHadaModule', 'FullModule', 'IA3Module', 'LycoKronModule'"
            "You may have other lyco extension that conflict with locon extension."
        )

    if hasattr(module, 'bias') and module.bias != None:
        updown = updown.reshape(module.bias.shape)
        updown += module.bias.to(orig_weight.device, dtype=orig_weight.dtype)
        updown = updown.reshape(output_shape)

    if len(output_shape) == 4:
        updown = updown.reshape(output_shape)

    if orig_weight.size().numel() == updown.size().numel():
        updown = updown.reshape(orig_weight.shape)
    # print(torch.sum(updown))
    return updown


def lyco_calc_updown(lyco, module, target, multiplier):
    with torch.no_grad():
        updown = rebuild_weight(module, target, lyco.dyn_dim)
        if lyco.dyn_dim and module.dim:
            dim = min(lyco.dyn_dim, module.dim)
        elif lyco.dyn_dim:
            dim = lyco.dyn_dim
        elif module.dim:
            dim = module.dim
        else:
            dim = None

        scale = (
            module.scale if module.scale is not None
            else module.alpha / dim if dim is not None and module.alpha is not None
            else 1.0
        )
        # print(scale, module.alpha, module.dim, lyco.dyn_dim)
        updown = updown * multiplier * scale
        return updown


def lyco_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Lycos to the weights of torch layer self.
    If weights already have this particular set of lycos applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to lycos.
    """

    lyco_layer_name = getattr(self, 'lyco_layer_name', None)
    if lyco_layer_name is None:
        return

    current_names = getattr(self, "lyco_current_names", ())
    lora_prev_names = getattr(self, "lora_prev_names", ())
    lora_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_lycos)

    # We take lora_changed as base_weight changed
    # but functional lora will not affect the weight so take it as unchanged
    lora_changed = lora_prev_names != lora_names
    lora_functional = getattr(shared.opts, 'lora_functional', False)
    lora_changed = lora_changed and not lora_functional

    lyco_changed = current_names != wanted_names

    weights_backup = getattr(self, "lyco_weights_backup", None)

    if ((len(loaded_lycos) and weights_backup is None)
        or (weights_backup is not None and lora_changed)):
        # backup when:
        #  * apply lycos but haven't backed up any weights
        #  * have outdated backed up weights
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (
                self.in_proj_weight.to(devices.cpu, copy=True),
                self.out_proj.weight.to(devices.cpu, copy=True)
            )
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)
        self.lyco_weights_backup = weights_backup
    elif len(loaded_lycos) == 0:
        # when we unload all the lycos and have no weights to backup
        # clean backup weights to save ram
        self.lyco_weights_backup = None

    if lyco_changed or lora_changed:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lyco in loaded_lycos:
            module = lyco.modules.get(lyco_layer_name, None)
            multiplier = (
                lyco.te_multiplier if 'transformer' in lyco_layer_name[:20]
                else lyco.unet_multiplier
            )
            if module is not None and hasattr(self, 'weight'):
                # print(lyco_layer_name, multiplier)
                updown = lyco_calc_updown(lyco, module, self.weight, multiplier)
                if len(self.weight.shape) == 4 and self.weight.shape[1] == 9:
                    # inpainting model. zero pad updown to make channel[1]  4 to 9
                    updown = F.pad(updown, (0, 0, 0, 0, 0, 5))
                self.weight += updown
                continue

            module_q = lyco.modules.get(lyco_layer_name + "_q_proj", None)
            module_k = lyco.modules.get(lyco_layer_name + "_k_proj", None)
            module_v = lyco.modules.get(lyco_layer_name + "_v_proj", None)
            module_out = lyco.modules.get(lyco_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lyco_calc_updown(lyco, module_q, self.in_proj_weight, multiplier)
                updown_k = lyco_calc_updown(lyco, module_k, self.in_proj_weight, multiplier)
                updown_v = lyco_calc_updown(lyco, module_v, self.in_proj_weight, multiplier)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lyco_calc_updown(lyco, module_out, self.out_proj.weight, multiplier)
                continue

            if module is None:
                continue

            print(3, f'failed to calculate lyco weights for layer {lyco_layer_name}')

        setattr(self, "lora_prev_names", lora_names)
        setattr(self, "lyco_current_names", wanted_names)


def lyco_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lyco_current_names", ())
    setattr(self, "lyco_weights_backup", None)


def lyco_Linear_forward(self, input):
    lyco_apply_weights(self)

    return torch.nn.Linear_forward_before_lyco(self, input)


def lyco_Linear_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lyco(self, *args, **kwargs)


def lyco_Conv2d_forward(self, input):
    lyco_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lyco(self, input)


def lyco_Conv2d_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lyco(self, *args, **kwargs)


def lyco_MultiheadAttention_forward(self, *args, **kwargs):
    lyco_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lyco(self, *args, **kwargs)


def lyco_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lyco(self, *args, **kwargs)


def list_available_lycos(model_dir=lora_dir):
    available_lycos.clear()

    os.makedirs(model_dir, exist_ok=True)

    candidates = \
        glob(os.path.join(model_dir, '**/*.pt'), recursive=True) + \
        glob(os.path.join(model_dir, '**/*.safetensors'), recursive=True) + \
        glob(os.path.join(model_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_lycos[name] = LycoOnDisk(name, filename)

if not hasattr(torch.nn, 'Linear_forward_before_lyco'):
    torch.nn.Linear_forward_before_lyco = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_lyco'):
    torch.nn.Linear_load_state_dict_before_lyco = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_lyco'):
    torch.nn.Conv2d_forward_before_lyco = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lyco'):
    torch.nn.Conv2d_load_state_dict_before_lyco = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lyco'):
    torch.nn.MultiheadAttention_forward_before_lyco = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lyco'):
    torch.nn.MultiheadAttention_load_state_dict_before_lyco = torch.nn.MultiheadAttention._load_from_state_dict


available_lycos = {}
loaded_lycos = []

available_loras = {}
loaded_loras = []
import safetensors

# (c) Alex Spirin 2023


def split_lora_from_prompts(prompts):
  re1 = '\<(.*?)\>'
  new_prompt_loras = {}
  new_prompts = {}

  #iterate through prompts keyframes and fill in lora schedules
  for key in prompts.keys():
    prompt_list = prompts[key]
    prompt_loras = []
    new_prompts[key] = []
    for i in range(len(prompt_list)):
      subp = prompt_list[i]

      #get a dict of loras:weights from a prompt
      prompt_loras+=re.findall(re1, subp)
      new_prompts[key].append(re.sub(re1, '', subp).strip(' '))

    prompt_loras_dict = dict([(o.split(':')[1], o.split(':')[-1]) for o in prompt_loras])

    #fill lora dict based on keyframe, lora:weight
    for lora_key in prompt_loras_dict.keys():
      try: new_prompt_loras[lora_key]
      except: new_prompt_loras[lora_key] = {}
      new_prompt_loras[lora_key][key] = float(prompt_loras_dict[lora_key])

      # remove lora keywords from prompts


  return new_prompts, new_prompt_loras

def get_prompt_weights(prompts):
  weight_re = r":\s*([\d.]+)\s*$"
  new_prompts = {}
  prompt_weights_dict = {}
  max_len = 0
  for key in prompts.keys():
    prompt_list = prompts[key]
    if len(prompt_list) == 1:
      prompt_weights_dict[key] = [1] #if 1 prompt set weight to 1
      new_prompts[key] = prompt_list
    else:
      weights = []
      new_prompt = []
      for i in range(len(prompt_list)):
        subprompt = prompt_list[i]
        m = re.findall(weight_re, subprompt) #find :number at the end of the string
        new_prompt.append(re.sub(weight_re, '', subprompt).strip(' '))
        m = m[0] if len(m)>0 else 1
        weights.append(m)

      prompt_weights_dict[key] = weights
      new_prompts[key] = new_prompt
    max_len = max(max_len,len(prompt_weights_dict[key]))

  for key in prompt_weights_dict.keys():
    weights = prompt_weights_dict[key]
    if len(weights)<max_len:
      weights+=[0]*(max_len-len(weights))
    weights = np.array(weights).astype('float16')
    if normalize_prompt_weights: weights = weights/weights.sum() #normalize to 1 - optional
    prompt_weights_dict[key] = weights
  return new_prompts, prompt_weights_dict

def get_loras_weights_for_frame(frame_num, loras_dict):
  loras = list(loras_dict.keys())
  loras_weights = [get_scheduled_arg(frame_num, loras_dict[o]) for o in loras]
  return loras, loras_weights

list_available_loras()
print('Loras/LyCos detected:\n','\n'.join(list(available_loras.keys())))

# list_available_lycos()
# print('LyCos detected:\n','\n'.join(list(available_lycos.keys())))

"""# GUI"""

#@title gui

#@markdown Load settings from txt file or output frame image
gui_difficulty_dict = {
    "I'm too young to die.":["flow_warp", "warp_strength","warp_mode","padding_mode","padding_ratio",
      "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k","warp_forward",
      "blend_json_schedules", "VERBOSE","offload_model", "do_softcap", "softcap_thresh",
      "softcap_q", "user_comment","turbo_mode","turbo_steps", "colormatch_turbo",
      "turbo_frame_skips_steps","soften_consistency_mask_for_turbo_frames", "check_consistency",
      "missed_consistency_weight","overshoot_consistency_weight", "edges_consistency_weight",
      "soften_consistency_mask","consistency_blur","match_color_strength","mask_result",
      "use_patchmatch_inpaiting","normalize_latent","normalize_latent_offset","latent_fixed_mean",
      "latent_fixed_std","latent_norm_4d","use_karras_noise", "cond_image_src", "inpainting_mask_source",
      "inverse_inpainting_mask", "inpainting_mask_weight", "init_grad", "grad_denoised",
      "image_scale_schedule","blend_latent_to_init","dynamic_thresh","rec_cfg", "rec_source",
      "rec_steps_pct", "controlnet_multimodel_mode",
      "overwrite_rec_noise",
      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp"],
    "Hey, not too rough.":["flow_warp", "warp_strength","warp_mode",
      "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k","warp_forward",

      "check_consistency",

      "use_patchmatch_inpaiting","init_grad", "grad_denoised",
      "image_scale_schedule","blend_latent_to_init","rec_cfg",

      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp"],
    "Hurt me plenty.":"",
    "Ultra-Violence.":[]
}
import traceback
gui_difficulty = "Hey, not too rough." #@param ["I'm too young to die.", "Hey, not too rough.", "Ultra-Violence."]
print(f'Using "{gui_difficulty}" gui difficulty. Please switch to another difficulty\nto unlock up to {len(gui_difficulty_dict[gui_difficulty])} more settings when you`re ready :D')
settings_path = '-1' #@param {'type':'string'}
load_settings_from_file = True #@param {'type':'boolean'}
#@markdown Disable to load settings into GUI from colab cells. You will need to re-run colab cells you've edited to apply changes, then re-run the gui cell.\
#@markdown Enable to keep GUI state.
keep_gui_state_on_cell_rerun = True #@param {'type':'boolean'}
settings_out = batchFolder+f"/settings"
from  ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, SelectionSlider, Valid

def desc_widget(widget, desc, width=80, h=True):
    if isinstance(widget, Checkbox): return widget
    if isinstance(width, str):
        if width.endswith('%') or width.endswith('px'):
            layout = Layout(width=width)
    else: layout = Layout(width=f'{width}')

    text = Label(desc, layout = layout, tooltip = widget.tooltip, description_tooltip = widget.description_tooltip)
    return HBox([text, widget]) if h else VBox([text, widget])

class ControlNetControls(HBox):
    def __init__(self,  name, values, **kwargs):
        self.label  = HTML(
                description=name,
                description_tooltip=name,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px', width='200px'))

        self.enable = Checkbox(value=values['weight']>0,description='',indent=True, description_tooltip='Enable model.',
                               style={'description_width': '25px' },layout=Layout(width='70px', left='-25px'))
        self.weight = FloatText(value = values['weight'], description=' ', step=0.05,
                                description_tooltip = 'Controlnet model weights. ', layout=Layout(width='100px', visibility= 'visible' if values['weight']>0 else 'hidden'),
                                style={'description_width': '25px' })
        self.start_end = FloatRangeSlider(
          value=[values['start'],values['end']],
          min=0,
          max=1,
          step=0.01,
          description=' ',
          description_tooltip='Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
          disabled=False,
          continuous_update=False,
          orientation='horizontal',
          readout=True,
          layout = Layout(width='300px', visibility= 'visible' if values['weight']>0 else 'hidden'),
          style={'description_width': '50px' }
        )

        self.enable.observe(self.on_change)
        self.weight.observe(self.on_change)

        super().__init__([self.enable, self.label, self.weight, self.start_end], layout = Layout(valign='center'))

    def on_change(self, change):
      # print(change)
      if change['name'] == 'value':
        # print(change)
        if self.enable.value:
              self.weight.disabled = False
              self.weight.layout.visibility = 'visible'
              if change['old'] == False and self.weight.value==0:
                self.weight.value = 1
              # if self.weight.value>0:
              self.start_end.disabled = False
              self.label.disabled = False
              self.start_end.layout.visibility = 'visible'
        else:
              self.weight.disabled = True
              self.start_end.disabled = True
              self.label.disabled = True
              self.weight.layout.visibility = 'hidden'
              self.start_end.layout.visibility = 'hidden'

    def __getattr__(self, attr):
        if attr == 'value':
            weight = 0
            if self.weight.value>0 and self.enable.value: weight = self.weight.value
            (start,end) = self.start_end.value
            return {
                  "weight": weight,
                  "start":start,
                  "end":end
                }
        else:
            return super.__getattr__(attr)

class ControlGUI(VBox):
  def __init__(self, args):
    enable_label = HTML(
                    description='Enable',
                    description_tooltip='Enable',  style={'description_width': '50px' },
                    layout = Layout(width='40px', left='-50px', ))
    model_label = HTML(
                    description='Model name',
                    description_tooltip='Model name',  style={'description_width': '100px' },
                    layout = Layout(width='265px'))
    weight_label = HTML(
                    description='weight',
                    description_tooltip='Model weight. 0 weight effectively disables the model. The total sum of all the weights will be normalized to 1.',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='125px'))#65
    range_label = HTML(
                    description='active range (% or total steps)',
                    description_tooltip='Model`s active range. % of total steps when the model is active.\n Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='200px'))
    controls_list = [HBox([enable_label,model_label, weight_label, range_label ])]
    controls_dict = {}
    self.possible_controlnets = ['control_sd15_depth',
        'control_sd15_canny',
        'control_sd15_softedge',
        'control_sd15_mlsd',
        'control_sd15_normalbae',
        'control_sd15_openpose',
        'control_sd15_scribble',
        'control_sd15_seg',
        'control_sd15_temporalnet',
        'control_sd15_face',
        'control_sd15_ip2p',
        'control_sd15_inpaint',
        'control_sd15_lineart',
        'control_sd15_lineart_anime',
        'control_sd15_shuffle']
    for key in self.possible_controlnets:
      if key in args.keys():
        w = ControlNetControls(key, args[key])
      else:
        w = ControlNetControls(key, {
            "weight":0,
            "start":0,
            "end":1
        })
      controls_list.append(w)
      controls_dict[key] = w

    self.args = args
    self.ws = controls_dict
    super().__init__(controls_list)

  def __getattr__(self, attr):
        if attr == 'value':
            res = {}
            for key in self.possible_controlnets:
              if self.ws[key].value['weight'] > 0:
                res[key] = self.ws[key].value
            return res
        else:
            return super.__getattr__(attr)

def set_visibility(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
          obj[key].layout.visibility = value


#try keep settings on occasional run cell
if keep_gui_state_on_cell_rerun:
  try:

    latent_scale_schedule=eval(get_value('latent_scale_schedule',guis))
    init_scale_schedule=eval(get_value('init_scale_schedule',guis))
    steps_schedule=eval(get_value('steps_schedule',guis))
    style_strength_schedule=eval(get_value('style_strength_schedule',guis))
    cfg_scale_schedule=eval(get_value('cfg_scale_schedule',guis))
    flow_blend_schedule=eval(get_value('flow_blend_schedule',guis))
    image_scale_schedule=eval(get_value('image_scale_schedule',guis))

    user_comment= get_value('user_comment',guis)
    blend_json_schedules=get_value('blend_json_schedules',guis)
    VERBOSE=get_value('VERBOSE',guis)

    #mask
    use_background_mask=get_value('use_background_mask',guis)
    invert_mask=get_value('invert_mask',guis)
    background=get_value('background',guis)
    background_source=get_value('background_source',guis)
    (mask_clip_low, mask_clip_high) = get_value('mask_clip',guis)


    #turbo
    turbo_mode=get_value('turbo_mode',guis)
    turbo_steps=get_value('turbo_steps',guis)
    colormatch_turbo=get_value('colormatch_turbo',guis)
    turbo_frame_skips_steps=get_value('turbo_frame_skips_steps',guis)
    soften_consistency_mask_for_turbo_frames=get_value('soften_consistency_mask_for_turbo_frames',guis)

    #warp
    flow_warp= get_value('flow_warp',guis)
    apply_mask_after_warp=get_value('apply_mask_after_warp',guis)
    warp_num_k=get_value('warp_num_k',guis)
    warp_forward=get_value('warp_forward',guis)
    warp_strength=get_value('warp_strength',guis)
    flow_override_map=eval(get_value('flow_override_map',guis))
    warp_mode=get_value('warp_mode',guis)
    warp_towards_init=get_value('warp_towards_init',guis)

    #cc
    check_consistency=get_value('check_consistency',guis)
    missed_consistency_weight=get_value('missed_consistency_weight',guis)
    overshoot_consistency_weight=get_value('overshoot_consistency_weight',guis)
    edges_consistency_weight=get_value('edges_consistency_weight',guis)
    consistency_blur=get_value('consistency_blur',guis)
    consistency_dilate=get_value('consistency_dilate',guis)
    padding_ratio=get_value('padding_ratio',guis)
    padding_mode=get_value('padding_mode',guis)
    match_color_strength=get_value('match_color_strength',guis)
    soften_consistency_mask=get_value('soften_consistency_mask',guis)
    mask_result=get_value('mask_result',guis)
    use_patchmatch_inpaiting=get_value('use_patchmatch_inpaiting',guis)

    #diffusion
    text_prompts=eval(get_value('text_prompts',guis))
    negative_prompts=eval(get_value('negative_prompts',guis))
    prompt_patterns_sched = eval(get_value('prompt_patterns_sched',guis))
    cond_image_src=get_value('cond_image_src',guis)
    set_seed=get_value('set_seed',guis)
    clamp_grad=get_value('clamp_grad',guis)
    clamp_max=get_value('clamp_max',guis)
    sat_scale=get_value('sat_scale',guis)
    init_grad=get_value('init_grad',guis)
    grad_denoised=get_value('grad_denoised',guis)
    blend_latent_to_init=get_value('blend_latent_to_init',guis)
    fixed_code=get_value('fixed_code',guis)
    code_randomness=get_value('code_randomness',guis)
    # normalize_code=get_value('normalize_code',guis)
    dynamic_thresh=get_value('dynamic_thresh',guis)
    sampler = get_value('sampler',guis)
    use_karras_noise = get_value('use_karras_noise',guis)
    inpainting_mask_weight = get_value('inpainting_mask_weight',guis)
    inverse_inpainting_mask = get_value('inverse_inpainting_mask',guis)
    inpainting_mask_source = get_value('mask_source',guis)

    #colormatch
    normalize_latent=get_value('normalize_latent',guis)
    normalize_latent_offset=get_value('normalize_latent_offset',guis)
    latent_fixed_mean=eval(str(get_value('latent_fixed_mean',guis)))
    latent_fixed_std=eval(str(get_value('latent_fixed_std',guis)))
    latent_norm_4d=get_value('latent_norm_4d',guis)
    colormatch_frame=get_value('colormatch_frame',guis)
    color_match_frame_str=get_value('color_match_frame_str',guis)
    colormatch_offset=get_value('colormatch_offset',guis)
    colormatch_method=get_value('colormatch_method',guis)
    colormatch_regrain=get_value('colormatch_regrain',guis)
    colormatch_after=get_value('colormatch_after',guis)
    image_prompts = {}

    fixed_seed = get_value('fixed_seed',guis)

    #rec noise
    rec_cfg = get_value('rec_cfg',guis)
    rec_steps_pct = get_value('rec_steps_pct',guis)
    rec_prompts = eval(get_value('rec_prompts',guis))
    rec_randomness = get_value('rec_randomness',guis)
    use_predicted_noise = get_value('use_predicted_noise',guis)
    overwrite_rec_noise  = get_value('overwrite_rec_noise',guis)

    #controlnet
    save_controlnet_annotations = get_value('save_controlnet_annotations',guis)
    control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face',guis)
    control_sd15_depth_detector  = get_value('control_sd15_depth_detector',guis)
    control_sd15_softedge_detector = get_value('control_sd15_softedge_detector',guis)
    control_sd15_seg_detector = get_value('control_sd15_seg_detector',guis)
    control_sd15_scribble_detector = get_value('control_sd15_scribble_detector',guis)
    control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse',guis)
    control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source',guis)
    control_sd15_shuffle_source = get_value('control_sd15_shuffle_source',guis)
    control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source',guis)
    controlnet_multimodel = get_value('controlnet_multimodel',guis)

    controlnet_preprocess = get_value('controlnet_preprocess',guis)
    detect_resolution  = get_value('detect_resolution',guis)
    bg_threshold = get_value('bg_threshold',guis)
    low_threshold = get_value('low_threshold',guis)
    high_threshold = get_value('high_threshold',guis)
    value_threshold = get_value('value_threshold',guis)
    distance_threshold = get_value('distance_threshold',guis)
    temporalnet_source = get_value('temporalnet_source',guis)
    temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame',guis)
    controlnet_multimodel_mode = get_value('controlnet_multimodel_mode',guis)
    max_faces = get_value('max_faces',guis)

    do_softcap = get_value('do_softcap',guis)
    softcap_thresh = get_value('softcap_thresh',guis)
    softcap_q = get_value('softcap_q',guis)

    masked_guidance = get_value('masked_guidance',guis)
    cc_masked_diffusion = get_value('cc_masked_diffusion',guis)
    alpha_masked_diffusion = get_value('alpha_masked_diffusion',guis)
    invert_alpha_masked_diffusion = get_value('invert_alpha_masked_diffusion',guis)

    normalize_prompt_weights = get_value('normalize_prompt_weights',guis)
    sd_batch_size = get_value('sd_batch_size',guis)
    controlnet_low_vram = get_value('controlnet_low_vram',guis)
    mask_paths = eval(get_value('mask_paths',guis))
    deflicker_scale = get_value('deflicker_scale',guis)
    deflicker_latent_scale = get_value('deflicker_latent_scale',guis)
  except:
    print('Error keeping state')
    print(traceback.format_exc())
    pass

gui_misc = {
    "user_comment": Textarea(value=user_comment,layout=Layout(width=f'80%'),  description = 'user_comment:',  description_tooltip = 'Enter a comment to differentiate between save files.'),
    "blend_json_schedules": Checkbox(value=blend_json_schedules, description='blend_json_schedules',indent=True, description_tooltip = 'Smooth values between keyframes.', tooltip = 'Smooth values between keyframes.'),
    "VERBOSE": Checkbox(value=VERBOSE,description='VERBOSE',indent=True, description_tooltip = 'Print all logs'),
    "offload_model": Checkbox(value=offload_model,description='offload_model',indent=True, description_tooltip = 'Offload unused models to CPU and back to GPU to save VRAM. May reduce speed.'),
    "do_softcap": Checkbox(value=do_softcap,description='do_softcap',indent=True, description_tooltip = 'Softly clamp latent excessive values. Reduces feedback loop effect a bit.'),
    "softcap_thresh":FloatSlider(value=softcap_thresh, min=0, max=1, step=0.05, description='softcap_thresh:', readout=True, readout_format='.1f', description_tooltip='Scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected'),
    "softcap_q":FloatSlider(value=softcap_q, min=0, max=1, step=0.05, description='softcap_q:', readout=True, readout_format='.1f', description_tooltip='Percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%'),
    "sd_batch_size":IntText(value = sd_batch_size, description='sd_batch_size:', description_tooltip='Diffusion batch size. Default=2 for 1 positive + 1 negative prompt. '),

}

gui_mask = {
    "use_background_mask":Checkbox(value=use_background_mask,description='use_background_mask',indent=True, description_tooltip='Enable masking. In order to use it, you have to either extract or provide an existing mask in Video Masking cell.\n'),
    "invert_mask":Checkbox(value=invert_mask,description='invert_mask',indent=True, description_tooltip='Inverts the mask, allowing to process either backgroung or characters, depending on your mask.'),
    "background": Dropdown(description='background',
                           options = ['image', 'color', 'init_video'], value = background,
                           description_tooltip='Background type. Image - uses static image specified in background_source, color - uses fixed color specified in background_source, init_video - uses raw init video for masked areas.'),
    "background_source": Text(value=background_source, description = 'background_source', description_tooltip='Specify image path or color name of hash.'),
    "apply_mask_after_warp": Checkbox(value=apply_mask_after_warp,description='apply_mask_after_warp',indent=True, description_tooltip='On to reduce ghosting. Apply mask after warping and blending warped image with current raw frame. If off, only current frame will be masked, previous frame will be warped and blended wuth masked current frame.'),
    "mask_clip" : IntRangeSlider(
      value=mask_clip,
      min=0,
      max=255,
      step=1,
      description='Mask clipping:',
      description_tooltip='Values below the selected range will be treated as black mask, values above - as white.',
      disabled=False,
      continuous_update=False,
      orientation='horizontal',
      readout=True),
    "mask_paths":Textarea(value=str(mask_paths),layout=Layout(width=f'80%'),  description = 'mask_paths:',
                          description_tooltip='A list of paths to prompt mask files/folders/glob patterns. Format: ["/somepath/somefile.mp4", "./otherpath/dirwithfiles/*.jpg]'),

}

gui_turbo = {
    "turbo_mode":Checkbox(value=turbo_mode,description='turbo_mode',indent=True, description_tooltip='Turbo mode skips diffusion process on turbo_steps number of frames. Frames are still being warped and blended. Speeds up the render at the cost of possible trails an ghosting.' ),
    "turbo_steps": IntText(value = turbo_steps, description='turbo_steps:', description_tooltip='Number of turbo frames'),
    "colormatch_turbo":Checkbox(value=colormatch_turbo,description='colormatch_turbo',indent=True, description_tooltip='Apply frame color matching during turbo frames. May increease rendering speed, but may add minor flickering.'),
    "turbo_frame_skips_steps" :  SelectionSlider(description='turbo_frame_skips_steps',
                                                 options = ['70%','75%','80%','85%', '80%', '95%', '100% (don`t diffuse turbo frames, fastest)'], value = '100% (don`t diffuse turbo frames, fastest)', description_tooltip='Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.'),
    "soften_consistency_mask_for_turbo_frames": FloatSlider(value=soften_consistency_mask_for_turbo_frames, min=0, max=1, step=0.05, description='soften_consistency_mask_for_turbo_frames:', readout=True, readout_format='.1f', description_tooltip='Clips the consistency mask, reducing it`s effect'),

}

gui_warp = {
    "flow_warp":Checkbox(value=flow_warp,description='flow_warp',indent=True, description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),

    "flow_blend_schedule" : Textarea(value=str(flow_blend_schedule),layout=Layout(width=f'80%'),  description = 'flow_blend_schedule:',  description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
    "warp_num_k": IntText(value = warp_num_k, description='warp_num_k:', description_tooltip='Nubmer of clusters in forward-warp mode. The more - the smoother is the motion. Lower values move larger chunks of image at a time.'),
    "warp_forward": Checkbox(value=warp_forward,description='warp_forward',indent=True,  description_tooltip='Experimental. Enable patch-based flow warping. Groups pixels by motion direction and moves them together, instead of moving individual pixels.'),
    # "warp_interp": Textarea(value='Image.LANCZOS',layout=Layout(width=f'80%'),  description = 'warp_interp:'),
    "warp_strength": FloatText(value = warp_strength, description='warp_strength:', description_tooltip='Experimental. Motion vector multiplier. Provides a glitchy effect.'),
    "flow_override_map":  Textarea(value=str(flow_override_map),layout=Layout(width=f'80%'),  description = 'flow_override_map:', description_tooltip='Experimental. Motion vector maps mixer. Allows changing frame-motion vetor indexes or repeating motion, provides a glitchy effect.'),
    "warp_mode": Dropdown(description='warp_mode', options = ['use_latent', 'use_image'],
                          value = warp_mode, description_tooltip='Experimental. Apply warp to latent vector. May get really blurry, but reduces feedback loop effect for slow movement'),
    "warp_towards_init": Dropdown(description='warp_towards_init',
                                  options = ['stylized', 'off'] , value = warp_towards_init, description_tooltip='Experimental. After a frame is stylized, computes the difference between output and input for that frame, and warps the output back to input, preserving its shape.'),
    "padding_ratio": FloatSlider(value=padding_ratio, min=0, max=1, step=0.05, description='padding_ratio:', readout=True, readout_format='.1f', description_tooltip='Amount of padding. Padding is used to avoid black edges when the camera is moving out of the frame.'),
    "padding_mode": Dropdown(description='padding_mode', options = ['reflect','edge','wrap'],
                             value = padding_mode),
}

# warp_interp = Image.LANCZOS

gui_consistency = {
    "check_consistency":Checkbox(value=check_consistency,description='check_consistency',indent=True, description_tooltip='Enables consistency checking (CC). CC is used to avoid ghosting and trails, that appear due to lack of information while warping frames. It allows replacing motion edges, frame borders, incorrectly moved areas with raw init frame data.'),
    "missed_consistency_weight":FloatSlider(value=missed_consistency_weight, min=0, max=1, step=0.05, description='missed_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for incorrectly predicted\moved areas. For example, if an object moves and background appears behind it. We can predict what to put in that spot, so we can either duplicate the object, resulting in trail, or use init video data for that region.'),
    "overshoot_consistency_weight":FloatSlider(value=overshoot_consistency_weight, min=0, max=1, step=0.05, description='overshoot_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for areas that appeared out of the frame. We can either leave them black or use raw init video.'),
    "edges_consistency_weight":FloatSlider(value=edges_consistency_weight, min=0, max=1, step=0.05, description='edges_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for motion edges. Moving objects are most likely to leave trails, this option together with missed consistency weight helps prevent that, but in a more subtle manner.'),
    "soften_consistency_mask" :  FloatSlider(value=soften_consistency_mask, min=0, max=1, step=0.05, description='soften_consistency_mask:', readout=True, readout_format='.1f'),
    "consistency_blur": FloatText(value = consistency_blur, description='consistency_blur:'),
    "consistency_dilate": FloatText(value = consistency_dilate, description='consistency_dilate:', description_tooltip='expand consistency mask without blurring the edges'),
    "barely used": Label(' '),
    "match_color_strength" : FloatSlider(value=match_color_strength, min=0, max=1, step=0.05, description='match_color_strength:', readout=True, readout_format='.1f', description_tooltip='Enables colormathing raw init video pixls in inconsistent areas only to the stylized frame. May reduce flickering for inconsistent areas.'),
    "mask_result": Checkbox(value=mask_result,description='mask_result',indent=True, description_tooltip='Stylizes only inconsistent areas. Takes consistent areas from the previous frame.'),
    "use_patchmatch_inpaiting": FloatSlider(value=use_patchmatch_inpaiting, min=0, max=1, step=0.05, description='use_patchmatch_inpaiting:', readout=True, readout_format='.1f', description_tooltip='Uses patchmatch inapinting for inconsistent areas. Is slow.'),
}

gui_diffusion = {
    "use_karras_noise":Checkbox(value=use_karras_noise,description='use_karras_noise',indent=True, description_tooltip='Enable for samplers that have K at their name`s end.'),
    "sampler": Dropdown(description='sampler',options= [('sample_euler', sample_euler),
                                  ('sample_euler_ancestral',sample_euler_ancestral),
                                  ('sample_heun',sample_heun),
                                  ('sample_dpm_2', sample_dpm_2),
                                  ('sample_dpm_2_ancestral',sample_dpm_2_ancestral),
                                  ('sample_lms', sample_lms),
                                  ('sample_dpm_fast', sample_dpm_fast),
                                  ('sample_dpm_adaptive',sample_dpm_adaptive),
                                  ('sample_dpmpp_2s_ancestral', sample_dpmpp_2s_ancestral),
                                  ('sample_dpmpp_sde', sample_dpmpp_sde),
                                  ('sample_dpmpp_2m', sample_dpmpp_2m)], value = sampler),
    "prompt_patterns_sched": Textarea(value=str(prompt_patterns_sched),layout=Layout(width=f'80%'),  description = 'Replace patterns:'),
    "text_prompts" : Textarea(value=str(text_prompts),layout=Layout(width=f'80%'),  description = 'Prompt:'),
    "negative_prompts" :  Textarea(value=str(negative_prompts), layout=Layout(width=f'80%'), description = 'Negative Prompt:'),
    "cond_image_src":Dropdown(description='cond_image_src', options = ['init', 'stylized','cond_video'] ,
                            value = cond_image_src, description_tooltip='Depth map source for depth model. It can either take raw init video frame or previously stylized frame.'),
    "inpainting_mask_source":Dropdown(description='inpainting_mask_source', options = ['none', 'consistency_mask', 'cond_video'] ,
                           value = inpainting_mask_source, description_tooltip='Inpainting model mask source. none - full white mask (inpaint whole image), consistency_mask - inpaint inconsistent areas only'),
    "inverse_inpainting_mask":Checkbox(value=inverse_inpainting_mask,description='inverse_inpainting_mask',indent=True, description_tooltip='Inverse inpainting mask'),
    "inpainting_mask_weight":FloatSlider(value=inpainting_mask_weight, min=0, max=1, step=0.05, description='inpainting_mask_weight:', readout=True, readout_format='.1f',
                                         description_tooltip= 'Inpainting mask weight. 0 - Disables inpainting mask.'),
    "set_seed": IntText(value = set_seed, description='set_seed:', description_tooltip='Seed. Use -1 for random.'),
    "clamp_grad":Checkbox(value=clamp_grad,description='clamp_grad',indent=True, description_tooltip='Enable limiting the effect of external conditioning per diffusion step'),
    "clamp_max": FloatText(value = clamp_max, description='clamp_max:',description_tooltip='limit the effect of external conditioning per diffusion step'),
    "latent_scale_schedule":Textarea(value=str(latent_scale_schedule),layout=Layout(width=f'80%'),  description = 'latent_scale_schedule:', description_tooltip='Latents scale defines how much minimize difference between output and input stylized image in latent space.'),
    "init_scale_schedule": Textarea(value=str(init_scale_schedule),layout=Layout(width=f'80%'),  description = 'init_scale_schedule:', description_tooltip='Init scale defines how much minimize difference between output and input stylized image in RGB space.'),
    "sat_scale": FloatText(value = sat_scale, description='sat_scale:', description_tooltip='Saturation scale limits oversaturation.'),
    "init_grad": Checkbox(value=init_grad,description='init_grad',indent=True,  description_tooltip='On - compare output to real frame, Off - to stylized frame'),
    "grad_denoised" : Checkbox(value=grad_denoised,description='grad_denoised',indent=True, description_tooltip='Fastest, On by default, calculate gradients with respect to denoised image instead of input image per diffusion step.' ),
    "steps_schedule" : Textarea(value=str(steps_schedule),layout=Layout(width=f'80%'),  description = 'steps_schedule:',
                               description_tooltip= 'Total diffusion steps schedule. Use list format like [50,70], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:50, 20:70} format to specify keyframes only.'),
    "style_strength_schedule" : Textarea(value=str(style_strength_schedule),layout=Layout(width=f'80%'),  description = 'style_strength_schedule:',
                                          description_tooltip= 'Diffusion (style) strength. Actual number of diffusion steps taken (at 50 steps with 0.3 or 30% style strength you get 15 steps, which also means 35 0r 70% skipped steps). Inverse of skep steps. Use list format like [0.5,0.35], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:0.5, 20:0.35} format to specify keyframes only.'),
    "cfg_scale_schedule": Textarea(value=str(cfg_scale_schedule),layout=Layout(width=f'80%'),  description = 'cfg_scale_schedule:', description_tooltip= 'Guidance towards text prompt. 7 is a good starting value, 1 is off (text prompt has no effect).'),
    "image_scale_schedule": Textarea(value=str(image_scale_schedule),layout=Layout(width=f'80%'),  description = 'image_scale_schedule:', description_tooltip= 'Only used with InstructPix2Pix Model. Guidance towards text prompt. 1.5 is a good starting value'),
    "blend_latent_to_init": FloatSlider(value=blend_latent_to_init, min=0, max=1, step=0.05, description='blend_latent_to_init:', readout=True, readout_format='.1f', description_tooltip = 'Blend latent vector with raw init'),
    # "use_karras_noise": Checkbox(value=False,description='use_karras_noise',indent=True),
    # "end_karras_ramp_early": Checkbox(value=False,description='end_karras_ramp_early',indent=True),
    "fixed_seed": Checkbox(value=fixed_seed,description='fixed_seed',indent=True, description_tooltip= 'Fixed seed.'),
    "fixed_code":  Checkbox(value=fixed_code,description='fixed_code',indent=True, description_tooltip= 'Fixed seed analog. Fixes diffusion noise.'),
    "code_randomness": FloatSlider(value=code_randomness, min=0, max=1, step=0.05, description='code_randomness:', readout=True, readout_format='.1f', description_tooltip= 'Fixed seed amount/effect strength.'),
    # "normalize_code":Checkbox(value=normalize_code,description='normalize_code',indent=True, description_tooltip= 'Whether to normalize the noise after adding fixed seed.'),
    "dynamic_thresh": FloatText(value = dynamic_thresh, description='dynamic_thresh:', description_tooltip= 'Limit diffusion model prediction output. Lower values may introduce clamping/feedback effect'),
    "use_predicted_noise":Checkbox(value=use_predicted_noise,description='use_predicted_noise',indent=True, description_tooltip='Reconstruct initial noise from init / stylized image.'),
    "rec_prompts" : Textarea(value=str(rec_prompts),layout=Layout(width=f'80%'),  description = 'Rec Prompt:'),
    "rec_randomness":   FloatSlider(value=rec_randomness, min=0, max=1, step=0.05, description='rec_randomness:', readout=True, readout_format='.1f', description_tooltip= 'Reconstructed noise randomness. 0 - reconstructed noise only. 1 - random noise.'),
    "rec_cfg": FloatText(value = rec_cfg, description='rec_cfg:', description_tooltip= 'CFG scale for noise reconstruction. 1-1.9 are the best values.'),
    "rec_source": Dropdown(description='rec_source', options = ['init', 'stylized'] ,
                            value = rec_source, description_tooltip='Source for noise reconstruction. Either raw init frame or stylized frame.'),
    "rec_steps_pct":FloatSlider(value=rec_steps_pct, min=0, max=1, step=0.05, description='rec_steps_pct:', readout=True, readout_format='.2f', description_tooltip= 'Reconstructed noise steps in relation to total steps. 1 = 100% steps.'),
    "overwrite_rec_noise":Checkbox(value=overwrite_rec_noise,description='overwrite_rec_noise',indent=True,
                               description_tooltip= 'Overwrite reconstructed noise cache. By default reconstructed noise is not calculated if the settings haven`t changed too much. You can eit prompt, neg prompt, cfg scale,  style strength, steps withot reconstructing the noise every time.'),

    "masked_guidance":Checkbox(value=masked_guidance,description='masked_guidance',indent=True,
                               description_tooltip= 'Use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas.'),
    "cc_masked_diffusion": FloatSlider(value=cc_masked_diffusion, min=0, max=1, step=0.05,
                                 description='cc_masked_diffusion:', readout=True, readout_format='.2f', description_tooltip= '0 - off. 0.5-0.7 are good values. Make inconsistent area passes only before this % of actual steps, then diffuse whole image.'),
    "alpha_masked_diffusion": FloatSlider(value=alpha_masked_diffusion, min=0, max=1, step=0.05,
                                 description='alpha_masked_diffusion:', readout=True, readout_format='.2f', description_tooltip= '0 - off. 0.5-0.7 are good values. Make alpha masked area passes only before this % of actual steps, then diffuse whole image.'),
    "invert_alpha_masked_diffusion":Checkbox(value=invert_alpha_masked_diffusion,description='invert_alpha_masked_diffusion',indent=True,
                               description_tooltip= 'invert alpha ask for masked diffusion'),
    "normalize_prompt_weights":Checkbox(value=normalize_prompt_weights,description='normalize_prompt_weights',indent=True,
                               description_tooltip='Scale prompt weights to sum up to 1.'),
    "deflicker_scale": FloatText(value = deflicker_scale, description='deflicker_scale:',
                                 description_tooltip= 'Deflicker loss scale in image pixel space'),
    "deflicker_latent_scale": FloatText(value = deflicker_latent_scale,
                                        description='deflicker_latent_scale:', description_tooltip= 'Deflicker loss scale in image latent space'),



}
gui_colormatch = {
    "normalize_latent": Dropdown(description='normalize_latent',
                                 options = ['off', 'user_defined', 'color_video', 'color_video_offset',
    'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], value =normalize_latent ,description_tooltip= 'Normalize latent to prevent it from overflowing. User defined: use fixed input values (latent_fixed_*) Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset field below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset filed below).'),
    "normalize_latent_offset":IntText(value = normalize_latent_offset, description='normalize_latent_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "latent_fixed_mean": FloatText(value = latent_fixed_mean, description='latent_fixed_mean:', description_tooltip= 'User defined mean value for normalize_latent=user_Defined mode'),
    "latent_fixed_std": FloatText(value = latent_fixed_std, description='latent_fixed_std:', description_tooltip= 'User defined standard deviation value for normalize_latent=user_Defined mode'),
    "latent_norm_4d": Checkbox(value=latent_norm_4d,description='latent_norm_4d',indent=True, description_tooltip= 'Normalize on a per-channel basis (on by default)'),
    "colormatch_frame": Dropdown(description='colormatch_frame', options = ['off', 'stylized_frame', 'color_video', 'color_video_offset', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'],
                                 value = colormatch_frame,
                                 description_tooltip= 'Match frame colors to prevent it from overflowing.  Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset filed below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset field below).'),
    "color_match_frame_str": FloatText(value = color_match_frame_str, description='color_match_frame_str:', description_tooltip= 'Colormatching strength. 0 - no colormatching effect.'),
    "colormatch_offset":IntText(value =colormatch_offset, description='colormatch_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "colormatch_method": Dropdown(description='colormatch_method', options = ['LAB', 'PDF', 'mean'], value =colormatch_method ),
    # "colormatch_regrain": Checkbox(value=False,description='colormatch_regrain',indent=True),
    "colormatch_after":Checkbox(value=colormatch_after,description='colormatch_after',indent=True, description_tooltip= 'On - Colormatch output frames when saving to disk, may differ from the preview. Off - colormatch before stylizing.'),

}

gui_controlnet = {
    "controlnet_preprocess": Checkbox(value=controlnet_preprocess,description='controlnet_preprocess',indent=True,
                                      description_tooltip= 'preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing.'),
    "detect_resolution":IntText(value = detect_resolution, description='detect_resolution:', description_tooltip= 'Control net conditioning image resolution. The size of the image passed into controlnet preprocessors. Suggest keeping this as high as you can fit into your VRAM for more details.'),
    "bg_threshold":FloatText(value = bg_threshold, description='bg_threshold:', description_tooltip='Control net depth/normal bg cutoff threshold'),
    "low_threshold":IntText(value = low_threshold, description='low_threshold:', description_tooltip= 'Control net canny filter parameters'),
    "high_threshold":IntText(value = high_threshold, description='high_threshold:', description_tooltip= 'Control net canny filter parameters'),
    "value_threshold":FloatText(value = value_threshold, description='value_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "distance_threshold":FloatText(value = distance_threshold, description='distance_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "temporalnet_source":Dropdown(description ='temporalnet_source', options = ['init', 'stylized'] ,
                            value = temporalnet_source, description_tooltip='Temporalnet guidance source. Previous init or previous stylized frame'),
    "temporalnet_skip_1st_frame": Checkbox(value = temporalnet_skip_1st_frame,description='temporalnet_skip_1st_frame',indent=True,
                                      description_tooltip='Skip temporalnet for 1st frame (if not skipped, will use raw init for guidance'),
    "controlnet_multimodel_mode":Dropdown(description='controlnet_multimodel_mode', options = ['internal','external'], value =controlnet_multimodel_mode, description_tooltip='internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.' ),
    "max_faces":IntText(value = max_faces, description='max_faces:', description_tooltip= 'Max faces to detect. Control net face parameters'),
    "controlnet_low_vram":Checkbox(value = controlnet_low_vram,description='controlnet_low_vram',indent=True,
                                      description_tooltip='Only load currently used controlnet to gpu. Slow, saves VRAM.'),
    "save_controlnet_annotations": Checkbox(value = save_controlnet_annotations,description='save_controlnet_annotations',indent=True,
                                      description_tooltip='Save controlnet annotator predictions. They will be saved to your project dir /controlnetDebug folder.'),
    "control_sd15_openpose_hands_face":Checkbox(value = control_sd15_openpose_hands_face,description='control_sd15_openpose_hands_face',indent=True,
                                      description_tooltip='Enable full openpose mode with hands and facial features.'),
    "control_sd15_depth_detector" :Dropdown(description='control_sd15_depth_detector', options = ['Zoe','Midas'], value =control_sd15_depth_detector,
                                            description_tooltip='Depth annotator model.' ),
    "control_sd15_softedge_detector":Dropdown(description='control_sd15_softedge_detector', options = ['HED','PIDI'], value =control_sd15_softedge_detector,
                                            description_tooltip='Softedge annotator model.' ),
    "control_sd15_seg_detector":Dropdown(description='control_sd15_seg_detector', options = ['Seg_OFCOCO', 'Seg_OFADE20K', 'Seg_UFADE20K'], value =control_sd15_seg_detector,
                                            description_tooltip='Segmentation annotator model.' ),
    "control_sd15_scribble_detector":Dropdown(description='control_sd15_scribble_detector', options = ['HED','PIDI'], value =control_sd15_scribble_detector,
                                            description_tooltip='Sccribble annotator model.' ),
    "control_sd15_lineart_coarse":Checkbox(value = control_sd15_lineart_coarse,description='control_sd15_lineart_coarse',indent=True,
                                      description_tooltip='Coarse strokes mode.'),
    "control_sd15_inpaint_mask_source":Dropdown(description='control_sd15_inpaint_mask_source', options = ['consistency_mask', 'None', 'cond_video'], value =control_sd15_inpaint_mask_source,
                                            description_tooltip='Inpainting controlnet mask source. consistency_mask - inpaints inconsistent areas, None - whole image, cond_video - loads external mask' ),
    "control_sd15_shuffle_source":Dropdown(description='control_sd15_shuffle_source', options = ['color_video', 'init', 'prev_frame', 'first_frame'], value =control_sd15_shuffle_source,
                                            description_tooltip='Shuffle controlnet source. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (stylized+warped with consistency mask and flow_blend opacity), prev_frame - uses previously stylized frame (stylized, not warped), first_frame - first stylized frame' ),
    "control_sd15_shuffle_1st_source":Dropdown(description='control_sd15_shuffle_1st_source', options = ['color_video', 'init', 'None'], value =control_sd15_shuffle_1st_source,
                                            description_tooltip='Set 1st frame source for shuffle model. If you need to geet the 1st frame style from your image, and for the consecutive frames you want to use the resulting stylized images. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (raw video frame), None - skips this controlnet for the 1st frame. For example, if you like the 1st frame you`re getting and want to keep its style, but don`t want to use an external image as a source.'),
    "controlnet_multimodel":ControlGUI(controlnet_multimodel)

}

colormatch_regrain = False

guis = [gui_diffusion, gui_controlnet, gui_warp, gui_consistency, gui_turbo, gui_mask, gui_colormatch, gui_misc]

for key in gui_difficulty_dict[gui_difficulty]:
  for gui in guis:
    set_visibility(key, 'hidden', gui)

class FilePath(HBox):
    def __init__(self,  **kwargs):
        self.model_path = Text(value='',  continuous_update = True,**kwargs)
        self.path_checker = Valid(
        value=False, layout=Layout(width='2000px')
        )

        self.model_path.observe(self.on_change)
        super().__init__([self.model_path, self.path_checker])

    def __getattr__(self, attr):
        if attr == 'value':
            return self.model_path.value
        else:
            return super.__getattr__(attr)

    def on_change(self, change):
        if change['name'] == 'value':
            if os.path.exists(change['new']):
                self.path_checker.value = True
                self.path_checker.description = ''
            else:
                self.path_checker.value = False
                self.path_checker.description = 'The file does not exist. Please specify the correct path.'

def add_labels_dict(gui):
    style = {'description_width': '250px' }
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        gui[key].style = style
        # temp = gui[key]
        # temp.observe(dump_gui())
        # gui[key] = temp
        if isinstance(gui[key], ControlGUI):
          continue
        if not isinstance(gui[key], Textarea) and not isinstance( gui[key],Checkbox ):
            # vis = gui[key].layout.visibility
            # gui[key].layout = layout
            gui[key].layout.width = '500px'
        if isinstance( gui[key],Checkbox ):
            html_label = HTML(
                description=gui[key].description,
                description_tooltip=gui[key].description_tooltip,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px'))
            gui_labels[key] = HBox([gui[key],html_label])
            gui_labels[key].layout.visibility = gui[key].layout.visibility
            gui[key].description = ''
            # gui_labels[key] = gui[key]

        else:

            gui_labels[key] = gui[key]
            # gui_labels[key].layout.visibility = gui[key].layout.visibility
        # gui_labels[key].observe(print('smth changed', time.time()))

    return gui_labels


gui_diffusion_label, gui_controlnet_label, gui_warp_label, gui_consistency_label, gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_label = [add_labels_dict(o) for o in guis]

cond_keys = ['latent_scale_schedule','init_scale_schedule','clamp_grad',
             'clamp_max','init_grad','grad_denoised','masked_guidance','deflicker_scale','deflicker_latent_scale' ]
conditioning_w = Accordion([VBox([gui_diffusion_label[o] for o in cond_keys])])
conditioning_w.set_title(0, 'External Conditioning...')

seed_keys = ['set_seed', 'fixed_seed', 'fixed_code', 'code_randomness']
seed_w = Accordion([VBox([gui_diffusion_label[o] for o in seed_keys])])
seed_w.set_title(0, 'Seed...')

rec_keys = ['use_predicted_noise','rec_prompts','rec_cfg','rec_randomness', 'rec_source', 'rec_steps_pct', 'overwrite_rec_noise']
rec_w = Accordion([VBox([gui_diffusion_label[o] for o in rec_keys])])
rec_w.set_title(0, 'Reconstructed noise...')

prompt_keys = ['text_prompts', 'negative_prompts', 'prompt_patterns_sched',
'steps_schedule', 'style_strength_schedule',
'cfg_scale_schedule', 'blend_latent_to_init', 'dynamic_thresh',
'cond_image_src', 'cc_masked_diffusion', 'alpha_masked_diffusion', 'invert_alpha_masked_diffusion', 'normalize_prompt_weights']
if model_version == 'v1_instructpix2pix':
  prompt_keys.append('image_scale_schedule')
if  model_version == 'v1_inpainting':
  prompt_keys+=['inpainting_mask_source', 'inverse_inpainting_mask', 'inpainting_mask_weight']
prompt_keys = [o for o in prompt_keys if o not in seed_keys+cond_keys]
prompt_w = [gui_diffusion_label[o] for o in prompt_keys]

gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'],
gui_diffusion_label['use_karras_noise'], conditioning_w, seed_w, rec_w]

control_annotator_keys = ['controlnet_preprocess','save_controlnet_annotations','detect_resolution','bg_threshold','low_threshold','high_threshold','value_threshold',
                          'distance_threshold', 'max_faces', 'control_sd15_openpose_hands_face','control_sd15_depth_detector' ,'control_sd15_softedge_detector',
'control_sd15_seg_detector','control_sd15_scribble_detector','control_sd15_lineart_coarse','control_sd15_inpaint_mask_source',
'control_sd15_shuffle_source','control_sd15_shuffle_1st_source', 'temporalnet_source', 'temporalnet_skip_1st_frame',]
control_annotator_w = Accordion([VBox([gui_controlnet_label[o] for o in control_annotator_keys])])
control_annotator_w.set_title(0, 'Controlnet annotator settings...')
controlnet_model_w = Accordion([gui_controlnet['controlnet_multimodel']])
controlnet_model_w.set_title(0, 'Controlnet models settings...')
control_keys = [ 'controlnet_multimodel_mode', 'controlnet_low_vram']
control_w = [gui_controlnet_label[o] for o in control_keys]
gui_control_list = [controlnet_model_w, control_annotator_w, *control_w]

#misc
misc_keys = ["user_comment","blend_json_schedules","VERBOSE","offload_model",'sd_batch_size']
misc_w = [gui_misc_label[o] for o in misc_keys]

softcap_keys = ['do_softcap','softcap_thresh','softcap_q']
softcap_w = Accordion([VBox([gui_misc_label[o] for o in softcap_keys])])
softcap_w.set_title(0, 'Softcap settings...')

load_settings_btn = Button(description='Load settings')
def btn_eventhandler(obj):
  load_settings(load_settings_path.value)
load_settings_btn.on_click(btn_eventhandler)
load_settings_path = FilePath(placeholder='Please specify the path to the settings file to load.', description_tooltip='Please specify the path to the settings file to load.')
settings_w = Accordion([VBox([load_settings_path, load_settings_btn])])
settings_w.set_title(0, 'Load settings...')
gui_misc_list = [*misc_w, softcap_w, settings_w]

guis_labels_source = [gui_diffusion_list]
guis_titles_source = ['diffusion']
if 'control' in model_version:
  guis_labels_source += [gui_control_list]
  guis_titles_source += ['controlnet']

guis_labels_source += [gui_warp_label, gui_consistency_label,
gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_list]
guis_titles_source += ['warp', 'consistency', 'turbo', 'mask', 'colormatch', 'misc']

guis_labels = [VBox([*o.values()]) if isinstance(o, dict) else VBox(o) for o in guis_labels_source]

app = Tab(guis_labels)
for i,title in enumerate(guis_titles_source):
    app.set_title(i, title)

def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key].value
        else:
            for o in obj.keys():
                res = get_value(key, obj[o])
                if res is not None: return res
    if isinstance(obj, list):
        for o in obj:
            res = get_value(key, o)
            if res is not None: return res
    return None

def set_value(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].value = value
        else:
            for o in obj.keys():
                set_value(key, value, obj[o])

    if isinstance(obj, list):
        for o in obj:
            set_value(key, value, o)



import json
def infer_settings_path(path):
    default_settings_path = path
    if default_settings_path == '-1':
      settings_files = sorted(glob(os.path.join(settings_out, '*.txt')))
      if len(settings_files)>0:
        default_settings_path = settings_files[-1]
      else:
        print('Skipping load latest run settings: no settings files found.')
        return ''
    else:
      try:
        if type(eval(default_settings_path)) == int:
          files = sorted(glob(os.path.join(settings_out, '*.txt')))
          for f in files:
            if f'({default_settings_path})' in f:
              default_settings_path = f
      except: pass

    path = default_settings_path
    return path

def load_settings(path):
    path = infer_settings_path(path)

    global guis, load_settings_path, output
    if not os.path.exists(path):
        output.clear_output()
        print('Please specify a valid path to a settings file.')
        return
    if path.endswith('png'):
      img = PIL.Image.open(path)
      exif_data = img._getexif()
      settings = json.loads(exif_data[37510])

    else:
      print('Loading settings from: ', path)
      with open(path, 'rb') as f:
          settings = json.load(f)

    for key in settings:
        try:
            val = settings[key]
            if key == 'normalize_latent' and val == 'first_latent':
              val = 'init_frame'
              settings['normalize_latent_offset'] = 0
            if key == 'turbo_frame_skips_steps' and val == None:
                val = '100% (don`t diffuse turbo frames, fastest)'
            if key == 'seed':
                key = 'set_seed'
            if key == 'grad_denoised ':
                key = 'grad_denoised'
            if type(val) in [dict,list]:
                if type(val) in [dict]:
                  temp = {}
                  for k in val.keys():
                    temp[int(k)] = val[k]
                  val = temp
                val = json.dumps(val)
            if key == 'mask_clip':
              val = eval(val)
            if key == 'sampler':
              val = getattr(K.sampling, val)
            if key == 'controlnet_multimodel':
              val = val.replace('control_sd15_hed', 'control_sd15_softedge')
              val = json.loads(val)
            # print(key, val)
            set_value(key, val, guis)
            # print(get_value(key, guis))
        except Exception as e:
            print(key), print(settings[key] )
            print(e)
    # output.clear_output()
    print('Successfully loaded settings from ', path )

def dump_gui():
  print('smth changed', time.time())

output = Output()
if settings_path != '' and load_settings_from_file:
  load_settings(settings_path)


display.display(app)

"""### Reference controlnet (attention injection)
By Lvmin Zhang (https://github.com/lllyasviel)

https://github.com/Mikubill/sd-webui-controlnet
"""

# Attention Injection by Lvmin Zhang
# https://github.com/Mikubill/sd-webui-controlnet

from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import BasicTransformerBlock
from enum import Enum



class AttentionAutoMachine(Enum):
    """
    Lvmin's algorithm for Attention AutoMachine States.
    """

    Read = "Read"
    Write = "Write"

# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def hacked_basic_transformer_inner_forward(self, x, context=None):
            x_norm1 = self.norm1(x)
            self_attn1 = 0
            if self.disable_self_attn:
                # Do not use self-attention
                self_attn1 = self.attn1(x_norm1, context=context)
            else:
                # Use self-attention
                self_attention_context = x_norm1
                if outer.attention_auto_machine == AttentionAutoMachine.Write:
                    self.bank.append(self_attention_context.detach().clone())
                if outer.attention_auto_machine == AttentionAutoMachine.Read:
                    if outer.attention_auto_machine_weight > self.attn_weight:
                        self_attention_context = torch.cat([self_attention_context] + self.bank, dim=1)
                    self.bank.clear()
                self_attn1 = self.attn1(x_norm1, context=self_attention_context)

            x = self_attn1 + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
            return x



# Attention Injection by Lvmin Zhang
# https://github.com/lllyasviel
# https://github.com/Mikubill/sd-webui-controlnet
outer = sd_model.model.diffusion_model
def control_forward(x, timesteps=None, context=None, control=None, only_mid_control=False, self=outer, **kwargs):
        if reference_latent is not None:
          # print('Using reference')
          query_size = int(x.shape[0])
          used_hint_cond_latent = reference_latent
          try:
            uc_mask_shape = outer.uc_mask_shape
          except:
            uc_mask_shape = [0,1]
          uc_mask = torch.tensor(uc_mask_shape, dtype=x.dtype, device=x.device)[:, None, None, None]
          ref_cond_xt = sd_model.q_sample(used_hint_cond_latent, torch.round(timesteps.float()).long())

          if reference_mode=='Controlnet':
                              ref_uncond_xt = x.clone()
                              # print('ControlNet More Important -  Using standard cfg for reference.')
          elif reference_mode=='Prompt':
                              ref_uncond_xt = ref_cond_xt.clone()
                              # print('Prompt More Important -  Using no cfg for reference.')
          else:
                              ldm_time_max = getattr(sd_model, 'num_timesteps', 1000)
                              time_weight = (timesteps.float() / float(ldm_time_max)).clip(0, 1)[:, None, None, None]
                              time_weight *= torch.pi * 0.5
                              # We should use sin/cos to make sure that the std of weighted matrix follows original ddpm schedule
                              ref_uncond_xt = x * torch.sin(time_weight) + ref_cond_xt.clone() * torch.cos(time_weight)
                              # print('Balanced - Using time-balanced cfg for reference.')
          for module in outer.attn_module_list:
                module.bank = []
          ref_xt = ref_cond_xt * uc_mask + ref_uncond_xt * (1 - uc_mask)
          outer.attention_auto_machine = AttentionAutoMachine.Write
          # print('ok')
          outer.original_forward(x=ref_xt, timesteps=timesteps, context=context)
          outer.attention_auto_machine = AttentionAutoMachine.Read
          outer.attention_auto_machine_weight = reference_weight

        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)




use_reference = False #@param {'type':'boolean'}
reference_weight = 0.5 #@param
reference_source = 'init' #@param ['stylized', 'init', 'prev_frame','color_video']
reference_mode = 'Balanced' #@param ['Balanced', 'Controlnet', 'Prompt']

reference_active = reference_weight>0 and use_reference and reference_source != 'None'
if reference_active:
  # outer = sd_model.model.diffusion_model
  try:
    outer.forward = outer.original_forward
  except: pass
  outer.original_forward = outer.forward
  outer.attention_auto_machine_weight = reference_weight
  outer.forward = control_forward
  outer.attention_auto_machine = AttentionAutoMachine.Read
  print('Using reference control.')

  attn_modules = [module for module in torch_dfs(outer) if isinstance(module, BasicTransformerBlock)]
  attn_modules = sorted(attn_modules, key=lambda x: - x.norm1.normalized_shape[0])

  for i, module in enumerate(attn_modules):
              module._original_inner_forward = module._forward
              module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
              module.bank = []
              module.attn_weight = float(i) / float(len(attn_modules))

  outer.attn_module_list = attn_modules
  for module in outer.attn_module_list:
                  module.bank = []

"""# 4. Diffuse!
if you are having OOM or PIL error here click "restart and run all" once.
"""

#@title Do the Run!
#@markdown Preview max size
deflicker_scale = 0. #makes glitches :D
deflicker_latent_scale = 0.
fft_scale = 0.
fft_latent_scale = 0.

sd_model.low_vram = True if controlnet_low_vram else False

mask_frames_many = None
if mask_paths != []:
  mask_frames_many = []
  for i in range(len(mask_paths)) :
    mask_path = mask_paths[i]
    prefix = f'mask_{i}'
    mask_frames_many.append(FrameDataset(mask_path, outdir_prefix=prefix,
                                        videoframes_root=f'{batchFolder}/videoFrames'))

from glob import glob
controlnet_multimodel = get_value('controlnet_multimodel',guis)
image_prompts = {}
controlnet_multimodel_temp = {}
for key in controlnet_multimodel.keys():

  weight = controlnet_multimodel[key]["weight"]
  if weight !=0 :
    controlnet_multimodel_temp[key] = controlnet_multimodel[key]
controlnet_multimodel = controlnet_multimodel_temp

inverse_mask_order = False
can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(getattr(torch.nn.functional, "scaled_dot_product_attention")) # not everyone has torch 2.x to use sdp
if can_use_sdp:
  shared.opts.xformers = False
  shared.cmd_opts.xformers = False

import copy
apply_depth = None;
apply_canny = None; apply_mlsd = None;
apply_hed = None; apply_openpose = None;
apply_seg = None;
loaded_controlnets = {}
torch.cuda.empty_cache(); gc.collect();
sd_model.control_scales = ([1]*13)
if model_version == 'control_multi':
  sd_model.control_model.cpu()
  print('Checking downloaded Annotator and ControlNet Models')
  for controlnet in controlnet_multimodel.keys():
    controlnet_settings = controlnet_multimodel[controlnet]
    weight = controlnet_settings["weight"]
    if weight!=0:
      small_url = control_model_urls[controlnet]
      local_filename = small_url.split('/')[-1]
      small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
      if use_small_controlnet and os.path.exists(model_path) and not os.path.exists(small_controlnet_model_path):
        print(f'Model found at {model_path}. Small model not found at {small_controlnet_model_path}.')
        if not os.path.exists(small_controlnet_model_path) or force_download:
          try:
            pathlib.Path(small_controlnet_model_path).unlink()
          except: pass
          print(f'Downloading small {controlnet} model... ')
          wget.download(small_url,  small_controlnet_model_path)
          print(f'Downloaded small {controlnet} model.')


      # helper_names = control_helpers[controlnet]
      # if helper_names is not None:
      #     if type(helper_names) == str: helper_names = [helper_names]
      #     for helper_name in helper_names:
      #       helper_model_url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/'+helper_name
      #       helper_model_path = f'{root_dir}/ControlNet/annotator/ckpts/'+helper_name
      #       if not os.path.exists(helper_model_path) or force_download:
      #         try:
      #           pathlib.Path(helper_model_path).unlink()
      #         except: pass
      #         wget.download(helper_model_url, helper_model_path)

  print('Loading ControlNet Models')
  loaded_controlnets = {}
  for controlnet in controlnet_multimodel.keys():
    controlnet_settings = controlnet_multimodel[controlnet]
    weight = controlnet_settings["weight"]
    if weight!=0:
      loaded_controlnets[controlnet] = copy.deepcopy(sd_model.control_model)
      small_url = control_model_urls[controlnet]
      local_filename = small_url.split('/')[-1]
      small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
      if os.path.exists(small_controlnet_model_path):
          ckpt = small_controlnet_model_path
          print(f"Loading model from {ckpt}")
          if ckpt.endswith('.safetensors'):
            pl_sd = {}
            with safe_open(ckpt, framework="pt", device=load_to) as f:
              for key in f.keys():
                  pl_sd[key] = f.get_tensor(key)
          else: pl_sd = torch.load(ckpt, map_location=load_to)

          if "global_step" in pl_sd:
              print(f"Global Step: {pl_sd['global_step']}")
          if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
          else: sd = pl_sd
          if "control_model.input_blocks.0.0.bias" in sd:


            sd = dict([(o.split('control_model.')[-1],sd[o]) for o in sd.keys() if o != 'difference'])

            # print('control_model in sd')
          del pl_sd

          gc.collect()
          m, u = loaded_controlnets[controlnet].load_state_dict(sd, strict=True)
          loaded_controlnets[controlnet].half()
          if len(m) > 0 and verbose:
              print("missing keys:")
              print(m, len(m))
          if len(u) > 0 and verbose:
              print("unexpected keys:")
              print(u, len(u))
      else:
        print('Small controlnet model not found in path but specified in settings. Please adjust settings or check controlnet path.')
        sys.exit(0)


# print('Loading annotators.')
controlnet_keys = controlnet_multimodel.keys() if model_version == 'control_multi' else model_version
if "control_sd15_depth" in controlnet_keys or "control_sd15_normal" in controlnet_keys:
        if control_sd15_depth_detector == 'Midas' or "control_sd15_normal" in controlnet_keys:
          from annotator.midas import MidasDetector
          apply_depth = MidasDetector()
          print('Loaded MidasDetector')
        if control_sd15_depth_detector == 'Zoe':
          from annotator.zoe import ZoeDetector
          apply_depth = ZoeDetector()
          print('Loaded ZoeDetector')

if "control_sd15_normalbae" in controlnet_keys:
        from annotator.normalbae import NormalBaeDetector
        apply_normal = NormalBaeDetector()
        print('Loaded NormalBaeDetector')
if 'control_sd15_canny' in controlnet_keys :
        from annotator.canny import CannyDetector
        apply_canny = CannyDetector()
        print('Loaded CannyDetector')
if 'control_sd15_softedge' in controlnet_keys:
        if control_sd15_softedge_detector == 'HED':
          from annotator.hed import HEDdetector
          apply_softedge = HEDdetector()
          print('Loaded HEDdetector')
        if control_sd15_softedge_detector == 'PIDI':
          from annotator.pidinet import PidiNetDetector
          apply_softedge = PidiNetDetector()
          print('Loaded PidiNetDetector')
if 'control_sd15_scribble' in controlnet_keys:
        from annotator.util import nms
        if control_sd15_scribble_detector == 'HED':
          from annotator.hed import HEDdetector
          apply_scribble = HEDdetector()
          print('Loaded HEDdetector')
        if control_sd15_scribble_detector == 'PIDI':
          from annotator.pidinet import PidiNetDetector
          apply_scribble = PidiNetDetector()
          print('Loaded PidiNetDetector')

if "control_sd15_mlsd" in controlnet_keys:
        from annotator.mlsd import MLSDdetector
        apply_mlsd = MLSDdetector()
        print('Loaded MLSDdetector')
if "control_sd15_openpose" in controlnet_keys:
        from annotator.openpose import OpenposeDetector
        apply_openpose = OpenposeDetector()
        print('Loaded OpenposeDetector')
if "control_sd15_seg" in controlnet_keys:
        if control_sd15_seg_detector == 'Seg_OFCOCO':
          from annotator.oneformer import OneformerCOCODetector
          apply_seg = OneformerCOCODetector()
          print('Loaded OneformerCOCODetector')
        elif control_sd15_seg_detector == 'Seg_OFADE20K':
          from annotator.oneformer import OneformerADE20kDetector
          apply_seg = OneformerADE20kDetector()
          print('Loaded OneformerADE20kDetector')
        elif control_sd15_seg_detector == 'Seg_UFADE20K':
          from annotator.uniformer import UniformerDetector
          apply_seg = UniformerDetector()
          print('Loaded UniformerDetector')
if "control_sd15_shuffle" in controlnet_keys:
        from annotator.shuffle import ContentShuffleDetector
        apply_shuffle = ContentShuffleDetector()
        print('Loaded ContentShuffleDetector')

# if "control_sd15_ip2p" in controlnet_keys:
#   #no annotator
#   pass
# if "control_sd15_inpaint" in controlnet_keys:
#   #no annotator
#   pass
if "control_sd15_lineart" in controlnet_keys:
  from annotator.lineart import LineartDetector
  apply_lineart = LineartDetector()
  print('Loaded LineartDetector')
if "control_sd15_lineart_anime" in controlnet_keys:
  from annotator.lineart_anime import LineartAnimeDetector
  apply_lineart_anime = LineartAnimeDetector()
  print('Loaded LineartAnimeDetector')

def deflicker_loss(processed2, processed1, raw1, raw2, criterion1, criterion2):
  raw_diff = criterion1(raw2, raw1)
  proc_diff = criterion1(processed1, processed2)
  return criterion2(raw_diff, proc_diff)




unload(torch)
sd_model.cuda()
sd_hijack.model_hijack.hijack(sd_model)
sd_hijack.model_hijack.embedding_db.add_embedding_dir(custom_embed_dir)
sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(sd_model, force_reload=True)


latent_scale_schedule=eval(get_value('latent_scale_schedule',guis))
init_scale_schedule=eval(get_value('init_scale_schedule',guis))
steps_schedule=eval(get_value('steps_schedule',guis))
style_strength_schedule=eval(get_value('style_strength_schedule',guis))
cfg_scale_schedule=eval(get_value('cfg_scale_schedule',guis))
flow_blend_schedule=eval(get_value('flow_blend_schedule',guis))
image_scale_schedule=eval(get_value('image_scale_schedule',guis))

latent_scale_schedule_bkup = copy.copy(latent_scale_schedule)
init_scale_schedule_bkup = copy.copy(init_scale_schedule)
steps_schedule_bkup = copy.copy(steps_schedule)
style_strength_schedule_bkup = copy.copy(style_strength_schedule)
flow_blend_schedule_bkup = copy.copy(flow_blend_schedule)
cfg_scale_schedule_bkup = copy.copy(cfg_scale_schedule)
image_scale_schedule_bkup = copy.copy(image_scale_schedule)

if make_schedules:
  if diff is None and diff_override == []: sys.exit(f'\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous cell, run it, and then run this cell again\n')
  if diff_override != []: diff = diff_override

  print('Applied schedules:')
  latent_scale_schedule = check_and_adjust_sched(latent_scale_schedule, latent_scale_template, diff, respect_sched)
  init_scale_schedule = check_and_adjust_sched(init_scale_schedule, init_scale_template, diff, respect_sched)
  steps_schedule = check_and_adjust_sched(steps_schedule, steps_template, diff, respect_sched)
  style_strength_schedule = check_and_adjust_sched(style_strength_schedule, style_strength_template, diff, respect_sched)
  flow_blend_schedule = check_and_adjust_sched(flow_blend_schedule, flow_blend_template, diff, respect_sched)
  cfg_scale_schedule = check_and_adjust_sched(cfg_scale_schedule, cfg_scale_template, diff, respect_sched)
  image_scale_schedule = check_and_adjust_sched(image_scale_schedule, cfg_scale_template, diff, respect_sched)
  for sched, name in zip([latent_scale_schedule,   init_scale_schedule,  steps_schedule,  style_strength_schedule,  flow_blend_schedule,
  cfg_scale_schedule, image_scale_schedule], ['latent_scale_schedule',   'init_scale_schedule',  'steps_schedule',  'style_strength_schedule',  'flow_blend_schedule',
  'cfg_scale_schedule', 'image_scale_schedule']):
    if type(sched) == list:
      if len(sched)>2:
        print(name, ': ', sched[:100])

use_karras_noise = False
end_karras_ramp_early = False
# use_predicted_noise = False
warp_interp = Image.LANCZOS
start_code_cb = None #variable for cb_code
guidance_start_code = None #variable for guidance code

display_size = 720 #@param

user_comment= get_value('user_comment',guis)
blend_json_schedules=get_value('blend_json_schedules',guis)
VERBOSE=get_value('VERBOSE',guis)
use_background_mask=get_value('use_background_mask',guis)
invert_mask=get_value('invert_mask',guis)
background=get_value('background',guis)
background_source=get_value('background_source',guis)
(mask_clip_low, mask_clip_high) = get_value('mask_clip',guis)

#turbo
turbo_mode=get_value('turbo_mode',guis)
turbo_steps=get_value('turbo_steps',guis)
colormatch_turbo=get_value('colormatch_turbo',guis)
turbo_frame_skips_steps=get_value('turbo_frame_skips_steps',guis)
soften_consistency_mask_for_turbo_frames=get_value('soften_consistency_mask_for_turbo_frames',guis)

#warp
flow_warp= get_value('flow_warp',guis)
apply_mask_after_warp=get_value('apply_mask_after_warp',guis)
warp_num_k=get_value('warp_num_k',guis)
warp_forward=get_value('warp_forward',guis)
warp_strength=get_value('warp_strength',guis)
flow_override_map=eval(get_value('flow_override_map',guis))
warp_mode=get_value('warp_mode',guis)
warp_towards_init=get_value('warp_towards_init',guis)

#cc
check_consistency=get_value('check_consistency',guis)
missed_consistency_weight=get_value('missed_consistency_weight',guis)
overshoot_consistency_weight=get_value('overshoot_consistency_weight',guis)
edges_consistency_weight=get_value('edges_consistency_weight',guis)
consistency_blur=get_value('consistency_blur',guis)
consistency_dilate=get_value('consistency_dilate',guis)
padding_ratio=get_value('padding_ratio',guis)
padding_mode=get_value('padding_mode',guis)
match_color_strength=get_value('match_color_strength',guis)
soften_consistency_mask=get_value('soften_consistency_mask',guis)
mask_result=get_value('mask_result',guis)
use_patchmatch_inpaiting=get_value('use_patchmatch_inpaiting',guis)

#diffusion
text_prompts=eval(get_value('text_prompts',guis))
negative_prompts=eval(get_value('negative_prompts',guis))
prompt_patterns_sched = eval(get_value('prompt_patterns_sched',guis))
cond_image_src=get_value('cond_image_src',guis)
set_seed=get_value('set_seed',guis)
clamp_grad=get_value('clamp_grad',guis)
clamp_max=get_value('clamp_max',guis)
sat_scale=get_value('sat_scale',guis)
init_grad=get_value('init_grad',guis)
grad_denoised=get_value('grad_denoised',guis)
blend_latent_to_init=get_value('blend_latent_to_init',guis)
fixed_code=get_value('fixed_code',guis)
code_randomness=get_value('code_randomness',guis)
# normalize_code=get_value('normalize_code',guis)
dynamic_thresh=get_value('dynamic_thresh',guis)
sampler = get_value('sampler',guis)
use_karras_noise = get_value('use_karras_noise',guis)
inpainting_mask_weight = get_value('inpainting_mask_weight',guis)
inverse_inpainting_mask = get_value('inverse_inpainting_mask',guis)
inpainting_mask_source = get_value('mask_source',guis)

#colormatch
normalize_latent=get_value('normalize_latent',guis)
normalize_latent_offset=get_value('normalize_latent_offset',guis)
latent_fixed_mean=eval(str(get_value('latent_fixed_mean',guis)))
latent_fixed_std=eval(str(get_value('latent_fixed_std',guis)))
latent_norm_4d=get_value('latent_norm_4d',guis)
colormatch_frame=get_value('colormatch_frame',guis)
color_match_frame_str=get_value('color_match_frame_str',guis)
colormatch_offset=get_value('colormatch_offset',guis)
colormatch_method=get_value('colormatch_method',guis)
colormatch_regrain=get_value('colormatch_regrain',guis)
colormatch_after=get_value('colormatch_after',guis)
image_prompts = {}

fixed_seed = get_value('fixed_seed',guis)

rec_cfg = get_value('rec_cfg',guis)
rec_steps_pct = get_value('rec_steps_pct',guis)
rec_prompts = eval(get_value('rec_prompts',guis))
rec_randomness = get_value('rec_randomness',guis)
use_predicted_noise = get_value('use_predicted_noise',guis)
overwrite_rec_noise  = get_value('overwrite_rec_noise',guis)

#controlnet
save_controlnet_annotations = get_value('save_controlnet_annotations',guis)
control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face',guis)
control_sd15_depth_detector  = get_value('control_sd15_depth_detector',guis)
control_sd15_softedge_detector = get_value('control_sd15_softedge_detector',guis)
control_sd15_seg_detector = get_value('control_sd15_seg_detector',guis)
control_sd15_scribble_detector = get_value('control_sd15_scribble_detector',guis)
control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse',guis)
control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source',guis)
control_sd15_shuffle_source = get_value('control_sd15_shuffle_source',guis)
control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source',guis)
controlnet_preprocess = get_value('controlnet_preprocess',guis)

detect_resolution  = get_value('detect_resolution',guis)
bg_threshold = get_value('bg_threshold',guis)
low_threshold = get_value('low_threshold',guis)
high_threshold = get_value('high_threshold',guis)
value_threshold = get_value('value_threshold',guis)
distance_threshold = get_value('distance_threshold',guis)
temporalnet_source = get_value('temporalnet_source',guis)
temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame',guis)
controlnet_multimodel_mode = get_value('controlnet_multimodel_mode',guis)
max_faces = get_value('max_faces',guis)

do_softcap = get_value('do_softcap',guis)
softcap_thresh = get_value('softcap_thresh',guis)
softcap_q = get_value('softcap_q',guis)

masked_guidance = get_value('masked_guidance',guis)
cc_masked_diffusion = get_value('cc_masked_diffusion',guis)
alpha_masked_diffusion = get_value('alpha_masked_diffusion',guis)

normalize_prompt_weights = get_value('normalize_prompt_weights',guis)
sd_batch_size = get_value('sd_batch_size',guis)
controlnet_low_vram = get_value('controlnet_low_vram',guis)
mask_paths = eval(get_value('mask_paths',guis))
deflicker_scale = get_value('deflicker_scale',guis)
deflicker_latent_scale = get_value('deflicker_latent_scale',guis)

if turbo_frame_skips_steps == '100% (don`t diffuse turbo frames, fastest)':
  turbo_frame_skips_steps = None
else:
  turbo_frame_skips_steps = int(turbo_frame_skips_steps.split('%')[0])/100

disable_cc_for_turbo_frames = False

colormatch_method_fn = PT.lab_transfer
if colormatch_method == 'PDF':
  colormatch_method_fn = PT.pdf_transfer
if colormatch_method == 'mean':
  colormatch_method_fn = PT.mean_std_transfer

turbo_preroll = 1
intermediate_saves = None
intermediates_in_subfolder = True
steps_per_checkpoint = None

forward_weights_clip = soften_consistency_mask
forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
inpaint_blend = 0

if animation_mode == 'Video Input':
  max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

def split_prompts(prompts):
  prompt_series = pd.Series([np.nan for a in range(max_frames)])
  for i, prompt in prompts.items():
    prompt_series[i] = prompt
  # prompt_series = prompt_series.astype(str)
  prompt_series = prompt_series.ffill().bfill()
  return prompt_series

key_frames = True
interp_spline = 'Linear'
perlin_init = False
perlin_mode = 'mixed'

if warp_towards_init != 'off':
  if flow_lq:
          raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        # raft_model = torch.nn.DataParallel(RAFT(args2))
  else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()


def printf(*msg, file=f'{root_dir}/log.txt'):
  now = datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
  with open(file, 'a') as f:
      msg = f'{dt_string}> {" ".join([str(o) for o in (msg)])}'
      print(msg, file=f)
printf('--------Beginning new run------')
##@markdown `n_batches` ignored with animation modes.
display_rate =  9999999
##@param{type: 'number'}
n_batches =  1
##@param{type: 'number'}
start_code = None
first_latent = None
first_latent_source = 'not set'
os.chdir(root_dir)
n_mean_avg = None
n_std_avg = None
n_smooth = 0.5
#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps

batch_size = 1

def move_files(start_num, end_num, old_folder, new_folder):
    for i in range(start_num, end_num):
        old_file = old_folder + f'/{batch_name}({batchNum})_{i:06}.png'
        new_file = new_folder + f'/{batch_name}({batchNum})_{i:06}.png'
        os.rename(old_file, new_file)

noise_upscale_ratio = int(noise_upscale_ratio)
#@markdown ---
#@markdown Frames to run. Leave empty or [0,0] to run all frames.
frame_range = [0,0] #@param
resume_run = False #@param{type: 'boolean'}
run_to_resume = 'latest' #@param{type: 'string'}
resume_from_frame = 'latest' #@param{type: 'string'}
retain_overwritten_frames = False #@param{type: 'boolean'}
if retain_overwritten_frames is True:
  retainFolder = f'{batchFolder}/retained'
  createPath(retainFolder)




if animation_mode == 'Video Input':
  frames = sorted(glob(in_path+'/*.*'));
  if len(frames)==0:
    sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
  flows = glob(flo_folder+'/*.*')
  if (len(flows)==0) and flow_warp:
    sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")
settings_out = batchFolder+f"/settings"
if resume_run:
  if run_to_resume == 'latest':
    try:
      batchNum
    except:
      batchNum = len(glob(f"{settings_out}/{batch_name}(*)_settings.txt"))-1
  else:
    batchNum = int(run_to_resume)
  if resume_from_frame == 'latest':
    start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
    if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(turbo_steps))
  else:
    start_frame = int(resume_from_frame)+1
    if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(turbo_steps))
    if retain_overwritten_frames is True:
      existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
      frames_to_save = existing_frames - start_frame
      print(f'Moving {frames_to_save} frames to the Retained folder')
      move_files(start_frame, existing_frames, batchFolder, retainFolder)
else:
  start_frame = 0
  batchNum = len(glob(settings_out+"/*.txt"))
  while os.path.isfile(f"{settings_out}/{batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True:
    batchNum += 1

print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed' or set_seed == -1:
    random.seed()
    seed = random.randint(0, 2**32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

new_prompt_loras = {}
prompt_weights = {}
if text_prompts:
  _, new_prompt_loras = split_lora_from_prompts(text_prompts)

  print('Inferred loras schedule:\n', new_prompt_loras)
  _, prompt_weights = get_prompt_weights(text_prompts)

  print('---prompt_weights---', prompt_weights, text_prompts)
if new_prompt_loras not in [{}, [], '', None]:
#inject lora even with empty weights to unload?
  if use_lycoris:
    unload(torch)
    inject_lyco(sd_model)
  else:
    unload(torch)
    inject_lora(sd_model)

else:
  unload(torch)

args = {
    'batchNum': batchNum,
    'prompts_series':text_prompts if text_prompts else None,
    'rec_prompts_series':rec_prompts if rec_prompts else None,
    'neg_prompts_series':negative_prompts if negative_prompts else None,
    'image_prompts_series':image_prompts if image_prompts else None,

    'seed': seed,
    'display_rate':display_rate,
    'n_batches':n_batches if animation_mode == 'None' else 1,
    'batch_size':batch_size,
    'batch_name': batch_name,
    'steps': steps,
    'diffusion_sampling_mode': diffusion_sampling_mode,
    'width_height': width_height,
    'clip_guidance_scale': clip_guidance_scale,
    'tv_scale': tv_scale,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'cutn_batches': cutn_batches,
    'init_image': init_image,
    'init_scale': init_scale,
    'skip_steps': skip_steps,
    'side_x': side_x,
    'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'animation_mode': animation_mode,
    'video_init_path': video_init_path,
    'extract_nth_frame': extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'key_frames': key_frames,
    'max_frames': max_frames if animation_mode != "None" else 1,
    'interp_spline': interp_spline,
    'start_frame': start_frame,
    'padding_mode': padding_mode,
    'text_prompts': text_prompts,
    'image_prompts': image_prompts,
    'intermediate_saves': intermediate_saves,
    'intermediates_in_subfolder': intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'perlin_init': perlin_init,
    'perlin_mode': perlin_mode,
    'set_seed': set_seed,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'skip_augs': skip_augs,
}
if frame_range not in [None, [0,0], '', [0], 0]:
  args['start_frame'] = frame_range[0]
  args['max_frames'] = min(args['max_frames'],frame_range[1])
args = SimpleNamespace(**args)

import traceback

gc.collect()
torch.cuda.empty_cache()
try:
  do_run()
except:
  traceback.print_exc()
  sys.exit()

print('n_stats_avg (mean, std): ', n_mean_avg, n_std_avg)

gc.collect()
torch.cuda.empty_cache()

"""# 5. Create the video"""

import PIL
#@title ### **Create video**
#@markdown Video file will save in the same folder as your images.
from tqdm.notebook import trange
skip_video_for_run_all = False #@param {type: 'boolean'}
#@markdown ### **Video masking (post-processing)**
#@markdown Use previously generated background mask during video creation
use_background_mask_video = False #@param {type: 'boolean'}
invert_mask_video = False #@param {type: 'boolean'}
#@markdown Choose background source: image, color, init video.
background_video = "init_video" #@param ['image', 'color', 'init_video']
#@markdown Specify the init image path or color depending on your background video source choice.
background_source_video = 'red' #@param {type: 'string'}
blend_mode = "optical flow" #@param ['None', 'linear', 'optical flow']
# if (blend_mode == "optical flow") & (animation_mode != 'Video Input Legacy'):
#@markdown ### **Video blending (post-processing)**
#   print('Please enable Video Input mode and generate optical flow maps to use optical flow blend mode')
blend =  0.5#@param {type: 'number'}
check_consistency = True #@param {type: 'boolean'}
postfix = ''
missed_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
# bitrate = 10 #@param {'type':'slider', 'min':'5', 'max':'28', 'step':'1'}
failed_frames = []

def try_process_frame(i, func):
    global failed_frames
    try:
        func(i)
    except:
        print('Error processing frame ', i)

        print('retrying 1 time')
        gc.collect()
        torch.cuda.empty_cache()
        try:
          func(i)
        except Exception as e:
          print('Error processing frame ', i, '. Please lower thread number to 1-3.', e)
          failed_frames.append(i)




if use_background_mask_video:
  postfix+='_mask'
  if invert_mask_video:
    postfix+='_inv'
#@markdown #### Upscale settings
upscale_ratio = "1" #@param [1,2,3,4]
upscale_ratio = int(upscale_ratio)
upscale_model = 'realesr-animevideov3' #@param ['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus', 'realesr-animevideov3', 'realesr-general-x4v3']

#@markdown #### Multithreading settings
#@markdown Suggested range - from 1 to number of cores on SSD and double number of cores - on HDD. Mostly limited by your drive bandwidth.
#@markdown Results for 500 frames @ 6 cores: 5 threads - 2:38, 10 threads - 0:55, 20 - 0:56, 1: 5:53
threads = 12#@param {type:"number"}
threads = max(min(threads, 64),1)
frames = []
if upscale_ratio>1:
  try:
    for key in loaded_controlnets.keys():
      loaded_controlnets[key].cpu()
  except: pass
  try:
    sd_model.model.cpu()
    sd_model.cond_stage_model.cpu()
    sd_model.cpu()
    sd_model.first_stage_model.cpu()
    model_wrap.inner_model.cpu()
    model_wrap.cpu()
    model_wrap_cfg.cpu()
    model_wrap_cfg.inner_model.cpu()
  except: pass
  torch.cuda.empty_cache()
  gc.collect()
  torch.cuda.empty_cache()
  gc.collect()

  os.chdir(f'{root_dir}/Real-ESRGAN')
  print(f'Upscaling to x{upscale_ratio}  using {upscale_model}')
  from realesrgan.archs.srvgg_arch import SRVGGNetCompact
  from basicsr.utils.download_util import load_file_from_url
  from realesrgan import RealESRGANer
  from basicsr.archs.rrdbnet_arch import RRDBNet
  os.chdir(root_dir)
  # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
  # netscale = 4
  # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']


  up_model_name = upscale_model
  if up_model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
  elif  up_model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
  elif  up_model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
  elif  up_model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
  elif  up_model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        up_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
  elif  up_model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        up_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
  model_path = os.path.join('weights', up_model_name + '.pth')
  if not os.path.isfile(model_path):
          ROOT_DIR = root_dir
          for url in file_url:
              # model_path will be updated
              model_path = load_file_from_url(
                  url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

  dni_weight = None

  upsampler = RealESRGANer(
          scale=netscale,
          model_path=model_path,
          dni_weight=dni_weight,
          model=up_model,
          tile=0,
          tile_pad=10,
          pre_pad=0,
          half=True,
          device='cuda',
      )

#@markdown ### **Video settings**
use_deflicker = False #@param {'type':'boolean'}
if platform.system() != 'Linux':
   use_deflicker = False
   print('Disabling ffmpeg deflicker filter for windows install, as it is causing a crash.')
if skip_video_for_run_all == True:
  print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

else:
  # import subprocess in case this cell is run without the above cells
  import subprocess
  from base64 import b64encode

  from multiprocessing.pool import ThreadPool as Pool

  pool = Pool(threads)

  latest_run = batchNum

  folder = batch_name #@param
  run =  latest_run#@param
  final_frame = 'final_frame'


  init_frame = 1#@param {type:"number"} This is the frame where the video will start
  last_frame = final_frame#@param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
  fps = 30#@param {type:"number"}
  output_format = 'mp4' #@param ['mp4','mov']
  # view_video_in_cell = True #@param {type: 'boolean'}

  # tqdm.write('Generating video...')

  if last_frame == 'final_frame':
    last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
    print(f'Total frames: {last_frame}')

  image_path = f"{outDirPath}/{folder}/{folder}({run})_%06d.png"
  # filepath = f"{outDirPath}/{folder}/{folder}({run}).{output_format}"

  if upscale_ratio>1:
      postfix+=f'_x{upscale_ratio}_{upscale_model}'
  if use_deflicker:
      postfix+='_dfl'
  if (blend_mode == 'optical flow') & (True) :
    image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%06d.png"
    postfix += '_flow'


    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/flow/{folder}({run})_*.png"))
    flo_out = batchFolder+f"/flow"
    # !rm -rf {flo_out}/*

    # !mkdir "{flo_out}"
    os.makedirs(flo_out, exist_ok=True)

    frames_in = sorted(glob(batchFolder+f"/{folder}({run})_*.png"))

    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    if upscale_ratio>1:
          frame0 = np.array(frame0)[...,::-1]
          output, _ = upsampler.enhance(frame0, outscale=upscale_ratio)
          frame0 = PIL.Image.fromarray((output)[...,::-1].astype('uint8'))
    frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])

    def process_flow_frame(i):
        frame1_path = frames_in[i-1]
        frame2_path = frames_in[i]

        frame1 = Image.open(frame1_path)
        frame2 = Image.open(frame2_path)
        frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):06}.jpg"
        flo_path = f"{flo_folder}/{frame1_stem}.npy"
        weights_path = None
        if check_consistency:
          if reverse_cc_order:
            weights_path = f"{flo_folder}/{frame1_stem}-21_cc.jpg"
          else:
            weights_path = f"{flo_folder}/{frame1_stem}_12-21_cc.jpg"
        tic = time.time()
        printf('process_flow_frame warp')
        frame = warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path,
            pad_pct=padding_ratio, padding_mode=padding_mode, inpaint_blend=0, video_mode=True)
        if use_background_mask_video:
          frame = apply_mask(frame, i, background_video, background_source_video, invert_mask_video)
        if upscale_ratio>1:
          frame = np.array(frame)[...,::-1]
          output, _ = upsampler.enhance(frame.clip(0,255), outscale=upscale_ratio)
          frame = PIL.Image.fromarray((output)[...,::-1].clip(0,255).astype('uint8'))
        frame.save(batchFolder+f"/flow/{folder}({run})_{i:06}.png")

    with Pool(threads) as p:
      fn = partial(try_process_frame, func=process_flow_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))

  if blend_mode == 'linear':
    image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%06d.png"
    postfix += '_blend'

    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/blend/{folder}({run})_*.png"))
    blend_out = batchFolder+f"/blend"
    os.makedirs(blend_out, exist_ok = True)
    frames_in = glob(batchFolder+f"/{folder}({run})_*.png")

    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    if upscale_ratio>1:
          frame0 = np.array(frame0)[...,::-1]
          output, _ = upsampler.enhance(frame0.clip(0,255), outscale=upscale_ratio)
          frame0 = PIL.Image.fromarray((output)[...,::-1].clip(0,255).astype('uint8'))
    frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])

    def process_blend_frame(i):
      frame1_path = frames_in[i-1]
      frame2_path = frames_in[i]

      frame1 = Image.open(frame1_path)
      frame2 = Image.open(frame2_path)
      frame = Image.fromarray((np.array(frame1)*(1-blend) + np.array(frame2)*(blend)).round().astype('uint8'))
      if use_background_mask_video:
        frame = apply_mask(frame, i, background_video, background_source_video, invert_mask_video)
      if upscale_ratio>1:
          frame = np.array(frame)[...,::-1]
          output, _ = upsampler.enhance(frame.clip(0,255), outscale=upscale_ratio)
          frame = PIL.Image.fromarray((output)[...,::-1].clip(0,255).astype('uint8'))
      frame.save(batchFolder+f"/blend/{folder}({run})_{i:06}.png")

    with Pool(threads) as p:
      fn = partial(try_process_frame, func=process_blend_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))
  deflicker_str = ''
  # if use_deflicker:
  #   deflicker_str = ['-filter:v', '"deflicker,mode=pm,size=10"']
  if output_format == 'mp4':
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec',
        'png',
        '-framerate',
        str(fps),
        '-start_number',
        str(init_frame),
        '-i',
        image_path,
        '-frames:v',
        str(last_frame+1),
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p'
    ]
  if output_format == 'mov':
      cmd = [
      'ffmpeg',
      '-y',
      '-vcodec',
      'png',
      '-r',
      str(fps),
      '-start_number',
      str(init_frame),
      '-i',
      image_path,
      '-frames:v',
      str(last_frame+1),
      '-c:v',
      'qtrle',
      '-vf',
      f'fps={fps}'
  ]
  if use_deflicker:
    cmd+=['-vf','deflicker=pm:size=10']
  cmd+=[filepath]

  if upscale_ratio>1:
    del up_model, upsampler
    gc.collect()
  process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
      print(stderr)
      raise RuntimeError(stderr)
  else:
      print(f"The video is ready and saved to {filepath}")
  keep_audio = True #@param {'type':'boolean'}
  if keep_audio:
    f_audio  = filepath[:-4]+'_audio'+filepath[-4:]
    if os.path.exists(filepath) and os.path.exists(video_init_path):

      cmd_a = ['ffmpeg', '-y', '-i', filepath, '-i', video_init_path, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-shortest', f_audio]
      process = subprocess.Popen(cmd_a, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = process.communicate()
      if process.returncode != 0:
          print(stderr)
          raise RuntimeError(stderr)
      else:
          print(f"The video with added audio is saved to {f_audio}")
    else: print('Error adding audio from init video to output video: either init or output video don`t exist.')

  # if view_video_in_cell:
  #     mp4 = open(filepath,'rb').read()
  #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

#@title Shutdown runtime
#@markdown Useful with the new Colab policy.\
#@markdown If on, shuts down the runtime after every cell has been run successfully.

shut_down_after_run_all = False #@param {'type':'boolean'}
if shut_down_after_run_all and is_colab:
  from google.colab import runtime
  runtime.unassign()

#@title Beep
beep = True #@param {'type':'boolean'}
if beep:
  if not is_colab:
    from IPython.display import Audio

    # Define the beep sound parameters
    duration = 1  # Duration of the beep sound in seconds
    freq = 440  # Frequency of the beep sound in Hz

    # Generate the beep sound
    beep_sound = 0.1 * np.sin(2 * np.pi * freq * np.arange(duration * 44100) / 44100)

    # Play the beep sound
    display.display(Audio(beep_sound, rate=44100, autoplay=True))

  if is_colab:
    from google.colab import output
    output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')

"""# Extras

## Compare settings
"""

#@title Insert paths to two settings.txt files to compare

file1 = '' #@param {'type':'string'}
file2 = '' #@param {'type':'string'}

import json
from  glob import glob
import os

changes = []
added = []
removed = []

def infer_settings_path(path):
    default_settings_path = path
    if default_settings_path == '-1':
      settings_files = sorted(glob(os.path.join(settings_out, '*.txt')))
      if len(settings_files)>0:
        default_settings_path = settings_files[-1]
      else:
        print('Skipping load latest run settings: no settings files found.')
        return ''
    else:
      try:
        if type(eval(default_settings_path)) == int:
          files = sorted(glob(os.path.join(settings_out, '*.txt')))
          for f in files:
            if f'({default_settings_path})' in f:
              default_settings_path = f
      except: pass

    path = default_settings_path
    return path

file1 = infer_settings_path(file1)
file2 = infer_settings_path(file2)

if file1 != '' and file2 != '':
  import json
  with open(file1, 'rb') as f:
    f1 = json.load(f)
  with open(file2, 'rb') as f:
    f2 = json.load(f)
  joint_keys = set(list(f1.keys())+list(f2.keys()))
  print(f'Comparing\n{file1.split("/")[-1]}\n{file2.split("/")[-1]}\n')
  for key in joint_keys:
    if key in f1.keys() and key in f2.keys() and f1[key] != f2[key]:
      changes.append(f'{key}: {f1[key]} -> {f2[key]}')
      # print(f'{key}: {f1[key]} -> {f2[key]}')
    if key in f1.keys() and key not in f2.keys():
      removed.append(f'{key}: {f1[key]} -> <variable missing>')
      # print(f'{key}: {f1[key]} -> <variable missing>')
    if key not in f1.keys() and key in f2.keys():
      added.append(f'{key}: <variable missing> -> {f2[key]}')
      # print(f'{key}: <variable missing> -> {f2[key]}')

print('Changed:\n')
for o in changes:
  print(o)

print('\n\nAdded in file2:\n')
for o in added:
  print(o)

print('\n\nRemoved in file2:\n')
for o in removed:
  print(o)

"""## Masking and tracking"""

#@title Install SAMTrack-CLI
#@markdown originally from https://github.com/z-x-yang/Segment-and-Track-Anything \
#@markdown Restart the notebook after install.
import os
try:
  #cd to root if root dir defined
  os.chdir(root_dir)
except:
  root_dir = os.getcwd()

subprocess.run(['git', 'clone', "https://github.com/Sxela/Segment-and-Track-Anything-CLI"], check=True)
# !git clone https://github.com/Sxela/Segment-and-Track-Anything-CLI
os.chdir(os.path.join(root_dir,'Segment-and-Track-Anything-CLI'))

subprocess.run(['python', "-m", 'pip', 'install', '-e', './sam'], check=True)
subprocess.run(['python', "-m", 'pip', 'install', '-e', 'git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO'], check=True)
subprocess.run(['python', "-m", 'pip', 'install', 'numpy', 'opencv-python', "pycocotools", "matplotlib", "Pillow", "scikit-image"], check=True)
subprocess.run(['python', "-m", 'pip', 'install', 'zip', "gdown", "ffmpeg"], check=True)
subprocess.run(['python', "-m", 'pip', 'install', '-e', "./Pytorch-Correlation-extension"], check=True)
# !python -m pip install -e ./sam
# !python -m pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO
# !python -m pip install numpy opencv-python pycocotools matplotlib Pillow scikit-image
# !python -m pip install zip gdown ffmpeg
subprocess.run(['git', 'clone', "https://github.com/ClementPinard/Pytorch-Correlation-extension.git"], check=True)
# !git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
# !python -m pip install -e ./Pytorch-Correlation-extension

os.chdir(os.path.join(root_dir,'Segment-and-Track-Anything-CLI'))
os.makedirs(os.path.join(root_dir,'Segment-and-Track-Anything-CLI', 'ckpt'), exist_ok=True)

subprocess.run(['gdown', "--id", "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ", '--output', './ckpt/R50_DeAOTL_PRE_YTB_DAV.pth'], check=True)
# download aot-ckpt
# !gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
subprocess.run(['wget', "-P", ".ckpt", 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'], check=True)
# !wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
subprocess.run(['wget', "-P", ".ckpt", 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth'], check=True)
# download grounding-dino ckpt
# !wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

#@title Detection setup
#@markdown Use this cell to tweak detection settings, that will be later used on the whole video.
#@markdown Run this cell to get detection preview.\
#@markdown Code mostly taken from https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/demo_instseg.ipynb
import os, pathlib, shutil, sys, subprocess
from glob import glob
try:
  #cd to root if root dir defined
  os.chdir(root_dir)
except:
  root_dir = os.getcwd()

os.chdir(os.path.join(root_dir,'Segment-and-Track-Anything-CLI'))

#(c) Alex Spirin 2023

import hashlib
# We use input file hashes to automate video extraction
#
def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)


def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  createPath(output_path)
  print(f"Exporting Video Frames (1 every {nth_frame})...")
  try:
    for f in [o.replace('\\','/') for o in glob(output_path+'/*.jpg')]:
    # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
      pathlib.Path(f).unlink()
  except:
    print('error deleting frame ', f)
  # vf = f'select=not(mod(n\\,{nth_frame}))'
  vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
  if os.path.exists(video_path):
    try:
        # subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    except:
        subprocess.run([f'{root_dir}/ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

  else:
    sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')


class FrameDataset():
  def __init__(self, source_path, outdir_prefix, videoframes_root):
    self.frame_paths = None
    image_extenstions = ['jpeg', 'jpg', 'png', 'tiff', 'bmp', 'webp']

    if not os.path.exists(source_path):
      if len(glob(source_path))>0:
        self.frame_paths = sorted(glob(source_path))
      else:
        raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    if os.path.exists(source_path):
      if os.path.isfile(source_path):
        if os.path.splitext(source_path)[1][1:].lower() in image_extenstions:
          self.frame_paths = [source_path]
        hash = generate_file_hash(source_path)[:10]
        out_path = os.path.join(videoframes_root, outdir_prefix+'_'+hash)

        extractFrames(source_path, out_path,
                        nth_frame=1, start_frame=0, end_frame=999999999)
        self.frame_paths = glob(os.path.join(out_path, '*.*'))
        if len(self.frame_paths)<1:
            raise Exception(f'Couldn`t extract frames from {source_path}\nPlease specify an existing source path.')
      elif os.path.isdir(source_path):
        self.frame_paths = glob(os.path.join(source_path, '*.*'))
        if len(self.frame_paths)<1:
          raise Exception(f'Found 0 frames in {source_path}\nPlease specify an existing source path.')
    extensions = []
    if self.frame_paths is not None:
      for f in self.frame_paths:
            ext = os.path.splitext(f)[1][1:]
            if ext not in image_extenstions:
              raise Exception(f'Found non-image file extension: {ext} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')
            if not ext in extensions:
              extensions+=[ext]
            if len(extensions)>1:
              raise Exception(f'Found multiple file extensions: {extensions} in {source_path}. Please provide a folder with image files of the same extension, or specify a glob pattern.')

      self.frame_paths = sorted(self.frame_paths)

    else: raise Exception(f'Frame source for {outdir_prefix} not found at {source_path}\nPlease specify an existing source path.')
    print(f'Found {len(self.frame_paths)} frames at {source_path}')

  def __getitem__(self, idx):
    idx = min(idx, len(self.frame_paths)-1)
    return self.frame_paths[idx]

  def __len__(self):
    return len(self.frame_paths)

# mostly taken from https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/demo_instseg.ipynb

import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0

    return img_mask.astype(img.dtype)

video_path = '/content/drive/MyDrive/vids/init/3 girls dancing1.mp4' #@param {'type':'string'}
video_name = video_path.split('/')[-1]
io_args = {
    'input_video': video_path,
    'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
    'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization
}
prefix = ''
try:
  videoframes_root = f'{batchFolder}/videoFrames'
except:
  videoframes_root = f'{root_dir}/videoFrames'

frames = FrameDataset(video_path, outdir_prefix=prefix, videoframes_root=videoframes_root)

# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args['generator_args'] = {
        'points_per_side': 60,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

# Set Text args
'''
parameter:
    grounding_caption: Text prompt to detect objects in key-frames
    box_threshold: threshold for box
    text_threshold: threshold for label(text)
    box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
    reset_image: reset the image embeddings for SAM
'''
frame_number = 1  #@param {'type':'number'}
frame_number = int(frame_number)
#@markdown Text prompt to detect objects in key-frames
grounding_caption = "person" #@param {'type':'string'}
#@markdown Box detection confidence threshold
box_threshold = 0.3 #@param {'type':'number'}
#@markdown Text confidence threshold
text_threshold = 0.3 #@param {'type':'number'}
#@markdown Box to Image ratio threshold (with box_size_threshold = 0.8 detections over 80% of the image will be ignored)
box_size_threshold = 0.8 #@param {'type':'number'}

reset_image = True

frame_idx = 0
segtracker = SegTracker(segtracker_args,sam_args,aot_args)
segtracker.restart_tracker()

with torch.cuda.amp.autocast():
    frame = cv2.imread(frames[frame_number])
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pred_mask, annotated_frame = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold)
    torch.cuda.empty_cache()
    obj_ids = np.unique(pred_mask)
    obj_ids = obj_ids[obj_ids!=0]
    print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')
    init_res = draw_mask(annotated_frame, pred_mask,id_countour=False)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(init_res)
    plt.show()
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(colorize_mask(pred_mask))
    plt.show()

    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

#@title Mask whole video.
use_cli = False #@param {'type':'boolean'}
import subprocess
start_frame = 0 #@param {'type':'number'}
end_frame = 0 #@param {'type':'number'}
#@markdown The interval to run SAM to segment new objects
sam_gap = 50 #@param {'type':'number'}
#@markdown minimal mask area to add a new mask as a new object
min_area = 200  #@param {'type':'number'}
#@markdown maximal object number to track in a video
max_obj_num = 255 #@param {'type':'number'}
#@markdown the area of a new object in the background should > 80%
min_new_obj_iou = 0.8 #@param {'type':'number'}
save_separate_masks = True
save_joint_mask = False #@param {'type':'boolean'}
save_mask = save_joint_mask
save_video = False #@param {'type':'boolean'}
save_gif = False #@param {'type':'boolean'}
# grounding_caption
# box_threshold
# text_threshold
# box_size_threshold
# video_path
output_multimask_dir = os.path.join(videoframes_root, f'{generate_file_hash(video_path)[:10]}_masks')
if use_cli:
  def run_command(cmd, cwd='./'):
      with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
          while True:
              line = p.stdout.readline()
              if not line:
                  break
              print(line)
          exit_code = p.poll()
      return exit_code

  # !python /content/Segment-and-Track-Anything/run.py\
  #  --video_path /content/SaveInsta.App_-_3067564057762969265_1317509610.mp4\
  #  --save_separate_masks --outdir /content/out/


  cmd = ['python', 'run.py','--video_path', video_path, '--save_separate_masks', '--outdir', output_multimask_dir,
        '--caption', grounding_caption, '--box_threshold', box_threshold, '--text_threshold', text_threshold, '--box_size_threshold', box_size_threshold,
        '--sam_gap', sam_gap, '--min_area', min_area, '--max_obj_num', max_obj_num, '--min_new_obj_iou',min_new_obj_iou]
  cmd = [str(o) for o in cmd]
  returncode = run_command(cmd, cwd=os.path.join(root_dir,'Segment-and-Track-Anything-CLI'))
  if process.returncode != 0:
    raise RuntimeError(returncode)
  else:
    print(f"The video is ready and saved to {output_multimask_dir}")
else:
  os.makedirs('./debug/seg_result', exist_ok=True)
  os.makedirs('./debug/aot_result', exist_ok=True)
  segtracker_args = {
    'sam_gap': sam_gap,
    'min_area': min_area,
    'max_obj_num': max_obj_num,
    'min_new_obj_iou': min_new_obj_iou
  }

  if save_mask:
    output_dir = io_args['output_mask_dir']
    os.makedirs(output_dir, exist_ok=True)
  pred_list = []
  masked_pred_list = []

  segtracker = SegTracker(segtracker_args, sam_args, aot_args)
  segtracker.restart_tracker()
  from tqdm.notebook import tqdm, trange
  if start_frame == 0 and end_frame == 0:
    frame_range = trange(len(frames))
  else:
    frame_range = trange(start_frame, end_frame)
  for frame_idx in frame_range:
    frame = cv2.imread(frames[frame_idx])
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if frame_idx == 0:
      pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
      torch.cuda.empty_cache()
      gc.collect()
      segtracker.add_reference(frame, pred_mask)
    elif (frame_idx % sam_gap) == 0:
      seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold,
                                                    box_size_threshold, reset_image)
      # save_prediction(seg_mask, './debug/seg_result', str(frame_idx)+'.png')
      torch.cuda.empty_cache()
      gc.collect()
      track_mask = segtracker.track(frame)
      # save_prediction(track_mask, './debug/aot_result', str(frame_idx)+'.png')

      # find new objects, and update tracker with new objects
      new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
      if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
        new_obj_mask = np.zeros_like(new_obj_mask)
      if save_mask: save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
      pred_mask = track_mask + new_obj_mask
      segtracker.add_reference(frame, pred_mask)
    else:
      pred_mask = segtracker.track(frame,update_memory=True)
    torch.cuda.empty_cache()
    gc.collect()

    if save_mask: save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')

    pred_list.append(pred_mask)

    print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')


  if  save_video:
  # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(io_args['input_video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if io_args['input_video'][-3:]=='mp4':
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    elif io_args['input_video'][-3:] == 'avi':
        fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0

    progress_bar = tqdm(total=num_frames)
    progress_bar.set_description("Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame,pred_mask)
        # masked_frame = masked_pred_list[frame_idx]
        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
        progress_bar.update(1)
    out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

  if  save_gif:
    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'],pred_list,fps=fps)
    print("{} saved".format(io_args['output_gif']))

  from multiprocessing.pool import ThreadPool as Pool
  from functools import partial
  import PIL

  threads = 12

  def write_masks_frame(frame_num,  predicted_masks, output_folder, max_ids=255):
    predicted_masks_frame = predicted_masks[frame_num]
    for i in range(max_ids):
      img_out = PIL.Image.fromarray(((predicted_masks_frame==i+1)*255).astype('uint8'))
      img_out.save(os.path.join(output_folder, f'mask{i:03}', f'alpha_{frame_num:06}.jpg'))

  def write_masks_frame_multi(predicted_masks, output_folder, max_ids):
    for i in range(max_ids):
      os.makedirs(os.path.join(output_folder, f'mask{i:03}'), exist_ok=True)

    with Pool(threads) as p:
      fn = partial(write_masks_frame, predicted_masks=predicted_masks, output_folder=output_folder, max_ids=max_ids)
      result = list(tqdm(p.imap(fn, range(len(predicted_masks))), total=len(predicted_masks)))

  if save_separate_masks:
    print('Saving Separate masks')
    write_masks_frame_multi(predicted_masks=pred_list, output_folder=output_multimask_dir, max_ids=segtracker.get_obj_num())
    print(f'Saved masks to {output_multimask_dir}')
