print("hererrererer !!!!!!!!")

#@title Prepare folders & Install
import subprocess, os, sys

#cell execution check thanks to #soze

if len(sys.argv) > 4:
  user_seed = int(sys.argv[4])
user_prompt = sys.argv[1]
user_width = int(float(sys.argv[2]))
user_height = int(float(sys.argv[3]))

print("Prompt:", user_prompt)

executed_cells = {
    'prepare_folders':False,
    # 'install_pytorch':False,
    # 'install_sd_dependencies':False,
    'import_dependencies':False,
    # 'basic_settings':False,
    # 'animation_settings':False,
    'video_input_settings':False,
    # 'video_masking':False,
    # 'generate_optical_flow':False,
    'load_model':False,
    # 'tiled_vae':False,
    'save_loaded_model':False,
    # 'clip_guidance':False,
    # 'brightness_adjustment':False,
    'content_aware_scheduling':False,
    'plot_threshold_vs_frame_difference':False,
    'create_schedules':False,
    'frame_captioning':False,
    'flow_and_turbo_settings':False,
    'consistency_maps_mixing':False,
    'seed_and_grad_settings':False,
    'prompts':False,
    'warp_turbo_smooth_settings':False,
    'video_mask_settings': False,
    'frame_correction':False,
    'main_settings':False,
    'advanced':False,
    'lora':False,
    # 'reference_controlnet':False,
    'GUI':False,
    'do_the_run':False
}

executed_cells_errors = {
    'prepare_folders': 'Prepare folders & Install',
    # 'install_pytorch':'1.2 Install pytorch',
    # 'install_sd_dependencies':'1.3 Install SD Dependencies',
    'import_dependencies':'Import dependencies, define functions',
    # 'basic_settings':'2.Settings - Basic Settings',
    # 'animation_settings': '2.Settings - Animation Settings',
    'video_input_settings':'2.Settings - Video Input Settings',
    # 'video_masking':'2.Settings - Video Masking',
    # 'generate_optical_flow':'Optical map settings - Generate optical flow and consistency maps',
    'load_model':'Load up a stable. - define SD + K functions, load model',
    # 'tiled_vae':'Extra features - Tiled VAE',
    'save_loaded_model':'Extra features - Save loaded model',
    # 'clip_guidance':'CLIP guidance - CLIP guidance settings',
    # 'brightness_adjustment':'Automatic Brightness Adjustment',
    'content_aware_scheduling':'Content-aware scheduing - Content-aware scheduing',
    'plot_threshold_vs_frame_difference':'Content-aware scheduing - Plot threshold vs frame difference',
    'create_schedules':'Content-aware scheduing - Create schedules from frame difference',
    'frame_captioning':'Frame captioning - Generate captions for keyframes',
    'flow_and_turbo_settings':'Render settings - Non-gui - Flow and turbo settings',
    'consistency_maps_mixing':'Render settings - Non-gui - Consistency map mixing',
    'seed_and_grad_settings':'Render settings - Non-gui - Seed and grad Settings',
    'prompts':'Render settings - Non-gui - Prompts',
    'warp_turbo_smooth_settings':'Render settings - Non-gui - Warp Turbo Smooth Settings',
    'video_mask_settings':'Render settings - Non-gui - Video mask settings',
    'frame_correction':'Render settings - Non-gui - Frame correction',
    'main_settings':'Render settings - Non-gui - Main settings',
    'advanced':'Render settings - Non-gui - Advanced',
    'lora': 'LORA & embedding paths',
    # 'reference_controlnet': 'Reference controlnet (attention injection)',
    'GUI':'GUI',
    'do_the_run':'Diffuse! - Do the run'

}


def check_execution(cell_name):
  for key in executed_cells.keys():
    if key == cell_name:
      #reached current cell successfully, exit
      return
    if executed_cells[key] == False:
      raise RuntimeError(f'The {executed_cells_errors[key]} cell was not run successfully and must be executed to continue. \
RUN ALL after starting runtime (CTRL-F9)');

cell_name = 'prepare_folders'
# check_execution(cell_name)


def gitclone(url, recursive=False, dest=None, branch=None):
  command = ['git', 'clone']
  if branch is not None:
    command.append(['-b', branch])
  command.append(url)
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
        model_path = '/content/models'
        createPath(model_path)
    if google_drive and save_models_to_google_drive:
        model_path = f'{root_path}/models'
        createPath(model_path)
else:
    root_dir = root_path
    model_path = f'{root_path}/models'
    createPath(model_path)

#(c) Alex Spirin 2023

class FrameDataset():
  def __init__(self, source_path, outdir_prefix='', videoframes_root=''):
    self.frame_paths = None
    image_extenstions = ['jpeg', 'jpg', 'png', 'tiff', 'bmp', 'webp']
    if "{" in source_path:
      source_path_e = eval(source_path)
      if isinstance(source_path_e, dict):
        self.frame_paths = []
        for i in range(max(np.array([int(o) for o in source_path_e.keys()]).max(),1)):
          frame = get_sched_from_json(i, source_path_e, blend=False)
          assert os.path.exists(frame), f'The source frame {frame} doesn`t exist. Please provide an existing file name.'
          self.frame_paths.append(frame)
        return
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

gpu = None

import requests
installer_url = 'https://raw.githubusercontent.com/Sxela/WarpTools/main/installersw/warp_installer_032.py'
r = requests.get(installer_url, allow_redirects=True)
open('warp_installer.py', 'wb').write(r.content)

import warp_installer

force_os = 'off'
force_torch_reinstall = False # \ @param {'type':'boolean'}
force_xformers_reinstall = False
# \ @markdown Use v2 by default.
use_torch_v2 = True # \ @param {'type':'boolean'}

import subprocess, sys
import os, platform

# simple_nvidia_smi_display = True
# if simple_nvidia_smi_display:
#   #!nvidia-smi
#   nvidiasmi_output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#   print(nvidiasmi_output)
# else:
#   #!nvidia-smi -i 0 -e 0
#   nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#   print(nvidiasmi_output)
#   nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0', '-e', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#   print(nvidiasmi_ecc_note)


if force_torch_reinstall:
  warp_installer.uninstall_pytorch(is_colab)

if platform.system() != 'Linux' or force_os == 'Windows':
  warp_installer.install_torch_windows(force_torch_reinstall, use_torch_v2)

try:
  if os.environ["IS_DOCKER"] == "1":
    print('Docker found. Skipping install.')
except:
  os.environ["IS_DOCKER"] = "0"

if (is_colab or (platform.system() == 'Linux') or force_os == 'Linux') and os.environ["IS_DOCKER"]=="0":
  from subprocess import getoutput
  # from IPython.display import HTML
  # from IPython.display import clear_output
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

if is_colab:
  warp_installer.install_torch_colab(force_torch_reinstall, use_torch_v2)

# from IPython.utils import io
import shutil, traceback
import pathlib, shutil, os, sys

#@markdown Enable skip_install to avoid reinstalling dependencies after the initial setup.
skip_install = False #@param {'type':'boolean'}
os.makedirs('./embeddings', exist_ok=True)

if os.environ["IS_DOCKER"]=="1":
  skip_install = True
  print('Docker detected. Skipping install.')

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

if skip_install:
  # pass
  warp_installer.pull_repos(is_colab)
else:
  warp_installer.install_dependencies_colab(is_colab, root_dir)

sys.path.append(f'{PROJECT_DIR}/BLIP')

#@markdown For a local install use the following guides: \
#@markdown [Windows](https://github.com/Sxela/WarpFusion/tree/main#local-installation-guide-for-windows-venv) \
#@markdown [Linux](https://github.com/Sxela/WarpFusion/tree/main#local-installation-guide-for-linux-ubuntu-2204-venv) / kudos to [Daniel Salvatierra](https://github.com/dpsalvatierra)
executed_cells[cell_name] = True

"""# 2. Settings"""

# Commented out IPython magic to ensure Python compatibility.
#@title ### Import dependencies, define functions


cell_name = 'import_dependencies'
# check_execution(cell_name)

user_settings_keys = ['latent_scale_schedule', 'init_scale_schedule', 'steps_schedule', 'style_strength_schedule',
                           'cfg_scale_schedule', 'flow_blend_schedule', 'image_scale_schedule', 'flow_override_map',
                           'text_prompts', 'negative_prompts', 'prompt_patterns_sched', 'latent_fixed_mean',
                           'latent_fixed_std', 'rec_prompts', 'cc_masked_diffusion_schedule', 'mask_paths','user_comment', 'blend_json_schedules', 'VERBOSE', 'use_background_mask', 'invert_mask', 'background',
                      'background_source', 'mask_clip_low', 'mask_clip_high', 'turbo_mode', 'turbo_steps', 'colormatch_turbo',
                      'turbo_frame_skips_steps', 'soften_consistency_mask_for_turbo_frames', 'flow_warp', 'apply_mask_after_warp',
                      'warp_num_k', 'warp_forward', 'warp_strength', 'warp_mode', 'warp_towards_init', 'check_consistency',
                       'padding_ratio', 'padding_mode', 'match_color_strength',
                      'mask_result', 'use_patchmatch_inpaiting', 'cond_image_src', 'set_seed', 'clamp_grad', 'clamp_max', 'sat_scale',
                      'init_grad', 'grad_denoised', 'blend_latent_to_init', 'fixed_code', 'code_randomness', 'dynamic_thresh',
                      'sampler', 'use_karras_noise', 'inpainting_mask_weight', 'inverse_inpainting_mask', 'inpainting_mask_source',
                      'normalize_latent', 'normalize_latent_offset', 'latent_norm_4d', 'colormatch_frame', 'color_match_frame_str',
                      'colormatch_offset', 'colormatch_method', 'colormatch_regrain', 'colormatch_after',
                      'fixed_seed', 'rec_cfg', 'rec_steps_pct', 'rec_randomness', 'use_predicted_noise', 'overwrite_rec_noise',
                      'save_controlnet_annotations', 'control_sd15_openpose_hands_face', 'control_sd15_depth_detector',
                      'control_sd15_softedge_detector', 'control_sd15_seg_detector', 'control_sd15_scribble_detector',
                      'control_sd15_lineart_coarse', 'control_sd15_inpaint_mask_source', 'control_sd15_shuffle_source',
                      'control_sd15_shuffle_1st_source', 'controlnet_multimodel', 'controlnet_mode', 'normalize_cn_weights',
                      'controlnet_preprocess', 'detect_resolution', 'bg_threshold', 'low_threshold', 'high_threshold',
                      'value_threshold', 'distance_threshold', 'temporalnet_source', 'temporalnet_skip_1st_frame',
                      'controlnet_multimodel_mode', 'max_faces', 'do_softcap', 'softcap_thresh', 'softcap_q', 'masked_guidance',
                      'alpha_masked_diffusion', 'invert_alpha_masked_diffusion', 'normalize_prompt_weights', 'sd_batch_size',
                      'controlnet_low_vram', 'deflicker_scale', 'deflicker_latent_scale', 'pose_detector','apply_freeu_after_control',
                      'do_freeunet','batch_length','batch_overlap','looped_noise',
                      'overlap_stylized','context_length','context_overlap','blend_batch_outputs',
                      'force_flow_generation','use_legacy_cc','flow_threads','num_flow_workers','flow_lq',
                      'flow_save_img_preview','num_flow_updates','lazy_warp', 'blend_prompts_b4_diffusion',
                      'clip_skip','qr_cn_mask_clip_high','qr_cn_mask_clip_low','qr_cn_mask_thresh',
                      'use_manual_splits','scene_split_thresh','scene_splits','qr_cn_mask_invert',
                      'qr_cn_mask_grayscale','fill_lips','flow_maxsize','use_reference', 'reference_weight',
                      'reference_source', 'reference_mode',
                      'missed_consistency_schedule', 'overshoot_consistency_schedule', 'edges_consistency_schedule',
                      'consistency_blur_schedule','consistency_dilate_schedule','soften_consistency_schedule', 'offload_model',
                      'use_tiled_vae', 'num_tiles','force_mask_overwrite','mask_source','extract_background_mask',
                      'enable_adjust_brightness',
                      'high_brightness_threshold',
                      'high_brightness_adjust_ratio',
                      'high_brightness_adjust_fix_amount',
                      'max_brightness_threshold',
                      'low_brightness_threshold',
                      'low_brightness_adjust_ratio',
                      'low_brightness_adjust_fix_amount',
                      'min_brightness_threshold','color_video_path','color_extract_nth_frame','b1','b2','s1','s2']
user_settings_eval_keys = ['scene_splits','latent_scale_schedule', 'init_scale_schedule', 'steps_schedule', 'style_strength_schedule',
                           'cfg_scale_schedule', 'flow_blend_schedule', 'image_scale_schedule', 'flow_override_map',
                           'text_prompts', 'negative_prompts', 'prompt_patterns_sched', 'latent_fixed_mean',
                           'latent_fixed_std', 'rec_prompts', 'cc_masked_diffusion_schedule', 'mask_paths',
                           'missed_consistency_schedule', 'overshoot_consistency_schedule', 'edges_consistency_schedule',
                            'consistency_blur_schedule','consistency_dilate_schedule','soften_consistency_schedule','num_tiles']
#init settings
user_settings = {} #init empty to check for missing keys
# user_settings = dict([(key,'') for key in user_settings_keys])
image_prompts = {}

import os, random, torch
import numpy as np
animation_mode = 'Video Input'

def seed_everything(seed, deterministic=False):
    print(f'Set global seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
# import timm
# from IPython import display
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
# from resize_right import resize
# from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
# from ipywidgets import Output
import hashlib
from functools import partial
if is_colab:
  os.chdir('/content')
  from google.colab import files
else:
  os.chdir(f'{PROJECT_DIR}')
# from IPython.display import Image as ipyimg
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

def mask_color_and_add_strokes(image_path, target_color, stroke_color=(0, 0, 255), tolerance=15, stroke_width=3):
    """
    Masks a specific color in an image and adds a stroke to the remaining lines.

    :param image_path: Path to the image file.
    :param target_color: The color to mask as a BGR tuple, e.g., (B, G, R).
    :param stroke_color: The color of the stroke as a BGR tuple.
    :param tolerance: Tolerance for color masking.
    :param stroke_width: Width of the stroke.
    :return: Image with color masked and strokes added.
    """
    # Load the image
    image =  image_path

    # Convert the target color to numpy array
    target_color = np.array(target_color)

    # Define the lower and upper bounds for the color to mask
    lower_bound = np.clip(target_color - tolerance, 0, 255)
    upper_bound = np.clip(target_color + tolerance, 0, 255)

    # Create a mask and apply it to the image
    mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Detect edges in the image
    edges = cv2.Canny(result, 100, 200)

    # Create a stroke effect
    kernel = np.ones((stroke_width, stroke_width), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Draw strokes on the image
    result[dilated_edges > 0] = stroke_color

    return result

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

# def regen_perlin():
#     if perlin_mode == 'color':
#         init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
#         init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
#     elif perlin_mode == 'gray':
#         init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
#         init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
#     else:
#         init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
#         init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

#     init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
#     del init2
#     return init.expand(batch_size, -1, -1, -1)

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
            # print('frame between keys, no blend')
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
  frame1_init = init_image
  init_image = img2tensor(init_image, size)
  flo = get_flow(init_image, sample, raft_model, half=flow_lq)
  # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
  warped = warp(sample_pil, sample_pil, flo_path=flo, blend=1, weights_path=None,
                          forward_clip=0, pad_pct=padding_ratio, padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
  return warped

def do_3d_step(img_filepath, frame_num, forward_clip):
            printf('frame_num', frame_num,'do 3d step',  file='./logs/resume_run_test.txt')
            ts = time.time()
            global warp_mode, filename, match_frame, first_frame
            global first_frame_source
            if warp_mode == 'use_image':
              prev = Image.open(img_filepath)

            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            frame2 = Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg')


            flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

            if flow_override_map not in [[],'', None]:
                 mapped_frame_num = int(get_scheduled_arg(frame_num, flow_override_map))
                 frame_override_path = f'{videoFramesFolder}/{mapped_frame_num:06}.jpg'
                 flo_path = f"{flo_folder}/{frame_override_path.split('/')[-1]}.npy"

            if use_background_mask and not apply_mask_after_warp:

                if VERBOSE:print('creating bg mask for frame ', frame_num)
                frame2 = apply_mask(frame2, frame_num, background, background_source, invert_mask)

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
                frame1_init = f'{videoFramesFolder}/{frame_num:06}.jpg'
                flow21, forward_weights = get_flow_and_cc(frame1_init, frame2, flo_path,
                                                          cc_path=weights_path)

                warped = warp(prev, frame2, flo_path=flow21, blend=flow_blend, weights_path=forward_weights,
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

            if use_background_mask and apply_mask_after_warp:

              if VERBOSE: print('creating bg mask for frame ', frame_num)
              if warp_mode == 'use_latent':
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
              else:
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
            printf('warping took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')
            return warped

from tqdm.notebook import trange
import copy

def get_frame_from_color_mode(mode, offset, frame_num):
                      if mode == 'color_video':
                        if VERBOSE:print(f'the color video frame number {offset} = {frame_num-offset+1}.')
                        filename = f'{colorVideoFramesFolder}/{offset+1:06}.jpg'
                      if mode == 'color_video_offset':
                        if VERBOSE:print(f'the color video frame with offset {offset} = {frame_num-offset+1}.')
                        filename = f'{colorVideoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'stylized_frame_offset':
                        if VERBOSE:print(f'the stylized frame with offset {offset} = {frame_num-offset+1}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-offset:06}.{save_img_format}'
                      if mode == 'stylized_frame':
                        if VERBOSE:print(f'the stylized frame number {offset} = {frame_num-offset+1}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.{save_img_format}'
                        if not os.path.exists(filename):
                          filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{args.start_frame+offset:06}.{save_img_format}'
                      if mode == 'init_frame_offset':
                        if VERBOSE:print(f'the raw init frame with offset {offset} = {frame_num-offset+1}.')
                        filename = f'{videoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'init_frame':
                        if VERBOSE:print(f'the raw init frame number {offset} = {frame_num-offset+1}.')
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

  for frame_num in range(args.start_frame, args.max_frames):
      frame_ts = time.time()
      ts_b4_sd = time.time()
      printf('----- Starting frame ', file='./logs/profiling.txt')
      if do_run_cast == 'cpu':
        try:
          sd_model.cpu()
          sd_model.model.cpu()
          sd_model.cond_stage_model.cpu()
          sd_model.first_stage_model.cpu()
          if 'control' in model_version:
            for key in loaded_controlnets.keys():
              loaded_controlnets[key].cpu()
        except: pass
        try:
          apply_openpose.body_estimation.model.cpu()
          apply_openpose.hand_estimation.model.cpu()
          apply_openpose.face_estimation.model.cpu()
        except: pass
        try:
          sd_model.model.diffusion_model.cpu()
        except: pass
        try:
          apply_softedge.netNetwork.cpu()
        except: pass
        try:
          apply_normal.netNetwork.cpu()
        except: pass
      elif do_run_cast == 'cuda':
        try:
          sd_model.cuda()
          sd_model.model.cuda()
          sd_model.cond_stage_model.cuda()
          sd_model.first_stage_model.cuda()
          if 'control' in model_version:
            for key in loaded_controlnets.keys():
              loaded_controlnets[key].cuda()
        except: pass
        try:
          apply_openpose.body_estimation.model.cuda()
          apply_openpose.hand_estimation.model.cuda()
          apply_openpose.face_estimation.model.cuda()
        except: pass
        try:
          sd_model.model.diffusion_model.cuda()
        except: pass
        try:
          apply_softedge.netNetwork.cuda()
        except: pass
        try:
          apply_normal.netNetwork.cuda()
        except: pass
      if cuda_empty_cache: torch.cuda.empty_cache()
      if gc_collect: gc.collect()
      if stop_on_next_loop:
        break

      global missed_consistency_weight, overshoot_consistency_weight, edges_consistency_weight
      global consistency_blur, consistency_dilate, soften_consistency_mask

      missed_consistency_weight = get_scheduled_arg(frame_num, missed_consistency_schedule)
      overshoot_consistency_weight = get_scheduled_arg(frame_num, overshoot_consistency_schedule)
      edges_consistency_weight = get_scheduled_arg(frame_num, edges_consistency_schedule)
      consistency_blur = get_scheduled_arg(frame_num, consistency_blur_schedule)
      consistency_dilate = get_scheduled_arg(frame_num, consistency_dilate_schedule)
      soften_consistency_mask = get_scheduled_arg(frame_num, soften_consistency_schedule)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        # display.display(batchBar.container)
        batchBar.n = frame_num
        batchBar.update(1)
        batchBar.refresh()

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
        if not resume_run and (frame_num == scene_start):
            print('First frame.')
            printf('frame_num', frame_num,'First frame.',  file='./logs/resume_run_test.txt')
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
              init_image_pil.save(f'init_alpha_{frame_num}.{save_img_format}')
              init_image = f'init_alpha_{frame_num}.{save_img_format}'
            if (args.init_image != '') and  args.init_image is not None:
              init_image = args.init_image
              if use_background_mask:
                init_image_pil = Image.open(init_image)
                init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
                init_image_pil.save(f'init_alpha_{frame_num}.{save_img_format}')
                init_image = f'init_alpha_{frame_num}.{save_img_format}'
            if VERBOSE:print('init image', args.init_image)
        else:
          printf('frame_num', frame_num,'not 1st frame.' , file='./logs/resume_run_test.txt')
          # print(frame_num)

          first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.{save_img_format}"
          if os.path.exists(first_frame_source):
              first_frame = Image.open(first_frame_source)
          else:
              first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame-1:06}.{save_img_format}"
              first_frame = Image.open(first_frame_source)

          if not fixed_seed:
            seed += 1
          if resume_run and frame_num == start_frame:
            print('if resume_run and frame_num == start_frame')
            img_filepath = batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:06}.{save_img_format}"
            if turbo_mode and frame_num > turbo_preroll:
              shutil.copyfile(img_filepath, f'oldFrameScaled.{save_img_format}')
            else:
              shutil.copyfile(img_filepath, f'{tempdir}/run({args.batchNum})prevFrame.{save_img_format}')
          else:
            img_filepath = f'{tempdir}/run({args.batchNum})prevFrame.{save_img_format}'

          next_step_pil = do_3d_step(img_filepath, frame_num,  forward_clip=forward_weights_clip)
          if warp_mode == 'use_image':
            next_step_pil.save(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}')
            prevframescaled_global = next_step_pil
          else:
            torch.save(next_step_pil, 'prevFrameScaled_lat.pt')

          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)
          # skip_steps = args.calc_frames_skip_steps

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if turbo_mode:
            if frame_num == turbo_preroll: #start tracking oldframe
              if warp_mode == 'use_image':
                next_step_pil.save(f'oldFrameScaled.{save_img_format}')#stash for later blending
              if warp_mode == 'use_latent':
                # lat_from_img = get_lat/_from_pil(next_step_pil)
                torch.save(next_step_pil, 'oldFrameScaled_lat.pt')
            elif frame_num > turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              if warp_mode == 'use_image':
                old_frame = do_3d_step(f'oldFrameScaled.{save_img_format}', frame_num, forward_clip=forward_weights_clip_turbo_step)
                old_frame.save(f'oldFrameScaled.{save_img_format}')
              if warp_mode == 'use_latent':
                old_frame = do_3d_step(f'oldFrameScaled.{save_img_format}', frame_num, forward_clip=forward_weights_clip_turbo_step)

                # lat_from_img = get_lat_from_pil(old_frame)
                torch.save(old_frame, 'oldFrameScaled_lat.pt')
              if frame_num % int(turbo_steps) != 0:
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.{save_img_format}'
                blend_factor = ((frame_num % int(turbo_steps))+1)/int(turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                if warp_mode == 'use_image':
                  newWarpedImg = cv2.imread(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}')#this is already updated..
                  oldWarpedImg = cv2.imread(f'oldFrameScaled.{save_img_format}')
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
                      oldWarpedImg = cv2.imread(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}')
                      cv2.imwrite(f'oldFrameScaled.{save_img_format}',oldWarpedImg)#swap in for blending later
                    print('clip/diff this frame - generate clip diff image')
                    if warp_mode == 'use_latent':
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later
                    skip_steps = math.floor(steps * turbo_frame_skips_steps)
                else: continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                if warp_mode == 'use_image':
                      oldWarpedImg = cv2.imread(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}')
                      cv2.imwrite(f'oldFrameScaled.{save_img_format}',oldWarpedImg)#swap in for blending later
                print('clip/diff this frame - generate clip diff image')
                if warp_mode == 'use_latent':
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later

                print('clip/diff this frame - generate clip diff image')
          if warp_mode == 'use_image':
            init_image = f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}'
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

      # image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            # display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          # display.display(image_display)
          if gc_collect: gc.collect()
          if cuda_empty_cache: torch.cuda.empty_cache()
          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)


          # if perlin_init:
          #     init = regen_perlin()

          consistency_mask = None
          if (check_consistency or (model_version == 'v1_inpainting') or (
              'control_sd15_inpaint' in controlnet_multimodel.keys())) and (frame_num>scene_start or resume_run):
            printf('frame_num', frame_num,'loading cc map',   file='./logs/resume_run_test.txt')
            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if reverse_cc_order:
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
            else:
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
            consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

          if diffusion_model == 'stable_diffusion':
            if VERBOSE: print(args.side_x, args.side_y, init_image)

            text_prompt = copy.copy(get_sched_from_json(frame_num, args.prompts_series, blend=False))
            if VERBOSE:print(f'Frame {frame_num} Prompt: {text_prompt}')
            text_prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in text_prompt] #remove loras from prompt
            text_prompt = [re.sub(":\s*([\d.]+)\s*$", '', o).strip(' ') for o in text_prompt] #remove weights from prompt
            used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, new_prompt_loras)
            frame_prompt_weights = get_sched_from_json(frame_num, prompt_weights, blend=blend_json_schedules)

            if VERBOSE:
              print('used_loras, used_loras_weights', used_loras, used_loras_weights)
              print('prompt weights, frame_prompt_weights', prompt_weights , frame_prompt_weights)

            load_networks(names=used_loras, te_multipliers=used_loras_weights, unet_multipliers=used_loras_weights, dyn_dims=[None]*len(used_loras), sd_model=sd_model)

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

            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps-steps*style_strength)
            cfg_scale = get_scheduled_arg(frame_num, cfg_scale_schedule)
            image_scale = get_scheduled_arg(frame_num, image_scale_schedule)
            cc_masked_diffusion = get_scheduled_arg(frame_num, cc_masked_diffusion_schedule)




            if VERBOSE:printf('skip_steps b4 run_sd: ', skip_steps)

            deflicker_src = {
                'processed1':f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}',
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
                ref_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}'
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
                shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{0:06}.{save_img_format}'
              elif control_sd15_shuffle_source == 'prev_frame':
                shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}'
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

            prev_frame = f'{videoFramesFolder}/{frame_num:06}.jpg'
            next_frame = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            #setup temporal source
            if temporalnet_source =='init':
              prev_frame = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if temporalnet_source == 'stylized':
              prev_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}'
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
            inpaint_cns = set(['control_sd15_inpaint', 'control_sd15_inpaint_softedge', 'control_sdxl_inpaint'])
            if len(inpaint_cns.intersection(set(controlnet_multimodel.keys())))>0:
            # if 'control_sd15_inpaint' in controlnet_multimodel.keys() or 'control_sd15_inpaint_softedge' in controlnet_multimodel.keys():
              if control_sd15_inpaint_mask_source == 'consistency_mask':
                  control_inpainting_mask = consistency_mask
              if control_sd15_inpaint_mask_source in ['none', None,'', 'None', 'off'] and frame_num>args.start_frame:
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
            controlnet_sources = {}
            if controlnet_multimodel != {}:
              controlnet_sources = get_control_source_images(frame_num, controlnet_multimodel_inferred, stylized_image=init_image)
            elif 'control_' in model_version:
              controlnet_sources[model_version] = cond_image
            controlnet_sources['next_frame'] = next_frame

            printf('b4 run_sd took ', f'{time.time()-ts_b4_sd:4.2}', file='./logs/profiling.txt')
            ts_sd = time.time()
            printf('frame_num', frame_num,'init_image = ', init_image,  file='./logs/resume_run_test.txt')
            sample, latent, depth_img = run_sd(args, init_image=init_image, skip_timesteps=skip_steps, H=args.side_y,
                             W=args.side_x, text_prompt=text_prompt, neg_prompt=neg_prompt, steps=steps,
                             seed=seed, init_scale = init_scale, init_latent_scale=init_latent_scale, cond_image=cond_image,
                             cfg_scale=cfg_scale, image_scale = image_scale, cond_fn=None,
                             init_grad_img=init_grad_img, consistency_mask=consistency_mask,
                             frame_num=frame_num, deflicker_src=deflicker_src, prev_frame=prev_frame,
                             rec_prompt=rec_prompt, rec_frame=rec_frame,control_inpainting_mask=control_inpainting_mask, shuffle_source=shuffle_source,
                             ref_image=ref_image, alpha_mask=np_alpha, prompt_weights=frame_prompt_weights,
                                               mask_current_frame_many=mask_current_frame_many, controlnet_sources=controlnet_sources, cc_masked_diffusion =cc_masked_diffusion )
            ts_after_sd = time.time()
            printf(' run_sd took ', f'{time.time()-ts_sd:4.2}', file='./logs/profiling.txt')
            settings_json = save_settings(skip_save=True)
            settings_exif = json2exif(settings_json)

            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.{save_img_format}'

            if warp_mode == 'use_latent':
              torch.save(latent,f'{batchFolder}/{filename[:-4]}_lat.pt')
            samples = sample*(steps-skip_steps)
            samples = [{"pred_xstart": sample} for sample in samples]

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

            # display.clear_output(wait=True)
            fit(image, display_size).save(f'progress.{save_img_format}', exif=settings_exif)
            # display.display(display.Image(f'progress.{save_img_format}'))

            if mask_result and check_consistency and frame_num>0:

                        if VERBOSE:print('imitating inpaint')
                        frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                        weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                        consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

                        consistency_mask = cv2.GaussianBlur(consistency_mask,
                                                (diffuse_inpaint_mask_blur,diffuse_inpaint_mask_blur),cv2.BORDER_DEFAULT)
                        if diffuse_inpaint_mask_thresh<1:
                          consistency_mask = np.where(consistency_mask<diffuse_inpaint_mask_thresh, 0, 1.)

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


                          torch.save(latent, 'prevFrame_lat.pt')
                          image_masked = np.array(image)*(1-consistency_mask) + np.array(image_prev)*(consistency_mask)
                          image_masked = Image.fromarray(image_masked.round().astype('uint8'))
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
                image.save(f'{tempdir}/run({args.batchNum})prevFrame.{save_img_format}', exif=settings_exif)
              else:
                torch.save(latent, 'prevFrame_lat.pt')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.{save_img_format}'
            image.save(f'{batchFolder}/{filename}', exif=settings_exif)
            # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
            if args.animation_mode == 'Video Input':
                          # If turbo, save a blended image
                          if turbo_mode and frame_num > args.start_frame:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(turbo_steps)
                            if warp_mode == 'use_image':
                              newFrame = cv2.imread(f'{tempdir}/run({args.batchNum})prevFrame.{save_img_format}') # This is already updated..
                              prev_frame_warped = cv2.imread(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}')
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
              image.save(f'{tempdir}/run({args.batchNum})prevFrameScaled.{save_img_format}', exif=settings_exif)
            printf('after run_sd took ', f'{time.time()-ts_after_sd:4.2}', file='./logs/profiling.txt')
          plt.plot(np.array(loss_values), 'r')

      printf('---- frame took ', f'{time.time()-frame_ts:4.2}', file='./logs/profiling.txt')
  batchBar.close()

def upload_settings_to_s3(file_path, user_id, num):
  import hashlib
  hashed_string = hashlib.sha256(user_id.encode()).hexdigest()
  bucket_name = 'flush-user-images'
  folder_name = 'settings_files'
  s3_file_name = f'{folder_name}/{hashed_string}/settings_{num}.txt'

  # Initialize S3 client
  s3 = boto3.client('s3')

  # Read the file in binary mode
  with open(file_path, 'rb') as file:
      s3.put_object(
          Body=file,
          Bucket=bucket_name,
          Key=s3_file_name,
          ContentType='text/plain',
          ACL='public-read'
      )

  url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_name}"
  print(f'Settings file uploaded: {url}')

def save_settings(skip_save=False, path=None):
  settings_out = batchFolder+f"/settings"
  os.makedirs(settings_out, exist_ok=True)
  setting_list = {
    'text_prompts': text_prompts,
    'user_comment':user_comment,
    'image_prompts': image_prompts,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'max_frames': max_frames,
    # 'interp_spline': interp_spline,
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
    'cc_masked_template':cc_masked_template,
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
    'force_mask_overwrite':force_mask_overwrite,
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
    'cc_masked_diffusion_schedule':cc_masked_diffusion_schedule_bkup,
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
    'pose_detector':pose_detector,
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
    'mask_paths':mask_paths,
    'controlnet_mode':controlnet_mode,
    'normalize_cn_weights':normalize_cn_weights,

    'apply_freeu_after_control':apply_freeu_after_control,
    'do_freeunet':do_freeunet,

    'batch_length':batch_length_bkup,
    'batch_overlap':batch_overlap,
    'looped_noise':looped_noise,
    'overlap_stylized':overlap_stylized,
    'context_length':context_length,
    'context_overlap':context_overlap,
    'blend_batch_outputs':blend_batch_outputs,

    'force_flow_generation':force_flow_generation,
    'use_legacy_cc':use_legacy_cc,
    'flow_threads':flow_threads,
    'num_flow_workers':num_flow_workers,
    'flow_lq':flow_lq,
    'flow_save_img_preview':flow_save_img_preview,
    'num_flow_updates':num_flow_updates,
    'lazy_warp':lazy_warp,
    'clip_skip':clip_skip,
    'qr_cn_mask_clip_high':qr_cn_mask_clip_high,
    'qr_cn_mask_clip_low':qr_cn_mask_clip_low,
    'qr_cn_mask_invert':qr_cn_mask_invert,
    'qr_cn_mask_grayscale':qr_cn_mask_grayscale,

    'use_manual_splits':use_manual_splits,
    'scene_split_thresh':scene_split_thresh,
    'scene_splits':scene_splits,
    'blend_prompts_b4_diffusion':blend_prompts_b4_diffusion,

    'fill_lips':fill_lips,
    'flow_maxsize':flow_maxsize,

    'missed_consistency_schedule':missed_consistency_schedule_bkup,
    'overshoot_consistency_schedule':overshoot_consistency_schedule_bkup,
    'edges_consistency_schedule':edges_consistency_schedule_bkup,
    'consistency_blur_schedule':consistency_blur_schedule_bkup,
    'consistency_dilate_schedule':consistency_dilate_schedule_bkup,
    'soften_consistency_schedule':soften_consistency_schedule_bkup,

    'use_tiled_vae':use_tiled_vae,
    'num_tiles':num_tiles,

    'color_video_path':color_video_path,
    'color_extract_nth_frame':color_extract_nth_frame,

    'b1':b1,
    'b2':b2,
    's1':s1,
    's2':s2,
  }
  if not skip_save:
    try:
      if path is not None:
        t = time.time()
        settings_fname = f"{settings_out}/{path}/{batch_name}({path})_({t})_settings.txt"
        os.makedirs(f"{settings_out}/{path}", exist_ok=True)
      else:
        settings_fname = f"{settings_out}/{batch_name}({batchNum})_settings.txt"
      # setting_list['settings_filename'] = settings_fname
      if os.path.exists(settings_fname):
        s_meta = os.path.getmtime(settings_fname)
        settings_fname = settings_fname[:-4]+'_'+str(s_meta)+'.txt'
        # os.rename(settings_fname,settings_fname[:-4]+'_'+str(s_meta)+'.txt' )
      setting_list['settings_filename'] = settings_fname
      with open(settings_fname, "w+") as f:   #save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)
      upload_settings_to_s3(settings_fname, "user_id_2", 3)
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
# from pytorch_lightning import seed_everything

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
        global lpips_model

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
          if lpips_model is None:
            lpips_model = init_lpips()
          loss += lpips_model(denoised_img, init_image_sd[:,:,::2,::2]).sum() * init_scale

        if deflicker_scale>0 and deflicker_fn is not None:
          # print('deflicker_fn(denoised_img).sum() * deflicker_scale',deflicker_fn(denoised_img).sum() * deflicker_scale)
          loss += deflicker_fn(processed2=denoised_img).sum() * deflicker_scale
          print('deflicker ', loss)

        if deflicker_latent_scale>0 and deflicker_lat_fn is not None:

          loss += deflicker_lat_fn(processed2=denoised, processed1=processed1).sum() * deflicker_latent_scale
          print('deflicker lat', loss)

        # if face_scale >0:
        #   with torch.autocast('cuda'):
        #     denoised_small = denoised[:,:,::2,::2]
        #     denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised_small)
        #     face_embed_pred = g_insight_face_model.run_model(denoised_img)[0]
        #     face_embed_gc = g_insight_face_model.run_model(init_image_sd[:,:,::2,::2])[0]
        #   loss+=spherical_dist_loss(face_embed_pred,face_embed_gc)*face_scale

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
def torch_interp1d(x, xp, fp):
    # print('x, xp, fp.device')
    # print(x.device, xp.device, fp.device)
    """
    Performs 1D linear interpolation.

    Args:
    x (Tensor): The x-coordinates at which to evaluate the interpolated values.
    xp (Tensor): The x-coordinates of the data points.
    fp (Tensor): The y-coordinates of the data points, same length as xp.

    Returns:
    Tensor: Interpolated values for each element of x.
    """
    idxs = torch.searchsorted(xp, x)
    idxs = idxs.clamp(1, len(xp) - 1).cuda()
    left = xp[idxs - 1]
    right = xp[idxs]
    alpha = (x - left) / (right - left)
    # print('alpha.device, fp.device, idxd.device')
    # print(alpha.device, fp.device, idxs.device)
    fp=fp.cuda()
    return fp[idxs - 1] + alpha * (fp[idxs] - fp[idxs - 1])

def pdf_transfer_nd_torch(self, arr_in=None, arr_ref=None, step_size=1):
    # print('arr_in.device, arr_ref.device')
    # print(arr_in.device, arr_ref.device)
    """Apply n-dim probability density function transfer in PyTorch.

    Args:
        arr_in: shape=(n, x), PyTorch tensor.
        arr_ref: shape=(n, x), PyTorch tensor.
        step_size: arr = arr + step_size * delta_arr.
    Returns:
        arr_out: shape=(n, x), PyTorch tensor.
    """
    # Initialize the output tensor
    arr_out = arr_in.clone()

    # Loop through rotation matrices
    for rotation_matrix in self.rotation_matrices_torch:
        # Rotate input and reference arrays
        rot_arr_in = torch.matmul(rotation_matrix, arr_out)
        rot_arr_ref = torch.matmul(rotation_matrix, arr_ref)

        # Initialize the rotated output array
        rot_arr_out = torch.zeros_like(rot_arr_in)

        # Loop over the first dimension
        for i in range(rot_arr_out.shape[0]):
            # Apply 1D PDF transfer (assuming _pdf_transfer_1d is adapted for PyTorch)
            rot_arr_out[i] = self._pdf_transfer_1d_torch(rot_arr_in[i], rot_arr_ref[i])

        # Calculate the delta array
        rot_delta_arr = rot_arr_out - rot_arr_in
        delta_arr = torch.matmul(rotation_matrix.transpose(0, 1), rot_delta_arr)

        # Update the output array
        arr_out = arr_out + step_size * delta_arr

    return arr_out

import torch

def _pdf_transfer_1d_torch(self, arr_in=None, arr_ref=None, n=300):
    # print('arr_in.device, arr_ref.device')
    # print(arr_in.device, arr_ref.device)
    """Apply 1-dim probability density function transfer using PyTorch.

    Args:
        arr_in: 1d PyTorch tensor input array.
        arr_ref: 1d PyTorch tensor reference array.
        n: discretization num of distribution of image's pixels.
    Returns:
        arr_out: transferred input tensor.
    """
    arr = torch.cat((arr_in, arr_ref))
    min_v = torch.min(arr) - self.eps
    max_v = torch.max(arr) + self.eps
    xs = torch.linspace(min_v, max_v, steps=n+1).to(arr.device)

    # Compute histograms
    hist_in = torch.histc(arr_in, bins=n, min=min_v, max=max_v)
    hist_ref = torch.histc(arr_ref, bins=n, min=min_v, max=max_v)
    xs = xs[:-1]

    # Compute cumulative distributions
    cum_in = torch.cumsum(hist_in, dim=0)
    cum_ref = torch.cumsum(hist_ref, dim=0)
    d_in = cum_in / cum_in[-1]
    d_ref = cum_ref / cum_ref[-1]

    # Transfer function
    t_d_in = torch_interp1d(d_in, d_ref, xs)
    t_d_in[d_in <= d_ref[0]] = min_v
    t_d_in[d_in >= d_ref[-1]] = max_v
    arr_out = torch_interp1d(arr_in, xs, t_d_in)

    return arr_out

import torch

def pdf_transfer_torch(self, img_arr_in=None, img_arr_ref=None, regrain=False):
    """Apply probability density function transfer using PyTorch.

    img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
    where f_{img}(r, g, b) is the probability density function of img's rgb values.

    Args:
        img_arr_in: BGR PyTorch tensor of input image.
        img_arr_ref: BGR PyTorch tensor of reference image.
    Returns:
        img_arr_out: Transferred BGR PyTorch tensor of input image.
    """

    # Ensure input is a PyTorch tensor
    img_arr_in = torch.tensor(img_arr_in, dtype=torch.float32, device='cuda') / 255.0
    img_arr_ref = torch.tensor(img_arr_ref, dtype=torch.float32, device='cuda') / 255.0
    # print('img_arr_in.device, img_arr_ref.device')
    # print(img_arr_in.device, img_arr_ref.device)
    # Reshape (h, w, c) to (c, h*w)
    [h, w, c] = img_arr_in.shape
    reshape_arr_in = img_arr_in.view(-1, c).permute(1, 0)
    reshape_arr_ref = img_arr_ref.view(-1, c).permute(1, 0)

    # PDF transfer
    reshape_arr_out = self.pdf_transfer_nd_torch(arr_in=reshape_arr_in,
                                                 arr_ref=reshape_arr_ref)

    # Reshape (c, h*w) to (h, w, c)
    reshape_arr_out.clamp_(0, 1)  # Ensure values are between 0 and 1
    reshape_arr_out = (255.0 * reshape_arr_out).to(torch.uint8)
    img_arr_out = reshape_arr_out.permute(1, 0).view(h, w, c)

    if regrain:
        img_arr_in = (255.0 * img_arr_in).to(torch.uint8)
        img_arr_out = self.RG.regrain(img_arr_in=img_arr_in,
                                      img_arr_col=img_arr_out)

    return img_arr_out.detach().cpu().numpy()

PT._pdf_transfer_1d_torch = _pdf_transfer_1d_torch.__get__(PT)
PT.pdf_transfer_nd_torch = pdf_transfer_nd_torch.__get__(PT)
PT.pdf_transfer = pdf_transfer_torch.__get__(PT)
PT.rotation_matrices_torch = [torch.from_numpy(o).float().cuda() for o in PT.rotation_matrices]

def match_color_var(stylized_img, raw_img, opacity=1., f=PT.pdf_transfer, regrain=False):
  ts = time.time()
  img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
  img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
  img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC )

  # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
  img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
  if regrain: img_arr_col = RG.regrain     (img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
  img_arr_col = img_arr_col*opacity+img_arr_in*(1-opacity)
  img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'),cv2.COLOR_BGR2RGB)
  printf('Match color took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')
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
# from ldm.modules.midas.api import load_midas_transform
# midas_tfm = load_midas_transform("dpt_hybrid")

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
#todo add batching for >2 cond size
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
    'cc_masked_template':cc_masked_template,
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
    'cc_masked_diffusion_schedule':cc_masked_diffusion_schedule,
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
    'pose_detector':pose_detector,
    'control_sd15_openpose_hands_face':control_sd15_openpose_hands_face,
    'control_sd15_depth_detector':control_sd15_openpose_hands_face,
    'control_sd15_softedge_detector':control_sd15_softedge_detector,
    'control_sd15_seg_detector':control_sd15_seg_detector,
    'control_sd15_scribble_detector':control_sd15_scribble_detector,
    'control_sd15_lineart_coarse':control_sd15_lineart_coarse,
    'control_sd15_inpaint_mask_source':control_sd15_inpaint_mask_source,
    'control_sd15_shuffle_source':control_sd15_shuffle_source,
    'control_sd15_shuffle_1st_source':control_sd15_shuffle_1st_source,
    'consistency_dilate':consistency_dilate,
    'apply_freeu_after_control':apply_freeu_after_control,
    'do_freeunet':do_freeunet
    }
    settings_hash = hashlib.sha256(json.dumps(rec_noise_setting_list).encode('utf-8')).hexdigest()[:16]
    filepath = f'{recNoiseCacheFolder}/{settings_hash}_{frame_num:06}.pt'
    if os.path.exists(filepath) and not overwrite_rec_noise:
      print(filepath)
      noise = torch.load(filepath)
      print('loading existing noise')
      return noise
    steps = int(copy.copy(steps)*rec_steps_pct)

    cfg_scale=rec_cfg
    if 'sdxl' in model_version:
      cond = sd_model.get_learned_conditioning(prompt)
      uncond = sd_model.get_learned_conditioning([''])
    else:
      cond = prompt_parser.get_learned_conditioning(sd_model, prompt, steps)
      uncond = prompt_parser.get_learned_conditioning(sd_model, [''], steps)
      cond = prompt_parser.reconstruct_cond_batch(cond, 0)
      uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)

    x = init_latent

    s_in = x.new_ones([x.shape[0]])
    if sd_model.parameterization == "v" or model_version == 'control_multi_v2_768':
        dnw = K.external.CompVisVDenoiser(sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(sd_model)
        skip = 0
    sigmas = dnw.get_sigmas(steps).flip(0)

    if 'sdxl' in model_version:
          vector = cond['vector']
          uc_vector = uncond['vector']
          y  = vector_in = torch.cat([uc_vector, vector])
          cond = cond['crossattn']
          uncond = uncond['crossattn']
          sd_model.conditioner.vector_in = vector_in

    if cond.shape[1]>77:
      cond = cond[:,:77,:]
      # print('Prompt length > 77 detected. Shorten your prompt or split into multiple prompts.')
    uncond = uncond[:,:77,:]
    for i in trange(1, len(sigmas)):


        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        uc_mask_shape = torch.ones(cond_in.shape[0], device=cond_in.device)
        uc_mask_shape[0] = 0
        sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape

        # image_conditioning = torch.cat([image_conditioning] * 2)
        # cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}
        if model_version == 'control_multi' and controlnet_multimodel_mode == 'external':
          raise Exception("Predicted noise not supported for external mode. Please turn predicted noise off or use internal mode.")
        if image_conditioning is not None:
          if 'control_multi' not in model_version:
            if model_version in ['sdxl_base', 'sdxl_refiner']:
              sd_model.conditioner.vector_in = vector_in[i*batch_size:(i+1)*batch_size]
            if img_zero_uncond:
              img_in = torch.cat([torch.zeros_like(image_conditioning),
                                          image_conditioning])
            else:
              img_in = torch.cat([image_conditioning]*2)
            cond_in={"c_crossattn": [cond_in],'c_concat': [img_in]}

          if 'control_multi' in model_version and controlnet_multimodel_mode != 'external':
            img_in = {}
            for key in image_conditioning.keys():
                  img_in[key] = torch.cat([torch.zeros_like(image_conditioning[key]),
                                              image_conditioning[key]]) if img_zero_uncond else torch.cat([image_conditioning[key]]*2)

            cond_in = {"c_crossattn": [cond_in],  'c_concat': img_in,
              'controlnet_multimodel':controlnet_multimodel_inferred,
              'loaded_controlnets':loaded_controlnets}
            if 'sdxl' in model_version:
                cond_in['y'] = y


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
      # if VERBOSE: print('Applying callback at step ', args['i'])
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

vae_cache = {}

pred_noise = None
def run_sd(opt, init_image, skip_timesteps, H, W, text_prompt, neg_prompt, steps, seed,
           init_scale,  init_latent_scale, cond_image, cfg_scale, image_scale,
           cond_fn=None, init_grad_img=None, consistency_mask=None, frame_num=0,
           deflicker_src=None, prev_frame=None, rec_prompt=None, rec_frame=None,
           control_inpainting_mask=None, shuffle_source=None, ref_image=None, alpha_mask=None,
           prompt_weights=None, mask_current_frame_many=None, controlnet_sources={}, cc_masked_diffusion=[0]):

  # sampler = sample_euler

  # if model_version in ['sdxl_base', 'sdxl_refiner']:
  #   print('Disabling init_scale for sdxl')
  #   init_scale = 0


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
  prompts =  text_prompt

  if VERBOSE:print('prompts', prompts, text_prompt)

  precision_scope = autocast

  t_enc = ddim_steps-skip_timesteps

  if init_image is not None:
    if isinstance(init_image, str):
      if not init_image.endswith('_lat.pt'):
        init_image_sd = load_img_sd(init_image, size=(W,H)).cuda()
        with torch.no_grad():
          with torch.autocast('cuda'):
            init_image_hash = f'{generate_file_hash(init_image)}_{W}_{H}'
            if init_image_hash in vae_cache.keys() and use_vae_cache:
              init_latent = x0 = vae_cache[init_image_hash]
              printf('using vae cache', file='./logs/profiling.txt')
            else:

              if no_half_vae:
                sd_model.first_stage_model.float()
                init_image_sd = init_image_sd.float()
              init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(init_image_sd))
              x0 = init_latent
              if len(vae_cache.keys())>vae_cache_size: vae_cache.pop(list(vae_cache.keys())[0])
              vae_cache[init_image_hash] = x0

      if init_image.endswith('_lat.pt'):
        init_latent = torch.load(init_image).cuda()
        init_image_sd = None
        x0 = init_latent

  reference_latent = None
  if ref_image is not None and reference_active:
    if os.path.exists(ref_image):
      with torch.no_grad(), torch.cuda.amp.autocast():
        init_image_hash = f'{generate_file_hash(ref_image)}_{W}_{H}'
        if init_image_hash in vae_cache.keys() and use_vae_cache:
          reference_latent = vae_cache[init_image_hash]
          printf('using vae cache', file='./logs/profiling.txt')
        else:
          reference_img = load_img_sd(ref_image, size=(W,H)).cuda()
          reference_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(reference_img))
          if len(vae_cache.keys())>vae_cache_size: vae_cache.pop(list(vae_cache.keys())[0])
          vae_cache[init_image_hash] = reference_latent
    else:
      print('Failed to load reference image')
      ref_image = None



  if use_predicted_noise:
    if rec_frame is not None:
      with torch.cuda.amp.autocast():
        init_image_hash = f'{generate_file_hash(rec_frame)}_{W}_{H}'
        if init_image_hash in vae_cache.keys() and use_vae_cache:
          rec_frame_latent = vae_cache[init_image_hash]
          printf('using vae cache', file='./logs/profiling.txt')
        else:
          rec_frame_img = load_img_sd(rec_frame, size=(W,H)).cuda()
          rec_frame_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(rec_frame_img))
          if len(vae_cache.keys())>vae_cache_size: vae_cache.pop(list(vae_cache.keys())[0])
          vae_cache[init_image_hash] = rec_frame_latent

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

  # sampler = sample_lcm; print('overriding sampler to sample lem')
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
                            ts = time.time()
                            uc = sd_model.get_learned_conditioning([neg_prompt])
                            printf('UC encoding took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        ts = time.time()
                        c = sd_model.get_learned_conditioning(prompts)
                        printf('C encoding took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')

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
                                                          ).float()
                        else:
                          sigmas = model_wrap.get_sigmas(ddim_steps).float()
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

                        depth_img = None
                        depth_cond = None
                        if model_version == 'v2_depth':
                          if VERBOSE: print('using depth')
                          depth_cond, depth_img = make_depth_cond(cond_image, x0)
                        if 'control_' in model_version:
                          input_image = np.array(Image.open(cond_image).resize(size=(W,H))); #print(type(input_image), 'input_image', input_image.shape)


                        if 'control_multi' in model_version:
                          if offload_model and not controlnet_low_vram:
                                        for key in loaded_controlnets.keys():
                                          loaded_controlnets[key].cuda()

                          models = list(controlnet_multimodel.keys()); print(models)
                        else: models = model_version

                        face_cc = None
                        if 'control_' in model_version:

                          controlnet_sources['control_inpainting_mask'] = control_inpainting_mask
                          controlnet_sources['shuffle_source'] = shuffle_source
                          controlnet_sources['prev_frame'] = prev_frame
                          controlnet_sources['init_image'] = init_image
                          init_image = np.array(Image.open(controlnet_sources['init_image']).convert('RGB').resize(size=(W,H)))
                          ts = time.time()
                          detected_maps, models, face_cc = get_controlnet_annotations(model_version, W, H, models, controlnet_sources)
                          printf('controlnet annotation took ', f'{time.time()- ts:4.2}', file='./logs/profiling.txt')

                          if gc_collect: gc.collect()
                          if cuda_empty_cache: torch.cuda.empty_cache()
                          if gc_collect: gc.collect()
                          if VERBOSE: print('Postprocessing cond maps')
                          def postprocess_map(detected_map):
                            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                            control = torch.stack([control for _ in range(num_samples)], dim=0)
                            depth_cond = einops.rearrange(control, 'b h w c -> b c h w').clone()
                            # if VERBOSE: print('depth_cond', depth_cond.min(), depth_cond.max(), depth_cond.mean(), depth_cond.std(), depth_cond.shape)
                            return depth_cond

                          if 'control_multi' in model_version:
                            print('init shape', init_latent.shape, H,W)
                            if render_mode == 'render, preview controlnet':
                              imgs = [fit(add_text_below_pil_image(PIL.Image.fromarray(detected_maps[m].astype('uint8')), m), maxsize=display_size) for m in models]

                              if use_background_mask:
                                imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.open(f'{videoFramesAlpha}/{frame_num+1:06}.jpg').convert('L'), 'background_mask'),
                                            maxsize=display_size))
                              if consistency_mask is not None:
                                imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.fromarray((consistency_mask*255).astype('uint8')).convert('L'),'consistency_mask'), maxsize=display_size))
                              if isinstance(init_image, str):
                                imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.open(init_image).convert('L'),'init_image'), maxsize=display_size))
                              else:
                                imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.fromarray(init_image.astype('uint8')),'init_image'), maxsize=display_size))
                                print('init image', type(init_image))
                              imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg').convert('RGB'), 'raw_frame'),
                                            maxsize=display_size))
                              if stack_previews:
                                if hstack_previews:
                                  imgs = hstack(imgs)
                                else:
                                  imgs = vstack(imgs)
                                if fit_previews: imgs = fit(imgs, maxsize=display_size)
                                # display.display(imgs)
                              else:
                                for img in imgs:
                                   pass
                                  # display.display(img)
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
                        if face_cc is not None and consistency_mask is not None:
                          # print('consistency_mask_t.shape,face_cc.shape, consistency_mask_t.max(), face_cc.max() ')
                          # print(consistency_mask_t.shape,face_cc.shape, consistency_mask_t.max(), face_cc.max() )
                          consistency_mask_t = torch.minimum(consistency_mask_t,torch.from_numpy(face_cc[...,0][None, None,...]).to(consistency_mask_t.device))
                          # torch.save(consistency_mask_t, './test.pt')
                        if frame_num > args.start_frame:
                          def absdiff(a,b):
                            return abs(a-b)
                          if deflicker_scale>0:
                            for key in deflicker_src.keys():
                              deflicker_src[key] = load_img_sd(deflicker_src[key], size=(W,H)).cuda()
                            deflicker_fn = partial(deflicker_loss, processed1=deflicker_src['processed1'][:,:,::2,::2],
                            raw1=deflicker_src['raw1'][:,:,::2,::2], raw2=deflicker_src['raw2'][:,:,::2,::2], criterion1= absdiff, criterion2=lpips_model)
                            fft_fn = partial(high_frequency_loss, image2=init_image_sd)
                          if deflicker_latent_scale>0:
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
                        if cc_masked_diffusion > 0 and consistency_mask is not None or alpha_masked_diffusion and alpha_mask is not None:
                          if cb_fixed_code:
                            if start_code_cb is None:
                              if VERBOSE:print('init start code')
                              start_code_cb = torch.randn_like(x0)
                          else:
                            start_code_cb = torch.randn_like(x0)
                          # start_code = torch.randn_like(x0)
                          callback_steps = []
                          callback_masks = []
                          if (cc_masked_diffusion > 0) and (consistency_mask is not None):
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
                          printf('frame_num', frame_num,'do cc callback',  file='./logs/resume_run_test.txt')
                          callback_partial = partial(masked_callback,
                                                     callback_steps=callback_steps,
                                                     masks=callback_masks,
                                                     init_latent=init_latent, start_code=start_code_cb)
                        if new_prompt_loras == {}: #todo - test this off
                          # only use cond fn when loras are off
                          model_fn = make_cond_model_fn(model_wrap_cfg, cond_fn_partial)
                          # model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
                        else:
                          model_fn = model_wrap_cfg

                        model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)


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
                          if offload_model:
                            sd_model.model.cuda()
                            sd_model.model.diffusion_model.cuda()
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
                            # with torch.autocast('cuda'):
                            # with torch.autocast('cuda', dtype=torch.float16):
                            ts = time.time()
                            # with torch.autocast('cuda'):
                            samples_ddim = sampler(model_fn, xi, sigma_sched,
                                                    extra_args=extra_args, callback=callback_partial)
                            printf('sampling took ', f'{time.time()- ts:4.2}', file='./logs/profiling.txt')
                          else:
                            samples_ddim = x0

                          if offload_model:
                            sd_model.model.cpu()
                            sd_model.model.diffusion_model.cpu()
                            if cuda_empty_cache: torch.cuda.empty_cache()
                            if gc_collect_offload: gc.collect()
                        else:
                          if offload_model:
                            sd_model.model.cuda()
                            sd_model.model.diffusion_model.cuda()
                          # if use_predicted_noise and frame_num>0:
                          if use_predicted_noise:
                              print('using predicted noise')
                              rand_noise = torch.randn_like(x0)
                              # with torch.autocast('cuda'):
                              rec_noise = find_noise_for_image_sigma_adjustment(init_latent=rec_frame_latent, prompt=rec_prompt, image_conditioning=depth_cond, cfg_scale=scale, steps=ddim_steps, frame_num=frame_num)
                              combined_noise = ((1 - rec_randomness) * rec_noise + rec_randomness * rand_noise) / ((rec_randomness**2 + (1-rec_randomness)**2) ** 0.5)
                              x = combined_noise# - (x0 / sigmas[0])

                          else: x = torch.randn([batch_size, *shape], device=device)
                          x = x * sigmas[0]
                          # with torch.autocast('cuda',dtype=torch.float16):
                          ts = time.time()
                          # with torch.autocast('cuda'):
                          samples_ddim = sampler(model_fn, x, sigmas, extra_args=extra_args, callback=callback_partial)
                          printf('sampling took ', f'{time.time()- ts:4.2}', file='./logs/profiling.txt')
                          if offload_model:
                            sd_model.model.cpu()
                            sd_model.model.diffusion_model.cpu()
                            if cuda_empty_cache: torch.cuda.empty_cache()
                            if gc_collect_offload: gc.collect()
                        if first_latent is None:
                          if VERBOSE:print('setting 1st latent')
                          first_latent_source = 'samples ddim (1st frame output)'
                          first_latent = samples_ddim

                        if offload_model:
                          sd_model.cond_stage_model.cpu()
                          if 'control_multi' in model_version:
                            for key in loaded_controlnets.keys():
                              loaded_controlnets[key].cpu()

                        if gc_collect: gc.collect()
                        if cuda_empty_cache: torch.cuda.empty_cache()
                        if offload_model:
                          sd_model.first_stage_model.cuda()
                        if no_half_vae:
                          sd_model.first_stage_model.float()
                          x_samples_ddim = sd_model.decode_first_stage(samples_ddim.float())
                        else:
                          x_samples_ddim = sd_model.decode_first_stage(samples_ddim)
                        if offload_model:
                          sd_model.first_stage_model.cpu()
                        printf('x_samples_ddim', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())
                        scale_raw_sample = False
                        if scale_raw_sample:
                          m = x_samples_ddim.mean()
                          x_samples_ddim-=m;
                          r = (x_samples_ddim.max()-x_samples_ddim.min())/2

                          x_samples_ddim/=r
                          x_samples_ddim+=m;
                          if VERBOSE:printf('x_samples_ddim scaled', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())

                        if x_samples_ddim.isnan().any():
                          print("""
Error: NaN encountered in VAE decode. You may get a black image.

To avoid this you can try:
1) enabling no_half_vae in load model cell, then re-running it.
2) disabling tiled vae and re-running tiled vae cell
3) If you are using SDXL, you can try keeping no_half_vae off,
then downloading and using this vae checkpoint as your external vae_ckpt: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors""")

                        all_samples.append(x_samples_ddim)
  return all_samples, samples_ddim, depth_img

def get_batch(keys, value_dict, N, device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "txt":
            if len(value_dict["prompt"]) != N[0]:
              batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            else: batch["txt"] = value_dict["prompt"]
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

diffusion_model = "stable_diffusion"
diffusion_sampling_mode = 'ddim'

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
def init_lpips():
  return lpips.LPIPS(net='vgg').to(device)
lpips_model = None

ipadapter_embeds_cache = {}

#todo
#offload face model
#offload canny, mlsd
def get_controlnet_annotations(model_version, W, H, models, controlnet_sources):
                        face_cc = None
                        detected_maps = {}
                        #controlnet sources have image paths
                        prev_frame = controlnet_sources['prev_frame']
                        if prev_frame is None:
                          controlnet_sources.pop('next_frame')
                        elif not os.path.exists(controlnet_sources['next_frame']):
                          if 'control_sd15_temporal_depth' in controlnet_multimodel_inferred.keys():
                            controlnet_sources['next_frame'] = controlnet_sources['control_sd15_temporal_depth']

                        init_image = controlnet_sources['init_image']
                        init_image = np.array(Image.open(controlnet_sources['init_image']).convert('RGB').resize(size=(W,H)))
                        control_inpainting_mask = controlnet_sources['control_inpainting_mask']

                        #todo: check that input images are hwc3 and int8, because loading grayscale images may return hw and 0-1 float images
                        shuffle_source = controlnet_sources['shuffle_source']
                        models_out = copy.deepcopy(models)

                        controlnet_sources_pil = dict([(o,np.array(Image.open(controlnet_sources[o]).convert('RGB').resize(size=(W,H)))) for o in models])

                        models_to_preprocess = [o for o in models if controlnet_multimodel_inferred[o]['preprocess']]

                        ts_clear_ip = time.time()
                        clear_all_ip_adapter()
                        printf('ts_clear_ip took ', f'{time.time()-ts_clear_ip:4.2}', file='./logs/profiling.txt')
                        for control_key in models:
                          if 'ipadapter' in control_key:

                            if control_key in ["ipadapter_sd15","ipadapter_sd15_light","ipadapter_sd15_plus","ipadapter_sd15_plus_face",
                                              "ipadapter_sd15_full_face", "ipadapter_sdxl_vit_h","ipadapter_sdxl_plus_vit_h",
                                              "ipadapter_sdxl_plus_face_vit_h", "ipadapter_sd15_faceid_plus",
                                               "ipadapter_sd15_faceid_plus_v2"]:
                              clip_vision_model = clip_vit_h
                            else:
                              clip_vision_model = clip_vit_g

                            """cache embeds"""
                            ipadapter_source_hash = str(generate_file_hash(os.path.join(root_dir, controlnet_sources[control_key])))+'_'+control_key
                            if ipadapter_source_hash in ipadapter_embeds_cache.keys() and cache_ipadapter:
                              clip_embed = ipadapter_embeds_cache[ipadapter_source_hash]
                            else:
                              ts_encode_ip = time.time()
                              print('controlnet_sources[control_key]', controlnet_sources[control_key])
                              image = img_loader_node.load_image(os.path.join(root_dir, controlnet_sources[control_key]))
                              if not ('faceid' in control_key and not 'plus' in control_key):
                                clip_vision_model.model.cuda().half()
                                clip_embed = clip_vision_model.encode_image(image[0].half().cuda())
                              if 'faceid' in control_key:
                                face_embed = g_insight_face_model.run_model(controlnet_sources_pil[control_key])[0]
                                if 'plus' in control_key:
                                  clip_embed = [face_embed, clip_embed]
                                else:
                                  clip_embed = face_embed

                              printf('ts_encode_ip took ', f'{time.time()-ts_encode_ip:4.2}', file='./logs/profiling.txt')
                              if len(ipadapter_embeds_cache.keys())>ipadapter_embeds_cache_size: ipadapter_embeds_cache.pop(list(ipadapter_embeds_cache.keys())[0])
                              ipadapter_embeds_cache[ipadapter_source_hash] = clip_embed
                              if offload_model:
                                if not ('faceid' in control_key and not 'plus' in control_key):
                                  clip_vision_model.model.cpu()
                                  if cuda_empty_cache: torch.cuda.empty_cache()
                                  if gc_collect_offload: gc.collect()
                            """cache embeds"""

                            weight = controlnet_multimodel_inferred[control_key]["weight"]
                            start = controlnet_multimodel_inferred[control_key]["start"]
                            end = controlnet_multimodel_inferred[control_key]["end"]
                            print(f'Applying {control_key} with source image {controlnet_sources[control_key]}')
                            ts_hook_ip = time.time()
                            if 'animatediff' in model_version:
                              eject_motion_module_from_unet(sd_model.model.diffusion_model, mm)
                            loaded_controlnets[control_key].hook(model=sd_model.model.diffusion_model, clip_vision_output=clip_embed,
                                         weight=weight, start=start, end=end)
                            models_out.remove(control_key)
                            if 'animatediff' in model_version:
                              inject_motion_module_to_unet(sd_model.model.diffusion_model, mm)

                            printf('ts_hook_ip took ', f'{time.time()-ts_hook_ip:4.2}', file='./logs/profiling.txt')

                          if control_key in ["control_sd15_inpaint_softedge"]:
                            #has detect res
                            #has preprocess option
                            if offload_model: apply_softedge.netNetwork.cuda()
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_softedge(resize_image(input_image, detect_resolution))
                              detected_map = HWC3(detected_map)
                              softedge_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                              # detected_maps[control_key] = detected_map
                            if offload_model: apply_softedge.netNetwork.cpu()

                            if control_inpainting_mask is None:
                              if VERBOSE: print(f'skipping {control_key} as control_inpainting_mask is None')
                              models_out = [o for o in models_out if o != control_key]
                              if VERBOSE: print('models after removing temp', models_out)
                            else:
                              print('Applying inpaint'
                              )
                              control_inpainting_mask *= 255
                              control_inpainting_mask = 255 - control_inpainting_mask
                              if VERBOSE: print('control_inpainting_mask',control_inpainting_mask.shape,
                                                control_inpainting_mask.min(), control_inpainting_mask.max())
                              if VERBOSE: print('control_inpainting_mask', (control_inpainting_mask[...,0] == control_inpainting_mask[...,0]).mean())
                              img = init_image #use prev warped frame
                              h, w, C = img.shape
                              #contolnet inpaint mask - H, W, 0-255 np array
                              detected_mask = cv2.resize(control_inpainting_mask[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
                              detected_map = img.astype(np.float32).copy()
                              detected_map[detected_mask > 127] = -255.0  # use -1 as inpaint value
                              detected_map = np.where(detected_map == -255, -1*softedge_map, detected_map)
                              detected_maps[control_key] = detected_map

                          if control_key in ['control_sd15_temporal_depth']:
                            if prev_frame is not None:
                              #no detect resolution
                              #no preprocessign option
                              #source options - prev raw, prev stylized
                              if offload_model:
                                apply_depth.model.cuda()

                              #has detect res
                              #has preprocess option
                              if offload_model: apply_depth.model.cuda()
                              input_image = controlnet_sources_pil[control_key]
                              detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                              if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                                detected_maps[control_key] = detected_map
                              else:
                                input_image = HWC3(np.array(input_image)); #print(type(input_image))
                                # Image.fromarray(input_image.astype('uint8')).save('./test.jpg')
                                input_image = resize_image(input_image, detect_resolution); #print((input_image.dtype), input_image.shape, input_image.size)
                                with torch.cuda.amp.autocast(False), torch.no_grad():
                                    detected_map = apply_depth(input_image)
                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                detected_maps[control_key] = detected_map
                                if 'next_frame' in controlnet_sources_pil.keys():
                                  next_frame = controlnet_sources_pil['next_frame']
                                  input_image  = HWC3(np.array(next_frame)); #print(type(input_image))

                                  input_image = resize_image(input_image, detect_resolution); #print((input_image.dtype), input_image.shape, input_image.size)
                                  with torch.cuda.amp.autocast(False), torch.no_grad():
                                      detected_map = apply_depth(input_image)
                                  detected_map = HWC3(detected_map)
                                  detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                  detected_maps[control_key][...,2] = detected_map[...,0]

                              if offload_model: apply_depth.model.cpu()

                              detected_map = np.array(Image.open(prev_frame).resize(size=(W,H)).convert('L')); #print(type(input_image), 'input_image', input_image.shape)
                              detected_maps[control_key][...,0]= detected_map
                              # Image.fromarray(input_image.astype('uint8')).save('./temp_test.jpg')
                            else:
                              if VERBOSE: print('skipping temporalnet as prev_frame is None')
                              models_out = [o for o in models_out if o != control_key]
                              if VERBOSE: print('models after removing temp', models_out)

                          if control_key in ['control_sd15_temporalnet', 'control_sdxl_temporalnet_v1']:
                            #no detect resolution
                            #no preprocessign option
                            #source options - prev raw, prev stylized
                            if prev_frame is not None:
                              detected_map = np.array(Image.open(prev_frame).resize(size=(W,H))); #print(type(input_image), 'input_image', input_image.shape)
                              detected_maps[control_key] = detected_map
                            else:
                              if VERBOSE: print('skipping temporalnet as prev_frame is None')
                              models_out = [o for o in models_out if o != control_key]
                              if VERBOSE: print('models after removing temp', models_out)

                          if control_key == 'control_sd15_face':
                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image

                            else:
                              input_image = resize_image(input_image,
                                                       detect_resolution)
                              detected_map = generate_annotation(input_image, max_faces)

                              lips_color = np.array([10,180,10])

                              # np.save('./test.np', detected_map)
                              if detected_map is not None and 'animatediff' not in model_version:
                                if fill_lips>0 and control_inpainting_mask is not None:
                                  face_cc = mask_color_and_add_strokes(detected_map, lips_color,
                                                                      stroke_color=(255, 255, 255), tolerance=0, stroke_width=fill_lips)
                                  face_cc = 1 - cv2.resize(face_cc, (W, H), interpolation=cv2.INTER_LINEAR)/255.
                                  # print('control_inpainting_mask.max(),type(control_inpainting_mask), face_cc.max()')
                                  # print(control_inpainting_mask.max(),type(control_inpainting_mask), face_cc.max())
                                  # print(control_inpainting_mask.dtype)
                                  control_inpainting_mask = np.minimum(control_inpainting_mask,face_cc.astype('uint8'))

                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                detected_maps[control_key] = detected_map
                              else:
                                if VERBOSE: print('No faces detected')
                                models_out = [o for o in models_out if o != control_key ]
                                if VERBOSE: print('models after removing face', models_out)

                          if control_key == 'control_sd15_normal':
                            #has detect res
                            #has preprocess option
                            if offload_model: apply_depth.model.cuda()
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image[:, :, ::-1]
                            else:
                              input_image = HWC3(np.array(input_image)); print(type(input_image))
                              input_image = resize_image(input_image, detect_resolution); print((input_image.dtype))
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                _,detected_map = apply_depth(input_image, bg_th=bg_threshold)
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                              detected_maps[control_key] = detected_map
                            if offload_model: apply_depth.model.cpu()

                          if control_key in ['control_sd15_normalbae',"control_sd21_normalbae"]:
                            #has detect res
                            #has preprocess option
                            if offload_model: apply_normal.model.cuda()
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]

                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image[:, :, ::-1]
                            else:
                              input_image = HWC3(np.array(input_image)); print(type(input_image))
                              input_image = resize_image(input_image, detect_resolution); print((input_image.dtype))
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_normal(input_image)
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                              detected_maps[control_key] = detected_map
                            if offload_model: apply_normal.model.cpu()

                          if control_key in ["control_sd21_depth",'control_sd15_depth','control_sdxl_depth',
                                             'control_sdxl_lora_128_depth','control_sdxl_lora_256_depth', 'control_sd15_depth_anything']:
                            if offload_model:
                              apply_depth.model.cuda()

                            #has detect res
                            #has preprocess option
                            if offload_model: apply_depth.model.cuda()
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]

                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(np.array(input_image))
                              # Image.fromarray(input_image.astype('uint8')).save('./test.jpg')
                              input_image = resize_image(input_image, detect_resolution)
                              printf('b4', control_sd15_depth_detector, input_image.max(), file='./depth.txt')
                              if control_sd15_depth_detector == 'Midas':
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                  detected_map,_ = apply_depth(input_image)
                              if control_sd15_depth_detector == 'Zoe':
                                with torch.cuda.amp.autocast(False), torch.no_grad():
                                  detected_map = apply_depth(input_image)
                              if control_sd15_depth_detector == 'depth_anything':
                                  detected_map = apply_depth(input_image, colored = False)
                                  if offload_model: apply_depth.model.to('cpu')

                              colored = 'depth_anything' in control_key
                              if colored:
                                #apply heatmap to any depth map to be usable with depth_anything controlnet
                                detected_map = cv2.applyColorMap(detected_map, cv2.COLORMAP_INFERNO)[:, :, ::-1]
                              printf(control_key, 'colored', colored, file='./depth.txt')
                              printf('after', control_sd15_depth_detector, input_image.max(), file='./depth.txt')
                              #print('dectected map depth',detected_map.shape, detected_map.min(), detected_map.max(), detected_map.mean(), detected_map.std(),  )
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                              detected_maps[control_key] = detected_map
                            if offload_model: apply_depth.model.cpu()

                          if control_key in ["control_sd21_canny",'control_sd15_canny','control_sdxl_canny', 'control_sdxl_lora_128_canny','control_sdxl_lora_256_canny']:
                            #has detect res
                            #has preprocess option

                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              detected_map = apply_canny(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map

                          if control_key in ["control_sd21_softedge",'control_sd15_softedge','control_sdxl_softedge', 'control_sdxl_lora_128_softedge', 'control_sdxl_lora_256_softedge']:
                            #has detect res
                            #has preprocess option
                            if offload_model:
                              apply_softedge.netNetwork.cuda()
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                apply_softedge.netNetwork.cuda()
                                detected_map = apply_softedge(resize_image(input_image, detect_resolution))
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                              detected_maps[control_key] = detected_map
                            if offload_model: apply_softedge.netNetwork.cpu()

                          if control_key == 'control_sd15_mlsd':
                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_mlsd(resize_image(input_image, detect_resolution), value_threshold, distance_threshold)
                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map

                          if control_key in ["control_sd21_openpose",'control_sd15_openpose','control_sdxl_openpose']:
                            #has detect res
                            #has preprocess option
                            if offload_model:
                              if pose_detector == 'openpose':
                                apply_openpose.body_estimation.model.cuda()
                                apply_openpose.hand_estimation.model.cuda()
                                apply_openpose.face_estimation.model.cuda()

                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              resized_img = resize_image(input_image,
                                    detect_resolution)
                              try:
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                  if pose_detector == 'openpose':
                                    detected_map = apply_openpose(resized_img, hand_and_face=control_sd15_openpose_hands_face)
                                  elif pose_detector == 'dw_pose':
                                    detected_map = apply_openpose(resized_img)
                              except:
                                detected_map = np.zeros_like(resized_img)

                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map
                            if offload_model:
                              if pose_detector == 'openpose':
                                apply_openpose.body_estimation.model.cpu()
                                apply_openpose.hand_estimation.model.cpu()
                                apply_openpose.face_estimation.model.cpu()

                          if control_key in ['control_sd15_scribble',"control_sd21_scribble"]:
                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)

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
                              detected_maps[control_key] = detected_map
                            if offload_model: apply_scribble.netNetwork.cpu()

                          if control_key in ["control_sd21_seg", "control_sd15_seg",'control_sdxl_seg']:

                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_seg(resize_image(input_image, detect_resolution))

                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map

                          if control_key in ["control_sd21_lineart", "control_sd15_lineart"]:
                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred['control_sd15_lineart']["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_lineart(resize_image(input_image, detect_resolution), coarse=control_sd15_lineart_coarse)

                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map

                          if control_key in ["control_sd15_lineart_anime"]:
                            #has detect res
                            #has preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if not controlnet_multimodel_inferred[control_key]["preprocess"]:
                              detected_maps[control_key] = input_image
                            else:
                              input_image = HWC3(input_image)
                              with torch.cuda.amp.autocast(True), torch.no_grad():
                                detected_map = apply_lineart_anime(resize_image(input_image, detect_resolution))

                              detected_map = HWC3(detected_map)
                              detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                              detected_maps[control_key] = detected_map

                          if control_key in ["control_sd15_ip2p"]:
                            #no detect res
                            #no preprocess option
                            #ip2p has no separate detect resolution
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            input_image = HWC3(input_image)
                            detected_map = input_image.copy()
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps[control_key] = detected_map

                          if control_key in ["control_sd15_gif","control_sd15_tile","control_sd15_qr", "control_sd21_qr","control_sd15_monster_qr"]:
                            #no detect res
                            #no preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            # print('qr', input_image.shape)
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            input_image = HWC3(input_image)
                            detected_map = input_image.copy()
                            # print('detected_map.shape')
                            # print(detected_map.shape)
                            if qr_cn_mask_grayscale:
                              detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2GRAY)[...,None].repeat(3,2)
                            # print(detected_map.shape)
                            if qr_cn_mask_invert:
                              detected_map = 255-detected_map
                            if qr_cn_mask_thresh>0:
                              detected_map = np.where(detected_map>qr_cn_mask_thresh, 255, 0)
                            detected_map = detected_map.clip(qr_cn_mask_clip_low, qr_cn_mask_clip_high)
                            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                            detected_maps[control_key] = detected_map

                          if control_key in ["control_sd15_shuffle"]:
                            #shuffle has no separate detect resolution
                            #no preprocess option
                            shuffle_image = np.array(Image.open(shuffle_source))
                            shuffle_image = HWC3(shuffle_image)
                            shuffle_image = cv2.resize(shuffle_image, (W, H), interpolation=cv2.INTER_NEAREST)

                            dH, dW, dC = shuffle_image.shape
                            detected_map = apply_shuffle(shuffle_image, w=dW, h=dH, f=256)
                            detected_maps[control_key] = detected_map

                          if control_key in ["control_sd15_inpaint", "control_sdxl_inpaint"]:
                            #defaults to init image (stylized prev frame)
                            #inpaint has no separate detect resolution
                            #no preprocess option
                            input_image = controlnet_sources_pil[control_key]
                            detect_resolution = controlnet_multimodel_inferred[control_key]["detect_resolution"]
                            if control_inpainting_mask is None:
                              printf('skip inpaint',  file='./logs/resume_run_test.txt')
                              if VERBOSE: print('skipping control_sd15_inpaint as control_inpainting_mask is None')
                              models_out = [o for o in models_out if o != control_key]
                              if VERBOSE: print('models after removing temp', models_out)
                            else:
                              printf('do inpaint',  file='./logs/resume_run_test.txt')
                              control_inpainting_mask *= 255
                              control_inpainting_mask = 255 - control_inpainting_mask
                              if VERBOSE: print('control_inpainting_mask',control_inpainting_mask.shape,
                                                control_inpainting_mask.min(), control_inpainting_mask.max())
                              if VERBOSE: print('control_inpainting_mask', (control_inpainting_mask[...,0] == control_inpainting_mask[...,0]).mean())
                              img = input_image
                              h, w, C = img.shape
                              #contolnet inpaint mask - H, W, 0-255 np array
                              detected_mask = cv2.resize(control_inpainting_mask[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
                              detected_map = img.astype(np.float32).copy()
                              detected_map[detected_mask > 127] = -255.0  # use -1 as inpaint value
                              detected_maps[control_key] = detected_map



                        return detected_maps, models_out, face_cc

#animatediff-warp code
# (c) Alex Spirin 2023
def make_ctx_sched(
    total_length = 32,
    context_length = 16,
    overlap = 4,
    steps = 15
):

    idxs=list(range(total_length))
    step = context_length-overlap
    step_ids = []

    for i in range(steps):
        inner_ids = [idxs[:context_length]]
        if idxs[-context_length:] not in inner_ids:
            inner_ids.append(idxs[-context_length:])

        start_offset = max(-step, -((i)*2%step))
        # print('start_offset', start_offset)
        for j in range(math.ceil(len(idxs)/(step))):

            start = j*step+start_offset
            end = j*step + context_length+start_offset
            if end>len(idxs):
                end = None
                start = -context_length

            ids = idxs[start:end]
            if ids not in inner_ids and ids!=[]:
                inner_ids.append(ids)
        random.shuffle(inner_ids)
        step_ids.append(inner_ids)
    return step_ids

def postprocess_map(detected_map):
  num_samples = 1
  control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
  control = torch.stack([control for _ in range(num_samples)], dim=0)
  depth_cond = einops.rearrange(control, 'b h w c -> b c h w').clone()
  # if VERBOSE: print('depth_cond', depth_cond.min(), depth_cond.max(), depth_cond.mean(), depth_cond.std(), depth_cond.shape)
  return depth_cond

def get_controlnet_annotations_by_frame_num(frame_num):
  models = list(controlnet_multimodel.keys())
  preprocess_models = [o for o in models if o not in no_preprocess_cn]
  controlnet_sources = {}
  if controlnet_multimodel != {}:
    W, H = width_height
    #we can only get stylized when we're overlapping, if not - use raw
    init_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
    current_stylized = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num:06}.{save_img_format}'
    if os.path.exists(current_stylized):
      # print('Using stylized cond src')
      init_image = current_stylized
    controlnet_sources = get_control_source_images(frame_num, controlnet_multimodel_inferred, stylized_image=init_image)
    # print('controlnet_sources',controlnet_sources)
    pseudo_init = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
    controlnet_sources['control_inpainting_mask'] = pseudo_init
    controlnet_sources['shuffle_source'] = pseudo_init
    controlnet_sources['init_image'] = pseudo_init
    controlnet_sources['prev_frame'] = pseudo_init
    controlnet_sources['next_frame'] = pseudo_init
    # print('models', models)
    if 'control_multi' in model_version:
      detected_maps, models, _ = get_controlnet_annotations(model_version, W, H, models, controlnet_sources)
      # if VERBOSE: print('Postprocessing cond maps')
      for m in models:
        if save_controlnet_annotations:
          PIL.Image.fromarray(detected_maps[m].astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_{m}_{frame_num:06}.jpg', quality=95)
        detected_maps[m] = postprocess_map(detected_maps[m])
      # print('detected_maps', str([(key, item.shape) for (key, item) in detected_maps.items()]))
    else:
      detected_maps = postprocess_map(detected_maps[model_version])
    return detected_maps, models
  return None, []

def preprocess_prompt(prompt, frame_num):
  prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in prompt] #remove loras from prompt
  prompt = [re.sub(":\s*([\d.]+)\s*$", '', o).strip(' ') for o in prompt] #remove weights from prompt

  #add captions
  caption = get_caption(frame_num)
  if caption:
    for i in range(len(prompt)):
      if '{caption}' in prompt[i]:
        print('Replacing ', '{caption}', 'with ', caption)
        prompt[0] = prompt[i].replace('{caption}', caption)

  #apply pattern replacement
  prompt_patterns = get_sched_from_json(frame_num, prompt_patterns_sched, blend=False)
  if prompt_patterns:
    for key in prompt_patterns.keys():
      for i in range(len(prompt)):
        if key in prompt[i]:
          print('Replacing ', key, 'with ', prompt_patterns[key])
          prompt[i] = prompt[i].replace(key, prompt_patterns[key])

  return prompt

def get_sched_args_from_frame_idxs(frame_idxs, args):
  scheduled_args = {}
  scheduled_args_keys = ['steps', 'style_strength', 'skip_steps', 'text_prompt',
                         'neg_prompt', 'cfg_scale', 'frame_paths','controlnet_annotations',
                         'used_loras', 'used_loras_weights', 'colormatch_file', 'ref_image','prompt_weights','prompt_uweights']

  for key in scheduled_args_keys:
    scheduled_args[key] = []

  for i, frame_num in enumerate(frame_idxs):
    scheduled_args['prompt_weights'].append(get_sched_from_json(frame_num, prompt_weights, blend=blend_json_schedules))
    scheduled_args['prompt_uweights'].append(get_sched_from_json(frame_num, prompt_uweights, blend=blend_json_schedules))
    scheduled_args['ref_image'].append(get_ref_source_image(frame_num))
    colormatch_file = None
    if colormatch_frame != 'off':
      if 'stylized' in colormatch_frame:
          for j in list(range(0,frame_num+1))[::-1]:
            # for stylized mode we most likely won't have the specified frame yet, so we loop back until we find a stylized frame
            colormatch_file_path = get_frame_from_color_mode(colormatch_frame, colormatch_offset, j)
            if os.path.exists(colormatch_file_path):
              colormatch_file = colormatch_file_path
              break
          if colormatch_file is None:
            colormatch_file_path = get_frame_from_color_mode(colormatch_frame.replace('stylized', 'init'), colormatch_offset, j)
            if os.path.exists(colormatch_file_path):
              colormatch_file = colormatch_file_path
      else:
          colormatch_file_path = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
          if os.path.exists(colormatch_file_path):
            colormatch_file = colormatch_file_path

    scheduled_args['colormatch_file'] = colormatch_file
    scheduled_args['steps'].append(int(get_scheduled_arg(frame_num, steps_schedule)))
    scheduled_args['style_strength'].append(get_scheduled_arg(frame_num, style_strength_schedule))
    scheduled_args['skip_steps'].append(int(scheduled_args['steps'][0]-scheduled_args['steps'][0]*scheduled_args['style_strength'][0]))
    text_prompt = copy.copy(get_sched_from_json(frame_num, args.prompts_series, blend=False))
    text_prompt = preprocess_prompt(text_prompt, frame_num)
    scheduled_args['text_prompt'].append(text_prompt)
    if args.neg_prompts_series is not None:
      neg_prompt = copy.copy(get_sched_from_json(frame_num, args.neg_prompts_series, blend=False))
    else:
      neg_prompt = copy.copy(text_prompt)
    neg_prompt = preprocess_prompt(neg_prompt, frame_num)
    scheduled_args['neg_prompt'].append(neg_prompt)
    scheduled_args['cfg_scale'].append(get_scheduled_arg(frame_num, cfg_scale_schedule))

    stylized_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}'
    if overlap_stylized and i<= batch_overlap and os.path.exists(stylized_frame):
      # print('Using stylized overlap input')
      scheduled_args['frame_paths'].append(stylized_frame)
    else:
      # print('Using raw overlap input')
      scheduled_args['frame_paths'].append(f'{videoFramesFolder}/{frame_num+1:06}.jpg')

    if 'animatediff' in model_version:
      detected_maps, models = get_controlnet_annotations_by_frame_num(frame_num)
      scheduled_args['controlnet_annotations'].append(detected_maps)
      scheduled_args['controlnet_models'] = models
    else:
      scheduled_args['controlnet_annotations'] = None
      scheduled_args['controlnet_models'] = []

    used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, new_prompt_loras)
    scheduled_args['used_loras_weights'].append(used_loras_weights)
    scheduled_args['used_loras'].append(used_loras)

  if 'animatediff' in model_version and scheduled_args['controlnet_models'] != []:
    result = {}
    for d in scheduled_args['controlnet_annotations']:
      for key, value in d.items():
        result.setdefault(key, []).append(value)
    for key  in result.keys():
      result[key] = torch.cat(result[key])
    scheduled_args['controlnet_annotations'] = result
  print('Using control models: ', scheduled_args['controlnet_models'])
  return scheduled_args

def path2latent(f, sz):
        im = PIL.Image.open(f).convert('RGB')
        im = im.resize(sz)
        im = np.array(im).astype(np.float32) / 255.0
        im = im[None].transpose(0, 3, 1, 2)
        im = torch.from_numpy(im)
        im = 2.*im - 1.
        with torch.autocast('cuda'):
            res = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(im.cuda()))
        return res

def prepare_latents(sigmas, scheduled_args):
  seed_everything(args.seed)
  total_length = len(scheduled_args['frame_paths'])
  skip_steps = scheduled_args['skip_steps'][0]
  ddim_steps = scheduled_args['steps'][0]
  if skip_steps > 0:
    sd_model.first_stage_model.cuda()
    out_f = []
    for f in tqdm(scheduled_args['frame_paths'], desc='Encoding input frames'):
      init_latent = path2latent(f, (W,H))
      out_f.append(init_latent)
    x = torch.cat(out_f)
    sd_model.first_stage_model.cpu()

    t_enc = ddim_steps-skip_steps
    noise = scheduled_args.get('noise', torch.randn_like(x, device='cpu')).cuda()
    if looped_noise: noise = torch.cat([noise[:batch_length]]*math.ceil(total_length/batch_length))[:total_length]

    noise *= sigmas[ddim_steps - t_enc -1].cuda()

    x = x + noise
    sigma_sched = sigmas[ddim_steps - t_enc - 1:]
  else:
    x = torch.randn([len(scheduled_args['steps']), 4, H//8, W//8], device='cpu').cuda()
    x = x * sigmas[0].cuda()
    sigma_sched = sigmas

  return x, sigma_sched

reference_latent = None
def prepare_ref_latent(scheduled_args):
    global reference_latent
    reference_latent = None
    ref_images = scheduled_args['ref_image']
    # print('ref image path', ref_images)
    sd_model.first_stage_model.cuda()
    if reference_active:
      out_ref = []
      for ref_image in ref_images:
        if ref_image is not None and os.path.exists(ref_image):
          with torch.no_grad(), torch.cuda.amp.autocast():
                reference_img = load_img_sd(ref_image, size=(W,H)).cuda()
                out_ref.append(sd_model.get_first_stage_encoding(sd_model.encode_first_stage(reference_img)).cuda())
        else:
          print('Failed to load reference image')
          return
      reference_latent = torch.cat(out_ref)

def split_stack_sdxl_cond(prompts):
  c = []
  vector = []
  for o in prompts:
    cond = sd_model.get_learned_conditioning(o)
    c.append(cond['crossattn'][:,:77,:])
    vector.append(cond['vector'])
  c = torch.stack(c)
  vector = torch.stack(vector)
  return {'crossattn': c,
          'vector':vector}

def process_prompts(scheduled_args):
  sd_model.cond_stage_model.cuda()
  # print(scheduled_args['text_prompt'])
  with torch.autocast('cuda'):
    if 'sdxl' in model_version:
      c = split_stack_sdxl_cond(scheduled_args['text_prompt'])
      uc = split_stack_sdxl_cond(scheduled_args['neg_prompt'])
    else:
      c = torch.stack([sd_model.get_learned_conditioning(o)[:,:77,:] for o in scheduled_args['text_prompt']])
      uc= torch.stack([sd_model.get_learned_conditioning(o)[:,:77,:] for o in scheduled_args['neg_prompt']])

  # print('c uc', c.shape, uc.shape)
  # print(c[0,0])
  # print(c[0,1])
  # print('c0 c1 mean', c[0,0].mean(), c[0,0].std(), c[0,0,0], c[0,1].mean(), c[0,1].std(), c[0,1,0])
  if offload_model:
    sd_model.cond_stage_model.cpu()
  if cuda_empty_cache: torch.cuda.empty_cache()
  if cuda_empty_cache: torch.cuda.ipc_collect()
  return c, uc

def repeat_list(l, n):
  out = []
  for o in l:
    for i in range(n):
      out.append(o)
  return out



adiff_pbar = None
ctx_ids = None
def run_adiff_batch(scheduled_args, args):
  global adiff_pbar, ctx_ids, adiff_pbar_total

  """juggle cuda"""
  sd_model.cuda()
  sd_model.model.cuda()
  sd_model.cuda()
  model_wrap.inner_model.cuda()
  model_wrap.cuda()
  model_wrap_cfg.cuda()
  model_wrap_cfg.inner_model.cuda()
  if offload_model:
    sd_model.first_stage_model.cpu()
  sd_model.cuda()

  """get sigmas"""
  sigmas = model_wrap.get_sigmas(scheduled_args['steps'][0]).float().cuda()

  """get latents"""
  latents, sigma_sched = prepare_latents(sigmas, scheduled_args)
  if reference_active:
    prepare_ref_latent(scheduled_args)

  """get text conds"""
  c, uc = process_prompts(scheduled_args)
  model_fn = model_wrap_cfg
  model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
  # print("scheduled_args['prompt_weights']", scheduled_args['prompt_weights'])
  scheduled_args['prompt_masks'] = None
  extra_args = {'cond': c, 'uncond': uc,
                'cond_scale': scheduled_args['cfg_scale'][0],
                'image_cond':scheduled_args['controlnet_annotations'] if ('control' in model_version and scheduled_args['controlnet_models'] != []) else None,
                'prompt_weights':scheduled_args['prompt_weights'],
                'prompt_uweights':scheduled_args['prompt_uweights'],
                'prompt_masks': scheduled_args['prompt_masks']
                }

  """make sampler inner schedule"""
  total_length = len(scheduled_args['frame_paths'])
  mm.set_video_length(context_length)
  ctx_ids = make_ctx_sched(
    total_length = total_length,
    context_length = context_length,
    overlap = context_overlap,
    steps = scheduled_args['steps'][0]-scheduled_args['skip_steps'][0]+1
  )
  if sampler.__name__ in ['sample_dpm_2','sample_dpm_2_ancestral','sample_dpmpp_2s_ancestral', 'sample_dpmpp_sde']:
    print('Repeating ctx for ', sampler.__name__)
    ctx_ids = repeat_list(ctx_ids, 2)
  total_substeps = len([o for a in ctx_ids for o in a])

  """jugle cuda"""
  if offload_model:
    sd_model.first_stage_model.cpu()
  if cuda_empty_cache: torch.cuda.empty_cache()
  if cuda_empty_cache: torch.cuda.ipc_collect()
  if gc_collect: gc.collect()

  # eject_motion_module_from_unet(sd_model.model.diffusion_model, mm)#todo remove it, temp test
  """diffuse"""
  seed_everything(args.seed)
  with torch.autocast('cuda'), torch.inference_mode():
    adiff_pbar_total = tqdm(total=total_substeps, desc='Diffusion substeps/batch:')
    adiff_pbar = tqdm(total=len(ctx_ids[0]), desc='Diffusion substeps/step:')
    samples_ddim = sampler(model_fn, latents, sigma_sched, extra_args=extra_args)
  # inject_motion_module_to_unet(sd_model.model.diffusion_model, mm)#todo remove it, temp test
  """decode latents"""
  with torch.autocast('cuda'), torch.inference_mode():
      sd_model.first_stage_model.cuda()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      gc.collect()
      x_samples_ddim = torch.cat([sd_model.decode_first_stage(o[None,...].float()) for o in tqdm(samples_ddim)])

  return x_samples_ddim

def save_adiff_frames(frame_idxs, pil_frames, settings_exif, blend=True, batch_overlap=0):
  with tqdm(total=len(frame_idxs), desc='Saving output frames:' ) as save_pbar:
    for j,(i,img) in enumerate(zip(frame_idxs, pil_frames)):
      fname = f'{batchFolder}/{args.batch_name}({args.batchNum})_{i:06}.{save_img_format}'
      if blend==True and batch_overlap>0 and os.path.exists(fname) and j<=batch_overlap:
        old_img = PIL.Image.open(fname)
        lerp = min(j/batch_overlap, 1) #goes from 0 to 1 when overlap ends
        # print('blending overlap with lerp ', lerp)
        img = np.array(img)*lerp + (1-lerp)*np.array(old_img)
        img = PIL.Image.fromarray(img.astype('uint8'))
      img.save(fname, exif=settings_exif)
      save_pbar.update(1)

def load_loras(scheduled_args):
    if 'animatediff' in model_version:
      eject_motion_module_from_unet(sd_model.model.diffusion_model, mm)
    used_loras = scheduled_args['used_loras'][0]
    used_loras_weights = scheduled_args['used_loras_weights'][0]
    if used_loras != []:
      print('Using loras: ', used_loras, ' with weights: ', used_loras_weights)
      printf('Using loras: ', used_loras, ' with weights: ', used_loras_weights)
    load_networks(names=used_loras, te_multipliers=used_loras_weights,
                  unet_multipliers=used_loras_weights, dyn_dims=[None]*len(used_loras), sd_model=sd_model)
    if 'animatediff' in model_version:
      inject_motion_module_to_unet(sd_model.model.diffusion_model, mm)

def apply_colormatch_batch(pil_frames, scheduled_args):
  for i in trange(len(pil_frames)):
    color_file = scheduled_args['colormatch_file']
    if color_file is not None and os.path.exists(color_file):
      color_img = PIL.Image.open(color_file).convert('RGB').resize(pil_frames[i].size)
      pil_frames[i] = PIL.Image.fromarray(match_color_var(color_img,
                        pil_frames[i], opacity=color_match_frame_str, f=colormatch_method_fn,
                        regrain=colormatch_regrain))
  return pil_frames

def do_run_adiff(args):
  """dump settings"""
  settings_json = save_settings()
  settings_exif = json2exif(settings_json)

  """set frames batches"""
  #todo: make edge cases with scene length < context_length work
  start_frame = scene_start
  end_frame = scene_end
  total_length = end_frame-start_frame+1
  assert total_length>= context_length, 'Total number of frames must be equal or higher than context_length. Please extract more frames or decrease context_length.'

  max_batches = 1 if total_length==batch_length else math.ceil(total_length/(batch_length-batch_overlap))
  batchBar = trange(max_batches, desc='Animatediff batches')
  display_frame = None
  big_noise = torch.randn((end_frame+1,4,H//8,W//8), device='cpu').half()
  # big_noise = big_noise[0:1].repeat(end_frame+1, 1,1,1 )

  for i in batchBar:
    """update display"""
    # display.clear_output(wait=True)
    print(f'Rendering frames {start_frame}-{end_frame}.\nSplitting {total_length} total frames in {len(batchBar)} batches of size {batch_length} with overlap of {batch_overlap}')
    # display.display(batchBar.container)
    if display_frame is not None:
       pass
      # display.display(display_frame)
    batchBar.n = i
    batchBar.refresh()

    """get frame ids for a batch"""
    frame_idxs = list(range(start_frame+i*(batch_length-batch_overlap),
                            min(start_frame+i*(batch_length-batch_overlap)+batch_length,end_frame+1)))
    # print('frameidxs', frame_idxs, batch_length, total_length, scene_start, scene_end)

    """
    repeat final frames to fill up the batch to reach batch_length
    pad noise for those frames from the beginning of the vector (can't repeat noise)
    """
    if len(frame_idxs) < batch_length:
      noise = torch.cat([big_noise[frame_idxs],big_noise[:batch_length-len(frame_idxs)]])
      frame_idxs+=[frame_idxs[-1]]*(batch_length-len(frame_idxs))
    else:
      noise = big_noise[frame_idxs]

    """get settings from schedules for current batch frame ids"""
    scheduled_args = get_sched_args_from_frame_idxs(frame_idxs, args)
    scheduled_args['noise'] = noise

    """load loras"""
    load_loras(scheduled_args)

    print(f"Doing {scheduled_args['steps'][0]-scheduled_args['skip_steps'][0]} diffusion steps. Using context {context_length} with overlap {context_overlap}")

    """run diffusion"""
    tensor_frames = run_adiff_batch(scheduled_args, args)

    """apply softcap"""
    if do_softcap:
      tensor_frames = torch.cat([softcap(o[None,...]) for o in tensor_frames])
      # tensor_frames = softcap(o, thresh=softcap_thresh, q=softcap_q)

    """convert frames from tensor to pil"""
    pil_frames = [TF.to_pil_image(o) for o in tensor_frames.add(1).div(2).clamp(0, 1)]

    #todo add masking here
    """colormatching"""
    if colormatch_frame != 'off' and colormatch_after:
      pil_frames = apply_colormatch_batch(pil_frames, scheduled_args)

    """save frames with possible overlap blending"""
    save_adiff_frames(frame_idxs, pil_frames, settings_exif, blend=blend_batch_outputs, batch_overlap=batch_overlap)
    display_frame = fit(pil_frames[0], display_size)

    """increment seed"""
    if not fixed_seed:
      args.seed += 1

  """display preview after run"""
  if display_frame is not None:
     pass
    # display.display(display_frame)

""" lazywarp """

def load_img_lz(img, size):
    if isinstance(img, str):
      img = Image.open(img)
    if isinstance(img, PIL.Image.Image):
      img = img.convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...]

def get_flow_and_cc(frame1_init, frame2, flow_path, cc_path=''):
  # print('frame1_init, frame2, flow_path, cc_path=', frame1_init, frame2, flow_path, cc_path)
  """
  lazy flow loader
  load flow and cc maps if exist, otherwise extracts them

  frame1_init: str, path to raw frame1
  frame2: str, path to raw frame 2

  """
  # print('flow_path, cc_path', flow_path, cc_path)
  cc = flow = None
  if cc_path is None: cc_path = ''
  if (os.path.exists(flow_path) and os.path.exists(cc_path)) and not force_flow_generation:
        flow = np.load(flow_path)
        flow_width_height = flow.shape[:2][::-1]
        # print('loading cc map')
        cc = load_cc(cc_path, blur=consistency_blur, dilate=consistency_dilate)

  else:
    sd_model.cpu();
    if cuda_empty_cache:
      torch.cuda.empty_cache();
      torch.cuda.ipc_collect()
    # p = Pool(flow_threads)
    if flow_maxsize not in [0,None,[],-1]:
      if isinstance(flow_maxsize, int):
        flow_width_height = fit_size(width_height, flow_maxsize)

      else:
        flow_width_height = flow_maxsize
      print(f'Resizing maps for flow to {flow_width_height}')
    else:
      flow_width_height = width_height
    frame1 = load_img_lz(frame1_init, flow_width_height)
    # print('frame1.shape b4 pad', frame1.shape)
    frame2 = load_img_lz(frame2, flow_width_height)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    # print('frame1.shape after pad', frame1.shape)
    batch = torch.cat([frame1, frame2])
    if normalize:
      batch = 2 * (batch / 255.0) - 1.0
    frame_1 = batch[0][None,...].cuda()
    frame_2 = batch[1][None,...].cuda()
    global raft_model
    raft_model = lazy_init_raft()
    raft_model.cuda().half()
    with torch.cuda.amp.autocast(), torch.no_grad(), torch.inference_mode():
      flow, cc = infer_flow_and_cc(frame_1, frame_2, flow_path=flow_path, cc_path=cc_path, pool=thread_pool, size=flow_width_height)
    raft_model.cpu()

    cc = load_cc(cc, blur=consistency_blur, dilate=consistency_dilate)


  cc = cv2.resize(cc, width_height, interpolation=cv2.INTER_LINEAR)
  flow = cv2.resize(flow, width_height, interpolation=cv2.INTER_LINEAR).astype('float32')
  flow_ratio = (np.array(flow_width_height)/np.array(width_height))[None,None,...].astype('float32')
  printf('flow_width_height, width_height, flow_ratio', flow_width_height, width_height, flow_ratio, file='./logs/flow.txt')
  flow = flow*flow_ratio

  return flow, cc


class InsightFaceModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            try:
              from insightface.app import FaceAnalysis
            except:
              print('Insightface not found. Installing.')
              pipi('insightface')
              pipi('pillow==9.0.0')
              print('Please restart the env and run all.')

            self.model = FaceAnalysis(
                name="buffalo_l",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                root=os.path.join(root_dir, "insightface"),
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))

    def run_model(self, img, **kwargs):
        self.load_model()
        img = HWC3(img)
        faces = self.model.get(img)
        if not faces:
            raise Exception("Insightface: No face found in image.")
        if len(faces) > 1:
            logger.warn("Insightface: More than one face is detected in the image. "
                        "Only the first one will be used")
        faceid_embeds = {
            "image_embeds": torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        }
        return faceid_embeds, False

# (c) Alex Spirin, 2024

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def add_text_below_pil_image(original_image, text, rectangle_height=60, font_size=40, font_path=None):
    """
    Adds a black rectangle with centered text below a PIL Image object, with resizable font.

    :param original_image: A PIL Image object of the original image.
    :param text: Text to add to the image.
    :param rectangle_height: Height of the black rectangle.
    :param font_size: Font size of the text.
    :param font_path: Path to a .ttf font file. If None, attempts to use a default resizable font.
    :return: A modified PIL Image object with the added text.
    """
    if not add_preview_label:
      return original_image

    original_width, original_height = original_image.size

    # Create a new image with extra space for the rectangle
    new_height = original_height + rectangle_height
    new_image = Image.new("RGB", (original_width, new_height), "black")
    new_image.paste(original_image, (0, 0))

    # Prepare to draw the rectangle and text
    draw = ImageDraw.Draw(new_image)

    # If a font path is provided, use it; otherwise, attempt to use a default TrueType font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # Attempt to use a common system font as a default
        try:
            # Common paths for a default TrueType font
            common_system_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "C:/Windows/Fonts/Arial.ttf",  # Windows
                "/System/Library/Fonts/SFCompactDisplay-Regular.otf"  # macOS
            ]
            # Try to find a valid font path
            font = next((ImageFont.truetype(font_path, font_size) for font_path in common_system_fonts if Path(font_path).exists()), ImageFont.load_default())
        except StopIteration:
            # Fallback if no font file is found
            font = ImageFont.load_default()

    # Calculate text width and height
    text_width, text_height = draw.textsize(text, font=font)

    # Calculate text position
    text_x = (original_width - text_width) / 2
    text_y = original_height + (rectangle_height - text_height) / 2

    # Draw the text
    draw.text((text_x, text_y), text, fill="white", font=font)

    return new_image


#\@title Generate optical flow and consistency maps
#\@markdown Run once per init video and width_height setting.
#if you're running locally, just restart this runtime, no need to edit PIL files.

lazy_warp = True
flow_warp = True
check_consistency = True


#\@title Setup Optical Flow
#\#@markdown Run once per session. Doesn't download again if model path exists.
#\#@markdown Use force download to reload raft models if needed
force_download = False #\@param {type:'boolean'}
# import wget
import zipfile, shutil

os.chdir(root_dir)
sys.path.append('./python-color-transfer')

#\@title Define color matching and brightness adjustment
os.chdir(f"{root_dir}/python-color-transfer")
from python_color_transfer.color_transfer import ColorTransfer, Regrain
os.chdir(root_path)

PT = ColorTransfer()
RG = Regrain()

def match_color(stylized_img, raw_img, opacity=1.):
  if opacity > 0:
    ts = time.time()
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    img_arr_reg = RG.regrain     (img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_reg = img_arr_reg*opacity+img_arr_in*(1-opacity)
    img_arr_reg = cv2.cvtColor(img_arr_reg.round().astype('uint8'),cv2.COLOR_BGR2RGB)
    printf('Match color took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')
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

##@title Define optical flow functions for Video input animation mode only
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
if animation_mode == 'Video Input':
  #the main idea comes from neural-style-tf frame warping with optical flow maps
  #https://github.com/cysmith/neural-style-tf

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
    if type(path) == str:
      img = Image.open(path)
    else:
      img = path
    multilayer_weights = np.array(img)/255
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
           pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1., frame1_init=''):
    printf('blend warp', blend)

    if isinstance(flo_path, str):
      flow21 = np.load(flo_path)
    else: flow21 = flo_path
    if weights_path is not None:
      if isinstance(weights_path, str):
        forward_weights = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)
      else: forward_weights = weights_path

    pad = int(max(flow21.shape)*pad_pct)
    flow21 = np.pad(flow21, pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')

    frame1pil = np.array(frame1.convert('RGB'))
    frame1pil = np.pad(frame1pil, pad_width=((pad,pad),(pad,pad),(0,0)),mode=padding_mode)
    if video_mode:
      warp_mul=1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0]-pad,pad:frame1_warped21.shape[1]-pad,:]

    frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    if forward_weights is not None:
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)

      forward_weights = forward_weights.clip(forward_clip,1.)
      if use_patchmatch_inpaiting>0 and warp_mode == 'use_image':
        if not is_colab: print('Patchmatch only working on colab/linux')
        else: print('PatchMatch disabled.')

      blended_w = frame2pil*(1-blend) + blend*(frame1_warped21*forward_weights+frame2pil*(1-forward_weights))
    else:
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
      blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)

    blended_w = Image.fromarray(blended_w.round().astype('uint8'))

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

  # in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
  # flo_folder = in_path+'_out_flo_fwd'

  # temp_flo = in_path+'_temp_flo'
  # flo_fwd_folder = in_path+'_out_flo_fwd'
  # flo_bck_folder = in_path+'_out_flo_bck'

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

def infer_flow_and_cc(frame_1, frame_2, flow_path, cc_path, pool, flow_save_img_preview=False, size=None):
            lazy_init_raft()
            out_flow21_fn = flow_path
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

            if flow_save_img_preview:
              pool.apply_async(save_preview, (flow21, out_flow21_fn+'.jpg') )
            if size is not None:
              flow21 = flow21[:size[1], :size[0],...]
            pool.apply_async(np.save, (out_flow21_fn, flow21))
            joint_mask = None
            if check_consistency:
              if use_jit_raft:
                _, flow12 = raft_model(frame_1, frame_2)
              else:
                flow12 = raft_model(frame_1, frame_2)[-1] #flow_fwd

              flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
              if flow_save_img_preview:
                pool.apply_async(save_preview, (flow12, out_flow21_fn+'_12'+'.jpg'))
              if use_legacy_cc:
                if size is not None:
                  flow12 = flow12[:size[1], :size[0],...]
                pool.apply_async(np.save, (out_flow21_fn+'_12', flow12))
                print('Legacy cc only available with non-lazywarp mode. Disable use_legacy_cc, or uncheck lazywarp, enable force_flow_generation, and run again.')
              joint_mask = make_cc_map(flow12, flow21_clamped, dilation=missed_consistency_dilation,
                                        edge_width=edge_consistency_width)
              if size is not None:
                joint_mask = joint_mask[:size[1], :size[0],...]
              joint_mask = PIL.Image.fromarray(joint_mask.astype('uint8'))
              joint_mask.save(cc_path)
            return flow21_clamped, joint_mask

def flow_batch(i, batch, pool, ds):
  with torch.cuda.amp.autocast():
          batch = batch[0]
          frame_1 = batch[0][None,...].cuda()
          frame_2 = batch[1][None,...].cuda()
          frame1 = ds.frames[i]
          frame1 = frame1.replace('\\','/')
          flow_path = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
          cc_path = f"{flo_fwd_folder}/{frame1.split('/')[-1]}-21_cc.jpg"
          save_img_preview = True if (flow_save_img_preview or i in range(0,len(ds),max(1, len(ds)//10))) else False
          infer_flow_and_cc(frame_1, frame_2, flow_path, cc_path, pool, save_img_preview)

from multiprocessing.pool import ThreadPool as Pool
import gc

##@markdown If you're having "process died" error on Windows, set num_flow_workers to 0


##@markdown Use lower quality model (half-precision).\
##@markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.

# \@markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
flow_save_img_preview = False  # \@param {type:'boolean'}

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
##@markdown Flow estimation quality (number of iterations, 12 - default. higher - better and slower)

#\@markdown Unreliable areas mask (missed consistency) width
#\@markdown Default = 1

#\@markdown Motion edge areas (edge consistency) width
#\@markdown Default = 11

flowframes_ds = None

def lazy_init_raft():
  global raft_model
  if raft_model is None:
    if use_jit_raft:
          if flow_lq:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
          else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()
    else:
          if raft_model is None or not compile_raft:
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
            from torchvision.models.optical_flow import raft_large, raft_small
            raft_weights = Raft_Large_Weights.C_T_SKHT_V1
            raft_device = "cuda" if torch.cuda.is_available() else "cpu"
            raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device)
            raft_model = raft_model.eval()
            if gpu != 'T4' and compile_raft: raft_model = torch.compile(raft_model)
            if flow_lq:
              raft_model = raft_model.half()
  return raft_model

def force_generate_flow():

    flowframes_ds = flowDataset(in_path, normalize=not use_jit_raft)

    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)<2:
      print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
    if len(frames)>=2:
      if __name__ == '__main__':

        dl = DataLoader(flowframes_ds, num_workers=num_flow_workers)
        try:
          sd_model.cpu()
          if cuda_empty_cache: torch.cuda.empty_cache()
          if gc_collect: gc.collect()
        except: pass
        raft_model = lazy_init_raft()
        raft_model.cuda().half()

        temp_flo = in_path+'_temp_flo'
        flo_fwd_folder = in_path+f'_out_flo_fwd/{side_x}_{side_y}/'
        for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
          f.unlink()

        os.makedirs(flo_fwd_folder, exist_ok=True)
        os.makedirs(temp_flo, exist_ok=True)
        cc_path = f'{root_dir}/flow_tools/check_consistency.py'
        with torch.no_grad():
          p = Pool(flow_threads)
          for i,batch in enumerate(tqdm(dl)):
              flow_batch(i, batch, p, flowframes_ds)
          p.close()
          p.join()

        del p, dl, flowframes_ds
        raft_model = None
        if cuda_empty_cache: torch.cuda.empty_cache()
        if gc_collect: gc.collect()
        if is_colab: locale.getpreferredencoding = getpreferredencoding
        if check_consistency and use_legacy_cc:
          fwd = f"{flo_fwd_folder}/*jpg.npy"
          bwd = f"{flo_fwd_folder}/*jpg_12.npy"
          if reverse_cc_order:
              print('Doing bwd->fwd cc check')
              subprocess.run([
                  'python', cc_path, '--flow_fwd', fwd, '--flow_bwd', bwd,
                  '--output', flo_fwd_folder + "/", '--image_output',
                  '--output_postfix', '-21_cc', '--blur', '0.',
                  '--save_separate_channels', '--skip_numpy_output'
              ])
          else:
              print('Doing fwd->bwd cc check')
              subprocess.run([
                  'python', cc_path, '--flow_fwd', bwd, '--flow_bwd', fwd,
                  '--output', flo_fwd_folder + "/", '--image_output',
                  '--output_postfix', '-21_cc', '--blur', '0.',
                  '--save_separate_channels', '--skip_numpy_output'
              ])

          # if reverse_cc_order:
          #   #old version, may be incorrect
          #   print('Doing bwd->fwd cc check')
          #   !python "{cc_path}" --flow_fwd "{fwd}" --flow_bwd "{bwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output
          # else:
          #   print('Doing fwd->bwd cc check')
          #   !python "{cc_path}" --flow_fwd "{bwd}" --flow_bwd "{fwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output

def make_flow_preview_grid():
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
  # display.display(fit(preview, 1024))
print('Flow is now generated in do the run cell.')

clip_type = 'ViT-H-14' # \@param ['ViT-L-14','ViT-B-32-quickgelu', 'ViT-H-14']
if clip_type == 'ViT-H-14' : clip_pretrain = 'laion2b_s32b_b79k'
if clip_type == 'ViT-L-14' : clip_pretrain = 'laion2b_s32b_b82k'
if clip_type == 'ViT-B-32-quickgelu' : clip_pretrain = 'laion400m_e32'

tile_size = 128 #\@param {'type':'number'}
stride = 96 #\@param {'type':'number'}
padding = [0.5,0.5] #\@param {'type':'raw'}

import types
#tiled vae from thttps://github.com/CompVis/latent-diffusion

def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
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
def encode_first_stage(self, x):
        ts = time.time()
        if hasattr(self, "split_input_params"):
          with torch.autocast('cuda'):
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

                if self.split_input_params["padding"] is not None:
                  padding = self.split_input_params["padding"]
                  stride = [int(ks[0]*padding[0]), int(ks[1]*padding[1])]
                else:
                  stride = self.split_input_params["stride"]  # eg. (64, 64)
                  stride = [o*(df) for o in stride]

                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape

                target_h = math.ceil(h/ks[0])*ks[0]
                target_w = math.ceil(w/ks[1])*ks[1]
                padh = target_h - h
                padw = target_w - w
                pad = (0, padw, 0, padh)
                if target_h != h or target_w != w:
                  print('Padding.')
                  x = torch.nn.functional.pad(x, pad, mode='reflect')
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")
                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                if no_half_vae:
                    self.disable_first_stage_autocast = True
                    self.first_stage_model.float()
                    z = z.float()
                    with torch.autocast('cuda', enabled=False):
                      output_list = [self.get_first_stage_encoding(self.first_stage_model.encode(z[:, :, :, :, i].float()), tiled_vae_call=True)
                                for i in range(z.shape[-1])]
                else:
                  output_list = [self.get_first_stage_encoding(self.first_stage_model.encode(z[:, :, :, :, i]), tiled_vae_call=True)
                                for i in range(z.shape[-1])]
                o = torch.stack(output_list, axis=-1)
                if 'sdxl' in model_version:
                  o = self.scale_factor * o
                o = o * weighting
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                decoded = fold(o)
                decoded = decoded / normalization
                printf('Tiled vae encoder took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')
                return decoded[...,:h//df, :w//df]
            else:
                print('Vae encoder took ', f'{time.time()-ts:.2}')
                return self.first_stage_model.encode(x)
        else:
            print('Vae encoder took ', f'{time.time()-ts:.2}')
            return self.first_stage_model.encode(x)

@torch.no_grad()
def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        ts = time.time()
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
          with torch.autocast('cuda'):

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
                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # print('z unfold, normalization, weighting',z.shape, normalization.shape, weighting.shape)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # print('z unfold view , normalization, weighting',z.shape)
                # 2. apply model loop over last dim

                if no_half_vae:
                  with torch.autocast('cuda', enabled=False):
                    self.disable_first_stage_autocast = True
                    self.first_stage_model.float()
                    z = z.float()
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i].float())
                                   for i in range(z.shape[-1])]
                else:
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
                printf('Tiled vae decoder took ', f'{time.time()-ts:4.2}', file='./logs/profiling.txt')
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

def get_first_stage_encoding(self, encoder_posterior, tiled_vae_call=False):
        if hasattr(self, "split_input_params") and not tiled_vae_call:
          #pass for tiled vae
          return encoder_posterior
        if self.is_sdxl:
          # print('skipping for sdxl')
          return encoder_posterior
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z


def get_weighting(self, h, w, Ly, Lx, device):
        weighting = delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

def meshgrid(h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr
def delta_border(h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

import subprocess
def prepare_mask(extract_background_mask, videoFramesFolder, mask_source, extract_nth_frame, start_frame, end_frame, force_mask_overwrite):
  if extract_background_mask:
    videoFramesAlpha = videoFramesFolder+'_alpha'
    createPath(videoFramesAlpha)
    if len(glob(videoFramesAlpha+'/*.*'))>=len(glob(videoFramesFolder+'/*.*')):
      if not force_mask_overwrite:
        print(f"Found {len(glob(videoFramesAlpha+'/*.*'))} mask frames for {len(glob(videoFramesFolder+'/*.*'))} existing video frames. Skipping background mask extraction. To force the mask extraction, check force_mask_overwrite. ")
        return videoFramesAlpha

    os.chdir(root_dir)
    pipi('av')
    pipi('pims')
    gitclone('https://github.com/Sxela/RobustVideoMattingCLI')
    print('Extracting background mask...')
    if mask_source == 'init_video':

      createPath(videoFramesAlpha)
      res = subprocess.run(['python',f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py",
                            '--input_path', f"{videoFramesFolder}",  '--output_alpha', f"{videoFramesAlpha}.mp4"], stdout=subprocess.PIPE).stdout.decode('utf-8')
      print(res)
      extractFrames(videoFramesAlpha+'.mp4', f"{videoFramesAlpha}", 1, 0, 999999999)
    else:

      maskVideoFrames = videoFramesFolder+'_mask'
      createPath(maskVideoFrames)
      extractFrames(mask_source, f"{maskVideoFrames}", extract_nth_frame, start_frame, end_frame)
      res = subprocess.run(['python',f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py",
                            '--input_path', f"{maskVideoFrames}",  '--output_alpha', f"{videoFramesAlpha}.mp4"], stdout=subprocess.PIPE).stdout.decode('utf-8')
      extractFrames(videoFramesAlpha+'.mp4', f"{videoFramesAlpha}", 1, 0, 999999999)
    print(f"Extracted {len(glob(videoFramesAlpha+'/*.*'))} mask frames")
  else:
    if mask_source in ['', None, 'none', 'None']: return ''
    if mask_source == 'init_video':
      videoFramesAlpha = videoFramesFolder
    else:
      videoFramesAlpha = videoFramesFolder+'_alpha'
      createPath(videoFramesAlpha)
      if len(glob(videoFramesAlpha+'/*.*'))>=len(glob(videoFramesFolder+'/*.*')):
        if not force_mask_overwrite:
          print(f"Found {len(glob(videoFramesAlpha+'/*.*'))} mask frames for {len(glob(videoFramesFolder+'/*.*'))} existing video frames. Skipping background mask extraction. To force the mask extraction, check force_mask_overwrite. ")
          return videoFramesAlpha
      extractFrames(mask_source, f"{videoFramesAlpha}", extract_nth_frame, start_frame, end_frame)
      #extract video
    print(f"Found {len(glob(videoFramesAlpha+'/*.*'))} mask frames")

  return videoFramesAlpha

#eof define cell
executed_cells[cell_name] = True

#@title ##Video Input Settings:
cell_name = 'video_input_settings'
# check_execution(cell_name)

#@markdown ###**Basic Settings:**

print("CAME HERERE CREATING FOLDER")
batch_name =  'stable_warpfusion_0.32.0' #'stable_warpfusion_0.17.0' #@param{type: 'string'}
steps =  50
##@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
# stop_early = 0  #@param{type: 'number'}
stop_early = 0
stop_early = min(steps-1,stop_early)


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


#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps


#Make folder for batch
print("MAKING FOLDER")

batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

print("BATCH FOLDER", batchFolder)

#@markdown  ###**Output Size  Settings**
#@markdown Specify desired output size here  [width,height] or use a single number to resize the frame keeping aspect ratio.\
#@markdown Don't forget to rerun all steps after changing the width height (including forcing optical flow generation)
width_height = 1280#@param{type: 'raw'}
#Get corrected sizes
#@markdown Make sure the resolution is divisible by that number. The Default 64 is the most stable.

force_multiple_of = "64" #@param [8,64]
force_multiple_of = int(force_multiple_of)
if isinstance(width_height, list):
  width_height = [int(o) for o in width_height]
  side_x = (width_height[0]//force_multiple_of)*force_multiple_of;
  side_y = (width_height[1]//force_multiple_of)*force_multiple_of;
  if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of {force_multiple_of}.')
  width_height = (side_x, side_y)
else:
  width_height = int(width_height)



animation_mode = 'Video Input'
import os, platform
if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
  print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")

#@markdown  ###**Video Input Settings**
#@markdown ---

video_source = 'video_init' #@param ['video_init', 'looped_init_image']

#@markdown Use video_init to process your video file.\
#@markdown  If you don't have a video file, you can looped_init_image to create a looping video from single init_image\
#@markdown Use this if you just want to test settings. This will create a small video (1 sec = 24 frames)\
#@markdown This way you will be able to iterate faster without the need to process flow maps for a long final video before even getting to testing prompts.
looped_video_duration_sec = 2 #@param {'type':'number'}

video_init_path = "video.mp4" #@param {type: 'string'}

def fit_size(size,maxsize=512):
    maxdim = max(size)
    ratio = maxsize/maxdim
    x,y = size
    size = (int(x*ratio)),(int(y*ratio))
    return size

if video_source=='looped_init_image':
  actual_size = Image.open(init_image).size
  if isinstance(width_height, int):
    width_height = fit_size(actual_size, width_height)

  force_multiple_of = int(force_multiple_of)
  side_x = (width_height[0]//force_multiple_of)*force_multiple_of;
  side_y = (width_height[1]//force_multiple_of)*force_multiple_of;
  if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of {force_multiple_of}.')
  width_height = (side_x, side_y)
  subprocess.run(['ffmpeg', '-loop', '1', '-i', init_image, '-c:v', 'libx264', '-t', str(looped_video_duration_sec), '-pix_fmt',
   'yuv420p', '-vf', f'scale={side_x}:{side_y}', f"{root_dir}/out.mp4", '-y'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print('Video saved to ', f"{root_dir}/out.mp4")
  video_init_path = f"{root_dir}/out.mp4"

extract_nth_frame =  1#@param {type: 'number'}
reverse = False #@param {type: 'boolean'}
no_vsync = True #@param {type: 'boolean'}
#@markdown *Specify frame range. end_frame=0 means fill the end of video*
start_frame = 0#@param {type: 'number'}
end_frame = 0#@param {type: 'number'}
end_frame_orig = end_frame
if end_frame<=0 or end_frame==None: end_frame = 99999999999999999999999999999
#@markdown ####**Separate guiding video** (optical flow source):
#@markdown Leave blank to use the first video.
flow_video_init_path = "" #@param {type: 'string'}
flow_extract_nth_frame =  1#@param {type: 'number'}
if flow_video_init_path == '':
  flow_video_init_path = None
#@markdown ####**Image Conditioning Video Source**:
#@markdown Used together with image-conditioned models, like controlnet, depth, or inpainting model.
#@markdown You can use your own video as depth mask or as inpaiting mask.
cond_video_path = "" #@param {type: 'string'}
cond_extract_nth_frame =  1#@param {type: 'number'}
if cond_video_path == '':
  cond_video_path = None

#@markdown Enable to store frames, flow maps, alpha maps on drive
store_frames_on_google_drive = False #@param {type: 'boolean'}
video_init_seed_continuity = False

def getFrameList(video_path):
  frames = []
  if os.path.isdir(video_path):
    frames = glob(os.path.join(video_path, '*.*'))
  else:
    frames = glob(video_path)
  assert len(frames)>0, f'No frames were found at {video_path}'
  return frames

def copyFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  frames = getFrameList(video_path)
  frames = frames[start_frame:end_frame:nth_frame]
  print(f"Copying Video Frames (1 every {nth_frame})...")
  for i in range(len(frames)):
    shutil.copy(frames[i], os.path.join(output_path, f'{i:06}.jpg'))

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  createPath(output_path)
  if not os.path.isfile(video_path):
    copyFrames(video_path, output_path, nth_frame, start_frame, end_frame)
    return
  print(f"Exporting Video Frames (1 every {nth_frame})...")
  try:
    for f in [o.replace('\\','/') for o in glob(output_path+'/*.jpg')]:
      pathlib.Path(f).unlink()
  except:
    print('error deleting frame ', f)
  vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
  if reverse: vf+=',reverse'
  if no_vsync: vsync='0'
  else: vsync = 'vfr'
  if os.path.exists(video_path):
    try:
        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}',
                        '-vsync', vsync, '-q:v', '2', '-loglevel', 'error', '-stats',
                        f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    except:
        subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}',
                        '-vsync', vsync, '-q:v', '2', '-loglevel', 'error', '-stats',
                        f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

  else:
    sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')

import cv2
def get_fps(video_init_path):
  if os.path.exists(video_init_path):
    if os.path.isfile(video_init_path):
        cap = cv2.VideoCapture(video_init_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
  return -1

if animation_mode == 'Video Input':
  if not os.path.isfile(video_init_path):
    detected_fps = 30.
    frames = getFrameList(video_init_path)
    postfix = f'{generate_file_hash(frames[0])[:10]}-{detected_fps:.6}_{start_frame}_{end_frame_orig}_{extract_nth_frame}'
  else:
    detected_fps = float(get_fps(video_init_path))
    postfix = f'{generate_file_hash(video_init_path)[:10]}-{detected_fps:.6}_{start_frame}_{end_frame_orig}_{extract_nth_frame}'
  print(f'Detected video fps of {detected_fps:.6}. With extract_nth_frame={extract_nth_frame} the suggested export fps would be {detected_fps/extract_nth_frame:.6}.')
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


actual_size = Image.open(sorted(glob(videoFramesFolder+'/*.*'))[0]).size
if isinstance(width_height, int):
  width_height = fit_size(actual_size, width_height)

force_multiple_of = int(force_multiple_of)
side_x = (width_height[0]//force_multiple_of)*force_multiple_of;
side_y = (width_height[1]//force_multiple_of)*force_multiple_of;
if side_x != width_height[0] or side_y != width_height[1]:
  print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of {force_multiple_of}.')
width_height = (side_x, side_y)


in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
temp_flo = in_path+'_temp_flo'
flo_fwd_folder = flo_folder = in_path+f'_out_flo_fwd/{side_x}_{side_y}/'
os.makedirs(flo_fwd_folder, exist_ok=True)
os.makedirs(temp_flo, exist_ok=True)

executed_cells[cell_name] = True

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

If you're having black frames with sdxl, turn off tiled vae, enable no_half_vae or use this vae - https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
"""

#@markdown specify path to your Stable Diffusion checkpoint (the "original" flavor)
#@title define SD + K functions, load model
cell_name = 'load_model'
# check_execution(cell_name)

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

try:
  del sd_model
except: pass
try:
  del model_wrap_cfg
  del model_wrap
except: pass


torch.cuda.empty_cache()
gc.collect()


model_urls = {
    "sd_v1_5":"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "dpt_hybrid-midas-501f0c75":"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
    "clip_vision_vit_h":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    "clip_vision_vit_bigg":"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
}

control_anno_urls = {
    "control_sd15_openpose":["https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
                             "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"],
    # "control_sd15_softedge":["https://github.com/Sxela/ControlNet-v1-1-nightly/releases/download/v0.1.0/table5_pidinet.pth"]

}

model_filenames = {
    "clip_vision_vit_h":"clip_vision_vit_h.safetensors",
    "clip_vision_vit_bigg":"clip_vision_vit_bigg.safetensors"
}

ipadapter_face_loras = {
    "ipadapter_sd15_faceid":["https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors"],
    "ipadapter_sd15_faceid_plus":["https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors"],
    "ipadapter_sd15_faceid_plus_v2":["https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"],
    "ipadapter_sdxl_faceid":["https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors"],
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
    "control_sd15_shuffle":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth",
    "control_sdxl_canny":"https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "control_sdxl_depth":"https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "control_sdxl_softedge":"https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-softedge-dexined/resolve/main/controlnet-sd-xl-1.0-softedge-dexined.safetensors",
    "control_sdxl_seg":"https://huggingface.co/SargeZT/sdxl-controlnet-seg/resolve/main/diffusion_pytorch_model.bin",
    "control_sdxl_openpose":"https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors",
    "control_sdxl_lora_128_depth":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors",
    "control_sdxl_lora_256_depth":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors",
    "control_sdxl_lora_128_canny":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors",
    "control_sdxl_lora_256_canny":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors",
    "control_sdxl_lora_128_softedge":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors",
    "control_sdxl_lora_256_softedge":"https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors",
    "control_sd15_tile":"https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
    "control_sd15_qr":"https://huggingface.co/DionTimmer/controlnet_qrcode/resolve/main/control_v1p_sd15_qrcode.safetensors",
    "control_sd21_qr":"https://huggingface.co/DionTimmer/controlnet_qrcode/resolve/main/control_v11p_sd21_qrcode.safetensors",
    "control_sd21_depth":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_zoedepth.safetensors",
    "control_sd21_scribble":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_scribble.safetensors",
    "control_sd21_openpose":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_openposev2.safetensors",
    "control_sd21_normalbae":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_normalbae.safetensors",
    "control_sd21_lineart":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_lineart.safetensors",
    "control_sd21_softedge":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_hed.safetensors",
    "control_sd21_canny":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_canny.safetensors",
    "control_sd21_seg":"https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_ade20k.safetensors",
    "control_sdxl_temporalnet_v1": "https://huggingface.co/CiaraRowles/controlnet-temporalnet-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    "control_sd15_inpaint_softedge":"https://huggingface.co/sxela/WarpControlnets/resolve/main/control_v01e_sd15_inpaint_softedge.pth",
    "control_sd15_temporal_depth":"https://huggingface.co/sxela/WarpControlnets/resolve/main/control_v01e_sd15_temporal_depth.pth",
    "control_sd15_monster_qr":"https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors",
    "control_sdxl_inpaint":"https://huggingface.co/sxela/out/resolve/main/checkpoint-15000/controlnet/diffusion_pytorch_model.safetensors",
    "ipadapter_sd15":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
    "ipadapter_sd15_light":"https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15_light.safetensors",
    "ipadapter_sd15_plus":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin",
    "ipadapter_sd15_plus_face":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin",
    "ipadapter_sd15_full_face":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.bin",
    "ipadapter_sd15_vit_G":"https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.bin",
    "ipadapter_sdxl":"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin",
    "ipadapter_sdxl_vit_h":"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.bin",
    "ipadapter_sdxl_plus_vit_h":"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
    "ipadapter_sdxl_plus_face_vit_h":"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
    "control_sd15_gif":"https://huggingface.co/crishhh/animatediff_controlnet/resolve/main/controlnet_checkpoint.ckpt?download=true",
    "ipadapter_sd15_faceid":"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin",
    "ipadapter_sd15_faceid_plus":"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin", #vit-h
    "ipadapter_sd15_faceid_plus_v2":"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin", #vit-h
    "ipadapter_sdxl_faceid":"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin",
    "control_sd15_depth_anything":"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_controlnet/diffusion_pytorch_model.safetensors"
}
control_model_filenames = {
    "control_sd15_gif":"control_v11p_sd15_gif.ckpt",
    "control_sdxl_canny":"diffusers-controlnet-canny-sdxl-1.0.fp16.safetensors",
    "control_sdxl_depth":"diffusers-controlnet-depth-sdxl-1.0.fp16.safetensors",
    "control_sdxl_softedge":"SargeZT-controlnet-sd-xl-1.0-softedge-dexined.safetensors",
    "control_sdxl_seg":"SargeZT-sdxl-controlnet-seg.bin",
    "control_sdxl_openpose":"thibaud-OpenPoseXL2.safetensors",
    "control_sdxl_lora_128_depth":"stability-control-lora-depth-rank128.safetensors",
    "control_sdxl_lora_256_depth":"stability-control-lora-depth-rank256.safetensors",
    "control_sdxl_lora_128_canny":"stability-control-lora-canny-rank128.safetensors",
    "control_sdxl_lora_256_canny":"stability-control-lora-canny-rank256.safetensors",
    "control_sdxl_lora_128_softedge":"stability-control-lora-sketch-rank128.safetensors",
    "control_sdxl_lora_256_softedge":"stability-control-lora-sketch-rank256.safetensors",
    "control_sdxl_temporalnet_v1":"CiaraRowles-temporalnet-sdxl-v1.safetensors", #old-style cn with 3 input channels
    "control_sdxl_inpaint":"control_v01e_sdxl_inpaint_diffusers_15k.safetensors",
    "control_sd15_depth_anything":"control_sd15_depth_anything.safetensors"
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
model_version = 'control_multi'#@param ['control_multi_animatediff', 'v1_animatediff', 'control_multi_v2','control_multi_v2_768','control_multi_sdxl','control_multi','sdxl_base','sdxl_refiner','v1','v1_inpainting','v1_instructpix2pix','v2_512', 'v2_768_v', 'control_multi_animatediff_sdxl']
if model_version in ['v1','v1_animatediff'] :
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
if 'sdxl' in model_version:
  os.chdir(f'{root_dir}/generative-models')
  import sgm
  from sgm.modules.diffusionmodules.discretizer import LegacyDDPMDiscretization
  os.chdir(f'{root_dir}')
else:
  try: del comfy
  except: pass
if model_version in  ['sdxl_base', 'control_multi_sdxl', 'control_multi_animatediff_sdxl']:
  config_path = f"{root_dir}/generative-models/configs/inference/sd_xl_base.yaml"
if model_version == 'sdxl_refiner':
  config_path = f"{root_dir}/generative-models/configs/inference/sd_xl_refiner.yaml"

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
    "control_sd15_face":None,
    "control_sdxl_temporalnet_v1":None
}

if model_version == 'v1_instructpix2pix':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1_instruct_pix2pix.yaml"
vae_ckpt = '' #@param {'type':'string'}
if vae_ckpt == '': vae_ckpt = None
load_to = 'cpu' #@param ['cpu','gpu']
if load_to == 'gpu': load_to = 'cuda'
quantize = True
#@markdown Enable no_half_vae if you are getting black frames.
no_half_vae = False #@param {'type':'boolean'}
import gc
init_dummy = True #@param {'type':'boolean'}
if 'sdxl' not in model_version:
   init_dummy = False
   print('disabling init dummy for non-sdxl models')

from accelerate.utils import named_module_tensors, set_module_tensor_to_device

def handle_size_mismatch(sd):
  context_dim = sd.get('model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight', None)
  # print(context_dim.shape[-1])
  suggested_model_version = ''
  base = ''
  if context_dim is None:
    return('Unknown model base type. Make sure you are not using LORA as your base checkpoint. Please check your checkpoint base model is the same that you have selected from the model_version dropdown.')

  else:
    context_dim = context_dim.shape[-1]
    if context_dim == 768:
      suggested_model_version = 'control_multi' #if 'control' in model_version else 'v1'
      base = 'v1.x'
    elif context_dim == 1024:
      suggested_model_version = 'control_multi_v2_512' #if 'control' in model_version else 'v2_512'
      base = 'v2.x'
    elif context_dim == 1280:
      suggested_model_version = 'sdxl_refiner'
      base = 'sdxl_refiner'
    elif context_dim == 2048:
      suggested_model_version = 'control_multi_sdxl' #if 'control' in model_version else 'sdxl_base'
      base = 'sdxl_base'
    else:
      return('Unknown model base type. Please check your checkpoint base model is the same that you have selected from the model_version dropdown.')

    return(f"""
Model version / checkpoint base type mismatch.
You have selected {model_version} model_version and provided a checkpoint with {base} base model version.
Double check your model checkpoint base model or try switching model_version to {suggested_model_version} and running this cell again.""")

#@markdown Custom motion module. For now only variations of animatediff v1.5-v2 are supported. Empty input will load the default v1.5-v2 model.
motion_module_path = 'animatediff\\mm_sd_v15_v2.ckpt' #@param {'type':'string'}

animatediff_model_paths = {
    'control_multi_animatediff':{
        'out_path':f'{root_dir}/animatediff/models/Motion_Module/mm_sd_v15_v2.ckpt',
        'model_url':'https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt'
    },
    'control_multi_animatediff_sdxl':{
        'out_path':f'{root_dir}/animatediff/models/Motion_Module/mm_sdxl_v10_beta.ckpt',
        'model_url':'https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt'
    }
}

if 'animatediff' in model_version:
  animatediff_out_path = animatediff_model_paths[model_version]['out_path']

  if motion_module_path in ['', None] or not os.path.exists(motion_module_path):
    if motion_module_path not in ['', None]:
      print(f'Motion module not found at {motion_module_path}, switching to default {animatediff_out_path.split("/")[-1]} file.')
    motion_module_path = animatediff_out_path

  if not os.path.exists(motion_module_path):
    animatediff_url = animatediff_model_paths[model_version]['model_url']
    print(f'Motion module not found {motion_module_path}, Downloading animatediff {animatediff_out_path.split("/")[-1]} file.')
    wget.download(animatediff_url, animatediff_out_path)
    print('Downloaded animatediff file.')

def move_tensors(module, device='cuda'):
  for name, _ in named_module_tensors(module):
    old_value = getattr(module, name)
    if device == torch.device('meta'):
      new_value = None
    else:
      new_value = torch.zeros_like(old_value, device=device)
    set_module_tensor_to_device(module, name, device, value=new_value)

def maybe_instantiate(ckpt, config):
      if ckpt.endswith('.pkl'):
        with open(ckpt, 'rb') as f:
              model = pickle.load(f).eval()
        return model #return loaded pickle

      dymmy_path = os.path.join(root_dir,f'{model_version}.dummypkl')
      if not os.path.exists(dymmy_path) and init_dummy:
        if model_version in ['sdxl_base', 'control_multi_sdxl']:
          #download dummmypkl
          dummypkl_out_path = dymmy_path
          dummypkl_url = 'https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/control_multi_sdxl.dummypkl'
          print('Downloading dummypkl file.')
          wget.download(dummypkl_url, dummypkl_out_path)
      #load dummy
      if (os.path.exists(dymmy_path) and init_dummy) or ckpt.endswith('.pkl'):
          try:
            print('Loading dummy pkl')
            #try load dummy pkl instead of initializing model
            with open(dymmy_path, 'rb') as f:
              model = pickle.load(f).eval()
            if model_version in ['sdxl_base', 'control_multi_sdxl']:

                model.conditioner.embedders[0].transformer.text_model.embeddings = model.conditioner.embedders[0].transformer.text_model.embeddings.to_empty(device='cuda').cuda()
                model.conditioner.embedders[0].transformer.text_model.encoder = model.conditioner.embedders[0].transformer.text_model.encoder.to_empty(device='cuda').cuda()
                model.conditioner.embedders[1].model.transformer = model.conditioner.embedders[1].model.transformer.to_empty(device='cuda').cuda()

            model.first_stage_model.encoder = model.first_stage_model.encoder.to_empty(device='cuda').cuda()
            model.first_stage_model.decoder = model.first_stage_model.decoder.to_empty(device='cuda').cuda()
            model.model.diffusion_model = model.model.diffusion_model.to_empty(device='cuda').cuda()

            # for key, value in model.named_parameters():
            #   if value.device == torch.device('meta'):
            #     print(key, 'meta')
            # print(next(model.parameters()))
            return model
          except:
            print(traceback.format_exc())
            model = None
            print('Found pkl file but failed loading. Probably codebase mismatch, try resaving.')
      else: model = None

      # instantiate and save dummy
      if model is None:
        # from IPython.utils import io
        # with io.capture_output(stderr=False) as captured:
        model = instantiate_from_config(config.model)

        if not os.path.exists(dymmy_path) and init_dummy:
            if use_torch_v2:
              model.half()
              if model_version in ['sdxl_base', 'control_multi_sdxl']:
                model.conditioner.embedders[0].transformer.text_model.encoder  = model.conditioner.embedders[0].transformer.text_model.encoder.to(torch.device('meta')).eval()
                model.conditioner.embedders[1].model.transformer = model.conditioner.embedders[1].model.transformer.to(torch.device('meta')).eval()
                model.conditioner.embedders[0].transformer.text_model.embeddings = model.conditioner.embedders[0].transformer.text_model.embeddings.to(torch.device('meta')).eval()
              model.first_stage_model.encoder = model.first_stage_model.encoder.to(torch.device('meta')).eval()
              model.first_stage_model.decoder = model.first_stage_model.decoder.to(torch.device('meta')).eval()
              model.model.diffusion_model = model.model.diffusion_model.to(torch.device('meta')).eval()
              torch.cuda.empty_cache()
              gc.collect()
            print('Saving dummy pkl')
            with open(dymmy_path, 'wb') as f:
              pickle.dump(model, f)
            model.half()
            #res
            if model_version in ['sdxl_base', 'control_multi_sdxl']:
                model.conditioner.embedders[0].transformer.text_model.embeddings = model.conditioner.embedders[0].transformer.text_model.embeddings.to_empty(device='cuda').cuda()
                model.conditioner.embedders[0].transformer.text_model.encoder = model.conditioner.embedders[0].transformer.text_model.encoder.to_empty(device='cuda').cuda()
                model.conditioner.embedders[1].model.transformer = model.conditioner.embedders[1].model.transformer.to_empty(device='cuda').cuda()
            model.first_stage_model.encoder = model.first_stage_model.encoder.to_empty(device='cuda').cuda()
            model.first_stage_model.decoder = model.first_stage_model.decoder.to_empty(device='cuda').cuda()
            model.model.diffusion_model = model.model.diffusion_model.to_empty(device='cuda').cuda()
          #save dummy model
      return model

def load_model_from_config(config, ckpt, vae_ckpt=None, controlnet=None, verbose=False):
    with torch.no_grad():
      # from IPython.utils import io

      model = maybe_instantiate(ckpt, config)

      if gpu != 'A100':
        model.half()


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
      if 'sdxl' in model_version:
        sd['denoiser.sigmas'] = torch.zeros(1000).to(load_to)
      try:
        m, u = model.load_state_dict(sd, strict=False)
      except Exception as e:
        if type(e) == RuntimeError:
          # print(e.args, e.with_traceback)
          if 'Error(s) in loading state_dict' in e.args[0]:
            print('Checkpoint and model_version size mismatch.')
            msg = handle_size_mismatch(sd)
            raise RuntimeError(msg)
      if gpu != 'A100':
        model.half()
      model.cuda()
      torch.cuda.empty_cache()
      gc.collect()
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
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach().half();# print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad(), torch.autocast('cuda'):
        # with torch.no_grad():
            # x = x.detach().requires_grad_()
            # print('x.shape, sigma', x.shape, sigma)
            denoised = model(x, sigma, **kwargs);# print(denoised.requires_grad)
        # with torch.enable_grad():
            # print(sigma**0.5, sigma, sigma**2)
            denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach().half();# print(cond_grad.requires_grad)
            # print('cond_grad.dtype')
            # print(cond_grad.dtype)
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
        if 'sdxl' in model_version:
          vector = cond['vector']
          uc_vector = uncond['vector']
          vector_in = torch.cat([uc_vector, vector])
          cond = cond['crossattn']
          uncond = uncond['crossattn']
        # else:
          # cond = prompt_parser.reconstruct_cond_batch(cond, 0)
          # uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)
        if cond.shape[1]>77:
            cond = cond[:,:77,:]
            # print('Prompt length > 77 detected. Shorten your prompt or split into multiple prompts.')
        uncond = uncond[:,:77,:]

        batch_size = sd_batch_size
        # print('batch size in cfgd ', batch_size, sd_batch_size)

        # print('cond_uncond_size',cond_uncond_size)
        x_in = torch.cat([x] * batch_size)
        sigma_in = torch.cat([sigma] * batch_size)
        # print('cond.shape, uncond.shape', cond.shape, uncond.shape)

        n_prompts = cond.shape[0]
        prompt_weights = preprocess_prompt_weights_old(prompt_weights, n_prompts).to(cond.device)

        if prompt_masks is not None:
          print('Using masked prompts')
          assert len(prompt_masks) == cond.shape[0], 'The number of masks doesn`t match the number of prompts-1.'
          prompt_masks = torch.tensor(prompt_masks).to(cond.device)
          # print('prompt_masks', prompt_masks.shape)
          # we use masks so that the 1st mask is full white, and others are applied on top of it
        elif blend_prompts_b4_diffusion:
          prompt_weights = einops.repeat(prompt_weights, 'b -> b c z', c=1, z=1).cuda()
          cond = (cond*prompt_weights).sum(dim=0, keepdim=True)
          prompt_weights = torch.ones(len(cond)).cuda()
          n_prompts = cond.shape[0]
        # print('cond.shape, prompt_weights, n_prompts', cond.shape, prompt_weights, n_prompts)

        cond_in = torch.cat([uncond, cond])
        res = None
        uc_mask_shape = torch.ones(cond_in.shape[0], device=cond_in.device)
        uc_mask_shape[0] = 0
        cond_uncond_size = cond.shape[0]+uncond.shape[0]

        n_batches = cond_uncond_size//batch_size if cond_uncond_size % batch_size == 0 else (cond_uncond_size//batch_size)+1
        # print('n_batches',n_batches)
        if image_cond is None:
          for i in range(n_batches):
            if model_version in ['sdxl_base', 'sdxl_refiner']:
              sd_model.conditioner.vector_in = vector_in[i*batch_size:(i+1)*batch_size]
            sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[i*batch_size:(i+1)*batch_size]
            pred = self.inner_model(x_in[i*batch_size:(i+1)*batch_size], sigma_in[i*batch_size:(i+1)*batch_size],
                                    cond=cond_in[i*batch_size:(i+1)*batch_size])
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
          if 'control_multi' not in model_version:
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


          if 'control_multi' in model_version and controlnet_multimodel_mode != 'external':
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
                  'controlnet_multimodel':controlnet_multimodel_inferred,
                  'loaded_controlnets':loaded_controlnets
                  }
              if 'sdxl' in model_version:
                y = vector_in[i*batch_size:(i+1)*batch_size]
                cond_dict['y'] = y
              x_in = torch.cat([x]*cond_dict["c_crossattn"][0].shape[0])
              sigma_in = torch.cat([sigma]*cond_dict["c_crossattn"][0].shape[0])
              # print(x_in.shape, cond_dict["c_crossattn"][0].shape)
              pred = self.inner_model(x_in,
                                     sigma_in, cond=cond_dict)
              # print(pred.shape)
              res = pred if res is None else torch.cat([res, pred])
              # print(res.shape)
              if sample_gc_collect: gc.collect()
              if sample_cuda_empty_cache: torch.cuda.empty_cache()
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
          if 'control_multi' in model_version and controlnet_multimodel_mode == 'external':

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
                    cond_dict = {"c_crossattn": [cond_in[i*batch_size:(i+1)*batch_size]],
                                 'c_concat':[img_in[i*batch_size:(i+1)*batch_size]]}
                    if 'sdxl' in model_version:
                      y = vector_in[i*batch_size:(i+1)*batch_size]
                      cond_dict['y'] = y
                    pred = self.inner_model(x_in[i*batch_size:(i+1)*batch_size], sigma_in[i*batch_size:(i+1)*batch_size], cond=cond_dict)
                    if gc_collect: gc.collect()
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

    def forward(self, z, sigma, cond, uncond, cond_scale, image_scale, image_cond, **kwargs):
        c = cond
        uc = uncond
        # c = prompt_parser.reconstruct_cond_batch(cond, 0)
        # uc = prompt_parser.reconstruct_cond_batch(uncond, 0)
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
controlnet_models_dir = "ControlNet" #@param {'type':'string'}
if not is_colab and (controlnet_models_dir.startswith('/content') or controlnet_models_dir=='' or controlnet_models_dir is None):
  controlnet_models_dir = f"{root_dir}/ControlNet/models"
  print('You have a controlnet path set up for google drive, but we are not on Colab. Defaulting controlnet model path to ', controlnet_models_dir)
os.makedirs(controlnet_models_dir, exist_ok=True)

#@markdown ---

import os
import sys

if 'control_multi' in model_version:
  os.chdir(f"{root_dir}/ControlNet/")
  print("subprocess pathhh")
  result = subprocess.run(['ls'], capture_output=True, text=True)
  print(result.stdout)
  if result.returncode != 0:
      print("Error:", result.stderr)
  import os

  print("OSSSSS PATH")
  print(os.getcwd())

  import sys
  # print(sys.path)

  sys.path.append(os.getcwd()) 
  from annotator.util import resize_image, HWC3
  from cldm.model import create_model, load_state_dict
  os.chdir('../')

if model_version in ['control_multi', 'control_multi_v2', 'control_multi_v2_768', 'control_multi_animatediff']:

  if model_version in ['control_multi', 'control_multi_animatediff']:
    config = OmegaConf.load(f"{root_dir}/ControlNet/models/cldm_v15.yaml")
  elif model_version in ['control_multi_v2', 'control_multi_v2_768']:
    config = OmegaConf.load(f"{root_dir}/ControlNet/models/cldm_v21.yaml")
  sd_model = load_model_from_config(config=config,
                                    ckpt=model_path, vae_ckpt=vae_ckpt, verbose=True)

  #legacy
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
      # from IPython.utils import io

      sd_model = load_model_from_config(config, model_path, vae_ckpt=vae_ckpt, verbose=True).cuda()

if 'animatediff' in model_version:
  os.chdir(f'{root_dir}/animatediff')
  from animatediff.models.motion_module import (get_motion_module,
                                                TemporalTransformerBlock, TemporalTransformer3DModel)
  os.chdir(f'{root_dir}/')
  get_motion_module = partial(get_motion_module,
                            motion_module_type='Vanilla',
                            motion_module_kwargs={
                                                    'temporal_position_encoding':True,
                                                    'temporal_position_encoding_max_len':32,
                                                    'num_transformer_block':1})
  context_length=16
  import types

  if 'sdxl' in model_version:
    from sgm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, TimestepBlock, SpatialTransformer

    def new_TimestepEmbedSequential_forward(
        self,
        x,
        emb,
        context=None,
        skip_time_mix=False,
        time_context=None,
        num_video_frames=None,
        time_context_cat=None,
        use_crossframe_attention_in_spatial_layers=False,
    ):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif type(layer).__name__ == "VanillaTemporalModule":
                x = layer(x, encoder_hidden_states=context)
            else:
                x = layer(x)
        return x
    TimestepEmbedSequential.forward = new_TimestepEmbedSequential_forward
  else:
    from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, TimestepBlock, SpatialTransformer
    def new_TimestepEmbedSequential_forward(self, x, emb, context=None):
            # print('type(layer).__name__', type(layer).__name__)
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                elif isinstance(layer, SpatialTransformer):
                    x = layer(x, context)
                elif type(layer).__name__ == "VanillaTemporalModule":
                  x = layer(x,encoder_hidden_states=context)
                  # x = layer(x, temb=None, encoder_hidden_states=context)
                else:
                    x = layer(x)
            return x

    TimestepEmbedSequential.forward = new_TimestepEmbedSequential_forward

  def new_TemporalTransformerBlock_forward(
          self,
          hidden_states,
          encoder_hidden_states=None,
          attention_mask=None,
          video_length=None,
      ):
          for attention_block, norm in zip(self.attention_blocks, self.norms):
              norm_hidden_states = norm(hidden_states)
              hidden_states = (
                  attention_block(
                      norm_hidden_states,
                      encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                      video_length=video_length,
                  )
                  + hidden_states
              )

          hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

          output = hidden_states
          return output

  TemporalTransformerBlock.forward = new_TemporalTransformerBlock_forward

  def new_TemporalTransformer3DModel_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
          batch, channel, height, weight = hidden_states.shape
          residual = hidden_states

          hidden_states = self.norm(hidden_states)
          inner_dim = hidden_states.shape[1]
          hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
          hidden_states = self.proj_in(hidden_states)

          # Transformer Blocks
          for block in self.transformer_blocks:
              hidden_states = block(
                  hidden_states,
                  encoder_hidden_states=encoder_hidden_states,
                  video_length=context_length,
              )

          # output
          hidden_states = self.proj_out(hidden_states)
          hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

          output = hidden_states + residual

          return output

  TemporalTransformer3DModel.forward = new_TemporalTransformer3DModel_forward
  print('Loading motion module from ', motion_module_path)
  if motion_module_path.endswith('.safetensors'):
        sd = {}
        with safe_open(motion_module_path, framework="pt") as f:
          for key in f.keys():
            sd[key] = f.get_tensor(key)
  else:  sd = torch.load(motion_module_path)

  from torch import Tensor

  sys.argv=['']
  sys.path.append(f'{root_dir}/comfyui-animatediff/')
  sys.path.append(f'{root_dir}/ComfyUI')

  from animatediff.motion_module import MotionWrapper
  from animatediff.motion_module import MotionModule, BlockType
  # from animatediff.sampler import eject_motion_module_from_unet

  # Merge from https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
  def get_encoding_max_len(mm_state_dict: dict[str, Tensor]) -> int:
      # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
      for key in mm_state_dict.keys():
          if key.endswith("pos_encoder.pe"):
              return mm_state_dict[key].size(1)  # get middle dim
      raise ValueError(f"No pos_encoder.pe found in mm_state_dict")

  def has_mid_block(mm_state_dict: dict[str, Tensor]):
      # check if keys contain mid_block
      for key in mm_state_dict.keys():
          if key.startswith("mid_block."):
              return True
      return False

  #taken from https://github.com/ArtVentureX/comfyui-animatediff
  class MotionWrapper(nn.Module):
    def __init__(self, mm_type: str, encoding_max_len: int = 24, is_v2=False, is_sdxl=False, is_v3=False):
        super().__init__()
        self.mm_type = mm_type
        self.is_v2 = is_v2
        self.is_sdxl = is_sdxl
        self.is_v3 = is_v3
        self.hack_gn = not (is_sdxl or is_v3)
        self.is_hotshot = False

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = encoding_max_len

        channels = [320, 640, 1280, 1280] if not is_sdxl else [320, 640, 1280]
        for c in channels:
            self.down_blocks.append(MotionModule(c, BlockType.DOWN, encoding_max_len=encoding_max_len))
            self.up_blocks.insert(0,MotionModule(c, BlockType.UP, encoding_max_len=encoding_max_len))
        if is_v2:
            self.mid_block = MotionModule(1280, BlockType.MID, encoding_max_len=encoding_max_len)

    @classmethod
    def from_state_dict(cls, mm_state_dict: dict[str, Tensor], mm_type: str, is_sdxl=False, is_v3=False):
        encoding_max_len = get_encoding_max_len(mm_state_dict)
        is_v2 = has_mid_block(mm_state_dict)

        mm = cls(mm_type, encoding_max_len=encoding_max_len, is_v2=is_v2, is_sdxl=is_sdxl, is_v3=is_v3)
        mm.load_state_dict(mm_state_dict, strict=False)
        return mm

    def set_video_length(self, video_length: int):
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)

  mm = MotionWrapper.from_state_dict(sd, mm_type='mm_sd_v15_v2.ckpt', is_sdxl= 'sdxl' in model_version, is_v3='v3' in motion_module_path).half().cuda()

  del sd
  from einops import rearrange

  def inject_motion_module_to_unet(diffusion_model, mm):
    if diffusion_model.mm_injected:
      print('mm already injected. exiting')
      return

    #insert motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ not in ["Downsample","Conv2d"]:
            # print('down', i,a,b)
            diffusion_model.input_blocks[i].append(mm.down_blocks[a].motion_modules[b-1])

        if type(diffusion_model.output_blocks[i][-1]).__name__ == "Upsample":
            # print('up', i,a,b)
            diffusion_model.output_blocks[i].insert(-1, mm.up_blocks[a].motion_modules[b])
        else:
            # print('up', i,a,b)
            diffusion_model.output_blocks[i].append(mm.up_blocks[a].motion_modules[b])
    if mm.is_v2:
      # pass
      diffusion_model.middle_block.insert(-1, mm.mid_block.motion_modules[0])
    elif mm.hack_gn:

            if mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            diffusion_model.gn32_original_forward = GroupNorm32.forward
            gn32_original_forward = diffusion_model.gn32_original_forward

            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
                x = gn32_original_forward(self, x)
                x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
                return x

            GroupNorm32.forward = groupnorm32_mm_forward

    diffusion_model.mm_injected = True

  def eject_motion_module_from_unet(diffusion_model, mm):
    if not diffusion_model.mm_injected:
      print('mm not injected. exiting')
      return
    #remove motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.input_blocks[i].pop(-1)

        if type(diffusion_model.output_blocks[i][-2]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-2)
        elif type(diffusion_model.output_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-1)
    if mm.is_v2:
      if type(diffusion_model.middle_block[-2]).__name__ == 'VanillaTemporalModule':
        # pass
        diffusion_model.middle_block.pop(-2)

    elif mm.hack_gn:
            if mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            GroupNorm32.forward = diffusion_model.gn32_original_forward
            diffusion_model.gn32_original_forward = None

    diffusion_model.mm_injected = False

  sd_model.model.diffusion_model.mm_injected = False
  inject_motion_module_to_unet(sd_model.model.diffusion_model, mm)
  #todo: fix this for sdxl
  if 'sdxl' not in model_version:
    sd_model.register_schedule(
            given_betas=None,
            beta_schedule="sqrt_linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.012,
            cosine_s=8e-3,
        )
  else:

      # https://github.com/continue-revolution/sd-webui-animatediff/blob/e9db9f287e73eeaee1d890c35a8f6f63b303159a/scripts/animatediff_mm.py#L9
      beta_start = 0.00085
      beta_end = 0.020
      betas = torch.linspace(beta_start**0.5, beta_end**0.5, 1000, dtype=torch.float32, device='cuda') ** 2
      alphas = 1.0 - betas
      alphas_cumprod = torch.cumprod(alphas, dim=0)
      sd_model.ad_alphas_cumprod = alphas_cumprod

  def preprocess_prompt_weights(prompt_weights, n_prompts):
        if prompt_weights is None:
          prompt_weights = [1.]*n_prompts
        if prompt_weights is not None:
          prompt_weights = torch.tensor(prompt_weights)
          assert prompt_weights.shape[1] >= n_prompts, 'The number of prompts is more than prompt weigths.'
          prompt_weights = prompt_weights[...,:n_prompts]
          if normalize_prompt_weights:
            prompt_weights = prompt_weights/prompt_weights.sum(dim=1, keepdim=True)
        return prompt_weights

  def get_cond_dict(cond, ids, i, image_cond=None, is_uc=False, y=None):
    if image_cond is not None:
              batch_img_in = {}
              for key in image_cond.keys():
                batch_img_in[key] = image_cond[key][ids]
              if is_uc:
                for key in image_cond.keys():
                  if img_zero_uncond or key == 'control_sd15_shuffle':
                    batch_img_in[key]*=0.
              cond_dict = {
                    "c_crossattn":  [cond[ids,i,...]],
                    'c_concat':  batch_img_in,
                    'controlnet_multimodel':controlnet_multimodel_inferred,
                    'loaded_controlnets':loaded_controlnets
              }
              # print('cond[ids,i,...].shape', cond[ids,i,...].shape)
              del batch_img_in
    else:
      if 'control' in model_version:
        cond_dict = {
                    "c_crossattn":  [cond[ids,i,...]],
                    'c_concat':  None,
                    'controlnet_multimodel':controlnet_multimodel_inferred,
                    'loaded_controlnets':loaded_controlnets
              }

      else:
              cond_dict = cond[ids,i,...]
    if y is not None:
                cond_dict['y'] = y[ids,i,...]
    # print('cond[ids,i,...]', cond[ids,i,...].max(), cond[ids,i,...].min())
    return cond_dict


  #inspired by sliding context feature from https://github.com/ArtVentureX/comfyui-animatediff
  class CFGDenoiser_adiff(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, cond, uncond, cond_scale, image_cond=None, prompt_weights=None,
                prompt_uweights=None, prompt_masks=None, **kwargs):
        global adiff_pbar, ctx_ids
        if 'sdxl' in model_version:
          vector = cond['vector']
          uc_vector = uncond['vector']
          cond = cond['crossattn']
          uncond = uncond['crossattn']
        c = cond.repeat(len(x),1,1,1).cuda() if len(cond) == 1 else cond.cuda()
        uc = uncond.repeat(len(x),1,1,1).cuda()  if len(uncond) == 1 else uncond.cuda()

        if 'sdxl' in model_version:
          vector = vector.repeat(len(x),1,1,1).cuda() if len(vector) == 1 else vector.cuda()
          uc_vector = uc_vector.repeat(len(x),1,1,1).cuda() if len(uc_vector) == 1 else uc_vector.cuda()
        else:
          vector = uc_vector = None

        if c.shape[2]>77:
          c = c[:,:,:77,:]
          # print('Prompt length > 77 detected. Shorten your prompt or split into multiple prompts.')
        uc = uc[:,:,:77,:]
        uc_mask_shape = torch.zeros(uc.shape[0], device=uc.device) #zero - uncond
        c_mask_shape  = torch.ones(c.shape[0], device=c.device) #ones - cond
        n_prompts = c.shape[1]
        prompt_weights = preprocess_prompt_weights(prompt_weights, n_prompts).to(c.device)
        n_uprompts = uc.shape[1]
        prompt_uweights = preprocess_prompt_weights(prompt_uweights, n_uprompts).to(uc.device)
        # print('prompt weights shape', prompt_weights.shape, prompt_weights)
        # print('prompt uweights shape', prompt_uweights.shape, prompt_uweights)
        if blend_prompts_b4_diffusion:
          # print('c shape', c.shape)
          # print('uc shape', uc.shape)
          # print('using simple weights')
          prompt_weights = einops.repeat(prompt_weights, 'l b -> l b c z', c=1, z=1).cuda()
          # print('prompt weights shape', prompt_weights)
          prompt_uweights = einops.repeat(prompt_uweights, 'l b ->  l b c z', c=1, z=1).cuda()
          # print('prompt uweights shape', prompt_weights)
          c = ((c*prompt_weights).sum(dim=1, keepdim=True))#/(prompt_weights*prompt_weights).sum().sqrt()
          uc = ((uc*prompt_uweights).sum(dim=1, keepdim=True))#/(prompt_uweights*prompt_uweights).sum().sqrt()
          prompt_weights = torch.ones((len(c),1)).cuda()
          prompt_uweights = torch.ones((len(uc),1)).cuda()
          n_prompts = c.shape[1]
          n_uprompts = uc.shape[1]
          # print('c shape', c.shape)
          # print('uc shape', uc.shape)

        if prompt_masks is not None:
          prompt_masks = torch.stack(prompt_masks)
          prompt_masks = torch.nn.functional.interpolate(prompt_masks, x.shape[2:])
          print('Using masked prompts')
          assert len(prompt_masks) == c.shape[1], 'The number of masks doesn`t match the number of prompts-1.'
          prompt_masks = torch.tensor(prompt_masks).to(c.device)

        cond = x.clone()
        uncond = x.clone()
        adiff_pbar.reset()

        cond_final = torch.zeros_like(x)
        uncond_final = torch.zeros_like(x)
        out_count_final = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)

        for ids in ctx_ids.pop(0):
            with torch.autocast('cuda', dtype=torch.float16), torch.inference_mode():
              """process cond"""
              out_cond = x[ids].clone()
              out_cond = einops.repeat(out_cond, pattern='b c h w -> l b c h w', l=n_prompts).clone()

              for i in range(n_prompts):
                # print('i', i)
                cond_dict = get_cond_dict(c, ids, i, image_cond, is_uc=False, y=vector)
                sd_model.model.diffusion_model.uc_mask_shape = c_mask_shape[ids]
                out_cond[i]   = self.inner_model(  cond[ids], sigma[ids], cond= cond_dict)

              if prompt_masks is None:
                # out_cond = out_cond[1]
                # print('(out_cond.shape, prompt_weights.transpose(1,0)[:, None, None, None].shape',out_cond.shape, prompt_weights.transpose(1,0)[:, None, None, None].shape)
                out_cond = (out_cond * (prompt_weights[ids].transpose(1,0)[..., None, None, None])).sum(dim=0, keepdim=False)
              else:
                out_cond_masked = out_cond[0]
                for i in range(len(out_cond)):
                  if i == 0: continue
                  out_cond_masked = (out_cond[i]*prompt_masks[i] + out_cond_masked*(1-prompt_masks[i]))
                out_cond = out_cond_masked
                del out_cond_masked

              # print('out_cond 0 ', str(out_cond[0].__repr__())[100:200])
              # print('out_cond 1 ', str(out_cond[1].__repr__())[100:200])

              cond_final  [ids] += out_cond
              del out_cond

              """process uncond"""
              out_uncond = x[ids].clone()
              out_uncond = einops.repeat(out_uncond, pattern='b c h w -> l b c h w', l=n_uprompts).clone()
              for i in range(n_uprompts):
                cond_dict = get_cond_dict(uc, ids, i, image_cond, is_uc=True, y=uc_vector)
                sd_model.model.diffusion_model.uc_mask_shape = uc_mask_shape[ids]
                out_uncond[i] = self.inner_model(uncond[ids], sigma[ids], cond= cond_dict)
              out_uncond = (out_uncond * (prompt_uweights[ids].transpose(1,0)[..., None, None, None])).sum(dim=0, keepdim=False)

              uncond_final[ids] += out_uncond
              del out_uncond

              out_count_final[ids]+=1

            adiff_pbar.update(1)
            adiff_pbar_total.update(1)

        cond_final /= out_count_final
        uncond_final /= out_count_final
        assert not (cond_final.isnan().any())
        assert not (uncond_final.isnan().any())
        res = uncond_final + (cond_final - uncond_final) * cond_scale
        return res

def preprocess_prompt_weights_old(prompt_weights, n_prompts):
        if isinstance(prompt_weights, list):
          ndim = 1
        else:
          if len(prompt_weights.shape) == 1:
            ndim = 0
          else:
            ndim = 1

        if prompt_weights is None:
          prompt_weights = [1.]*n_prompts
        if prompt_weights is not None:
          prompt_weights = torch.tensor(prompt_weights)
          assert prompt_weights.shape[ndim] >= n_prompts, 'The number of prompts is more than prompt weigths.'
          prompt_weights = prompt_weights[...,:n_prompts]
          if normalize_prompt_weights:
            prompt_weights = prompt_weights/prompt_weights.sum(dim=ndim, keepdim=True)
        return prompt_weights


sys.path.append('./stablediffusion/')
from modules import prompt_parser, sd_hijack
if 'sdxl' in model_version:
  discretizer = LegacyDDPMDiscretization()
  # sd_model.alphas_cumprod = torch.from_numpy(discretizer.alphas_cumprod)
  if 'animatediff' in model_version:
    sd_model.register_buffer('alphas_cumprod', sd_model.ad_alphas_cumprod)
  else:
    sd_model.register_buffer('alphas_cumprod', torch.from_numpy(discretizer.alphas_cumprod))
  sd_model.model.conditioning_key = 'c_crossattn'
  sd_model.cond_stage_model = sd_model.conditioner
  sd_model.parameterization = 'eps'

  def apply_model_vector(x_noisy, t, cond, self=sd_model, **kwargs):
    cond = {
        'crossattn': cond,
        'vector':self.conditioner.vector_in
        }
    return self.model.forward(x_noisy, t, cond, **kwargs)

  def get_first_stage_encoding_sdxl(z, self=sd_model):
        return z

  def get_unconditional_conditioning_sdxl(batch_c, self=sd_model.conditioner):

        def get_unique_embedder_keys_from_conditioner(conditioner):
          return list(set([x.input_key for x in conditioner.embedders]))

        W, H = width_height

        init_dict = {
                "orig_width": W,
                "orig_height": H,
                "target_width": W,
                "target_height": H,
            }

        prompt = batch_c
        negative_prompt = prompt
        num_samples = len(prompt)
        force_uc_zero_embeddings = []

        value_dict = init_dict
        value_dict["prompt"] = prompt
        value_dict["negative_prompt"] = ['']

        value_dict["crop_coords_top"] = 0
        value_dict["crop_coords_left"] = 0

        value_dict["aesthetic_score"] = 6.0
        value_dict["negative_aesthetic_score"] = 2.5

        batch_c, _ = get_batch(
                            get_unique_embedder_keys_from_conditioner(sd_model.conditioner),
                            value_dict,
                            [num_samples],
                        )

        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)

        [print(c[key].shape) for key in c.keys()]
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate

        return c

  sd_model.get_learned_conditioning = get_unconditional_conditioning_sdxl
  sd_model.apply_model = apply_model_vector
  sd_model.disable_first_stage_autocast = no_half_vae
  sd_model.get_first_stage_encoding = get_first_stage_encoding_sdxl

if sd_model.parameterization == "v" or model_version == 'control_multi_v2_768':
  model_wrap = K.external.CompVisVDenoiser(sd_model, quantize=quantize )
else:
  model_wrap = K.external.CompVisDenoiser(sd_model, quantize=quantize)
sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

if 'animatediff' in model_version:
  model_wrap_cfg =  CFGDenoiser_adiff(model_wrap)
else:
  model_wrap_cfg =  CFGDenoiser(model_wrap)
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

from torch import fft
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

b1= 1.2
b2= 1.4
s1= 0.9
s2= 0.2

def apply_freeu(h, _hs):
  if h.shape[1] == 1280:
    h[:,:640] = h[:,:640] * b1
    _hs = Fourier_filter(_hs.float(), threshold=1, scale=s1)
  if h.shape[1] == 640:
    h[:,:320] = h[:,:320] * b2
    _hs = Fourier_filter(_hs.float(), threshold=1, scale=s2)
  return h, _hs


def cldm_forward(x, timesteps=None, context=None, control=None, only_mid_control=False, self = sd_model.model.diffusion_model,**kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
          # print('control len', len(control), control[0].shape)
          h += control.pop()

        for i, module in enumerate(self.output_blocks):

            _hs = hs.pop()
            if do_freeunet and not apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

            if not only_mid_control and control is not None:
              _control = control.pop()
              _hs += _control

            if do_freeunet and apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

            h = cat8([h, _hs], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

def sdxl_cn_forward(x, timesteps=None, context=None, y=None, control=None, self=sd_model.model.diffusion_model,
                    only_mid_control=False,  **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None: h += control.pop()

        for module in self.output_blocks:
          _hs = hs.pop()
          if do_freeunet and not apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

          if not only_mid_control and control is not None:
              _control = control.pop()
              _hs += _control

          if do_freeunet and apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

          h = cat8([h, _hs], dim=1)
          h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)

if model_version in ['control_multi_sdxl', 'control_multi_animatediff_sdxl']:
  sd_model.model.diffusion_model.forward = sdxl_cn_forward

try:
  if 'sdxl' not in model_version:
    sd_model.model.diffusion_model.forward = cldm_forward

except Exception as e:
  print(e)
  # pass

if 'sdxl' in model_version:
  @torch.enable_grad()
  def differentiable_decode_first_stage(z, self=sd_model):
          z = 1.0 / self.scale_factor * z
          with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
              out = self.first_stage_model.decode(z)
          return out
  sd_model.differentiable_decode_first_stage = differentiable_decode_first_stage

#from colab
def apply_model_sdxl_cn(x_noisy, t, cond, self=sd_model, *args, **kwargs):
        if 'sdxl' in model_version:
          y = cond['y']
        else: y = None
        # print('apply model', type(cond), cond.keys(), cond['c_concat'].keys())
        t_ratio = 1-t[0]/self.num_timesteps;
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # y = torch.cat(cond['y'], 1)

        #if dict - we've got a multicontrolnet
        if cond['c_concat'] is None:
            control = None
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control, y=y)
            return eps

        if isinstance(cond['c_concat'], dict):
          # print('apply model got into if isinstance(cond:')
          try:
            uc_mask_shape = diffusion_model.uc_mask_shape[:, None, None, None]

          except:
            uc_mask_shape = torch.ones(x_noisy.shape[0], device=x_noisy.device)
          controlnet_multimodel = cond['controlnet_multimodel']
          loaded_controlnets = cond['loaded_controlnets']
          control_wsum = None
          #loop throught all controlnets to get controls
          active_models = {}
          for key in controlnet_multimodel.keys():
            settings = controlnet_multimodel[key]
            if settings['weight']!=0 and t_ratio>=settings['start'] and t_ratio<=settings['end']:
              active_models[key] = controlnet_multimodel[key]
          weights = np.array([active_models[m]["weight"] for m in active_models.keys()])
          if self.normalize_weights: weights = weights/weights.sum()

          if self.low_vram:
            diffusion_model.cpu()
            for key in active_models.keys():
               loaded_controlnets[key].cpu()
            if cuda_empty_cache:
              torch.cuda.empty_cache();
            if gc_collect: gc.collect()
          for i,key in enumerate(active_models.keys()):
            if self.debug:
                print('controlnet_multimodel keys ', controlnet_multimodel[key].keys())
                print('Using layer weights ', controlnet_multimodel[key]['layer_weights'], key)
            try:
                cond_hint = torch.cat([cond['c_concat'][key]], 1)
                if 'zero_uncond' in controlnet_multimodel[key].keys():
                    if controlnet_multimodel[key]['zero_uncond']:
                        if self.debug: print(f'Using zero uncond {list(uc_mask_shape.detach().cpu().numpy())} for {key}')
                        cond_hint*=uc_mask_shape # try zeroing the prediction, should mimic zero uncond, need to research more
                if self.low_vram:
                  loaded_controlnets[key].half().to(device=cond_hint.device)
                with torch.autocast('cuda'), torch.no_grad(), torch.inference_mode():
                  control = loaded_controlnets[key].cuda()(x=x_noisy, hint=cond_hint,
                                                               timesteps=t, context=cond_txt, y=y)

                if 'layer_weights' in controlnet_multimodel[key].keys():
                    control_scales = controlnet_multimodel[key]['layer_weights']
                    if self.debug: print('Using layer weights ', control_scales, key)
                    control_scales = control_scales[:len(control)]
                    control = [c * scale for c, scale in zip(control, control_scales)]
                if key == 'control_sd15_shuffle':
                    #apply avg pooling for shuffle control
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                if control_wsum is None: control_wsum = [weights[i]*o for o in control]
                else: control_wsum = [weights[i]*c+cs for c,cs in zip(control,control_wsum)]
                if self.low_vram:
                  loaded_controlnets[key].cpu()
                  if cuda_empty_cache:
                    torch.cuda.empty_cache();
                  if gc_collect: gc.collect()
            except Exception as e:
              # assert type(e) != torch.cuda.OutOfMemoryError, 'Got CUDA out of memory during ControlNet proccessing.'
              print(e)
          control = control_wsum
        else:
            cond_hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=cond_hint,
                                                               timesteps=t, context=cond_txt, y=y)

        if control is not None:
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
        if self.low_vram:
          for key in active_models.keys():
               loaded_controlnets[key].cpu()
          if cuda_empty_cache: torch.cuda.empty_cache();
          if gc_collect: gc.collect()
          diffusion_model.half().cuda()
        # print('got to eps', control)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, y=y, only_mid_control=self.only_mid_control)

        return eps
if model_version in ['control_multi', 'control_multi_animatediff']:
  #try using it with v1.5 cn as well
  sd_model.apply_model = apply_model_sdxl_cn
  #29.3
  sd_model.apply_model = apply_model_sdxl_cn
  os.chdir(f'{root_dir}/ComfyUI')
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath('./')), "comfy"))
  sys.argv=['']
  import os, sys
  print("subprocess pathhh")
  result = subprocess.run(['ls'], capture_output=True, text=True)
  print(result.stdout)
  if result.returncode != 0:
      print("Error:", result.stderr)
  import os
  print("OSSSSS PATH")
  print(os.getcwd())
  import sys
  sys.path.append(os.getcwd()) 
  sys.path.append('./comfy')
  from comfy.sd import load_controlnet
  os.chdir(f'{root_dir}')
  #29.3
if model_version in ['control_multi_sdxl','control_multi_animatediff_sdxl']:
  sd_model.apply_model = apply_model_sdxl_cn
  os.chdir(f'{root_dir}/ComfyUI')
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath('./')), "comfy"))
  sys.argv=['']
  import os, sys
  print("subprocess pathhh")
  result = subprocess.run(['ls'], capture_output=True, text=True)
  print(result.stdout)
  if result.returncode != 0:
      print("Error:", result.stderr)
  import os
  print("OSSSSS PATH")
  print(os.getcwd())
  import sys
  sys.path.append(os.getcwd()) 
  sys.path.append('./comfy')
  from comfy.sd import load_controlnet
  os.chdir(f'{root_dir}')

  sd_model.num_timesteps = 1000
  sd_model.debug = False
  sd_model.global_average_pooling = False
  sd_model.only_mid_control = False

  import comfy
  sd_model.model.model_config = nn.Module() #dummy module to assign config to it
  sd_model.model.model_config.unet_config = OmegaConf.load(config_path).model.params.network_config.params
  sd_model.model.model_config.unet_config = dict(sd_model.model.model_config.unet_config)
  sd_model.model.model_config.unet_config['out_channels'] = ''
  sd_model.model.model_config.unet_config.pop('spatial_transformer_attn_type')
  sd_model.model.model_config.unet_config['image_size'] = 32
  def get_dtype(self=sd_model.model):
        return next(self.parameters()).dtype
  sd_model.model.get_dtype = get_dtype

#from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/IPAdapterPlus.py
def load_ipadapter_model(ckpt_path):
        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            # sort keys
            model = {"image_proj": st_model["image_proj"], "ip_adapter": {}}
            sorted_keys = sorted(st_model["ip_adapter"].keys(), key=lambda x: int(x.split(".")[0]))
            for key in sorted_keys:
                model["ip_adapter"][key] = st_model["ip_adapter"][key]
            st_model = None

        if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
            raise Exception("invalid IPAdapter model {}".format(ckpt_path))

        return model

def clip_vision_encode(self, image):
        # comfy.model_management.load_model_gpu(self.patcher)
        pixel_values = clip_preprocess(image.to(self.model.device))

        if self.model.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(comfy.model_management.get_autocast_device(self.model.device), torch.float32):
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        for k in outputs:
            t = outputs[k]
            if t is not None:
                if k == 'hidden_states':
                    outputs["penultimate_hidden_states"] = t[-2].cpu()
                    outputs["hidden_states"] = t
                else:
                    outputs[k] = t.cpu()

        return outputs

def clip_preprocess(image, size=224):
    mean = torch.tensor([ 0.48145466,0.4578275,0.40821073], device=image.device, dtype=image.dtype)
    std = torch.tensor([0.26862954,0.26130258,0.27577711], device=image.device, dtype=image.dtype)
    scale = (size / min(image.shape[1], image.shape[2]))
    image = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(round(scale * image.shape[1]), round(scale * image.shape[2])), mode="bicubic", antialias=True)
    h = (image.shape[2] - size)//2
    w = (image.shape[3] - size)//2
    image = image[:,:,h:h+size,w:w+size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3,1,1])) / std.view([3,1,1])

os.chdir(f'{root_dir}/ComfyUI')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath('./')), "comfy"))
sys.argv=['']
import os, sys
sys.path.append('./comfy')
from comfy.k_diffusion.sampling import sample_lcm
os.chdir(f'{root_dir}')

from comfy.clip_vision import load as load_clip_vision
sys.path.append(f'{root_dir}/ComfyUI')
import comfy
import comfy.clip_vision
comfy.clip_vision.clip_preprocess = clip_preprocess

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

def control_forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
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
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
          # print('control len', len(control), control[0].shape)
          h += control.pop()

        for i, module in enumerate(self.output_blocks):

            _hs = hs.pop()
            if do_freeunet and not apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

            if not only_mid_control and control is not None:
              _control = control.pop()
              _hs += _control

            if do_freeunet and apply_freeu_after_control:
              h, _hs = apply_freeu(h, _hs)

            h = cat8([h, _hs], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)




use_reference = False #\@param {'type':'boolean'}
reference_weight = 0.5 #\@param
reference_source = 'init' #@\param ['stylized', 'init', 'prev_frame','color_video']
reference_mode = 'Balanced' #@\param ['Balanced', 'Controlnet', 'Prompt']


def apply_reference_cn():
  global reference_active
  if 'sdxl' in model_version:
    reference_active = False
    print('Temporarily disabling reference controlnet for SDXL')
  if reference_active:
    # outer = sd_model.model.diffusion_model
    try:
      outer.forward = outer.original_forward
    except: pass
    outer.original_forward = outer.forward
    outer.attention_auto_machine_weight = reference_weight
    # outer.forward = control_forward
    outer.forward = control_forward.__get__(outer, outer.__class__)
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

executed_cells[cell_name] = True

"""# Extra features"""

#@title Save loaded model
#@markdown For this cell to work you need to load model in the previous cell.\
#@markdown Saves an already loaded model as an object file, that weights less, loads faster, and requires less CPU RAM.\
#@markdown After saving model as pickle, you can then load it as your usual stable diffusion model in thecell above.\
#@markdown The model will be saved under the same name with .pkl extenstion.
cell_name = 'save_loaded_model'
# check_execution(cell_name)

save_model_pickle = False #@param {'type':'boolean'}
save_folder = "/content/drive/MyDrive/models" #@param {'type':'string'}
if save_folder != '' and save_model_pickle:
  os.makedirs(save_folder, exist_ok=True)
  out_path = save_folder+model_path.replace('\\', '/').split('/')[-1].split('.')[0]+'.pkl'
  with open(out_path, 'wb') as f:
    pickle.dump(sd_model, f)
  print('Model successfully saved as: ',out_path)

executed_cells[cell_name] = True

"""## Content-aware scheduling"""

#@title Content-aware scheduling
#@markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
#@markdown rmse function is faster than lpips, but less precise.\
#@markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.
cell_name = 'content_aware_scheduling'
# check_execution(cell_name)
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

diff_function = 'rmse' #@param ['rmse','lpips','rmse+lpips']

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
  if 'lpips' in diff_function:
    if lpips_model is None:
      lpips_model = init_lpips()
      diff_func = lpips_model
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

executed_cells[cell_name] = True

#@title Plot threshold vs frame difference
#@markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
cell_name = 'plot_threshold_vs_frame_difference'
# check_execution(cell_name)

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

executed_cells[cell_name] = True

#@title Create schedules from frame difference
cell_name = 'create_schedules'
# check_execution(cell_name)

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
style_strength_template = '' #@param {'type':'raw'}
flow_blend_template = [0.8, 0., 0.51, 2] #@param {'type':'raw'}
cc_masked_template = [0.7, 0, 0.51, 2] #@param {'type':'raw'}
cfg_scale_template = None #@param {'type':'raw'}
image_scale_template = None #@param {'type':'raw'}

#@markdown Turning this off will disable templates and will use schedules set in previous cell
make_schedules = False #@param {'type':'boolean'}
#@markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
respect_sched = True #@param {'type':'boolean'}
diff_override = [] #@param {'type':'raw'}

#shift+1 required
executed_cells[cell_name] = True

"""## Frame Captioning

"""

#@title Generate captions for keyframes
#@markdown Automatically generate captions for every n-th frame, \
#@markdown or keyframe list: at keyframe, at offset from keyframe, between keyframes.\
#@markdown keyframe source: Every n-th frame, user-input, Content-aware scheduling keyframes

cell_name = 'frame_captioning'
# check_execution(cell_name)

inputFrames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
make_captions = False #@param {'type':'boolean'}
keyframe_source = 'Every n-th frame' #@param ['Content-aware scheduling keyframes', 'User-defined keyframe list', 'Every n-th frame']
#@markdown This option only works with  keyframe source == User-defined keyframe list
user_defined_keyframes = [3,4,5] #@param
#@markdown This option only works with  keyframe source == Content-aware scheduling keyframes
diff_thresh = 0.33 #@param {'type':'number'}
#@markdown This option only works with  keyframe source == Every n-th frame
nth_frame = 10 #@param {'type':'number'}
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

executed_cells[cell_name] = True

"""# Render settings

## Non-gui
These settings are used as initial settings for the GUI unless you specify default_settings_path. Then the GUI settings will be loaded from the specified file.
"""

#@title Flow and turbo settings
#@markdown #####**Video Optical Flow Settings:**
cell_name = 'flow_and_turbo_settings'
# check_execution(cell_name)

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

executed_cells[cell_name] = True

#@title Consistency map mixing
#@markdown You can mix consistency map layers separately\
#@markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
#@markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
#@markdown edges_consistency_weight - masks moving objects' edges\
#@markdown The default values to simulate previous versions' behavior are 1,1,1
cell_name = 'consistency_maps_mixing'
# check_execution(cell_name)

missed_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}

executed_cells[cell_name] = True

#@title  ####**Seed and grad Settings:**
cell_name = 'seed_and_grad_settings'
# check_execution(cell_name)
set_seed = '4275770367' #@param{type: 'string'}


#@markdown *Clamp grad is used with any of the init_scales or sat_scale above 0*\
#@markdown Clamp grad limits the amount various criterions, controlled by *_scale parameters, are pushing the image towards the desired result.\
#@markdown For example, high scale values may cause artifacts, and clamp_grad removes this effect.
#@markdown 0.7 is a good clamp_max value.
eta = 0.55
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 2 #@param{type: 'number'}

executed_cells[cell_name] = True

"""### Prompts
`animation_mode: None` will only use the first set. `animation_mode: 2D / Video` will run through them per the set frames and hold on the last one.
"""

cell_name = 'prompts'
# check_execution(cell_name)


print("USER PROMPT", user_prompt)

text_prompts = {0: [user_prompt]} #{0: ["marblesh, marble, marblee style, soccer player, made of marble, renaissance, white marble skin, bodybuilder, veins, shredded, giant, in the stage, shirtless, <lora:marble:1>"]} #{0: [user_prompt]}
# text_prompts = {0: ['a beautiful highly detailed cyberpunk mechanical \
# augmented most beautiful (woman) ever, cyberpunk 2077, neon, dystopian, \
# hightech, trending on artstation']}

negative_prompts = {
    0: ["text, naked, nude, logo, cropped, two heads, four arms, lazy eye, blurry, unfocused"]
}

executed_cells[cell_name] = True

"""### Warp Turbo Smooth Settings

turbo_frame_skips_steps - allows to set different frames_skip_steps for turbo frames. None means turbo frames are warped only without diffusion

soften_consistency_mask - clip the lower values of consistency mask to this value. Raw video frames will leak stronger with lower values.

soften_consistency_mask_for_turbo_frames - same, but for turbo frames
"""

#@title ##Warp Turbo Smooth Settings
#@markdown Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.
cell_name = 'warp_turbo_smooth_settings'
# check_execution(cell_name)

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

#@markdown ###Warp settings

warp_mode = 'use_image' #@param ['use_latent', 'use_image']
warp_towards_init = 'off' #@param ['stylized', 'off']

if warp_towards_init != 'off':
  if flow_lq:
          raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        # raft_model = torch.nn.DataParallel(RAFT(args2))
  else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

cond_image_src = 'init' #@param ['init', 'stylized']

executed_cells[cell_name] = True

"""### Video masking (render-time)"""

#@title Video mask settings
#@markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
cell_name = 'video_mask_settings'
# check_execution(cell_name)

use_background_mask = False #@param {'type':'boolean'}
#@markdown Check to invert the mask.
invert_mask = False #@param {'type':'boolean'}
#@markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True #@param {'type':'boolean'}
#@markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video" #@param ['image', 'color', 'init_video']
#@markdown Specify the init image path or color depending on your background source choice.
background_source = 'red' #@param {'type':'string'}

executed_cells[cell_name] = True

"""### Frame correction (latent & color matching)"""

#@title Frame correction
#@markdown Match frame pixels or latent to other frames to preven oversaturation and feedback loop artifacts
#@markdown ###Latent matching
#@markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
cell_name = 'frame_correction'
# check_execution(cell_name)


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

executed_cells[cell_name] = True

"""### Main settings.

Duplicated in the GUI and can be loaded there.
"""

# @title Basic

cell_name = 'main_settings'
# check_execution(cell_name)
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

colormatch_after = True if 'animatediff' in model_version else False #colormatch after stylizing. On in previous notebooks.
colormatch_turbo = False #apply colormatching for turbo frames. On in previous notebooks

user_comment = 'v0.30'

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
mask_clip_low  = 0
mask_clip_high = 255
sampler = sample_euler
image_scale = 2
image_scale_schedule = {0:1.5, 1:2}
inpainting_mask_source = 'none'
fixed_seed = False #fixes seed
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
fill_lips = 20
masked_guidance = False #use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas
cc_masked_diffusion_schedule = [0.7]  # 0 - off. 0.5-0.7 are good values. make inconsistent area passes only before this % of actual steps, then diffuse whole image
alpha_masked_diffusion = 0.  # 0 - off. 0.5-0.7 are good values. make alpha masked area passes only before this % of actual steps, then diffuse whole image
invert_alpha_masked_diffusion = False

save_controlnet_annotations = True
pose_detector = 'dw_pose'
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
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ''
  },
  "control_sd15_canny": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_softedge": {
    "weight": 1,
    "start": 0,
    "end": 0.5,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_mlsd": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_normalbae": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_openpose": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_scribble": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_seg": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_temporalnet": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "source": ""
  },
  "control_sd15_face": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_ip2p": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "source": ""
  },
  "control_sd15_inpaint": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "source": "stylized"
  },
  "control_sd15_lineart": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_lineart_anime": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ""
  },
  "control_sd15_shuffle":{
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "source": ""
  }
}
if model_version == 'control_multi_sdxl':
  controlnet_multimodel = {
  "control_sdxl_canny": {
    "weight": 0,
    "start": 0,
    "end": 1,
    "preprocess": '',
    "mode": '',
    "detect_resolution": '',
    "source": ''
  },
  "control_sdxl_depth": {
    "weight": 0,
    "start": 0,
    "end": 1,
  },
  "control_sdxl_seg": {
    "weight": 0,
    "start": 0,
    "end": 1,
  },
   "control_sdxl_openpose": {
    "weight": 0,
    "start": 0,
    "end": 1,
  },
  "control_sdxl_softedge": {
    "weight": 1,
    "start": 0,
    "end": 1,
  }
  }
if model_version in ['control_multi_v2','control_multi_v2_768']:
  controlnet_multimodel = {
  "control_sd21_canny": {
    "weight": 0,
    "start": 0,
    "end": 1
  }
}
executed_cells[cell_name] = True

# @title Advanced

#these variables are not in the GUI and are not being loaded.
cell_name = 'advanced'
# check_execution(cell_name)

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

sd_batch_size = 2

mask_paths = []

# deflicker_scale = 0.
# deflicker_latent_scale = 0

controlnet_mode = 'balanced'
normalize_cn_weights = False if 'animatediff' in model_version else True
sd_model.normalize_weights = normalize_cn_weights
sd_model.debug = False

apply_freeu_after_control = False
do_freeunet = False

batch_length = 32 #context length
batch_overlap = 8
looped_noise = True
overlap_stylized = True #use prev stylized as init raw for the next batch if overlaps
context_length = 16
context_overlap = 10
blend_batch_outputs = True
clip_skip = 1
qr_cn_mask_clip_high = 255
qr_cn_mask_clip_low = 0
qr_cn_mask_thresh = 0
qr_cn_mask_invert = False
qr_cn_mask_grayscale = False
use_manual_splits = False
scene_split_thresh = 0.5
scene_splits = None
scenes = None

blend_prompts_b4_diffusion = True

fill_lips = 20
flow_maxsize = 0 # render flow in smaller size to reduce computational overhead

use_reference = False
reference_weight = 0.5
reference_source = 'init'
reference_mode = 'Balanced'

settings_queue = []

missed_consistency_schedule = [missed_consistency_weight]
overshoot_consistency_schedule = [overshoot_consistency_weight]
edges_consistency_schedule = [edges_consistency_weight]
consistency_blur_schedule = [consistency_blur]
consistency_dilate_schedule = [consistency_dilate]
soften_consistency_schedule = [soften_consistency_mask]

"""preview settings
"""
stack_previews = True #stacks previews into a single image
hstack_previews = True #stack horizontally
fit_previews = True #fit preview to max size
add_preview_label = True #add preview text labels

if add_preview_label and PIL.__version__ != '9.0.0':
  pipi('pillow==9.0.0')
  import PIL

offload_model = True #offload unused models to cpu
controlnet_low_vram = False #offload unused controlnet models to cpu (VERY slow)

#legacy settings (like it was in older notebooks)

gc_collect_offload = True #garbage collect during offload (slower)
cuda_empty_cache = True #empty cuda cache (slower)
gc_collect = True #garbage collect (slower)
do_run_cast = 'cpu' #['cuda', 'cpu', 'off'] #move modes to gpu at the beginning of every frame, faster / needs more vram. cpu - slower, less vram. off - do nothing
save_img_format = 'png' #['png', 'jpg', 'tiff'] #image save format, jpg is faster, png is slower, tiff is the slowest
sample_gc_collect = True
sample_cuda_empty_cache = True

#new - faster, may use more vram, so switch them 1 by 1 until you get enough vram
#just uncomment there lines to use
# gc_collect_offload = False #garbage collect during offload (slow)
# cuda_empty_cache = False #empty cuda cache (slow)
# gc_collect = False #garbage collect (slow)
# do_run_cast = 'cuda' #['cuda', 'cpu', 'off'] #move modes to gpu at the beginning of every frame, faster / needs more vram. cpu - slower, less vram. off - do nothing
# save_img_format = 'jpg' #['png', 'jpg', 'tiff'] #image save format, jpg is faster, png is slower, tiff is the slowest
# sample_gc_collect = False
# sample_cuda_empty_cache = False

force_flow_generation = False
use_legacy_cc = False
num_flow_workers = 0
flow_threads = 4
flow_lq = True
num_flow_updates = 20
missed_consistency_dilation = 2
edge_consistency_width = 11

use_tiled_vae = False  #enable if running oom during vae stage
num_tiles = [2,2]

force_mask_overwrite = False
extract_background_mask = False
mask_source = 'init_video' #replace with a path to video or frames folder if needed

enable_adjust_brightness = False
high_brightness_threshold = 180
high_brightness_adjust_ratio = 0.97
high_brightness_adjust_fix_amount = 2
max_brightness_threshold = 254
low_brightness_threshold = 40
low_brightness_adjust_ratio = 1.03
low_brightness_adjust_fix_amount = 2
min_brightness_threshold = 1

color_video_path = ""
color_extract_nth_frame =  1

#freeu
b1= 1.2
b2= 1.4
s1= 0.9
s2= 0.2

executed_cells[cell_name] = True

"""# Lora  & Embedding paths
Don't forget to set up if you use loras
"""

#@title LORA & embedding paths
cell_name = 'lora'
# check_execution(cell_name)

weight_load_location = 'cpu'
from modules import devices, shared
#@markdown Specify folders containing your Loras and Textual Inversion Embeddings. Detected loras will be listed after you run the cell.
lora_dir = './lora' #@param {'type':'string'}
if not is_colab and lora_dir.startswith('/content'):
  lora_dir = './loras'
  print('Overriding lora dir to ./loras for non-colab env because you path begins with /content. Change path to desired folder')

custom_embed_dir =   'c:\\code\\warp\\models\\embeddings\\' #@param {'type':'string'}
if not is_colab and custom_embed_dir.startswith('/content'):
  custom_embed_dir = './embeddings'
  os.makedirs(custom_embed_dir, exist_ok=True)
  print('Overriding embeddings dir to ./embeddings for non-colab env because you path begins with /content. Change path to desired folder')

# %cd C:\code\warp\18_venv\stablediffusion\modules\Lora

os.chdir(f'{root_dir}/stablediffusion/modules/Lora')
print("subprocess pathhh")
result = subprocess.run(['ls'], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
  print("Error:", result.stderr)
import os
print("OSSSSS PATH")
print(os.getcwd())
import sys
sys.path.append(os.getcwd()) 

from networks import list_available_networks, available_networks, load_networks, assign_network_names_to_compvis_modules, loaded_networks
import networks
os.chdir(root_dir)
list_available_networks(lora_dir)
import re

print('Found loras: ',[*available_networks.keys()])
if 'sdxl' in model_version: sd_model.is_sdxl = True
else:  sd_model.is_sdxl = False

if not hasattr(torch.nn, 'Linear_forward_before_network'):
    torch.nn.Linear_forward_before_network = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_network'):
    torch.nn.Linear_load_state_dict_before_network = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_network'):
    torch.nn.Conv2d_forward_before_network = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_network'):
    torch.nn.Conv2d_load_state_dict_before_network = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_network'):
    torch.nn.MultiheadAttention_forward_before_network = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_network'):
    torch.nn.MultiheadAttention_load_state_dict_before_network = torch.nn.MultiheadAttention._load_from_state_dict



def inject_network(sd_model):
  print('injecting loras')
  torch.nn.Linear.forward = networks.network_Linear_forward
  torch.nn.Linear._load_from_state_dict = networks.network_Linear_load_state_dict
  torch.nn.Conv2d.forward = networks.network_Conv2d_forward
  torch.nn.Conv2d._load_from_state_dict = networks.network_Conv2d_load_state_dict
  torch.nn.MultiheadAttention.forward = networks.network_MultiheadAttention_forward
  torch.nn.MultiheadAttention._load_from_state_dict = networks.network_MultiheadAttention_load_state_dict

  sd_model = assign_network_names_to_compvis_modules(sd_model)

def unload_network():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_network
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_network
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_network
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_network
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_network
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_network




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

executed_cells[cell_name] = True

"""# GUI"""



###############################################################################################################################################
#@title gui
cell_name = 'GUI'
# check_execution(cell_name)
global_keys = ['global', '', -1, '-1','global_settings']
if 'animatediff' in model_version:
  print('Animatediff has limited functionality at the moment. Stay tuned!')
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
      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp",
                             'flow_lq', 'num_flow_workers', 'flow_threads', 'use_legacy_cc', 'use_reference', 'reference_weight', 'reference_source', 'reference_mode'],
    "Hey, not too rough.":["flow_warp", "warp_strength","warp_mode",
      "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k","warp_forward",

      "check_consistency",

      "use_patchmatch_inpaiting","init_grad", "grad_denoised",
      "image_scale_schedule","blend_latent_to_init","rec_cfg",

      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp"],
    "Hurt me plenty.":"",
    "Ultra-Violence.":["warp_mode","use_patchmatch_inpaiting","warp_num_k","warp_forward","sat_scale",]
}
animatediff_exclusions = ['fixed_code', 'code_randomness',
                          'cc_masked_diffusion_schedule', 'alpha_masked_diffusion', 'invert_alpha_masked_diffusion',
                          "normalize_latent","normalize_latent_offset","latent_fixed_mean",
      "latent_fixed_std","latent_norm_4d"]

import traceback
gui_difficulty = "Ultra-Violence." #@param ["I'm too young to die.", "Hey, not too rough.", "Ultra-Violence."]
print(f'Using "{gui_difficulty}" gui difficulty. Please switch to another difficulty\nto unlock up to {len(gui_difficulty_dict[gui_difficulty])} more settings when you`re ready :D')
settings_path = 'settings.txt' #@param {'type':'string'}
load_settings_from_file = True #@param {'type':'boolean'}
#@markdown Disable to load settings into GUI from colab cells. You will need to re-run colab cells you've edited to apply changes, then re-run the gui cell.\
#@markdown Enable to keep GUI state.
keep_gui_state_on_cell_rerun = True #@param {'type':'boolean'}
settings_out = batchFolder+f"/settings"
# from  ipywidgets import HTML, IntSlider, IntRangeSlider, FloatRangeSlider, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, SelectionSlider, Valid

# def desc_widget(widget, desc, width=80, h=True):
#     if isinstance(widget, Checkbox): return widget
#     if isinstance(width, str):
#         if width.endswith('%') or width.endswith('px'):
#             layout = Layout(width=width)
#     else: layout = Layout(width=f'{width}')

#     text = Label(desc, layout = layout, tooltip = widget.tooltip, description_tooltip = widget.description_tooltip)
#     return HBox([text, widget]) if h else VBox([text, widget])

no_preprocess_cn = ['control_sd15_gif', 'control_sdxl_inpaint','control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
                            'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile',"control_sd15_monster_qr"]

no_resolution_cn = ['control_sd15_gif','control_sdxl_inpaint','control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
                            'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile',"control_sd15_monster_qr"]

# class ControlNetControls(HBox):
#     def __init__(self,  name, values, **kwargs):
#         self.label  = HTML(
#                 description=name,
#                 description_tooltip=name,  style={'description_width': 'initial' },
#                 layout = Layout(position='relative', left='-25px', width='200px'))
#         self.name = name
#         self.enable = Checkbox(value=values['weight']>0,description='',indent=True, description_tooltip='Enable model.',
#                                style={'description_width': '25px' },layout=Layout(width='70px', left='-25px'))
#         self.weight = FloatText(value = values['weight'], description=' ', step=0.05,
#                                 description_tooltip = 'Controlnet model weights. ',
#                                 layout=Layout(width='100px', visibility= 'visible' if values['weight']>0 else 'hidden'),
#                                 style={'description_width': '25px' })
#         self.start_end = FloatRangeSlider(
#           value=[values['start'],values['end']],
#           min=0,
#           max=1,
#           step=0.01,
#           description=' ',
#           description_tooltip='Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
#           disabled=False,
#           continuous_update=False,
#           orientation='horizontal',
#           readout=True,
#           layout = Layout(width='300px', visibility= 'visible' if values['weight']>0 else 'hidden'),
#           style={'description_width': '50px' }
#         )


#         if (not "preprocess" in values.keys()) or values["preprocess"] in global_keys:
#           values["preprocess"] = 'global'

#         if (not "mode" in values.keys()) or values["mode"] in global_keys:
#           values["mode"] = 'global'

#         if (not "detect_resolution" in values.keys()) or values["detect_resolution"] in global_keys:
#           values["detect_resolution"] = -1


#         if (not "source" in values.keys()) or values["source"] in global_keys:
#           if ('inpaint' in name) or ('ipadapter' in name): values["source"] = 'stylized'
#           else: values["source"] = 'global'
#         if values["source"] == 'init': values["source"] = 'raw_frame'


#         self.preprocess = Dropdown(description='',
#                            options = ['True', 'False', 'global'], value = values['preprocess'],
#                            description_tooltip='Preprocess input for this controlnet', layout=Layout(width='80px'))

#         self.mode = Dropdown(description='',
#                            options = ['balanced', 'controlnet', 'prompt', 'global'], value = values['mode'],
#                            description_tooltip='Controlnet mode. Pay more attention to controlnet prediction, to prompt or somewhere in-between.',
#                              layout=Layout(width='100px'))

#         self.detect_resolution = IntText(value = values['detect_resolution'], description='',
#                                          description_tooltip = 'Controlnet detect_resolution.',layout=Layout(width='80px'), style={'description_width': 'initial' })

#         self.source = Text(value=values['source'], description = '', layout=Layout(width='200px'),
#                            description_tooltip='controlnet input source, either a file or video, raw_frame, cond_video, color_video, or stylized - to use previously stylized frame ad input. leave empty for global source')

#         self.enable.observe(self.on_change)
#         self.weight.observe(self.on_change)
#         settings = [self.enable, self.label, self.weight, self.start_end, self.mode, self.source, self.detect_resolution, self.preprocess]
#         if name in no_preprocess_cn: self.preprocess.layout.visibility = 'hidden'
#         if name in no_resolution_cn: self.detect_resolution.layout.visibility = 'hidden'
#         if 'ipadapter' in name:
#           self.preprocess.layout.visibility = 'hidden'
#           self.preprocess.layout.width = '0px'
#           self.detect_resolution.layout.visibility = 'hidden'
#           self.detect_resolution.layout.width = '0px'
#           self.mode.layout.visibility = 'hidden'
#           self.mode.layout.width = '0px'
#           self.source.layout.width = '350px'
#           self.start_end.layout.width = '250px'

#         if values['weight']==0:
#               self.preprocess.layout.visibility = 'hidden'
#               self.mode.layout.visibility = 'hidden'
#               self.detect_resolution.layout.visibility = 'hidden'
#               self.source.layout.visibility = 'hidden'
#         super().__init__(settings, layout = Layout(valign='center'))

#     def on_change(self, change):
#       if change['name'] == 'value':
#         if self.enable.value:
#               self.weight.layout.visibility = 'visible'
#               if change['old'] == False and self.weight.value==0:
#                 self.weight.value = 1
#               self.start_end.layout.visibility = 'visible'
#               if not (self.name in no_preprocess_cn) and (not 'ipadapter' in self.name):
#                 self.preprocess.layout.visibility = 'visible'
#               if not 'ipadapter' in self.name:
#                 self.mode.layout.visibility = 'visible'
#               if (not self.name in no_resolution_cn) and (not 'ipadapter' in self.name):
#                 self.detect_resolution.layout.visibility = 'visible'
#               self.source.layout.visibility = 'visible'
#         else:
#               self.weight.layout.visibility = 'hidden'
#               self.start_end.layout.visibility = 'hidden'
#               self.preprocess.layout.visibility = 'hidden'
#               self.mode.layout.visibility = 'hidden'
#               self.detect_resolution.layout.visibility = 'hidden'
#               self.source.layout.visibility = 'hidden'

#     def __setattr__(self, attr, values):
#         if attr == 'value':
#           self.enable.value = values['weight']>0
#           self.weight.value = values['weight']
#           self.start_end.value=[values['start'],values['end']]
#           if (not "preprocess" in values.keys()) or values["preprocess"] in global_keys:
#                     values["preprocess"] = 'global'

#           if (not "mode" in values.keys()) or values["mode"] in global_keys:
#                     values["mode"] = 'global'

#           if (not "detect_resolution" in values.keys()) or values["detect_resolution"] in global_keys:
#                     values["detect_resolution"] = -1

#           if (not "source" in values.keys()) or values["source"] in global_keys:
#                     if self.name == 'control_sd15_inpaint': values["source"] = 'stylized'
#                     else: values["source"] = 'global'
#           if values["source"] == 'init': values["source"] = 'raw_frame'
#           self.preprocess.value = values['preprocess']
#           self.mode.value = values['mode']
#           self.detect_resolution.value = values['detect_resolution']
#           self.source.value=values['source']

#         else: super().__setattr__(attr, values)

#     def __getattr__(self, attr):
#         if attr == 'value':
#             weight = 0
#             if self.weight.value>0 and self.enable.value: weight = self.weight.value
#             (start,end) = self.start_end.value
#             values = {
#                   "weight": weight,
#                   "start":start,
#                   "end":end,

#                 }
#             if True:
#             # if self.preprocess.value not in global_keys:
#               values['preprocess'] = self.preprocess.value
#             # if self.mode.value not in global_keys:
#               values['mode'] = self.mode.value
#             # if self.detect_resolution.value not in global_keys:
#               values['detect_resolution'] = self.detect_resolution.value
#             # if self.source.value not in global_keys:
#               values['source'] = self.source.value
#             # print('returned values', values)
#             return values
#         if attr == 'name':
#           return self.name
#         else:
#             return super.__getattr__(attr)

# class ControlGUI(VBox):
#   def __init__(self, args):
#     enable_label = HTML(
#                     description='Enable',
#                     description_tooltip='Enable',  style={'description_width': '50px' },
#                     layout = Layout(width='40px', left='-50px', ))
#     model_label = HTML(
#                     description='Model name',
#                     description_tooltip='Model name',  style={'description_width': '100px' },
#                     layout = Layout(width='265px'))
#     weight_label = HTML(
#                     description='weight',
#                     description_tooltip='Model weight. 0 weight effectively disables the model. The total sum of all the weights will be normalized to 1.',  style={'description_width': 'initial' },
#                     layout = Layout(position='relative', left='-25px', width='125px'))#65
#     range_label = HTML(
#                     description='active range (% or total steps)',
#                     description_tooltip='Model`s active range. % of total steps when the model is active.\n Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',  style={'description_width': 'initial' },
#                     layout = Layout(position='relative', left='-25px', width='200px'))
#     mode_label = HTML(
#                     description='mode',
#                     description_tooltip='Controlnet mode. Pay more attention to controlnet prediction, to prompt or somewhere in-between.', layout = Layout(width='110px', left='0px', ))
#     source_label = HTML(
#                     description='source',
#                     description_tooltip='controlnet input source, either a file or video, raw_frame, cond_video, color_video, or stylized - to use previously stylized frame ad input. leave empty for global source',
#                     layout = Layout(width='210px', left='0px', ))
#     resolution_label = HTML(
#                     description='resolution',
#                     description_tooltip='Controlnet detect_resolution. The size of the image fed into annotator model if current controlnet has one.',
#                     layout = Layout(width='90px', left='0px', ))
#     preprocess_label = HTML(
#                     description='preprocess',
#                     description_tooltip='Preprocess (put through annotator model) input for this controlnet. When disabled, puts raw image from selected source into the controlnet. For example, if you have sequence of pdeth maps from your 3d software, you need to put path to those maps into source field and disable preprocessing.',
#                     layout = Layout(width='80px', left='0px', ))
#     controls_list = [HBox([enable_label,model_label, weight_label, range_label, mode_label, source_label, resolution_label, preprocess_label ])]
#     controls_dict = {}
#     possible_controlnets_v2 = [
#         'control_sd21_qr',
#         "control_sd21_depth",
#         "control_sd21_scribble",
#         "control_sd21_openpose",
#         "control_sd21_normalbae",
#         "control_sd21_lineart",
#         "control_sd21_softedge",
#         "control_sd21_canny",
#         "control_sd21_seg"
#     ]
#     #adiff accepts all controlnets, just not prev stylized frame and cc masks, need to test temporal, inpaint
#     possible_controlnets_adiff = ['control_sd15_depth',
#         'control_sd15_canny',
#         'control_sd15_softedge',
#         'control_sd15_mlsd',
#         'control_sd15_normalbae',
#         'control_sd15_openpose',
#         'control_sd15_scribble',
#         'control_sd15_seg',
#         # 'control_sd15_temporalnet',
#         'control_sd15_face',
#         'control_sd15_ip2p',
#         # 'control_sd15_inpaint',
#         'control_sd15_lineart',
#         'control_sd15_lineart_anime',
#         'control_sd15_shuffle',
#         'control_sd15_tile',
#         'control_sd15_qr',
#         'control_sd15_monster_qr',
#         # 'control_sd15_inpaint_softedge',
#         # 'control_sd15_temporal_depth',
#         "control_sd15_gif",
#         "ipadapter_sd15" ,
#         "ipadapter_sd15_light",
#         "ipadapter_sd15_plus",
#         "ipadapter_sd15_plus_face",
#         "ipadapter_sd15_full_face",
#         "ipadapter_sd15_vit_G",
#         "ipadapter_sd15_faceid",
#         "ipadapter_sd15_faceid_plus",
#         "ipadapter_sd15_faceid_plus_v2",
#         "control_sd15_depth_anything"
#                             ]
#     possible_controlnets_adiff_sdxl = [
#         'control_sdxl_canny',
#         'control_sdxl_depth',
#         'control_sdxl_softedge',
#         'control_sdxl_seg',
#         'control_sdxl_openpose',
#         'control_sdxl_lora_128_depth',
#         "control_sdxl_lora_256_depth",
#         "control_sdxl_lora_128_canny",
#         "control_sdxl_lora_256_canny",
#         "control_sdxl_lora_128_softedge",
#         "control_sdxl_lora_256_softedge",
#         "ipadapter_sdxl",
#         "ipadapter_sdxl_vit_h",
#         "ipadapter_sdxl_plus_vit_h",
#         "ipadapter_sdxl_plus_face_vit_h",
#         "ipadapter_sdxl_faceid"
#         ]
#     possible_controlnets_sdxl = ["control_sdxl_temporalnet_v1","control_sdxl_inpaint"]+possible_controlnets_adiff_sdxl
#     possible_controlnets = ['control_sd15_temporalnet',
#                             'control_sd15_inpaint_softedge',
#                             'control_sd15_inpaint',
#                             'control_sd15_temporal_depth']+possible_controlnets_adiff

#     self.possible_controlnets = possible_controlnets
#     if model_version == 'control_multi':
#       self.possible_controlnets = possible_controlnets
#     elif model_version == 'control_multi_sdxl':
#       self.possible_controlnets = possible_controlnets_sdxl
#     elif model_version in ['control_multi_v2','control_multi_v2_768']:
#       self.possible_controlnets = possible_controlnets_v2
#     elif model_version == 'control_multi_animatediff':
#       self.possible_controlnets = possible_controlnets_adiff
#     elif model_version == 'control_multi_animatediff_sdxl':
#       self.possible_controlnets = possible_controlnets_adiff_sdxl

#     for key in self.possible_controlnets:
#       if key in args.keys():
#         w = ControlNetControls(key, args[key])
#       else:
#         w = ControlNetControls(key, {
#             "weight":0,
#             "start":0,
#             "end":1
#         })
#         w.name = key
#       controls_list.append(w)

#       controls_dict[key] = w

#     self.args = args
#     self.ws = controls_dict
#     super(ControlGUI, self).__init__(controls_list)

#   def __setattr__(self, attr, values):
#         if attr == 'value':
#           keys = values.keys()
#           for i in range(len(self.children)):
#             w = self.children[i]
#             if isinstance(w, ControlNetControls) :
#               w.enable.value = False
#               for key in values.keys():
#                 if w.name == key:
#                   self.children[i].value = values[key]
#         else:
#           super().__setattr__(attr, values)

#   def __getattr__(self, attr):
#         if attr == 'value':
#             res = {}
#             for key in self.possible_controlnets:
#               if self.ws[key].value['weight'] > 0:
#                 res[key] = self.ws[key].value
#             return res
#         else:
#             return super.__getattr__(attr)

def set_visibility(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
          obj[key].layout.visibility = value

def get_settings_from_gui(user_settings_keys, guis):
  for key in user_settings_keys:
    if key in ['mask_clip_low', 'mask_clip_high']:
      value = get_value('mask_clip', guis)
    else:
      value = get_value(key, guis)

    if key in ['latent_fixed_mean', 'latent_fixed_std']:
      value = str(value)

    #apply eval for string schedules
    if key in user_settings_eval_keys:
      try:
        value = eval(value)
      except Exception as e:
        print(e, key, value)

    #load mask clip
    if key == 'mask_clip_low':
      value = value[0]
    if key == 'mask_clip_high':
      value = value[1]

    user_settings[key] = value
  return user_settings

def set_globals_from_gui(user_settings_keys, guis):

  for key in user_settings_keys:
    if key not in globals().keys():
      print(f'Variable {key} is not defined or present in globals()')
      continue
    #load mask clip

    if key in ['mask_clip_low', 'mask_clip_high']:
      value = get_value('mask_clip', guis)
    else:
      value = get_value(key, guis)

    if key in ['latent_fixed_mean', 'latent_fixed_std']:
      value = str(value)

    #apply eval for string schedules
    if key in user_settings_eval_keys:
      value = eval(value)

    if key == 'mask_clip_low':
      value = value[0]
    if key == 'mask_clip_high':
      value = value[1]

    globals()[key] = value

#try keep settings on occasional run cell
if keep_gui_state_on_cell_rerun:
  try:
    # user_settings = get_settings_from_gui(user_settings, guis)
    set_globals_from_gui(user_settings_keys, guis)
  except:
    if not "NameError: name 'get_value' is not defined" in traceback.format_exc() and not "NameError: name 'guis' is not defined" in traceback.format_exc():
      print('Error keeping state')
      print(traceback.format_exc())
    else:
      pass

gui_reference = {
    'use_reference':use_reference,
    'reference_weight': reference_weight,
    'reference_source':reference_source,
    'reference_mode':reference_mode,
}

gui_flow = {
    'lazy_warp':lazy_warp,
    'force_flow_generation':force_flow_generation,
    'use_legacy_cc':use_legacy_cc,
    'flow_threads':flow_threads,
    'num_flow_workers':num_flow_workers,
    'flow_save_img_preview':flow_save_img_preview,
    'num_flow_updates':num_flow_updates,
    "flow_maxsize": flow_maxsize

}

gui_adiff = {
    "batch_length":batch_length,
    "batch_overlap":batch_overlap,
    "context_length":context_length,
    "context_overlap" : context_overlap,
    "overlap_stylized":overlap_stylized,
    "blend_batch_outputs":blend_batch_outputs,
    "looped_noise":looped_noise
}

gui_misc = {
    "user_comment": user_comment,
    "blend_json_schedules": blend_json_schedules,
    "VERBOSE": VERBOSE,
    "offload_model": offload_model,
    "do_softcap": do_softcap,
    "softcap_thresh":softcap_thresh,
    "softcap_q":softcap_q,
    "sd_batch_size":sd_batch_size,
    "do_freeunet": do_freeunet,
    "apply_freeu_after_control": apply_freeu_after_control,
    'b1':b1,
    'b2':b2,
    's1':s1,
    's2':s2,
    'use_manual_splits':use_manual_splits,
    'scene_split_thresh': scene_split_thresh,
    'scene_splits':str(scene_splits),
    'use_tiled_vae':use_tiled_vae,
    'num_tiles':str(num_tiles),
    'enable_adjust_brightness': enable_adjust_brightness,
    'high_brightness_threshold':high_brightness_threshold,
    'high_brightness_adjust_ratio': high_brightness_adjust_ratio,
    'high_brightness_adjust_fix_amount':  high_brightness_adjust_fix_amount,
    'max_brightness_threshold':max_brightness_threshold,
    'low_brightness_threshold' :low_brightness_threshold,
    'low_brightness_adjust_ratio' :low_brightness_adjust_ratio,
    'low_brightness_adjust_fix_amount' :low_brightness_adjust_fix_amount,
    'min_brightness_threshold' :min_brightness_threshold,
}

gui_mask = {
    "use_background_mask":use_background_mask,
    "invert_mask":invert_mask,
    "background": background,
    "background_source": background_source,
    "apply_mask_after_warp": apply_mask_after_warp,
    "mask_clip" : (mask_clip_low, mask_clip_high),
    "mask_paths":str(mask_paths),
    "extract_background_mask":extract_background_mask,
    "force_mask_overwrite":force_mask_overwrite,
    "mask_source": mask_source,
}

gui_turbo = {
    "turbo_mode": turbo_mode,  # Assuming turbo_mode is a boolean
    "turbo_steps": turbo_steps,  # Assuming turbo_steps is an integer
    "colormatch_turbo": colormatch_turbo,  # Assuming colormatch_turbo is a boolean
    "turbo_frame_skips_steps": '100% (don`t diffuse turbo frames, fastest)',  # Assuming turbo_frame_skips_steps is a string from the options ['70%', '75%', '80%', '85%', '90%', '95%', '100% (don`t diffuse turbo frames, fastest)']
    "soften_consistency_mask_for_turbo_frames": soften_consistency_mask_for_turbo_frames,  # Assuming soften_consistency_mask_for_turbo_frames is a float
}

gui_warp = {
    "flow_warp": flow_warp,  # Assuming flow_warp is a boolean
    "flow_blend_schedule": str(flow_blend_schedule),  # Assuming flow_blend_schedule is a string or a serialized structure
    "warp_num_k": warp_num_k,  # Assuming warp_num_k is an integer
    "warp_forward": warp_forward,  # Assuming warp_forward is a boolean
    "warp_strength": warp_strength,  # Assuming warp_strength is a float
    "flow_override_map": str(flow_override_map),  # Assuming flow_override_map is a string or a serialized structure
    "warp_mode": warp_mode,  # Assuming warp_mode is a string from the options ['use_latent', 'use_image']
    "warp_towards_init": warp_towards_init,  # Assuming warp_towards_init is a string from the options ['stylized', 'off']
    "padding_ratio": padding_ratio,  # Assuming padding_ratio is a float
    "padding_mode": padding_mode,  # Assuming padding_mode is a string from the options ['reflect', 'edge', 'wrap']
}

# warp_interp = Image.LANCZOS

gui_consistency = {
    "check_consistency":check_consistency,
    "missed_consistency_schedule":str(missed_consistency_schedule),
    "overshoot_consistency_schedule":str(overshoot_consistency_schedule),
    "edges_consistency_schedule":str(edges_consistency_schedule),
    "consistency_blur_schedule":str(consistency_blur_schedule),
    "consistency_dilate_schedule":str(consistency_dilate_schedule),
    "soften_consistency_schedule":str(soften_consistency_schedule),
    "barely used": None,
    "match_color_strength" : match_color_strength,
    "mask_result": mask_result,
    "use_patchmatch_inpaiting": use_patchmatch_inpaiting,
}

gui_diffusion = {
    "clip_skip":clip_skip,
    "use_karras_noise":use_karras_noise,
    "sampler": sampler,
    'blend_prompts_b4_diffusion':blend_prompts_b4_diffusion,
    "prompt_patterns_sched": str(prompt_patterns_sched),
    "text_prompts" : str(text_prompts),
    "negative_prompts" : str(negative_prompts),
    "cond_image_src":cond_image_src,
    "inpainting_mask_source":inpainting_mask_source,
    "inverse_inpainting_mask":inverse_inpainting_mask,
    "inpainting_mask_weight":inpainting_mask_weight,
    "set_seed": set_seed,
    "clamp_grad":clamp_grad,
    "clamp_max": clamp_max,
    "latent_scale_schedule":str(latent_scale_schedule),
    "init_scale_schedule": str(init_scale_schedule),
    "sat_scale": sat_scale,
    "init_grad": init_grad,
    "grad_denoised" : grad_denoised,
    "steps_schedule" : str(steps_schedule),
    "style_strength_schedule" : str(style_strength_schedule),
    "cfg_scale_schedule": str(cfg_scale_schedule),
    "image_scale_schedule": str(image_scale_schedule),
    "blend_latent_to_init": blend_latent_to_init,

    "fixed_seed": fixed_seed,
    "fixed_code":  fixed_code,
    "code_randomness": code_randomness,
    "dynamic_thresh": dynamic_thresh,
    "use_predicted_noise":use_predicted_noise,
    "rec_prompts" : str(rec_prompts),
    "rec_randomness": rec_randomness,
    "rec_cfg": rec_cfg,
    "rec_source": rec_source,
    "rec_steps_pct":rec_steps_pct,
    "overwrite_rec_noise":overwrite_rec_noise,
    "masked_guidance":masked_guidance,
    "cc_masked_diffusion_schedule": str(cc_masked_diffusion_schedule),
    "alpha_masked_diffusion": alpha_masked_diffusion,
    "invert_alpha_masked_diffusion":invert_alpha_masked_diffusion,
    "normalize_prompt_weights":normalize_prompt_weights,
    "deflicker_scale": deflicker_scale,
    "deflicker_latent_scale": deflicker_latent_scale,
}
gui_colormatch = {
    "normalize_latent": normalize_latent,
    "normalize_latent_offset":normalize_latent_offset,
    "latent_fixed_mean": latent_fixed_mean,
    "latent_fixed_std": latent_fixed_std,
    "latent_norm_4d": latent_norm_4d,
    "colormatch_frame": colormatch_frame,
    "color_match_frame_str": color_match_frame_str,
    "colormatch_offset":colormatch_offset,
    "colormatch_method": colormatch_method,
    "colormatch_after":colormatch_after,
    'color_video_path':str(color_video_path),
    'color_extract_nth_frame': color_extract_nth_frame,
}

gui_controlnet = {
    'qr_cn_mask_clip_high':qr_cn_mask_clip_high,
    'qr_cn_mask_clip_low':qr_cn_mask_clip_low,
    'qr_cn_mask_thresh':qr_cn_mask_thresh,
    'qr_cn_mask_invert':qr_cn_mask_invert,
    'qr_cn_mask_grayscale':qr_cn_mask_grayscale,
    "controlnet_preprocess": controlnet_preprocess,
    "detect_resolution":detect_resolution,
    "bg_threshold":bg_threshold,
    "low_threshold":low_threshold,
    "high_threshold":high_threshold,
    "value_threshold":value_threshold,
    "distance_threshold":distance_threshold,
    "temporalnet_source":temporalnet_source,
    "temporalnet_skip_1st_frame": temporalnet_skip_1st_frame,
    "controlnet_multimodel_mode":controlnet_multimodel_mode,
    "max_faces":max_faces,
    "fill_lips":fill_lips,
    "controlnet_low_vram":controlnet_low_vram,
    "save_controlnet_annotations": save_controlnet_annotations,
    "control_sd15_openpose_hands_face":control_sd15_openpose_hands_face,
    "control_sd15_depth_detector" :control_sd15_depth_detector,
    "pose_detector" :pose_detector,
    "control_sd15_softedge_detector":control_sd15_softedge_detector,
    "control_sd15_seg_detector":control_sd15_seg_detector,
    "control_sd15_scribble_detector":control_sd15_scribble_detector,
    "control_sd15_lineart_coarse":control_sd15_lineart_coarse,
    "control_sd15_inpaint_mask_source":control_sd15_inpaint_mask_source,
    "control_sd15_shuffle_source":control_sd15_shuffle_source,
    "control_sd15_shuffle_1st_source":control_sd15_shuffle_1st_source,
    "controlnet_multimodel":controlnet_multimodel,
    "controlnet_mode":controlnet_mode,
    "normalize_cn_weights":normalize_cn_weights,
}

colormatch_regrain = False

guis = [gui_diffusion, gui_controlnet, gui_warp, gui_consistency, gui_turbo, gui_mask, gui_colormatch, gui_misc, gui_adiff, gui_flow, gui_reference]

# for key in gui_difficulty_dict[gui_difficulty]:
#   for gui in guis:
#     set_visibility(key, 'hidden', gui)

# if 'animatediff' in model_version:

#   for key in animatediff_exclusions:
#     for gui in guis:
#       set_visibility(key, 'hidden', gui)



# class FilePath(HBox):
#     def __init__(self,  **kwargs):
#         self.model_path = Text(value='',  continuous_update = True,**kwargs)
#         self.path_checker = Valid(
#         value=False, #layout=Layout(width='200px')
#         )

#         self.model_path.observe(self.on_change)
#         super().__init__([self.model_path, self.path_checker])

#     def __getattr__(self, attr):
#         if attr == 'value':
#             return self.model_path.value
#         else:
#             return super.__getattr__(attr)

#     def on_change(self, change):
#         if change['name'] == 'value':
#             path = infer_settings_path(change['new'])
#             if os.path.exists(path):
#                 self.path_checker.value = True
#                 self.path_checker.description = ''
#             else:
#                 self.path_checker.value = False
#                 self.path_checker.description = 'The file does not exist. Please specify the correct path.'

# def add_labels_dict(gui):
#     style = {'description_width': '250px' }
#     layout = Layout(width='500px')
#     gui_labels = {}
#     for key in gui.keys():
#         gui[key].style = style
#         # temp = gui[key]
#         # temp.observe(dump_gui())
#         # gui[key] = temp
#         if key == "controlnet_multimodel":
#            continue
#         # if isinstance(gui[key], ControlGUI):
#         #   continue
#         if not isinstance(gui[key], Textarea) and not isinstance( gui[key],Checkbox ):
#             # vis = gui[key].layout.visibility
#             # gui[key].layout = layout
#             gui[key].layout.width = '500px'
#         if isinstance( gui[key],Checkbox ):
#             html_label = HTML(
#                 description=gui[key].description,
#                 description_tooltip=gui[key].description_tooltip,  style={'description_width': 'initial' },
#                 layout = Layout(position='relative', left='-25px'))
#             gui_labels[key] = HBox([gui[key],html_label])
#             gui_labels[key].layout.visibility = gui[key].layout.visibility
#             gui[key].description = ''
#             # gui_labels[key] = gui[key]

#         else:

#             gui_labels[key] = gui[key]
#             # gui_labels[key].layout.visibility = gui[key].layout.visibility
#         # gui_labels[key].observe(print('smth changed', time.time()))

#     return gui_labels


# (gui_diffusion_label, gui_controlnet_label, gui_warp_label, gui_consistency_label,
#  gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_label, gui_adiff_label, gui_flow_label, gui_reference_label) = [add_labels_dict(o) for o in guis]

# flow_w = Accordion([VBox(list(gui_flow_label.values()))])
# flow_w.set_title(0, 'Optical Flow generation settings...')

# ref_w = Accordion([VBox(list(gui_reference_label.values()))])
# ref_w.set_title(0, 'Reference controlnet settings...')

# cond_keys = ['latent_scale_schedule','init_scale_schedule','clamp_grad',
#              'clamp_max','init_grad','grad_denoised','masked_guidance','deflicker_scale', 'deflicker_latent_scale']
# conditioning_w = Accordion([VBox([gui_diffusion_label[o] for o in cond_keys])])
# conditioning_w.set_title(0, 'External Conditioning...')

# seed_keys = ['set_seed', 'fixed_seed', 'fixed_code', 'code_randomness']
# seed_w = Accordion([VBox([gui_diffusion_label[o] for o in seed_keys])])
# seed_w.set_title(0, 'Seed...')

# rec_keys = ['use_predicted_noise','rec_prompts','rec_cfg','rec_randomness', 'rec_source', 'rec_steps_pct', 'overwrite_rec_noise']
# rec_w = Accordion([VBox([gui_diffusion_label[o] for o in rec_keys])])
# rec_w.set_title(0, 'Reconstructed noise...')

# prompt_keys = ['text_prompts', 'negative_prompts','blend_prompts_b4_diffusion', 'clip_skip','prompt_patterns_sched',
# 'steps_schedule', 'style_strength_schedule',
# 'cfg_scale_schedule', 'blend_latent_to_init', 'dynamic_thresh',
# 'cond_image_src', 'cc_masked_diffusion_schedule', 'alpha_masked_diffusion', 'invert_alpha_masked_diffusion', 'normalize_prompt_weights']
# if model_version == 'v1_instructpix2pix':
#   prompt_keys.append('image_scale_schedule')
# if  model_version == 'v1_inpainting':
#   prompt_keys+=['inpainting_mask_source', 'inverse_inpainting_mask', 'inpainting_mask_weight']
# prompt_keys = [o for o in prompt_keys if o not in seed_keys+cond_keys]
# prompt_w = [gui_diffusion_label[o] for o in prompt_keys]

# if 'animatediff' in model_version:
#   gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'],
#   gui_diffusion_label['use_karras_noise'], seed_w]
# else:
#   gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'],
#   gui_diffusion_label['use_karras_noise'], conditioning_w, seed_w, rec_w]

control_annotator_keys = ['normalize_cn_weights', 'save_controlnet_annotations','qr_cn_mask_clip_high','qr_cn_mask_clip_low','qr_cn_mask_thresh',
'qr_cn_mask_invert','qr_cn_mask_grayscale','bg_threshold','low_threshold','high_threshold','value_threshold',
                          'distance_threshold', 'max_faces', 'fill_lips', 'control_sd15_openpose_hands_face','control_sd15_depth_detector' ,'pose_detector','control_sd15_softedge_detector',
'control_sd15_seg_detector','control_sd15_scribble_detector','control_sd15_lineart_coarse','control_sd15_inpaint_mask_source',
'control_sd15_shuffle_source','control_sd15_shuffle_1st_source', 'temporalnet_source', 'temporalnet_skip_1st_frame']
control_global_keys = ['controlnet_preprocess', 'detect_resolution', 'controlnet_mode']
# control_global_w_list = [gui_controlnet_label[o] for o in control_global_keys]
# control_global_w_list.append(gui_diffusion_label["cond_image_src"])
# control_global_w = Accordion([VBox(control_global_w_list)])
# control_global_w.set_title(0, 'Controlnet global settings...')

# control_annotator_w = Accordion([VBox([gui_controlnet_label[o] for o in control_annotator_keys])])
# control_annotator_w.set_title(0, 'Controlnet annotator settings...')
# controlnet_model_w = Accordion([gui_controlnet['controlnet_multimodel']])
# controlnet_model_w.set_title(0, 'Controlnet models settings...')
# control_keys = [ 'controlnet_multimodel_mode', 'controlnet_low_vram']
# control_w = [gui_controlnet_label[o] for o in control_keys]
# gui_control_list = [controlnet_model_w, control_global_w, control_annotator_w, ref_w, *control_w]

# #misc
# misc_keys = ["user_comment","blend_json_schedules","VERBOSE","offload_model",'sd_batch_size']
# misc_w = [gui_misc_label[o] for o in misc_keys]

# freeu_w = [gui_misc_label[o] for o in ['do_freeunet','apply_freeu_after_control','b1', 'b2', 's1', 's2']]
# freeu_w = Accordion([VBox(freeu_w)])
# freeu_w.set_title(0, 'FreeU...')

# tiledvae_w = [gui_misc_label[o] for o in ['use_tiled_vae','num_tiles']]
# tiledvae_w = Accordion([VBox(tiledvae_w)])
# tiledvae_w.set_title(0, 'Tiled VAE...')

# splits_w = [gui_misc_label[o] for o in ['use_manual_splits','scene_split_thresh','scene_splits']]
# splits_w = Accordion([VBox(splits_w)])
# splits_w.set_title(0, 'Scene splits...')

# brightness_w = [gui_misc_label[o] for o in ['enable_adjust_brightness',
# 'high_brightness_threshold',
# 'high_brightness_adjust_ratio',
# 'high_brightness_adjust_fix_amount',
# 'max_brightness_threshold',
# 'low_brightness_threshold',
# 'low_brightness_adjust_ratio',
# 'low_brightness_adjust_fix_amount',
# 'min_brightness_threshold']]

# brightness_w = Accordion([VBox(brightness_w)])
# brightness_w.set_title(0, 'Automatic Brightness Adjustment...')

# softcap_keys = ['do_softcap','softcap_thresh','softcap_q']
# softcap_w = Accordion([VBox([gui_misc_label[o] for o in softcap_keys])])
# softcap_w.set_title(0, 'Softcap settings...')

# load_settings_btn = Button(description='Load settings')
# load_q_btn = Button(description='Add path to queue')
# save_settings_btn = Button(description='Save settings')
# save_settings_q_btn = Button(description='Save & queue')
# def btn_eventhandler(obj):
#   global guis
#   guis = load_settings(load_settings_path.value, guis)

# def stringify_settings_queue(settings_queue):
#   if settings_queue in [[], [''], [None]]:
#     return ''
#   else:
#     return str("\n".join(settings_queue))

# def flush_q_btn_eventhandler(obj):
#   print('Clearing queue')
#   global settings_queue
#   settings_queue = []
#   queue_box.value = stringify_settings_queue(settings_queue)

# flush_q_btn = Button(description='Clear queue')
# flush_q_btn.on_click(flush_q_btn_eventhandler)

# queue_box = Textarea(value=stringify_settings_queue(settings_queue),layout=Layout(width=f'80%'),  description = '',
#                      description_tooltip = 'settings_queue', disabled=True)

max_frames = None
seed = None
style_strength_schedule_bkup = None
latent_scale_schedule_bkup = None
init_scale_schedule_bkup = None
steps_schedule_bkup = None
style_strength_schedule_bkup = None
flow_blend_schedule_bkup = None
cfg_scale_schedule_bkup = None
image_scale_schedule_bkup = None
cc_masked_diffusion_schedule_bkup = None
missed_consistency_schedule_bkup = None
overshoot_consistency_schedule_bkup = None
edges_consistency_schedule_bkup = None
consistency_blur_schedule_bkup = None
consistency_dilate_schedule_bkup = None
soften_consistency_schedule_bkup = None

frame_range = [0,0]
inverse_mask_order = False
batch_length_bkup = None

def save_settings_from_gui():
  global latent_scale_schedule_bkup, init_scale_schedule_bkup, steps_schedule_bkup, style_strength_schedule_bkup, batch_length_bkup
  global flow_blend_schedule_bkup, cfg_scale_schedule_bkup, image_scale_schedule_bkup, cc_masked_diffusion_schedule_bkup, seed
  global missed_consistency_schedule_bkup, overshoot_consistency_schedule_bkup, edges_consistency_schedule_bkup, consistency_blur_schedule_bkup,consistency_dilate_schedule_bkup, soften_consistency_schedule_bkup
  user_settings = get_settings_from_gui(user_settings_keys, guis)
  #assign user_settings back to globals()
  #after here you can work with settings
  for key in user_settings.keys():
    globals()[key] = user_settings[key]
  latent_scale_schedule_bkup = copy.copy(latent_scale_schedule)
  init_scale_schedule_bkup = copy.copy(init_scale_schedule)
  steps_schedule_bkup = copy.copy(steps_schedule)
  style_strength_schedule_bkup = copy.copy(style_strength_schedule)
  flow_blend_schedule_bkup = copy.copy(flow_blend_schedule)
  cfg_scale_schedule_bkup = copy.copy(cfg_scale_schedule)
  image_scale_schedule_bkup = copy.copy(image_scale_schedule)
  cc_masked_diffusion_schedule_bkup = copy.copy(cc_masked_diffusion_schedule)
  batch_length_bkup = batch_length
  missed_consistency_schedule_bkup = copy.copy(missed_consistency_schedule)
  overshoot_consistency_schedule_bkup = copy.copy(overshoot_consistency_schedule)
  edges_consistency_schedule_bkup  = copy.copy(edges_consistency_schedule)
  consistency_blur_schedule_bkup = copy.copy(consistency_blur_schedule)
  consistency_dilate_schedule_bkup  = copy.copy(consistency_dilate_schedule)
  soften_consistency_schedule_bkup = copy.copy(soften_consistency_schedule)
  if set_seed == 'random_seed' or set_seed == -1:
    random.seed()
    seed = random.randint(0, 2**32)
      # print(f'Using seed: {seed}')
  else:
    seed = int(set_seed)

  return save_settings(path='manual_save')

# def save_btn_eventhandler(obj):
#   settings_filename = save_settings_from_gui()['settings_filename']
#   print('Saved settings manually to ', settings_filename)

# def save_q_btn_eventhandler(obj):
#   settings_filename = save_settings_from_gui()['settings_filename']
#   settings_queue.append(settings_filename)
#   print('Saved settings manually to ', settings_filename)
#   print('Added current settings to the queue. Total settings in the queue: ', len(settings_queue))
#   queue_box.value = stringify_settings_queue(settings_queue)

# def load_to_q_eventhandler(obj):
#   path = infer_settings_path(load_settings_path.value)
#   settings_queue.append(path)
#   print('Added ', load_settings_path.value ,' to the queue. Total settings in the queue: ', len(settings_queue))
#   queue_box.value = stringify_settings_queue(settings_queue)

# load_q_btn.on_click(load_to_q_eventhandler)
# load_settings_btn.on_click(btn_eventhandler)
# save_settings_btn.on_click(save_btn_eventhandler)
# save_settings_q_btn.on_click(save_q_btn_eventhandler)
# load_settings_path = FilePath(placeholder='Please specify the path to the settings file to load.',
#                               description_tooltip='Please specify the path to the settings file to load.')
# settings_lists = [[save_settings_btn, save_settings_q_btn,flush_q_btn]]
# settings_hboxes = [HBox(o) for o in settings_lists]
# settings_w = Accordion([VBox([HBox([load_settings_btn, load_q_btn, load_settings_path]), queue_box, *settings_hboxes])])
# settings_w.set_title(0, 'Settings and queue...')
# gui_misc_list = [*misc_w, tiledvae_w, freeu_w, softcap_w, splits_w, brightness_w, settings_w]

# guis_labels_source = [gui_diffusion_list]
# guis_titles_source = ['diffusion']
# if 'control' in model_version:
#   guis_labels_source += [gui_control_list]
#   guis_titles_source += ['controlnet']

# gui_warp_list = [*list(gui_warp_label.values()), flow_w]

# if 'animatediff' in model_version:
#   guis_titles_source+=['animatediff']
#   guis_labels_source+=[gui_adiff_label]
# else:
#   guis_labels_source += [gui_warp_list, gui_consistency_label,
#   gui_turbo_label, gui_mask_label, ]
#   guis_titles_source += ['warp', 'consistency', 'turbo', 'mask',]

# guis_labels_source += [gui_colormatch_label, gui_misc_list]
# guis_titles_source += [ 'colormatch', 'misc']

# guis_labels = [VBox([*o.values()]) if isinstance(o, dict) else VBox(o) for o in guis_labels_source]

# t_app = Tab(guis_labels)
# for i,title in enumerate(guis_titles_source):
#     t_app.set_title(i, title)

# app = VBox([settings_w, t_app])

def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key]#.value
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
          obj[key] = value
            # obj[key].value = value
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
      settings_files = sorted(glob(os.path.join(settings_out, '*.txt')),
                              key=os.path.getctime)
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

def load_settings(path, guis):
    path = infer_settings_path(path)

    # global guis, load_settings_path, output
    global output
    if not os.path.exists(path):
        output.clear_output()
        print('Please specify a valid path to a settings file.')
        return guis
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
            if key == 'cc_masked_diffusion':
              key = 'cc_masked_diffusion_schedule'
              val = f'[{val}]'
            if key == 'mask_clip':
              val = eval(val)
            if key == 'sampler':
              val = getattr(K.sampling, val)
            if key == 'controlnet_multimodel':
              val = val.replace('control_sd15_hed', 'control_sd15_softedge')
              val = json.loads(val)
              set_value(key, val, guis)
              set_value(key, val, guis)
            # print(key, val)
            set_value(key, val, guis)
            # print(get_value(key, guis))
        except Exception as e:
            print(key), print(settings[key] )
            print(e)
    # output.clear_output()
    print('Successfully loaded settings from ', path )
    return guis

def dump_gui():
  print('smth changed', time.time())

# output = Output()

# display.display(app)
if settings_path != '' and load_settings_from_file:
  guis = load_settings(settings_path, guis)



# executed_cells[cell_name] = True


###############################################################################################################################################

"""# 4. Diffuse!
if you are having OOM or PIL error here click "restart and run all" once.
"""

#@title Do the Run!
#@markdown Preview max size

ipadapter_embeds_cache = {}
cell_name = 'do_the_run'
# check_execution(cell_name)
list_available_networks(lora_dir)
# only_preview_controlnet = False #@param {'type':'boolean'}
# skip_diffuse_cell = False #@param {'type':'boolean'}



render_mode = 'render' #@param  ['render', 'render, preview controlnet', 'skip render, preview controlnet', 'skip cell']
if render_mode == 'skip cell':
  skip_diffuse_cell = True
  print('Skipping render. To do the render select render_mode = render')
else:
  skip_diffuse_cell = False

cache_ipadapter = True
vae_cache_size = 10
ipadapter_embeds_cache_size = 10
use_vae_cache = True

display_size = 512 #@param

thread_pool = Pool(flow_threads)
deflicker_scale = 0. #makes glitches :D
deflicker_latent_scale = 0.
fft_scale = 0.
fft_latent_scale = 0.

if 'sdxl' in model_version: sd_model.is_sdxl = True
else:  sd_model.is_sdxl = False

if settings_queue != []:
  print(f'Rendering {len(settings_queue)} settings.')
for i in trange(max(len(settings_queue),1)):
  if settings_queue != []:
    guis = load_settings(settings_queue[0], guis)

  try:
    sd_model.cpu()
    sd_model.model.cpu()
    sd_model.cond_stage_model.cpu()
    sd_model.first_stage_model.cpu()
    if 'control' in model_version:
      for key in loaded_controlnets.keys():
        loaded_controlnets[key].cpu()
  except: pass
  try:
    apply_openpose.body_estimation.model.cpu()
    apply_openpose.hand_estimation.model.cpu()
    apply_openpose.face_estimation.model.cpu()
  except: pass
  try:
    sd_model.model.diffusion_model.cpu()
  except: pass
  try:
    apply_softedge.netNetwork.cpu()
  except: pass
  try:
    apply_normal.netNetwork.cpu()
  except: pass
  try:
    apply_depth.model.cpu()
  except: pass
  torch.cuda.empty_cache()
  gc.collect()

  #@markdown ---
  #@markdown Frames to run. Leave empty or [0,0] to run all frames.
  frame_range = [0,0] #@param
  resume_run = False #@param{type: 'boolean'}
  load_settings_from_run = False #@param{type: 'boolean'}
  run_to_resume = 'latest' #@param{type: 'string'}
  resume_from_frame = 'latest' #@param{type: 'string'}
  retain_overwritten_frames = False #@param{type: 'boolean'}
  if retain_overwritten_frames is True:
    retainFolder = f'{batchFolder}/retained'
    createPath(retainFolder)

  user_settings = get_settings_from_gui(user_settings_keys, guis)
  for key in user_settings.keys():
    globals()[key] = user_settings[key]
    #loading settings from gui 1st so that resuming will consiter GUI settings

  settings_out = batchFolder+f"/settings"
  if resume_run:
    if run_to_resume == 'latest':
      try:
        batchNum
      except:
        renders = glob(f"{batchFolder}/*00.{save_img_format}")
        renders_s = sorted([int(o.split('(')[-1].split(')')[0]) for o in renders])
        batchNum = max(renders_s)
        # batchNum = len(glob(f"{settings_out}/{batch_name}(*)_settings.txt"))-1

    else:
      batchNum = int(run_to_resume)
    if resume_from_frame == 'latest':
      start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.{save_img_format}"))
      if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
        start_frame = start_frame - (start_frame % int(turbo_steps))
    else:
      start_frame = int(resume_from_frame)+1
      if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
        start_frame = start_frame - (start_frame % int(turbo_steps))
      if retain_overwritten_frames is True:
        existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.{save_img_format}"))
        frames_to_save = existing_frames - start_frame
        print(f'Moving {frames_to_save} frames to the Retained folder')
        move_files(start_frame, existing_frames, batchFolder, retainFolder)
    if 'animatediff' in model_version:
      #start with overlapping frames
      start_frame = max(0, start_frame - batch_overlap)
  else:
    start_frame = 0
    batchNum = len(glob(settings_out+"/*.txt"))
    while os.path.isfile(f"{settings_out}/{batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True:
      batchNum += 1

  print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')
  tempdir = os.path.join(batchFolder, 'temp')
  os.makedirs(tempdir, exist_ok=True)
  if resume_run and load_settings_from_run:
    resume_settings = sorted(glob(batchFolder+f"/{batch_name}({batchNum})_*.{save_img_format}"))[-1]
    if resume_settings != '':
      try:
        guis = load_settings(resume_settings, guis)
      except:
        print('Failed loading settings from the previous run.')

  user_settings = get_settings_from_gui(user_settings_keys, guis)
  #assign user_settings back to globals()
  #after here you can work with settings
  for key in user_settings.keys():
    globals()[key] = user_settings[key]

  if 'animatediff' in model_version:
    extract_background_mask = False

  """prepare color video"""
  if color_video_path in ['', 'None']:
    color_video_path = None
  if color_video_path:
    try:
      os.makedirs(colorVideoFramesFolder, exist_ok=True)
      Image.open(color_video_path).save(os.path.join(colorVideoFramesFolder,'000001.jpg'))
    except:
      print(color_video_path, colorVideoFramesFolder, color_extract_nth_frame)
      extractFrames(color_video_path, colorVideoFramesFolder, color_extract_nth_frame, start_frame, end_frame)

  """prepare or extract bg mask"""
  videoFramesAlpha = prepare_mask(extract_background_mask, videoFramesFolder, mask_source, extract_nth_frame, start_frame, end_frame, force_mask_overwrite)

  """apply reference cn"""
  outer = sd_model.model.diffusion_model
  reference_active = reference_weight>0 and use_reference and reference_source != 'None'
  apply_reference_cn()

  if use_tiled_vae:
    print(f'Splitting WxH {width_height} into {num_tiles[0]*num_tiles[1]} {width_height[0]//num_tiles[1]}x{width_height[1]//num_tiles[0]} tiles' )
  if num_tiles in [0, '', None]:
    num_tiles = None
  if padding in [0, '', None]:
    padding = None

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
          try:
            bkup_get_fold_unfold = sd_model.get_fold_unfold
          except:
            pass

          sd_model.split_input_params = split_input_params
          sd_model.decode_first_stage = types.MethodType(decode_first_stage,sd_model)
          sd_model.encode_first_stage = types.MethodType(encode_first_stage,sd_model)
          sd_model.get_first_stage_encoding = types.MethodType(get_first_stage_encoding,sd_model)
          sd_model.get_fold_unfold = types.MethodType(get_fold_unfold,sd_model)

  else:
          if hasattr(sd_model, "split_input_params"):
            delattr(sd_model, "split_input_params")
            try:
              sd_model.decode_first_stage = bkup_decode_first_stage
              sd_model.encode_first_stage = bkup_encode_first_stage
              sd_model.get_first_stage_encoding = bkup_get_first_stage_encoding
              sd_model.get_fold_unfold = bkup_get_fold_unfold
            except: pass


  if 'sdxl' in model_version:
    sd_model.get_weighting = types.MethodType(get_weighting,sd_model)


  def make_scenes(scene_splits):
    """
    splits frames into scenes based on a list of keyframes
    """
    # print('scene_splits',scene_splits)
    if scene_splits in [None,[]]:
      return None
    scenes = []
    start_frame = 0
    for split in scene_splits:
      scenes.append((start_frame, split))
      start_frame = split+1
    scenes.append((start_frame, -1)) #add final scene after the last split
    return scenes

  # print(diff, scenes, scene_splits)
  # if 'animatediff' in model_version:
  if True:
    if not use_manual_splits and (diff is not None):
      scene_splits = [i for (i,d) in enumerate(diff) if d>scene_split_thresh]
    # print(diff, scenes, scene_splits)
    scenes = make_scenes(scene_splits)
    if scenes not in [None, []]:
      print('Split the sequence into scenes: ', scenes)

  qr_cn_mask_clip_high = min(qr_cn_mask_clip_high, 255)
  qr_cn_mask_clip_low = max(qr_cn_mask_clip_low, 0)
  qr_cn_mask_thresh = min(max(qr_cn_mask_thresh,0),255)

  shared.opts.CLIP_stop_at_last_layers = min(max(clip_skip, 1),12)
  print('Using clip_skip value of ', shared.opts.CLIP_stop_at_last_layers)
  if (animation_mode == 'Video Input') and (flow_warp):
    flows = glob(flo_folder+'/*.jpg.npy')
    if (len(flows)>0) and not force_flow_generation and len(flows)>= len(glob(in_path+'/*.*'))-1:
      if 'animatediff' in model_version:
        print('Skipping flow generation for animate diff as it is not used. If you still wish to generate it, check force_flow_generation in GUI')
      else:
        print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

    if ((len(flows)==0) or force_flow_generation or (len(flows)< len(glob(in_path+'/*.*'))-1)) and (not lazy_warp):
      print('Flow files not found or incomplete. Generating.')
      force_generate_flow()
    if not lazy_warp and (len(flows)>0):
      make_flow_preview_grid()

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
  controlnet_multimodel_inferred = copy.deepcopy(controlnet_multimodel)

  #set global settings by default

  global_keys = ['global', '', -1, '-1','global_settings']
  fileDatasetsByPath = {}

  for key in controlnet_multimodel.keys():
    if (not "preprocess" in controlnet_multimodel[key].keys()) or controlnet_multimodel[key]["preprocess"] in global_keys:
      controlnet_multimodel_inferred[key]["preprocess"] = controlnet_preprocess
    else:
      controlnet_multimodel_inferred[key]["preprocess"] = eval(controlnet_multimodel_inferred[key]["preprocess"])

    if (not "mode" in controlnet_multimodel[key].keys()) or controlnet_multimodel[key]["mode"] in global_keys:
      controlnet_multimodel_inferred[key]["mode"] = controlnet_mode

    if (not "detect_resolution" in controlnet_multimodel[key].keys()) or controlnet_multimodel[key]["detect_resolution"] in global_keys:
      controlnet_multimodel_inferred[key]["detect_resolution"] = detect_resolution

    if (not "source" in controlnet_multimodel[key].keys()) or controlnet_multimodel[key]["source"] in global_keys:
      controlnet_multimodel_inferred[key]["source"] = cond_image_src
      if controlnet_multimodel_inferred[key]["source"] == 'init': controlnet_multimodel_inferred[key]["source"] = 'raw_frame'

    if controlnet_multimodel_inferred[key]["source"] == 'raw_frame':
      #cache file datasets with same sources
      if videoFramesFolder not in fileDatasetsByPath.keys():
        fileDatasetsByPath[videoFramesFolder] = FrameDataset(videoFramesFolder, f'{key}_source', '' )
      controlnet_multimodel_inferred[key]["source"] = fileDatasetsByPath[videoFramesFolder]

    elif controlnet_multimodel_inferred[key]["source"] == 'prev_frame':
      if videoFramesFolder not in fileDatasetsByPath.keys():
        fileDatasetsByPath[videoFramesFolder] = FrameDataset(videoFramesFolder, f'{key}_source', '' )
      controlnet_multimodel_inferred[key]["source"] = fileDatasetsByPath[videoFramesFolder]
      controlnet_multimodel_inferred[key]["source_raw"] = 'prev_frame'

    elif controlnet_multimodel_inferred[key]["source"] == 'cond_video':
      if condVideoFramesFolder not in fileDatasetsByPath.keys():
        fileDatasetsByPath[condVideoFramesFolder] = FrameDataset(condVideoFramesFolder, f'{key}_source', '' )
      controlnet_multimodel_inferred[key]["source"] = fileDatasetsByPath[condVideoFramesFolder]

    elif controlnet_multimodel_inferred[key]["source"] == 'color_video':
      if colorVideoFramesFolder not in fileDatasetsByPath.keys():
        fileDatasetsByPath[colorVideoFramesFolder] = FrameDataset(colorVideoFramesFolder, f'{key}_source', '' )
      controlnet_multimodel_inferred[key]["source"] = fileDatasetsByPath[colorVideoFramesFolder]

    elif controlnet_multimodel_inferred[key]["source"] not in ['raw_frame', 'stylized','prev_frame']:
      if controlnet_multimodel_inferred[key]["source"] not in fileDatasetsByPath.keys():
        fileDatasetsByPath[controlnet_multimodel_inferred[key]["source"]] = FrameDataset(controlnet_multimodel_inferred[key]["source"], f'{key}_source', '')
      controlnet_multimodel_inferred[key]["source"] = fileDatasetsByPath[controlnet_multimodel_inferred[key]["source"]]

    if controlnet_multimodel_inferred[key]["mode"] == 'balanced':
      controlnet_multimodel_inferred[key]["layer_weights"] = [1]*13
      controlnet_multimodel_inferred[key]["zero_uncond"] = False
    elif controlnet_multimodel_inferred[key]["mode"] == 'controlnet':
      controlnet_multimodel_inferred[key]["layer_weights"] =  [(0.825 ** float(12 - i)) for i in range(13)]
      controlnet_multimodel_inferred[key]["zero_uncond"] = True
    elif controlnet_multimodel_inferred[key]["mode"] == 'prompt':
      controlnet_multimodel_inferred[key]["layer_weights"] = [(0.825 ** float(12 - i)) for i in range(13)]
      controlnet_multimodel_inferred[key]["zero_uncond"] = False

  def get_control_source_images(frame_num, controlnet_multimodel_inferred, stylized_image):
    controlnet_sources = {}
    for key in controlnet_multimodel_inferred.keys():
      control_source = controlnet_multimodel_inferred[key]['source']
      if control_source == 'stylized':
        controlnet_sources[key] = stylized_image
      elif isinstance(control_source, FrameDataset):
        if controlnet_multimodel_inferred[key].get('source_raw',0) == 'prev_frame':
          controlnet_sources[key] = control_source[frame_num-1]
        else:
          controlnet_sources[key] = control_source[frame_num] #for raw, cond, color videos
    return controlnet_sources

  def get_ref_source_image(frame_num):
              init_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
              stylized_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num:06}.{save_img_format}'
              ref_image = None
              if reference_source == 'init':
                ref_image = init_image
              if reference_source == 'stylized':
                ref_image = stylized_image
              if reference_source == 'prev_frame':
                  ref_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.{save_img_format}'
              if reference_source == 'color_video':
                  if os.path.exists(f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'):
                    ref_image = f'{colorVideoFramesFolder}/{frame_num+1:06}.jpg'
                  elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                    ref_image = f'{colorVideoFramesFolder}/{1:06}.jpg'
                  else:
                    raise Exception("Reference mode specified with no color video or image. Please specify color video or disable the shuffle model")
              return ref_image

  image_prompts = {}
  controlnet_multimodel_temp = {}
  for key in controlnet_multimodel.keys():

    weight = controlnet_multimodel[key]["weight"]
    if weight !=0 :
      controlnet_multimodel_temp[key] = controlnet_multimodel[key]
  controlnet_multimodel = controlnet_multimodel_temp

  inverse_mask_order = False
  try:
    import xformers.ops
    xformers_available = True
  except:
    xformers_available = False
  can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(getattr(torch.nn.functional, "scaled_dot_product_attention")) # not everyone has torch 2.x to use sdp
  if can_use_sdp and not xformers_available:
    shared.opts.xformers = False
    shared.cmd_opts.xformers = False
  else:
    shared.opts.xformers = True
    shared.cmd_opts.xformers = True

  import copy
  apply_depth = None;
  apply_canny = None; apply_mlsd = None;
  apply_hed = None; apply_openpose = None;
  apply_seg = None;
  #loaded_controlnets = {}
  torch.cuda.empty_cache(); gc.collect();
  sd_model.control_scales = ([1]*13)
  from modules.controlmodel_ipadapter import PlugableIPAdapter, clear_all_ip_adapter

  if 'control_multi' in model_version:
    try:
      sd_model.control_model.cpu()
    except: pass
    print('Checking downloaded Annotator and ControlNet Models')
    for controlnet in controlnet_multimodel.keys():
      controlnet_settings = controlnet_multimodel[controlnet]
      weight = controlnet_settings["weight"]
      if weight!=0 and not skip_diffuse_cell:
        small_url = control_model_urls[controlnet]
        if controlnet in control_model_filenames.keys():
          local_filename = control_model_filenames[controlnet]
        else: local_filename = small_url.split('/')[-1]
        print(f"Loading {controlnet} from checkpoint: {local_filename}")
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

        """download annotators"""
        if controlnet in control_anno_urls.keys():
          for anno_url in control_anno_urls[controlnet]:
            anno_path = f'{root_dir}/ControlNet/annotator/ckpts'
            anno_fname = anno_url.split('/')[-1]
            os.makedirs(anno_path, exist_ok=True)
            anno_path = os.path.join(anno_path, anno_fname)
            if not os.path.exists(anno_path) or force_download:
              try:
                pathlib.Path(anno_path).unlink()
              except: pass
              print(f'Downloading {anno_fname} annotator for the {controlnet} model... ')
              wget.download(anno_url,  anno_path)
              print(f'Downloaded {anno_fname} annotator for the {controlnet} model... ')

        """download faceid loras"""
        if controlnet in ipadapter_face_loras.keys():
          for lora_url in ipadapter_face_loras[controlnet]:
            lora_dir = lora_dir
            lora_fname = lora_url.split('/')[-1]
            os.makedirs(lora_dir, exist_ok=True)
            lora_path = os.path.join(lora_dir, lora_fname)
            if not os.path.exists(lora_path) or force_download:
              try:
                pathlib.Path(lora_path).unlink()
              except: pass
              print(f'Downloading {lora_fname} lora for the {controlnet} model... ')
              wget.download(lora_url,  lora_path)
              print(f'Downloaded {lora_fname} lora for the {controlnet} model... ')

    print('Loading ControlNet Models')
    try:
      to_pop = set(loaded_controlnets.keys()).symmetric_difference(set( controlnet_multimodel.keys()))
      for key in to_pop:
        if key in loaded_controlnets.keys():
          loaded_controlnets.pop(key)
    except NameError:
      loaded_controlnets = {}

    for controlnet in controlnet_multimodel.keys():
      controlnet_settings = controlnet_multimodel[controlnet]
      weight = controlnet_settings["weight"]
      print('controlnet', controlnet)
      if weight!=0  and not skip_diffuse_cell:
        if controlnet in loaded_controlnets.keys():
          continue
        small_url = control_model_urls[controlnet]
        if controlnet in control_model_filenames.keys():
          local_filename = control_model_filenames[controlnet]
        else: local_filename = small_url.split('/')[-1]
        small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
        if 'ipadapter' in controlnet:
          print('Loading ipadapter')
          adapter_weights = load_ipadapter_model(small_controlnet_model_path)
          loaded_controlnets[controlnet] = PlugableIPAdapter(adapter_weights, is_v2='v2' in small_controlnet_model_path)
          loaded_controlnets[controlnet].half().cuda()
          del adapter_weights

          continue
        if model_version in ['control_multi_sdxl','control_multi_animatediff_sdxl']:
          # from IPython.utils import io
          # with io.capture_output(stderr=False) as captured:
          cn =  load_controlnet(small_controlnet_model_path)
          if type(cn) == comfy.sd.ControlLora:
            cn.pre_run(sd_model.model, lambda a: model_wrap.sigma_to_t(model_wrap.t_to_sigma(torch.tensor(a) * 999.0)))
          loaded_controlnets[controlnet] = cn.control_model.cpu().half()
        if model_version in ['control_multi', 'control_multi_v2','control_multi_v2_768', 'control_multi_animatediff']:
          loaded_controlnets[controlnet] = copy.deepcopy(sd_model.control_model)
          if os.path.exists(small_controlnet_model_path):
            if controlnet in ['control_sd15_gif', 'control_sd15_depth_anything']:
              cn =  load_controlnet(small_controlnet_model_path)
              loaded_controlnets[controlnet] = cn.control_model.cpu().half()
            else:
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
  # print('loaded_controlnets', loaded_controlnets)
  clip_vit_h = None
  clip_vit_g = None
  clip_model_dir = f"{controlnet_models_dir}/clip_vision"
  os.makedirs(clip_model_dir, exist_ok=True)
  sys.path.append(f'{root_dir}/ComfyUI')
  import comfy
  import comfy.clip_vision
  comfy.clip_vision.clip_preprocess = clip_preprocess
  from nodes import LoadImage
  img_loader_node = LoadImage()
  if not skip_diffuse_cell:
  # print('Loading annotators.')
    controlnet_keys = controlnet_multimodel.keys() if 'control_multi' in model_version else model_version
    preprocess_clip_vit_h = set([
          "ipadapter_sd15",
          "ipadapter_sd15_light",
          "ipadapter_sd15_plus",
          "ipadapter_sd15_plus_face",
          "ipadapter_sd15_full_face",
          "ipadapter_sdxl_vit_h",
          "ipadapter_sdxl_plus_vit_h",
          "ipadapter_sdxl_plus_face_vit_h",
          "ipadapter_sd15_faceid_plus",
          "ipadapter_sd15_faceid_plus_v2"
          ])
    if len(preprocess_clip_vit_h.intersection(set(controlnet_keys)))>0:
      clip_model_name = 'clip_vision_vit_h'
      clip_model_filename = model_filenames[clip_model_name]
      clip_url = model_urls[clip_model_name]
      clip_model_path = os.path.join(clip_model_dir, clip_model_filename)
      if not os.path.exists(clip_model_path):
        print(f'Downloading {clip_model_name} model... ')
        wget.download(clip_url,  clip_model_path)
        print(f'Downloaded {clip_model_name} model.')
      clip_vit_h = load_clip_vision(clip_model_path)
      clip_vit_h.encode_image = clip_vision_encode.__get__(clip_vit_h, clip_vit_h.__class__)

    preprocess_clip_vit_g = set(["ipadapter_sd15_vit_G","ipadapter_sdxl"])
    if len(preprocess_clip_vit_g.intersection(set(controlnet_keys)))>0:
      clip_model_name = 'clip_vision_vit_bigg'
      clip_model_filename = model_filenames[clip_model_name]
      clip_url = model_urls[clip_model_name]
      clip_model_path = os.path.join(clip_model_dir, clip_model_filename)
      if not os.path.exists(clip_model_path):
        print(f'Downloading {clip_model_name} model... ')
        wget.download(clip_url,  clip_model_path)
        print(f'Downloaded {clip_model_name} model.')
      clip_vit_g = load_clip_vision(clip_model_path)
      clip_vit_g.encode_image = clip_vision_encode.__get__(clip_vit_g, clip_vit_g.__class__)

    depth_cns = set(["control_sd21_depth", 'control_sd15_depth','control_sd15_normal', 'control_sd15_depth_anything',
                    'control_sdxl_depth', 'control_sdxl_lora_128_depth', 'control_sdxl_lora_256_depth', "control_sd15_temporal_depth"])
    if len(depth_cns.intersection(set(controlnet_keys)))>0:
            if control_sd15_depth_detector == 'Midas' or "control_sd15_normal" in controlnet_keys:
              from annotator.midas import MidasDetector
              apply_depth = MidasDetector()
              print('Loaded MidasDetector')
            elif control_sd15_depth_detector == 'Zoe':
              from annotator.zoe import ZoeDetector
              apply_depth = ZoeDetector()
              print('Loaded ZoeDetector')
            elif control_sd15_depth_detector == 'depth_anything':
              depth_anything_path = f"{root_dir}/ControlNet/annotator/ckpts/depth_anything_vitl14.pth"
              if not os.path.exists(depth_anything_path):
                depth_anything_url = 'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth'

                print('Downloading depth anything annotator checkpoint...')
                wget.download(depth_anything_url, depth_anything_path)
                print('Downloaded depth anything annotator checkpoint.')
              sys.path.append(f'{root_dir}/Depth-Anything')
              print("subprocess pathhh")
              result = subprocess.run(['ls'], capture_output=True, text=True)
              print(result.stdout)
              if result.returncode != 0:
                  print("Error:", result.stderr)
              import os
              print("OSSSSS PATH")
              print(os.getcwd())
              import sys
              sys.path.append(os.getcwd()) 
              from annotator.depth_anything import DepthAnythingDetector
              apply_depth = DepthAnythingDetector(ckpt=depth_anything_path)
              apply_depth.device = 'cuda'

    normalbae_cns = set(["control_sd15_normalbae", "control_sd21_normalbae"])
    if len(normalbae_cns.intersection(set(controlnet_keys)))>0:
            print("subprocess pathhh")
            result = subprocess.run(['ls'], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("Error:", result.stderr)
            import os
            print("OSSSSS PATH")
            print(os.getcwd())
            import sys
            sys.path.append(os.getcwd()) 
            from annotator.normalbae import NormalBaeDetector
            apply_normal = NormalBaeDetector()
            print('Loaded NormalBaeDetector')

    canny_cns = set(['control_sd15_canny','control_sdxl_canny',
                    'control_sdxl_lora_128_canny', 'control_sdxl_lora_256_canny'])
    if len(canny_cns.intersection(set(controlnet_keys)))>0:
            print("subprocess pathhh")
            result = subprocess.run(['ls'], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("Error:", result.stderr)
            import os
            print("OSSSSS PATH")
            print(os.getcwd())
            import sys
            sys.path.append(os.getcwd()) 
            from annotator.canny import CannyDetector
            apply_canny = CannyDetector()
            print('Loaded CannyDetector')

    faceid_adapters = set(["ipadapter_sd15_faceid",
    "ipadapter_sd15_faceid_plus",
    "ipadapter_sd15_faceid_plus_v2",
    "ipadapter_sdxl_faceid"])
    if len(faceid_adapters.intersection(set(controlnet_keys)))>0:
      g_insight_face_model = InsightFaceModel()
      g_insight_face_model.load_model()

    softedge_cns = set(["control_sd21_softedge", 'control_sd15_softedge', 'control_sdxl_softedge',
                        'control_sdxl_lora_128_softedge', 'control_sdxl_lora_256_softedge',"control_sd15_inpaint_softedge"])
    if len(softedge_cns.intersection(set(controlnet_keys)))>0:
            if control_sd15_softedge_detector == 'HED':
              from annotator.hed import HEDdetector
              apply_softedge = HEDdetector()
              print('Loaded HEDdetector')
            if control_sd15_softedge_detector == 'PIDI':
              annotator_path = f"{root_dir}/ControlNet/annotator/ckpts/table5_pidinet.pth"
              if not os.path.exists(annotator_path):
                annotator_url = 'https://github.com/Sxela/ControlNet-v1-1-nightly/releases/download/v0.1.0/table5_pidinet.pth'
                print(f'Downloading {control_sd15_softedge_detector} annotator checkpoint...')
                wget.download(annotator_url, annotator_path)
                print(f'Downloaded {control_sd15_softedge_detector} annotator checkpoint.')
              from annotator.pidinet import PidiNetDetector
              apply_softedge = PidiNetDetector()
              print('Loaded PidiNetDetector')
    scribble_cns = set(['control_sd15_scribble', "control_sd21_scribble"])
    if len(scribble_cns.intersection(set(controlnet_keys)))>0:
            from annotator.util import nms
            if control_sd15_scribble_detector == 'HED':
              from annotator.hed import HEDdetector
              apply_scribble = HEDdetector()
              print('Loaded HEDdetector')
            if control_sd15_scribble_detector == 'PIDI':
              annotator_path = f"{root_dir}/ControlNet/annotator/ckpts/table5_pidinet.pth"
              if not os.path.exists(annotator_path):
                annotator_url = 'https://github.com/Sxela/ControlNet-v1-1-nightly/releases/download/v0.1.0/table5_pidinet.pth'
                print(f'Downloading {control_sd15_softedge_detector} annotator checkpoint...')
                wget.download(annotator_url, annotator_path)
                print(f'Downloaded {control_sd15_softedge_detector} annotator checkpoint.')
              from annotator.pidinet import PidiNetDetector
              apply_scribble = PidiNetDetector()
              print('Loaded PidiNetDetector')

    if "control_sd15_mlsd" in controlnet_keys:
            from annotator.mlsd import MLSDdetector
            apply_mlsd = MLSDdetector()
            print('Loaded MLSDdetector')

    openpose_cns = set(["control_sd15_openpose", "control_sdxl_openpose",  "control_sd21_openpose"])
    if len(openpose_cns.intersection(set(controlnet_keys)))>0:
      if pose_detector == 'openpose':
            from annotator.openpose import OpenposeDetector
            apply_openpose = OpenposeDetector()
            print('Loaded OpenposeDetector')
      elif pose_detector == 'dw_pose':
        import gdown
        if not os.path.exists(f"{root_dir}/ControlNet/annotator/ckpts/dw-ll_ucoco_384.onnx"):
          gdown.download(id='12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2', output=f"{root_dir}/ControlNet/annotator/ckpts/dw-ll_ucoco_384.onnx")
        if not os.path.exists(f"{root_dir}/ControlNet/annotator/ckpts/yolox_l.onnx"):
          gdown.download(id='1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI', output=f"{root_dir}/ControlNet/annotator/ckpts/yolox_l.onnx")
        os.chdir(f"{root_dir}/ControlNet")
        from annotator.dwpose import DWposeDetector
        apply_openpose = DWposeDetector()
        print('Loaded DWposeDetector')
        os.chdir(root_dir)

    seg_cns = set(["control_sd15_seg", "control_sdxl_seg",  "control_sd21_seg"])
    if len(seg_cns.intersection(set(controlnet_keys)))>0:
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

    lineart_cns = set(["control_sd15_lineart",  "control_sd21_lineart"])
    if len(lineart_cns.intersection(set(controlnet_keys)))>0:
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

  # unload_network()
  sd_model.cuda()
  sd_hijack.model_hijack.hijack(sd_model)
  sd_hijack.model_hijack.embedding_db.add_embedding_dir(custom_embed_dir)
  if 'sdxl' not in model_version:
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(sd_model, force_reload=True)

  latent_scale_schedule_bkup = copy.copy(latent_scale_schedule)
  init_scale_schedule_bkup = copy.copy(init_scale_schedule)
  steps_schedule_bkup = copy.copy(steps_schedule)
  style_strength_schedule_bkup = copy.copy(style_strength_schedule)
  flow_blend_schedule_bkup = copy.copy(flow_blend_schedule)
  cfg_scale_schedule_bkup = copy.copy(cfg_scale_schedule)
  image_scale_schedule_bkup = copy.copy(image_scale_schedule)
  cc_masked_diffusion_schedule_bkup = copy.copy(cc_masked_diffusion_schedule)

  missed_consistency_schedule_bkup = copy.copy(missed_consistency_schedule)
  overshoot_consistency_schedule_bkup = copy.copy(overshoot_consistency_schedule)
  edges_consistency_schedule_bkup = copy.copy(edges_consistency_schedule)
  consistency_blur_schedule_bkup = copy.copy(consistency_blur_schedule)
  consistency_dilate_schedule_bkup = copy.copy(consistency_dilate_schedule)
  soften_consistency_schedule_bkup = copy.copy(soften_consistency_schedule)

  if make_schedules:
    if diff is None and diff_override == []: sys.exit(f'\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous cell, run it, and then run this cell again\n')
    if diff_override != []: diff = diff_override

    print('Applied schedules:')
    latent_scale_schedule = check_and_adjust_sched(latent_scale_schedule, latent_scale_template, diff, respect_sched)
    init_scale_schedule = check_and_adjust_sched(init_scale_schedule, init_scale_template, diff, respect_sched)
    steps_schedule = check_and_adjust_sched(steps_schedule, steps_template, diff, respect_sched)
    style_strength_schedule = check_and_adjust_sched(style_strength_schedule, style_strength_template, diff, respect_sched)
    flow_blend_schedule = check_and_adjust_sched(flow_blend_schedule, flow_blend_template, diff, respect_sched)
    cc_masked_diffusion_schedule = check_and_adjust_sched(flow_blend_schedule, cc_masked_template, diff, respect_sched)

    cfg_scale_schedule = check_and_adjust_sched(cfg_scale_schedule, cfg_scale_template, diff, respect_sched)
    image_scale_schedule = check_and_adjust_sched(image_scale_schedule, cfg_scale_template, diff, respect_sched)
    for sched, name in zip([cc_masked_diffusion_schedule, latent_scale_schedule,   init_scale_schedule,  steps_schedule,  style_strength_schedule,  flow_blend_schedule,
    cfg_scale_schedule, image_scale_schedule], ['cc_masked_diffusion_schedule','latent_scale_schedule',   'init_scale_schedule',  'steps_schedule',  'style_strength_schedule',  'flow_blend_schedule',
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

  image_prompts = {}
  sd_model.normalize_weights = normalize_cn_weights
  sd_model.low_vram = True if controlnet_low_vram else False

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
    max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))-1

  def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
      prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series

  key_frames = True
  # interp_spline = 'Linear'
  # perlin_init = False
  # perlin_mode = 'mixed'

  if warp_towards_init != 'off':
    if flow_lq:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
          # raft_model = torch.nn.DataParallel(RAFT(args2))
    else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

  os.makedirs(f'{root_dir}/logs', exist_ok=True)
  def printf(*msg, file=f'{root_dir}/logs/log.txt'):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    name_string = str(batchNum)
    log_name = ".".join(file.split('.')[:-1])+'_'+name_string+'.'+file.split('.')[-1]
    with open(log_name, 'a') as f:
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
          old_file = old_folder + f'/{batch_name}({batchNum})_{i:06}.{save_img_format}'
          new_file = new_folder + f'/{batch_name}({batchNum})_{i:06}.{save_img_format}'
          os.rename(old_file, new_file)

  noise_upscale_ratio = int(noise_upscale_ratio)

  if animation_mode == 'Video Input':
    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)==0:
      sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
    flows = glob(flo_folder+'/*.*')
    # if (len(flows)==0) and flow_warp:
    #   sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

  if set_seed == 'random_seed' or set_seed == -1:
      random.seed()
      seed = random.randint(0, 2**32)
      # print(f'Using seed: {seed}')
  else:
      seed = int(set_seed)

  new_prompt_loras = {}
  prompt_weights = {}
  prompt_uweights = {}
  if text_prompts:
    _, new_prompt_loras = split_lora_from_prompts(text_prompts)
    if new_prompt_loras != {}:
      print('Inferred loras schedule:\n', new_prompt_loras)
    _, prompt_weights = get_prompt_weights(text_prompts)

    if len(prompt_weights)>1:
      print('---prompt_weights---', prompt_weights, text_prompts)
  if negative_prompts:
    _, prompt_uweights = get_prompt_weights(negative_prompts)
  if new_prompt_loras not in [{}, [], '', None]:# and model_version not in ['sdxl_base', 'sdxl_refiner']:
  #inject lora even with empty weights to unload?
    inject_network(sd_model)
  else:
    loaded_networks.clear()

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
      # 'interp_spline': interp_spline,
      'start_frame': start_frame,
      'padding_mode': padding_mode,
      'text_prompts': text_prompts,
      'image_prompts': image_prompts,
      'intermediate_saves': intermediate_saves,
      'intermediates_in_subfolder': intermediates_in_subfolder,
      'steps_per_checkpoint': steps_per_checkpoint,
      # 'perlin_init': perlin_init,
      # 'perlin_mode': perlin_mode,
      'set_seed': set_seed,
      'clamp_grad': clamp_grad,
      'clamp_max': clamp_max,
      'skip_augs': skip_augs,
  }
  if frame_range not in [None, [0,0], '', [0], 0]:
    args['start_frame'] = max(frame_range[0], start_frame)
    args['max_frames'] = min(args['max_frames'],frame_range[1])


  """safeguard batch/context settings"""
  # print('batch_length, batch_overlap, context_overlap')
  # print(batch_length, batch_overlap, context_overlap)
  batch_length_bkup = batch_length
  if batch_length_bkup == -1:
      batch_length = args['max_frames'] - args['start_frame']
  batch_length = min(batch_length, args['max_frames'] - args['start_frame']+1)
  batch_overlap = max(0, min(batch_overlap, batch_length-1 ))
  context_overlap = max(0, min(context_overlap, context_length-1 ))
  # print(batch_length, batch_overlap, context_overlap)
  total_batch_length = batch_length

  args = SimpleNamespace(**args)

  import traceback

  gc.collect()
  torch.cuda.empty_cache()
  try:
    # if only_preview_controlnet:
    if render_mode == 'skip render, preview controlnet':
      if 'control_multi' in model_version:
        models = list(controlnet_multimodel.keys())
        models = [o for o in models if o not in no_preprocess_cn or 'qr' in o]; print(models)
        controlnet_sources = {}
        if controlnet_multimodel != {}:

          for i in trange(max(0, frame_range[0]), max_frames if frame_range[1] == 0 else min(frame_range[1],max_frames)):
            init_image = glob(videoFramesFolder+'/*.*')[i]
            alpha = glob(videoFramesAlpha+'/*.*')[i]
            W, H = width_height
            controlnet_sources = get_control_source_images(i, controlnet_multimodel_inferred, stylized_image=init_image)
            controlnet_sources['control_inpainting_mask'] = init_image
            controlnet_sources['shuffle_source'] = init_image
            controlnet_sources['init_image'] = init_image
            controlnet_sources['prev_frame'] = init_image
            controlnet_sources['next_frame'] = f'{videoFramesFolder}/{i+1:06}.jpg'
            detected_maps, models, _ = get_controlnet_annotations(model_version, W, H, models, controlnet_sources)
            for m in models:
              if save_controlnet_annotations:
                PIL.Image.fromarray(detected_maps[m].astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_{m}_{i:06}.jpg', quality=95)
            # detected_maps[m] = postprocess_map(detected_maps[m])
            gc.collect()
            torch.cuda.empty_cache()
            imgs = [fit(add_text_below_pil_image(PIL.Image.fromarray(detected_maps[m].astype('uint8')),m), maxsize=display_size) for m in models]

            if use_background_mask and os.path.exists(alpha):
              imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.open(alpha).convert('L'),'backgeound_mask'), maxsize=display_size))
            if isinstance(init_image, str):
              imgs.insert(0,fit(add_text_below_pil_image(PIL.Image.open(init_image),'init_image'), maxsize=display_size))
            if stack_previews:
              if hstack_previews:
                imgs = hstack(imgs)
              else:
                imgs = vstack(imgs)
              if fit_previews: imgs = fit(imgs, maxsize=display_size)
              # display.display(imgs)
            else:
              for img in imgs:
                 pass
                # display.display(img)
            # for m in models:
            #   display.display()
    elif not skip_diffuse_cell:
        total_frames = args.max_frames
        W, H = width_height
        if scenes in [[], None, [[0,0]],[[]]]:
          scenes = [[args.start_frame, args.max_frames]]
        filtered_scenes = []
        for scene in scenes:
          scene_start, scene_end = scene
          if scene_end == -1: scene_end = args.max_frames
          scene_start = max(args.start_frame, scene_start)
          scene_end = min(scene_end, args.max_frames)
          if scene_end-scene_start <= 0: continue
          filtered_scenes.append([scene_start, scene_end])

        for scene in filtered_scenes:
          if 'animatediff' in model_version:
              #reset batch length to max size each scene if == -1
              if batch_length_bkup == -1:
                batch_length = scene_end-scene_start+1
              else:
                batch_length = min(batch_length_bkup, scene_end-scene_start+1)
              # batch_length = min(total_batch_length, scene_end-scene_start+1) #make sure to not overflow with -1 batch size
              do_run_adiff(args)
          else:
            scene_start, scene_end = scene
            args.start_frame = scene_start
            scene_end = min(scene_end+1, total_frames)
            args.max_frames = scene_end
            # frame_range = [scene_start, scene_end]
            # print('scene_start, scene_end, frame_range', scene_start, scene_end, frame_range)
            # print(frame_range, args.max_frames, args.start_frame)
            do_run()
  except KeyboardInterrupt:
    break
  except:
    try:
      sd_model.cpu()
      if 'control' in model_version:
        for key in loaded_controlnets.keys():
          loaded_controlnets[key].cpu()
        torch.cuda.empty_cache()
      gc.collect()
    except: pass
    traceback.print_exc()

  if len(settings_queue)>0:
    settings_queue.pop(0)
    # queue_box.value = stringify_settings_queue(settings_queue)


print('n_stats_avg (mean, std): ', n_mean_avg, n_std_avg)
thread_pool.close()
thread_pool.join()
gc.collect()
torch.cuda.empty_cache()
executed_cells[cell_name] = True

"""# 5. Create the video"""

import PIL
#@title ### **Create video**
#@markdown Video file will save in the same folder as your images.
cell_name = 'create_video'
# check_execution(cell_name)

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
if   (blend_mode == "optical flow") and 'animatediff' in model_version:
  print('Disabling optical flow for animatediff mode.')
  blend_mode = 'None'
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
  os.makedirs(f'{root_dir}/Real-ESRGAN', exist_ok=True)
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
  upscaler_model_path = os.path.join('weights', up_model_name + '.pth')
  if not os.path.isfile(upscaler_model_path):
          ROOT_DIR = root_dir
          for url in file_url:
              # model_path will be updated
              upscaler_model_path = load_file_from_url(
                  url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

  dni_weight = None

  upsampler = RealESRGANer(
          scale=netscale,
          model_path=upscaler_model_path,
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
# if platform.system() != 'Linux' and use_deflicker:
#    use_deflicker = False
#    print('Disabling ffmpeg deflicker filter for windows install, as it is causing a crash.')
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

  try:
    batchNum
  except:
    batchNum = None

  """check we have frames in the latest run"""
  if batchNum is not None:
    frames_check = len(sorted(glob(batchFolder+f"/{folder}({batchNum})_*.{save_img_format}")))
  else:
    frames_check = 0

  if frames_check<=1:
    renders = glob(f"{batchFolder}/*00.{save_img_format}")
    renders_s = sorted([int(o.split('(')[-1].split(')')[0]) for o in renders])
    print(f'Could not find latest run "{batchNum}", setting lates run to "{max(renders_s)}"')
    batchNum = max(renders_s)
    latest_run = batchNum

  run =  latest_run#@param
  final_frame = 'final_frame'


  #@markdown This is the frame where the video will start
  init_frame = 1#@param {type:"number"}
  #@markdown You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

  last_frame = final_frame#@param {type:"number"}
  #@markdown Export fps. Leave as -1 to get fps from your init video divided by nth frame, and to keep video duration the same.
  fps = -1#@param {type:"number"}
  if fps == -1:
    if 'extract_nth_frame' in globals().keys():
      if 'detected_fps' in globals().keys():
        fps = detected_fps/extract_nth_frame
      elif 'video_init_path' in globals().keys():
        fps = get_fps(video_init_path)/extract_nth_frame

    assert fps != -1, 'please specify a valid FPS value > 0'
    print(f'Using detected fps of {fps}')
  output_format = 'h264_mp4' #@param ['h264_mp4','qtrle_mov','prores_mov']

  if last_frame == 'final_frame':
    last_frame = len(glob(batchFolder+f"/{folder}({run})_*.{save_img_format}"))
    print(f'Total frames: {last_frame}')

  video_out = batchFolder+f"/video"
  os.makedirs(video_out, exist_ok=True)
  image_path = f"{outDirPath}/{folder}/{folder}({run})_%06d.{save_img_format}"
  filepath = f"{video_out}/{folder}({run})_{'_noblend'}.{output_format.split('_')[-1]}"

  if upscale_ratio>1:
      postfix+=f'_x{upscale_ratio}_{upscale_model}'
  if use_deflicker:
      postfix+='_dfl'

  if use_background_mask_video and blend_mode == 'None':
    """if we have masked video, use linear blend with blend = 1, which mimics blend = None, but applies mask"""
    blend_mode = 'linear'
    blend = 1

  """flow blend mode"""

  if (blend_mode == 'optical flow') & (True) :
    image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%06d.{save_img_format}"
    postfix += '_flow'

    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format.split('_')[-1]}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/flow/{folder}({run})_*.{save_img_format}"))
    flo_out = batchFolder+f"/flow"

    os.makedirs(flo_out, exist_ok=True)

    frames_in = sorted(glob(batchFolder+f"/{folder}({run})_*.{save_img_format}"))
    assert len(frames_in)>1, 'Less than 1 frame found in the specified run, make sure you have specified correct batch name and run number.'
    flow_in = glob(flo_folder+f"/*jpg.npy")
    cc_in = glob(flo_folder+f"/*.jpg-21_cc.jpg")
    print(flo_folder, len(flow_in), len(frames_in)-1, len(cc_in))
    if len(flow_in) < len(frames_in)-1 or (check_consistency and len(cc_in) < len(frames_in)-1):

      print('Not enough flow files found. Will try to recreate flow now. If it fails, please force recreate flow or swith to other blend modes.')
      # force_generate_flow()
    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    if upscale_ratio>1:
          frame0 = np.array(frame0)[...,::-1]
          output, _ = upsampler.enhance(frame0, outscale=upscale_ratio)
          frame0 = PIL.Image.fromarray((output)[...,::-1].astype('uint8'))
    # frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])
    frame0.save(batchFolder+f"/flow/{folder}({run})_{0:06}.{save_img_format}")
    def process_flow_frame(i):

        frame1_path = frames_in[i-1]
        frame2_path = frames_in[i]
        # print(i, frame1_path, frame2_path)

        frame1 = Image.open(frame1_path) # 000000.png
        frame2 = Image.open(frame2_path) # 000001.png (for i == 1)
        frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):06}.jpg" # 000001.jpg
        flo_path = f"{flo_folder}/{frame1_stem}.npy"
        weights_path = None
        if check_consistency:
          if reverse_cc_order:
            weights_path = f"{flo_folder}/{frame1_stem}-21_cc.jpg"
          else:
            weights_path = f"{flo_folder}/{frame1_stem}_12-21_cc.jpg"
        tic = time.time()
        printf('process_flow_frame warp')
        # to warp frame 000000 -> 000001 we need to calc flow between raw frames 000001 and 000002
        frame1_init = f'{videoFramesFolder}/{i:06}.jpg'
        frame2_init = f'{videoFramesFolder}/{i+1:06}.jpg'
        flow21, forward_weights = get_flow_and_cc(frame1_init, frame2_init, flo_path,
                                                          cc_path=weights_path)
        frame = warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path,
            pad_pct=padding_ratio, padding_mode=padding_mode, inpaint_blend=0, video_mode=True)
        if use_background_mask_video:
          frame = apply_mask(frame, i, background_video, background_source_video, invert_mask_video)
        if upscale_ratio>1:
          frame = np.array(frame)[...,::-1]
          output, _ = upsampler.enhance(frame.clip(0,255), outscale=upscale_ratio)
          frame = PIL.Image.fromarray((output)[...,::-1].clip(0,255).astype('uint8'))
        frame.save(batchFolder+f"/flow/{folder}({run})_{i:06}.{save_img_format}")

    with Pool(threads) as p, Pool(flow_threads) as thread_pool:
      fn = partial(try_process_frame, func=process_flow_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))

  if blend_mode == 'linear':
    image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%06d.{save_img_format}"
    postfix += '_blend'

    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format.split('_')[-1]}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/blend/{folder}({run})_*.{save_img_format}"))
    blend_out = batchFolder+f"/blend"
    os.makedirs(blend_out, exist_ok = True)
    frames_in = glob(batchFolder+f"/{folder}({run})_*.{save_img_format}")

    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    if upscale_ratio>1:
          frame0 = np.array(frame0)[...,::-1]
          output, _ = upsampler.enhance(frame0.clip(0,255), outscale=upscale_ratio)
          frame0 = PIL.Image.fromarray((output)[...,::-1].clip(0,255).astype('uint8'))
    # frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])
    frame0.save(batchFolder+f"/blend/{folder}({run})_{0:06}.{save_img_format}")

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
      frame.save(batchFolder+f"/blend/{folder}({run})_{i:06}.{save_img_format}")

    with Pool(threads) as p:
      fn = partial(try_process_frame, func=process_blend_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))
  deflicker_str = ''
  input_format = 'png' if save_img_format == 'png' else 'mjpeg'
  if output_format == 'h264_mp4':
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec',
        input_format,
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
  if output_format == 'qtrle_mov':
      cmd = [
      'ffmpeg',
      '-y',
      '-vcodec',
      input_format,
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
  if output_format == 'prores_mov':
      cmd = [
      'ffmpeg',
      '-y',
      '-vcodec',
      input_format,
      '-r',
      str(fps),
      '-start_number',
      str(init_frame),
      '-i',
      image_path,
      '-frames:v',
      str(last_frame+1),
      '-c:v',
      'prores_aw',
      '-profile:v',
      '2',
      '-pix_fmt',
      'yuv422p10',
      '-vf',
      f'fps={fps}'
  ]
  if use_deflicker:
    cmd+=['-vf','deflicker=mode=pm:size=10']
  cmd+=[filepath]
  experimental_deflicker = False #@param {'type':'boolean'}

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
  if True:
    f_deflicker = filepath[:-4]+'_deflicker'+filepath[-4:]
    cmd_d=['ffmpeg', '-y','-fflags', '+genpts', '-i', filepath, '-fflags', '+genpts', '-i', filepath,
          '-filter_complex', "[0:v]setpts=PTS-STARTPTS[top]; [1:v]setpts=PTS-STARTPTS+.033/TB, format=yuva420p, colorchannelmixer=aa=0.5[bottom]; [top][bottom]overlay=shortest=1",
           f_deflicker]

    if os.path.exists(filepath):
      process = subprocess.Popen(cmd_d, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = process.communicate()
      if process.returncode != 0:
          print(stderr)
          raise RuntimeError(stderr)
      else:
          print(f"The deflickered video  is saved to {f_deflicker}")
    else: print('Error deflickering video: either init or output video don`t exist.')
    filepath = f_deflicker

  if True:
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
          pathlib.Path(filepath).unlink()

    else: print('Error adding audio from init video to output video: either init or output video don`t exist.')

save_settings(skip_save=False)
torch.cuda.empty_cache()
