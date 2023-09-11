import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp
from pathlib import Path
from tqdm import tqdm

from core.FlowFormer import build_flowformer

from utils.utils import InputPadder, forward_interpolate
import itertools

TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    # print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    # print(f"preparing image...")
    # print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    dirname = Path(fn1).parent.parent.stem
    filename = Path(fn1).stem

    viz_dir = osp.join(viz_root_dir, dirname)
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)

    viz_fn = osp.join(viz_dir, filename + '.png')

    return image1, image2, viz_fn

def build_model(args):
    print(f"building  model...")
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size):
    weights = None
    for img_pair in tqdm(img_pairs):
        fn1, fn2 = img_pair
        # print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)
        print(f"image1 shape = {image1.shape}, image2 shape = {image2.shape}")
        flow = compute_flow(model, image1, image2, weights)


        u_flow, v_flow = flow[:, :, 0], flow[:, :, 1]
        # hist, x_edges, y_edges = np.histogram2d(flow[..., 0].ravel(), flow[..., 1].ravel(), bins=(u_flow, v_flow))
        
        flow_mag = np.sqrt(u_flow ** 2 + v_flow ** 2)
        #flow_angle = np.arctan2(v_flow, u_flow)
        # convert to degrees
        #flow_angle = cv2.cartToPolar(u_flow, v_flow, angleInDegrees=True)[1]
        # accept angles from 100 to 280 and angles from 80 to 260
        #motion_mask = np.logical_or(np.logical_and(flow_angle > 100, flow_angle < 280), np.logical_and(flow_angle > 80, flow_angle < 260))
        
        flow_mean = np.mean(flow_mag)
        motion_mask = flow_mag - flow_mean
        # remove negative values
        motion_mask = np.clip(motion_mask, 0, None)
        motion_mask = motion_mask.astype(np.uint8) * 255

        # Draw the motion mask
        #cv2.imwrite(viz_fn, motion_mask.astype(np.uint8) * 255)
        
        flow_img = flow_viz.flow_to_image(flow)
        image = flow_img[:, :, [2, 1, 0]]
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("image shape = ", image.shape)
        # print("image type = ", image.dtype)
        #contour_image = np.ones_like(image)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        cv2.imwrite(viz_fn, image)

def process_sintel(sintel_dir):
    img_pairs = []
    for scene in os.listdir(sintel_dir):
        dirname = osp.join(sintel_dir, scene)
        image_list = sorted(glob(osp.join(dirname, '*.png')))
        for i in range(len(image_list)-1):
            img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs

def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img1 = osp.join(dirname, f'{idx:06}.png')
        img2 = osp.join(dirname, f'{idx+1:06}.png')
        # img1 = f'{idx:06}.png'
        # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs

def process_img_names(dirname):
    img_list = sorted(glob(osp.join(dirname, '*.png')))
    count_images = len(img_list)
    print(f"Found {count_images} images in {dirname}")
    for idx, img in enumerate(img_list):
        os.rename(img, osp.join(dirname, f'{idx+1:06}.png'))
    return count_images
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', default='sintel')
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
    parser.add_argument('--seq_dir', default='demo_data/mihoyo')
    parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=1200)    # ending index of the image sequence
    parser.add_argument('--viz_root_dir', default='viz_results')
    parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.
    parser.add_argument('--only-videoize', action='store_true')
    parser.add_argument('--small', action='store_true')

    args = parser.parse_args()

    root_dir = args.root_dir
    viz_root_dir = args.viz_root_dir

    if not args.only_videoize:

        model = build_model(args)

        if args.eval_type == 'sintel':
            img_pairs = process_sintel(args.sintel_dir)
        # Added this part to evaluate TUM dataset
        if args.eval_type == 'tum':
            args.end_idx = process_img_names(args.seq_dir)
            img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
        elif args.eval_type == 'seq':
            img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
        with torch.no_grad():
            visualize_flow(root_dir, viz_root_dir, model, img_pairs, args.keep_size)

    # read from viz_root_dir / dirname and write to a video viz_root_dir / dirname.mp4
    for dirname in Path(viz_root_dir).iterdir():
        if not dirname.is_dir():
            continue
        dirname = dirname.stem
        print(f"processing {dirname}...")
        filename = osp.join(viz_root_dir, f'{dirname}.mp4')
        print(f"writing to {filename}...")
        optical_list = sorted(glob(osp.join(viz_root_dir, dirname, '*.png')))
        img = cv2.imread(optical_list[0])
        height, width, layers = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
        for image in optical_list:
            video.write(cv2.imread(image))
        cv2.destroyAllWindows()
        video.release()