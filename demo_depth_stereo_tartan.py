import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid_slam.droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream_depth_stereo(datapath, use_depth=False, stride=1):
    """ image generator """

    image_list_l = sorted(glob.glob(os.path.join(datapath, 'image_left', '*.png')))[::stride]
    image_list_r = sorted(glob.glob(os.path.join(datapath, 'image_right', '*.png')))[::stride]
    
    depth_left_list = sorted(glob.glob(os.path.join(datapath, 'depth_left', '*.npy')))[::stride]
    depth_right_list = sorted(glob.glob(os.path.join(datapath, 'depth_right', '*.npy')))[::stride]

    image_size=[320, 512]

    intrinsics_vec = [320.0, 320.0, 320.0, 240.0]

    #for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
    for t, (imfile_l, imfile_r, depth_left_file, depth_right_file) in enumerate(zip(image_list_l, image_list_r, depth_left_list, depth_right_list)):
        #image = cv2.imread(image_file)
        images_left = imfile_l
        images_right = imfile_r

        img = cv2.imread(images_left)
        ht0, wd0, _ = img.shape

        images = [cv2.imread(images_left)]
        images += [cv2.imread(images_right)]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        # ----------------- depth
        depth_left = np.load(depth_left_file)
        depth_right = np.load(depth_right_file)
        depth = (depth_left + depth_right)/2
        print("depth values : ")
        print("depth min : ",np.min(depth))
        print("depth max : ",np.max(depth))
        print("depth shape : ",depth.shape)

        depth = torch.as_tensor(depth)
        
        depth = F.interpolate(depth[None,None], image_size, mode='nearest').squeeze()
        
        h1 = image_size[0]
        w1 = image_size[1]
        depth = depth[:h1-h1%8, :w1-w1%8]

        yield t, images, depth, intrinsics




def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--imagedir", default="./data" ,type=str, help="path to image directory")


    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")


    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()

    #args.stereo = False
    args.stereo = True
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    #for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
    stride=1

    print("args.imagedir : ",args.imagedir)

    for (t, image, depth, intrinsics) in tqdm(image_stream_depth_stereo(args.imagedir, use_depth=True, stride=stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        #droid.track(t, image, intrinsics=intrinsics)
        droid.track(t, image, depth, intrinsics=intrinsics)

    # del droid.frontend
    #
    # torch.cuda.empty_cache()
    # print("#" * 32)
    # droid.backend(7)
    # torch.cuda.empty_cache()
    # print("#" * 32)
    # droid.backend(12)



    # if args.reconstruction_path is not None:
    #     save_reconstruction(droid, args.reconstruction_path)

    #traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
