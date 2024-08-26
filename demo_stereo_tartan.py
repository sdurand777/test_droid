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

import droid_slam.droid_frontend
import droid_slam.depth_video
import droid_slam.factor_graph

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(imagedir, calib, stride):
    """ image generator """

    # calib = np.loadtxt(calib, delimiter=" ")
    # fx, fy, cx, cy = calib[:4]
    #
    # K = np.eye(3)
    # K[0,0] = fx
    # K[0,2] = cx
    # K[1,1] = fy
    # K[1,2] = cy

    #image_list = sorted(os.listdir(imagedir))[::stride]

    image_list = sorted(glob.glob(os.path.join(imagedir, 'rgb', '*.JPG')))[::stride]

    intrinsics = [322.580, 322.580, 259.260, 184.882]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])
        # print("--------- image rectification")
        #
        # print("++++++++++++++++++++++++++++++++++++ ", imfile)

        K_l = np.array([322.580, 0.0, 259.260, 0.0, 322.580, 184.882, 0.0, 0.0, 1.0]).reshape(3,3)
        d_l = np.array([-0.070162237, 0.07551153, 0.0012286149,  0.00099302817, -0.018171599])
        R_l = np.array([
    0.9999956354796169, -0.002172438871054654, 0.002002381349442793,
     0.002175041160237588, 0.9999967917532834, -0.00129833704855268,
     -0.001999554367437393, 0.001302686643787701, 0.9999971523908654
        ]).reshape(3,3)
        P_l = np.array([
    322.6092376708984, 0, 257.7363166809082, 0,
     0, 322.6092376708984, 186.6225147247314, 0,
     0, 0, 1, 0
            ]).reshape(3,4)
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (514, 376), cv2.CV_32F)

        image = cv2.remap(image, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)


        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        #intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics = torch.as_tensor(intrinsics)
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics



def image_stream_mono(imagedir, image_size=[320, 512], stereo=False, stride=1):
#def image_stream_stereo(imageadresse, image_size=[240, 320], stereo=False, stride=1):
    # recuperation nom images
    image_list = sorted(glob.glob(os.path.join(imagedir, 'left', '*.JPG')))[::stride]

    image_size=[320, 512]

    for t, imfile in enumerate(image_list):

        """ image generator """
        # print("------------ image generator ------------------")
        # print(imfile)
        # print("------------ Left pre rectification ------------------")
        K_l = np.array([322.580, 0.0, 259.260, 0.0, 322.580, 184.882, 0.0, 0.0, 1.0]).reshape(3,3)
        d_l = np.array([-0.070162237, 0.07551153, 0.0012286149,  0.00099302817, -0.018171599])
        R_l = np.array([
    0.9999956354796169, -0.002172438871054654, 0.002002381349442793,
     0.002175041160237588, 0.9999967917532834, -0.00129833704855268,
     -0.001999554367437393, 0.001302686643787701, 0.9999971523908654
        ]).reshape(3,3)
        P_l = np.array([
    322.6092376708984, 0, 257.7363166809082, 0,
     0, 322.6092376708984, 186.6225147247314, 0,
     0, 0, 1, 0
            ]).reshape(3,4)
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (514, 376), cv2.CV_32F)
       
        intrinsics_vec = [322.6092376708984, 322.6092376708984, 257.7363166809082, 186.6225147247314]
        ht0, wd0 = [376, 514]

        # # read all png images in folder
        # print("------- image paths ------")
        # #images_left = imageadresse
        images_left = imfile
        # print(images_left)
        # # images_right = images_left.replace('imgL','imgR')
        # # print(images_right)
        images = [cv2.remap(cv2.imread(images_left), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
#    images += [cv2.remap(cv2.imread(images_right), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        # #images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        # # Ensure either size or scale_factor is defined
        # print("++++++++ image_size  : ",image_size)
        if image_size is not None:
            images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError("image_size must be defined")
            
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        #yield images, intrinsics
        #return images, intrinsics
        yield t, images, intrinsics



def image_stream_stereo(imagedir, image_size=[320, 512], stereo=False, stride=1):
#def image_stream_stereo(imageadresse, image_size=[240, 320], stereo=False, stride=1):
    # recuperation nom images
    image_list_l = sorted(glob.glob(os.path.join(imagedir, 'image_left', '*.png')))[::stride]
    image_list_r = sorted(glob.glob(os.path.join(imagedir, 'image_right', '*.png')))[::stride]

    image_size=[320, 512]

    intrinsics_vec = [320.0, 320.0, 320.0, 240.0]

    for t, (imfile_l, imfile_r) in enumerate(zip(image_list_l,image_list_r)):

        # read all png images in folder
        #print("------- image paths ------")
        images_left = imfile_l
        images_right = imfile_r

        img = cv2.imread(images_left)
        ht0, wd0, _ = img.shape

        images = [cv2.imread(images_left)]

        tmp = torch.from_numpy(np.stack(images, 0))

        images += [cv2.imread(images_right)]

        images = torch.from_numpy(np.stack(images, 0))

        images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        #yield images, intrinsics
        #return images, intrinsics
        yield t, images, intrinsics





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
    #parser.add_argument("--imagedir", type=str, help="path to image directory")


    #parser.add_argument("--imagedir", default="/home/ivm/Selective-Stereo/Selective-IGEV/test_video" ,type=str, help="path to image directory")

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
    for (t, image, intrinsics) in tqdm(image_stream_stereo(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        # if not args.disable_vis:
        #     show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    del droid.frontend

    torch.cuda.empty_cache()
    print("#" * 32)
    droid.backend(7)
    torch.cuda.empty_cache()
    print("#" * 32)
    droid.backend(12)


    # if args.reconstruction_path is not None:
    #     save_reconstruction(droid, args.reconstruction_path)
    #
    # traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
