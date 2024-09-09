


from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import Config
from pycallgraph2 import GlobbingFilter


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
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    image_list = sorted(glob.glob(os.path.join(imagedir, 'rgb', '*.JPG')))[::stride]

    intrinsics = [320.0, 320.0, 320.0, 240.0]


    for t, imfile in enumerate(image_list):

        image = cv2.imread(os.path.join(imagedir, imfile))

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
    image_list = sorted(glob.glob(os.path.join(imagedir, 'image_left', '*.png')))[::stride]

    image_size=[320, 512]

    intrinsics_vec = [320.0, 320.0, 320.0, 240.0]

    for t, imfile in enumerate(image_list):

        #images_left = imageadresse
        images_left = imfile

        images = [cv2.imread(images_left)]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        
        img = cv2.imread(images_left)
        ht0, wd0, _ = img.shape

        if image_size is not None:
            images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError("image_size must be defined")
            
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

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

    parser.add_argument("--imagedir", default="./data/" ,type=str, help="path to image directory")


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

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True



    try:
        config = Config()
        
        #config.trace_filter = GlobbingFilter(exclude=['pycallgraph2.*'])
        config.trace_filter = GlobbingFilter(
                #include=['dpvo.net.*'],  # Inclure explicitement le module torch
                exclude=['numpy.*', 
                         'pdb.*',
                         'pycallgraph2.*' ,
                         '_*', 
                         'shutil', 
                         'os', 
                         're', 
                         'sys', 
                         'module_from_spec.*',
                         'module_from_spec',
                         'SourceFileLoader.*',
                         'FileFinder.*',
                         'find_spec', 
                         '<listcomp>',
                         '<genexpr>',
                         'spec_from_file_location',
                         'cache_from_source',
                         'cb',
                         '<lambda>',
                         'VFModule.*',
                         'ModuleSpec.*',
                         'dpvo.lietorch.*',
                         'dpvo.utils.*',
                         'dpvo.blocks.*',
                         'dpvo.altcorr.*',
                         'dpvo.projective_ops.*',
                         'dpvo.extractor.*',
                         'einops.*',
                         'einops.reduce',
                         'importlib.*',
                         'PatchGraph.*',
                         'pg.*',
                         'PatchGraph.reduce'])

        graphviz = GraphvizOutput()
        #graphviz.output_file = 'tmp.pdf'
        graphviz.output_file = 'callgraph.pdf'
        graphviz.output_type = 'pdf'  # Spécifier le format de sortie en PDF

        # graphviz.output_file = 'callgraph.png'
        # graphviz.output_type = 'png'  # Spécifier le format de sortie en PDF

        # Générer le graphe d'appel
        with PyCallGraph(output=graphviz, config=config):
            tstamps = []
            #for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
            for (t, image, intrinsics) in tqdm(image_stream_mono(args.imagedir, args.calib, args.stride)):
                if t < args.t0:
                    continue

                if not args.disable_vis:
                    show_image(image[0])

                if droid is None:
                    args.image_size = [image.shape[2], image.shape[3]]
                    droid = Droid(args)
                
                droid.track(t, image, intrinsics=intrinsics)

    except KeyboardInterrupt:
        print("Programme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
    finally:
        print("fin du programme")
        #sys.exit(0)        
        os._exit(0)  # Utilisé pour forcer la fermeture

 
