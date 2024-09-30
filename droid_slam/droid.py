import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

import matplotlib.pyplot as plt


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()

        # load model
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        # load weights
        self.net.load_state_dict(state_dict)
        # put net to cuda
        self.net.to("cuda:0").eval()



    # tracking
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """
        
        # print("-------- tracking")
        #
        # print("-------- depth shape : ", depth.shape)
        # 
        # Calculer et afficher les valeurs minimales et maximales
        # min_val = torch.min(depth)
        # max_val = torch.max(depth)
        # print("-------- min depth value : ", min_val.item())
        # print("-------- max depth value : ", max_val.item())
        
        # Afficher le tenseur depth sous forme d'image
        # plt.imshow(depth.cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Depth Image')
        # plt.show()

        with torch.no_grad():

            # before motion track
#             import pdb; pdb.set_trace()

            # check there is enough motion
            # append to video network and image
            self.filterx.track(tstamp, image, depth, intrinsics)

            # after track
#             import pdb; pdb.set_trace()

            # local bundle adjustment
            self.frontend()

            # local ba
#             import pdb; pdb.set_trace()


    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

