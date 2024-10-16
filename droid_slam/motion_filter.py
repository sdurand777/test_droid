import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock

# import droid
#
# import depth_video


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        


    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        # split latent and input for RAFT
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)


    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)



    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        #print("===== track motionfilter")
        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features for the current image or pair if stereo

        print("inputs.shape ", inputs.shape)

        gmap = self.__feature_encoder(inputs)
        
        print("gmap.shape : ", gmap.shape)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            #print("===== first frame")
            # extract features map

            print("inputs[:,[0]] ", inputs[:,[0]])
            net, inp = self.__context_encoder(inputs[:,[0]])

            # definition of self net inp and fmap
            self.net, self.inp, self.fmap = net, inp, gmap
            print("first definition of net inp and fmap")
            print("self.net.shape : ", self.net.shape)
            print("self.inp.shape : ", self.inp.shape)
            print("self.fmap.shape : ", self.fmap.shape)

            # share data with all process
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])
            print("self.net[0,0].shape : ", self.net[0,0].shape)
            print("self.inp[0,0].shape : ", self.inp[0,0].shape)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume meshgrid of the pixel coords for an image
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

            # gmap new image et fmap previous
            
            print("self.fmap.shape ", self.fmap.shape)
            print("self.fmap[None,[0]] ", self.fmap[None,[0]].shape)

            print("self.gmap.shape ", gmap.shape)
            print("self.gmap[None,[0]] ", gmap[None,[0]].shape)

            # on construit un objet Corrblock en utilisant les feature maps de gauche [0] avec un batch size de 1 avec Nonea et on applique la methode call avec (coords0) pour recuperer uniquement les correlations voulues
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)
            
            print("corr.shape : ", corr.shape)

            # approximate flow magnitude using 1 update iteration
            #print("===== compute 1 update for delta optical flow")

            # RAFT to get delta net inp and corr but no initial flow

            print("self.net[None].shape ", self.net[None].shape)
            print("self.inp[None].shape ", self.inp[None].shape)
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            
            print("delta.shape ", delta.shape)

            #print("delta.shape : ",delta.shape)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                #print("===== enough motion to update video")
                # reset count of frame with not enough motion
                self.count = 0
                # extract context features
                net, inp = self.__context_encoder(inputs[:,[0]])
                # update net inp and fmap for next iteration
                self.net, self.inp, self.fmap = net, inp, gmap

                print("first definition of net inp and fmap")
                print("self.net.shape : ", self.net.shape)
                print("self.inp.shape : ", self.inp.shape)
                print("self.fmap.shape : ", self.fmap.shape)

                # update video with new frame tstamp frame index to update video counter
                # None for pose will be estimated later
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])
                print("self.net[0].shape : ", self.net[0].shape)
                print("self.inp[0].shape : ", self.inp[0].shape)

            else:
                # update counter of frame with no enough motion
                self.count += 1





