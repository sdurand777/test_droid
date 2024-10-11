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

        #import pdb; pdb.set_trace()

        gmap = self.__feature_encoder(inputs)
        
        # extraction feature
        #import pdb; pdb.set_trace()

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            # extract features map
            # context feature only on left [0]
            net, inp = self.__context_encoder(inputs[:,[0]])

            # definition of self net inp and fmap
            self.net, self.inp, self.fmap = net, inp, gmap

            # share data with all process
            # on recup image[0] left uniquement
            # gmap stereo features net et inp left context features
            # gmap [2, 128, 40, 64] inp [1, 128, 40, 64] net [1, 128, 40, 64]
            
            # FAUX net et inp pas bon
            #self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])

            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0], inp[0])


        ### not first frame process correlation ###
        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume meshgrid of the pixel coords for an image
            # shape [1,1,40,64,2] size of feature maps
            # on a toutes les coords des pixels de img 40 par 64
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

            #import pdb; pdb.set_trace()

            # gmap new image et fmap previous
            
            # on construit un objet Corrblock en utilisant les feature maps de gauche [0] avec un batch size de 1 avec None et on applique la methode call avec (coords0) pour recuperer uniquement les correlations voulues
            # on a fmap gmap previous keyframe
            # on a gmap stereo feature current frame
            # on a [0] pour prendre que la gauche
            # on donne coords0 en arg donc on utilise direct la methode call de CorrBlock apres init
            # corr [1, 1 ,196, 40, 64] 196 pour 49(7*7 patch)*4 lvl pyramid
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            # RAFT to get delta net inp and corr but no initial flow

            # update prends en args
            # net avec None pour batch size [1,128,40,64]
            # inp avec None pour batch [1,128,40,64]
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            
            # we get delta [1,1,40,64,2]
            
            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                #print("===== enough motion to update video")
                # reset count of frame with not enough motion
                self.count = 0
                # extract context features from left image
                net, inp = self.__context_encoder(inputs[:,[0]])
                # update net inp and fmap for next iteration
                self.net, self.inp, self.fmap = net, inp, gmap

                # update video with new frame tstamp frame index to update video counter
                # None for pose will be estimated later
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])
    
            else:
                # update counter of frame with no enough motion
                self.count += 1

        # end of tracking motion
#         import pdb; pdb.set_trace()



