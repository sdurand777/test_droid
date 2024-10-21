import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ### save all feature maps extracted

        # stereo fearure maps
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # left only context feature maps

        # latent for conv gru
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # correlation features
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        
    def get_lock(self):
        return self.counter.get_lock()



    ## append launcher setter
    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        #import pdb; pdb.set_trace()

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)
        if item[5] is not None:
            self.intrinsics[index] = item[5]

        # gmap set to fmap [2,128,40,64] pour stereo
        if len(item) > 6:
            self.fmaps[index] = item[6]

        # net pour context encoder [1,128,40,64]
        if len(item) > 7:
            self.nets[index] = item[7]

        # inp from context encoder [1,128,40,64]
        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj


    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask




    # method to get distance between frame to optimize graph with this metric or add or replace KF in frontend update method frame_distance is on GPU
    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        #import pdb; pdb.set_trace()

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d



    # bundle adjustment performed in graph.update to update video poses
    # eta c'est le damping dans factorgraph
    # t0 vaut 1 car on optimise pas la premiere pose 0 on garde identite
    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """



        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1


            my_values = {
                'poses': self.poses.to('cpu'),
                'disps': self.disps.to('cpu'),
                'intrinsics': self.intrinsics[0].to('cpu'),
                'target': target.to('cpu'),
                'weight': weight.to('cpu'),
                'eta': eta.to('cpu'),
                'ii': ii.to('cpu'),
                'jj': jj.to('cpu'),
            }

            class Container(torch.nn.Module):
                def __init__(self, my_values):
                    super().__init__()
                    for key in my_values:
                        setattr(self, key, my_values[key])

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
            container = torch.jit.script(Container(my_values))
            container.save("container.pt")

            import pdb; pdb.set_trace()

            # eta c'est le damping
            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            # on clamp disp donc on retire les points trop loin
            self.disps.clamp_(min=0.001)
