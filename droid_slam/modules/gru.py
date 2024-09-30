import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, net, *inputs):
        # forward function to compute gru
        # inp concat inp corr et flow
        # concat donne [1,320,40,64]
        inp = torch.cat(inputs, dim=1)

        # concat inp avec net [1,448,40,64]
        net_inp = torch.cat([net, inp], dim=1)

        # get net shape [1,128,40,64]
        b, c, h, w = net.shape

        glo = torch.sigmoid(self.w(net)) * net
        # donne [1,128,1,1]
        glo = glo.view(b, c, h*w).mean(-1).view(b, c, 1, 1)

        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))

        net = (1-z) * net + z * q
        return net


