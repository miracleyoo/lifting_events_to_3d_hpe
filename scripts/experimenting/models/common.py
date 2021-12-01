import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, att_only=False):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.att_only = att_only

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W= x.size()
        queries = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1).contiguous()  # B x WH x C/8
        keys = self.key_conv(x).view(B, -1, W*H).contiguous()  # B x C/8 x WH
        values = self.value_conv(x).view(B, -1, W*H).contiguous()  # B x C x WH

        energy = torch.bmm(queries, keys)  # transpose check
        attention = self.softmax(energy)  # B x WH x WH
        if self.att_only:
            return attention, values

        out = torch.bmm(values, attention.permute(0, 2, 1).contiguous())  # B x C x WH
        out = out.view(B, C, H, W).contiguous()  # B x C x W x H
        out = self.gamma * out + x
        return out, attention

        # queries = self.query_conv(x).reshape(B, -1, W*H)  # B x C/8 x HW
        # keys = self.key_conv(x).reshape(B, -1, W*H)  # B x C/8 x WH
        # values = self.value_conv(x).reshape(B, -1, W*H)  # B x C x WH

        # # compute the unnormalized attention
        # QK = torch.einsum("bkl,bkf->blf", queries, keys) # compute the dot products
        # A = self.softmax(QK)
        # if self.att_only:
        #     return A, values
        
        # V = torch.einsum("bcl, blf->bcf", values, A)
        # V = V.reshape(B, C, H, W).contiguous()  # B x C x W x H
        # return V, A


class CLSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim=128, kernel_size=(3,3)):
        super(CLSelfAttention, self).__init__()
        self.sa = SelfAttention(in_dim, att_only=True)
        self.cl = ConvLSTM(1, 1, kernel_size, 1, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        att, v = self.sa(x)
        attuns = att.unsqueeze(0).unsqueeze(2).contiguous()
        # print(attuns.shape, v.shape)
        clatt = self.cl(attuns)[0].squeeze()
        # print(clatt.shape)

        # clatt = att
        # out = torch.einsum("bcl, blf->bcf", v, clatt)
        # out = out.reshape(B, C, H, W).contiguous()  # B x C x W x H
        # out = self.gamma * out + x

        out = torch.bmm(v, clatt.permute(0, 2, 1).contiguous())  # B x C x WH
        out = out.view(B, C, H, W)  # B x C x H x W
        out = self.gamma * out + x
        return out