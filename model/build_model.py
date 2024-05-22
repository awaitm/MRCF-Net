import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from modules.swin_transformer import Swin_T, Mlp
from modules.cross_transformer import PointFusion_side, PointFusion
from modules.transformer_decoder import *
from modules.VGG import LowFeatureExtract

from modules.BaseBlock import *
from modules.cmWR import *
from modules.multilevel_interaction import *
print(torch.cuda.is_available())
import torch

torch.cuda.current_device()
torch.cuda._initialized = True


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, bias=True, norm_layer=False):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential()
        self.basicconv.add_module(
            'conv', nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias))
        if norm_layer:
            self.basicconv.add_module('bn', nn.BatchNorm2d(out_planes))
        self.basicconv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

def resize_tensor(feature):
    element = feature.size()
    s = element[1]
    h = int(torch.sqrt(torch.tensor(s)))
    w = h

    # ????????????
    b = feature.shape[0]
    c = feature.shape[2]
    new_matrix = torch.reshape(feature, (b, c, h, w))
    # print(new_matrix.shape)
    return new_matrix



def ChannelHalvingConv(feature):
    in_c = feature.shape[1]
    out_c = in_c//2
    conv = nn.Conv2d(in_c, out_c, kernel_size=1).cuda()
    return conv(feature)


def ChannelHalvingConv_s(feature,s):
    in_c = feature.shape[1]
    out_c = in_c//s
    conv = nn.Conv2d(in_c, out_c, kernel_size=1).cuda()
    return conv(feature)

def reshape_input(input):
    if len(input.shape) == 3 and input.shape[-1] == 768:
        input = input.transpose(1, 2)
        input = input.reshape(-1, 384)
    return input

def restore_tensor(resized_tensor, original_shape):
    return resized_tensor.view(original_shape)

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        # ?????????????????????????????
        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        # ??????��?????
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.dconv = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=2,dilation=2)

    def forward(self, x):
        dx = x
        dx = self.dconv(dx)
        residual = x
        #print ('res',residual.shape)
        #print ('x',x.shape)
        out = self.conv1(x)
        #print ('out1',out.shape)
        out = self.bn1(out)
        #print ('out1',out.shape)
        out = self.relu(out)
        #print ('out1',out.shape)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        out = ChannelHalvingConv_s(out,4)

        #print ('res',residual.shape)
        #print ('out',out.shape)
        out += residual
        out += dx
        out = self.relu(out)

        return out


class MRCF(nn.Module):

    def __init__(self):
        super(PICR_Net, self).__init__()
        channels = [96, 96 * 2, 96 * 4, 96 * 8, 96 * 8]
        self.backbone = Swin_T(pretrained=True)
        self.low_feature_extract = LowFeatureExtract()

        self.dims = [96 * 2, 96 * 4, 96 * 8, 96 * 8]
        self.fusion = nn.ModuleList([])

        self.fusion.append(PointFusion(dim=self.dims[-1], depth=1, heads=3, dim_head=self.dims[-1] // 3))


        self.transformer_decoder = TransformerDecoder_side()
#################################################################
        dim =96
        embed_dim = 384
        self.proj1 = nn.Linear(dim * 2,784)
        self.proj2 = nn.Linear( dim * 4,196)
        self.proj3 = nn.Linear(dim * 8,49)
        self.proj4 = nn.Linear( dim * 8,49)
        self.proj5 = nn.Linear(768, 984)
        self.interact1 = MultilevelInteractionBlock(dim=dim * 8, dim1=dim * 8, embed_dim=embed_dim, num_heads=4,
                                                    mlp_ratio=3)
        self.interact2 = MultilevelInteractionBlock(dim=dim * 4, dim1=dim * 8, dim2=dim * 8, embed_dim=embed_dim,
                                                    num_heads=2, mlp_ratio=3)
        self.interact3 = MultilevelInteractionBlock(dim=dim * 2, dim1=dim * 4, dim2=dim * 8, embed_dim=embed_dim,
                                                    num_heads=1, mlp_ratio=3)
        feature_dims = [dim, dim * 2, dim * 4]

#################################################################

        self.ca = nn.ModuleList([])

        self.SpatialAttention=SpatialAttention(kernel_size=7)
        self.ChannelAttention3=ChannelAttention(self.dims[3])
        self.ChannelAttention2=ChannelAttention(self.dims[2])
        self.ChannelAttention1=ChannelAttention(self.dims[1])
        self.ChannelAttention0=ChannelAttention(self.dims[0])

        self.ChannelAttention112=ChannelAttention(128)
        self.ChannelAttention224=ChannelAttention(64)

        self.cmWR = nn.ModuleList([])

        self.conv_rgb = nn.ModuleList([])
        for i in range(len(channels)-1,0,-1):
            self.conv_rgb.append(BaseConv2d(channels[i], channels[i], kernel_size = 3, padding=1))
            self.cmWR.append(cmWR(channels[i], squeeze_ratio=1))

        self.conv_112 = BaseConv2d(96, 128)

        # ================================================================
        self.conv_224 = nn.Sequential(BaseConv2d(256, 64), BaseConv2d(64, 64))
        # =================================================================
        self.conv_atten = conv1x1(256, 256)
        self.conv_atten1 = conv1x1(128, 128)
        self.conv_map = nn.Sequential(BaseConv2d(128, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.B2N_0 = Bottle2neck(96,96)
        self.B2N_1 = Bottle2neck(192,192)
        self.B2N_2 = Bottle2neck(384, 384)
        self.B2N_3 = Bottle2neck(768, 768)
        self.B2N_4 = Bottle2neck(768, 768)

        self.B2N_f112 = Bottle2neck(128, 128)
        self.B2N_f224 = Bottle2neck(64, 64)


    def forward(self, image, depth):
        """
        Args:
            image: The input of RGB images, three channels.
            depth: The input of Depth images, single channels.

        Returns: The final saliency maps.

        """

        b, c, h, w = image.size()

        # rgb_depth = torch.cat([image, depth], dim=1)
        depth = torch.cat([depth, depth, depth], dim=1)  # ????????????3?????

        # Shared backbone to extract multi-level features
        # list, length=5 [patch_embd_feature, block_1_feature,block_2_feature, block_3_feature, block_4_feature]
        rgb_features = self.backbone(image)
        depth_features = self.backbone(depth)

        depth_hance = []
        depth_hance.append(self.B2N_0(resize_tensor(depth_features[0])))
        depth_hance.append(self.B2N_1(resize_tensor(depth_features[1])))
        depth_hance.append(self.B2N_2(resize_tensor(depth_features[2])))
        depth_hance.append(self.B2N_3(resize_tensor(depth_features[3])))
        depth_hance.append(self.B2N_3(resize_tensor(depth_features[4])))
        #print(len(depth_hance))

        fea = []
        fea_1_16_ = self.interact1(rgb_features[3], rgb_features[4])
        fea.append(fea_1_16_)
        fea_1_8_ = self.interact2(rgb_features[2], fea_1_16_, rgb_features[3])
        fea.append(fea_1_8_)
        fea_1_4_ = self.interact3(rgb_features[1], fea_1_8_, fea_1_16_)
        fea.append(fea_1_4_)
        rgb_SA = []
        depth_SA = []
        rgb_CA = []
        depth_CA = []
        for n in range(len(rgb_features)-1,0,-1):
          B,C,H,W = resize_tensor(rgb_features[n]).size()
          P = H*W
          rgb_SA_i = self.SpatialAttention(resize_tensor(rgb_features[n])).view(B, -1, P)
          depth_SA_i = self.SpatialAttention(resize_tensor(depth_features[n])).view(B, -1, P)
          rgb_SA.append(rgb_SA_i)
          depth_SA.append(depth_SA_i)

        B3, C3, H3, W3 = resize_tensor(rgb_features[4]).size()
        rgb_CA_3 = self.ChannelAttention3(resize_tensor(rgb_features[4])).view(B3,C3,-1)
        depth_CA_3 = self.ChannelAttention3(resize_tensor(depth_features[4])).view(B3,C3,-1)
        rgb_CA.append(rgb_CA_3)
        depth_CA.append(depth_CA_3)

        B2, C2, H2, W2 = resize_tensor(rgb_features[3]).size()
        rgb_CA_2 = self.ChannelAttention2(resize_tensor(rgb_features[3])).view(B2,C2,-1)
        depth_CA_2 = self.ChannelAttention2(resize_tensor(depth_features[3])).view(B2,C2,-1)
        rgb_CA.append(rgb_CA_2)
        depth_CA.append(depth_CA_2)

        B1, C1, H1, W1 = resize_tensor(rgb_features[2]).size()
        rgb_CA_1 = self.ChannelAttention1(resize_tensor(rgb_features[2])).view(B1,C1,-1)
        depth_CA_1 = self.ChannelAttention1(resize_tensor(depth_features[2])).view(B1,C1,-1)
        rgb_CA.append(rgb_CA_1)
        depth_CA.append(depth_CA_1)

        B0, C0, H0, W0 = resize_tensor(rgb_features[1]).size()
        rgb_CA_0 = self.ChannelAttention0(resize_tensor(rgb_features[1])).view(B0,C0,-1)
        depth_CA_0 = self.ChannelAttention0(resize_tensor(depth_features[1])).view(B0,C0,-1)
        rgb_CA.append(rgb_CA_0)
        depth_CA.append(depth_CA_0)


        # SA*CA
        rgb_M = []
        depth_M = []
        rgb_smAR = []
        depth_smAR = []
        rgb_cmWR = []
        depth_cmWR = []
        x=rgb_features[::-1]
        for m in range(0,len(rgb_CA)):
            #print(m)
            B, C, H, W = resize_tensor(x[m]).size()
            P = H * W
            #print(B, C, H, W)
            # print ("depth_CA,rgb_CA,depth_SA,rgb_CA",depth_CA[m].shape,rgb_CA[m].shape,depth_SA[m].shape,rgb_SA[m].shape)
            rgbM = torch.bmm(rgb_CA[m], rgb_SA[m]).view(B, C, H, W)
            depthM = torch.bmm(depth_CA[m], depth_SA[m]).view(B, C, H, W)
            rgb_smAR_i = resize_tensor(x[m]) * rgbM + resize_tensor(x[m])
            depth_smAR_i = resize_tensor(x[m]) * depthM + resize_tensor(x[m])
            # print("rgbM",rgbM.shape)
            # print("depthM",depthM.shape)
            rgb_smAR_i = self.conv_rgb[m](rgb_smAR_i)
            depth_smAR_i =self.conv_rgb[m](depth_smAR_i)
            # print("rgbsm",rgb_smAR_i.shape)
            # print("rgbsm",depth_smAR_i.shape)



            rgb_smAR.append(rgb_smAR_i)
            depth_smAR.append(depth_smAR_i)

            rgb_M.append(rgbM)
            depth_M.append(depthM)
        rgbd_smAR = torch.add(rgb_smAR[0], depth_smAR[0])


        # decoder
        x, sides, x_decoders = self.transformer_decoder(rgb_features, depth_features,rgb_smAR, depth_smAR, fea, depth_hance)  # [b, 3136, 96]
        # x = torch.cat([x, rgb_features[0]], dim=-1)  # [b, 3136, 192]
        x = x.view(b, 56, 56, 96).permute(0, 3, 1, 2)  # [b, 192, 56, 56]



        # CNN Based Saliency Maps Refinement Unit
        feature_224, feature_112 = self.low_feature_extract(image)
        x = self.conv_112(x)
        # ????????????x??????????
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # print(x.shape, feature_112.shape)
        # ????????????????
        # B, C, H, W = feature_112.size()
        # P = H * W
        # f112 = self.SpatialAttention(feature_112).view(B, -1, P)
        # # fx = self.SpatialAttention(x).view(B, -1, P)
        # #print ("feature_224, feature_112",feature_112.shape,feature_224.shape)
        # rgb112 = self.ChannelAttention112(feature_112).view(B,C,-1)
        # # rgbx = self.ChannelAttention112(x).view(B,C,-1)
        #
        # f112M = torch.bmm(rgb112, f112).view(B, C, H, W)
        # # xM = torch.bmm(rgbx, fx).view(B, C, H, W)
        #
        # # f112_BN = self.B2N_f112(feature_112)
        # x_BN = self.B2N_f112(x)
        #
        # #112____main
        # x = x_BN * f112M + x_BN
        # #x_____main
        # # x_smAR = f112_BN * xM + f112_BN
        # # print("f112_smAR", f112_smAR.shape)
        # # print("x_smAR", x_smAR.shape)


        x = torch.cat([x, feature_112], dim=1)  # torch.Size([1, 128, 112, 112]) torch.Size([1, 128, 112, 112])
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        x = torch.mul(x, atten) + x
        #print ("x = torch.mul(x, atten) + x",x.shape)

        x = self.conv_224(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


        # print(x.shape, feature_224.shape)
        x = torch.cat([x, feature_224], dim=1)  # torch.Size([1, 64, 224, 224]) torch.Size([1, 64, 224, 224])
        atten1 = F.avg_pool2d(x, x.size()[2:])
        atten1 = torch.sigmoid(self.conv_atten1(atten1))
        x = torch.mul(x, atten1) + x

        # B, C, H, W = feature_224.size()
        # P = H * W
        # f224 = self.SpatialAttention(feature_224).view(B, -1, P)
        # # fx = self.SpatialAttention(x).view(B, -1, P)
        # print ("feature_224, feature_112",feature_112.shape,feature_224.shape)
        # rgb224 = self.ChannelAttention224(feature_224).view(B,C,-1)
        # # rgbx = self.ChannelAttention224(x).view(B,C,-1)
        #
        # f224M = torch.bmm(rgb224, f224).view(B, C, H, W)
        # # xM = torch.bmm(rgbx, fx).view(B, C, H, W)
        #
        # # f224_BN = self.B2N_f112(feature_112)
        # x_BN = self.B2N_f224(x)
        #
        # #112____main
        # x = x_BN * f224M + x_BN
        # #x_____main
        # # x_smAR = f112_BN * xM + f112_BN
        # # print("f112_smAR", f112_smAR.shape)
        # # print("x_smAR", x_smAR.shape)
        # x = torch.cat([x, x], dim=1)
        smap = self.conv_map(x)



        return smap, sides


if __name__ == '__main__':
    rgb = torch.randn([1, 3, 224, 224])
    depth = torch.randn([1, 1, 224, 224])
    # model = C2TNet()
    # model(rgb, depth)
