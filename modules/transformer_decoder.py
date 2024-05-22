

import torch
import torch.nn as nn
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from modules.BaseBlock import *
from modules.cmWR import *

from modules.swin_transformer import Mlp, WindowAttention, SwinTransformerBlock
from timm.models.layers import trunc_normal_
from modules.cross_transformer import *


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


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        input: [B, H * W, C]
        output: [B, H*2 * W*2, C/2]
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer decoder layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dims = [96 * 2, 96 * 4, 96 * 8, 96 * 8]
        self.conv_stem = nn.ModuleList()
        for i in range(len(self.dims)):
            self.conv_stem.append(Conv_Stem(self.dims[i],self.dims[i]))
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch expanding layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                for n in range(len(self.dims)-2):
                    if self.dim == self.dims[n]:

                        x1 = self.conv_stem[n](resize_tensor(x))
                        #print("x1.shape",x1.shape)
                        x = checkpoint.checkpoint(blk, x)
                        #print("x.shape",x.shape)
                        x2 = x1 * resize_tensor(x)
                        #print("x2.shape",x2.shape)
                        x = restore_tensor(x2,x.shape)
                        #print("x.shape",x.shape)

            else:
                for n in range(len(self.dims)-2):
                    if self.dim == self.dims[n]:

  
                        x1 = self.conv_stem[n](resize_tensor(x))
                        #print("x1.shape",x1.shape)
                        x = blk(x)
                        #print("x.shape",x.shape)
                        x2 = x1 * resize_tensor(x)
                        #print("x2.shape",x2.shape)
                        x = restore_tensor(x2,x.shape)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class conv_base(nn.Module):
    def __init__(self, in_res, in_dim, out_res, out_dim):
        super(conv_base, self).__init__()
        # print(in_res, in_dim, out_res, out_dim)
        self.in_res = in_res
        self.in_dim = in_dim
        self.out_res = out_res
        self.out_dim = out_dim
        self.layer1 = BaseConv2d(in_dim, out_dim)


    def forward(self, x):
        B, _, _ = x.shape
        x = x.view(B, self.in_res, self.in_res, -1).permute(0, 3, 1, 2)
        x = self.layer1(x)
        x = F.interpolate(x, (self.out_res, self.out_res), mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 3, 1).view(B, self.out_res * self.out_res, self.out_dim)
        return x
def restore_tensor(resized_tensor, original_shape):
    return resized_tensor.view(original_shape)

def cat_deCh(x1, x2, x3):
    x = torch.cat([x1, x2, x3],dim=1)
    cc = x1.shape[1]
    con = nn.Conv2d(3*cc, cc, kernel_size=3,stride=1,padding=1).cuda()

    return con(x)

def resize_tensor_to_one(feature):

    c = feature.shape[1]
    conv_reduce = nn.Conv2d(c, 1, kernel_size=1).cuda()
    new_matrix = conv_reduce(feature)

    return new_matrix

def resize_tensor(feature):
    element = feature.size()
    s = element[1]
    h = int(torch.sqrt(torch.tensor(s)))
    w = h

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

# -----------------------------------------加在mcf模块后的二次融合模块-------------------------------------------------------
class ConvNormAct(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, act=True, bias=False):

        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=bias)
        #self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, norm=nn.BatchNorm3d):
        super().__init__()

        pad_size = 3//2

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, )
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size)
        self.residual = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size)
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)

        x = self.conv2(x)

        x = x + self.residual(shortcut)

        return x

class Conv_Stem(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        pad_size = 3//2

        self.conv1 = BasicBlock(in_ch, out_ch, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride = 1, padding=pad_size, bias=False)

    def forward(self, x):
        x_2 = self.conv1(x)
        x = self.conv2(x_2)
        return x
# ----------------------------------------------------------------------------------------------------------------------

class TransformerDecoder_side(nn.Module):

    def __init__(self, dim_list=[768, 768, 384, 192], decoder_depths=[2, 6, 2, 2], num_heads=[24, 12, 6, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, ):
        super().__init__()
        channels = [96, 96 * 2, 96 * 4, 96 * 8, 96 * 8]
        self.dim_list = dim_list
        self.num_layers = len(decoder_depths)
        self.patches_resolution = [(7, 7), (7, 7), (14, 14), (28, 28)]
        self.embed_dim = 96
        self.patch_norm = patch_norm
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dims = [96 * 2, 96 * 4, 96 * 8, 96 * 8]
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]  # stochastic depth decay rule
        self.SpatialAttention=SpatialAttention(kernel_size=7)
        self.ChannelAttention3=ChannelAttention(self.dims[3])
        self.ChannelAttention2=ChannelAttention(self.dims[2])
        self.ChannelAttention1=ChannelAttention(self.dims[1])
        self.ChannelAttention0=ChannelAttention(self.dims[0])
        self.cmWR = nn.ModuleList([])

        self.conv_rgb = nn.ModuleList([])
        self.ChannelAttention = nn.ModuleList([])
        for i in range(len(channels) - 1, 0, -1):
            if i > 1:
                self.ChannelAttention.append(ChannelAttention(self.dims[i-2]))
            self.conv_rgb.append(BaseConv2d(channels[i], channels[i], kernel_size=3, padding=1))
            self.cmWR.append(cmWR(channels[i], squeeze_ratio=1))
        # build decoder layers
        self.layers = nn.ModuleList()
        self.concat_layer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_layer = nn.Linear(2 * self.dim_list[i_layer],
                                     self.dim_list[i_layer]) if i_layer > 0 else nn.Identity()
            layer = BasicLayer_up(dim=self.dim_list[i_layer],
                                  input_resolution=self.patches_resolution[i_layer],
                                  depth=decoder_depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  upsample=PatchExpand if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)
            self.concat_layer.append(concat_layer)
            self.layers.append(layer)
        self.norm = norm_layer(self.embed_dim)
        self.apply(self._init_weights)
        self.fusion = nn.ModuleList([])
        for item in self.dims[:]:
            self.fusion.append(PointFusion_side(dim=item, depth=1, heads=3, dim_head=item // 3))
        #self.fusion.append(PointFusion(dim=self.dims[-1], depth=1, heads=3, dim_head=self.dims[-1] // 3))
        self.conv_map4 = nn.Sequential(BaseConv2d(768, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map3 = nn.Sequential(BaseConv2d(384, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map2 = nn.Sequential(BaseConv2d(192, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map1 = nn.Sequential(BaseConv2d(96, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.out = [self.conv_map4, self.conv_map3, self.conv_map2, self.conv_map1]
        self.relu = nn.ReLU(True)

        self.conv_stem = nn.ModuleList()
        for i in range(len(self.dims)-1,-1,-1):
            self.conv_stem.append(Conv_Stem(self.dims[i],self.dims[i]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_features, depth_features, rgb_smAR, depth_smAR,feas,hance_d):
        rgbd_smAR = torch.add(rgb_smAR[0], depth_smAR[0])
        rgbd_smAR_chen = rgb_smAR[0]*depth_smAR[0]
        rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR[0](rgb_smAR[0], depth_smAR[0], rgbd_smAR)
        feass = feas
        feas = feas[::-1]

        #print(fea[0].shape,fea[1].shape,fea[2].shape)
        B, _, _ = rgb_features[0].shape
        assert len(self.concat_layer) == len(self.layers)

        x_decoder = []

        sides = []

        l = len(self.layers) # 4
        for i in range(len(self.layers)):
            #print(fea[0].shape,fea[1].shape,fea[2].shape)
            if i != 0:
                # rgbd_smAR =torch.cat(())
                rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR[i](rgb_smAR_i, depth_smAR_i, rgbd_smARx)

            if i == 0:
                # rgb_cmWR_f = self.conv_stem[i](rgb_cmWR)
                # depth_cmWR_f = self.conv_stem[i](depth_cmWR)
                # rgbd_smAR_f = self.conv_stem[i](rgbd_smAR)
                # rgbd_i = cat_deCh(rgb_cmWR_f, depth_cmWR_f, rgbd_smAR_f)
                # rgbd_i = restore_tensor(rgbd_i,rgb_features[l-i].shape)
                rgbd_i = self.fusion[l - i - 1](restore_tensor(rgb_cmWR,rgb_features[l-i].shape), restore_tensor(depth_cmWR,rgb_features[l-i].shape), resize_tensor_to_one(rgbd_smAR_chen).sigmoid())
                # print(fea[0].shape,fea[1].shape,fea[2].shape)
                x = rgbd_i
                # print("x:",x.shape)
            else:
                rgb_cmWR_f = self.conv_stem[i](rgb_cmWR)
                depth_cmWR_f = self.conv_stem[i](depth_cmWR)
                rgbd_smAR_f = self.conv_stem[i](rgbd_smARx)

                maps = F.interpolate(sides[i - 1], self.patches_resolution[i], mode='bilinear', align_corners=True)
                # rgbd_smARx = rgbd_smARx * maps
                # rgbd_i = cat_deCh(rgb_cmWR_f, depth_cmWR_f, rgbd_smARx)
                # rgbd_i = restore_tensor(rgbd_i,rgb_features[l-i].shape)
                rgbd_i = self.fusion[l - i - 1](restore_tensor(rgb_cmWR,rgb_features[l-i].shape), restore_tensor(depth_cmWR,rgb_features[l-i].shape), maps.sigmoid())
                # print ("rgbd_i:",rgbd_i.shape)
                x = self.concat_layer[i](torch.cat([x, rgbd_i], dim=-1))
                #print("x_l:",x.shape)
            # print('concat', x.shape)



            x = self.layers[i](x)
            #print("x = self.layers[i](x)",x.shape)

            if i < 3:
                #x_rgb = torch.bmm(feass[i], x)
                x_rgb = resize_tensor(feass[i]) * resize_tensor(x)
                # x_rgb = ChannelHalvingConv(x_rgb)
                x_rgb = restore_tensor(x_rgb, feass[i].shape)
                # x_rgb.append(x_rgb)

                x_depth = hance_d[3-i] * resize_tensor(x)
                # x_depth = ChannelHalvingConv(x_depth)
                # x_depth.append(x_depth)

                B, C, H, W = resize_tensor(feass[i]).size()
                P = H * W
                x_rgb_SA_i = self.SpatialAttention(resize_tensor(x_rgb)).view(B, -1, P)
                x_depth_SA_i = self.SpatialAttention(x_depth).view(B, -1, P)

                B0, C0, H0, W0 = resize_tensor(x_rgb).size()
                x_rgb_CA_0 = self.ChannelAttention[i](resize_tensor(x_rgb)).view(B0, C0, -1)
                x_depth_CA_0 = self.ChannelAttention[i](x_depth).view(B0, C0, -1)

                B3, C3, H3, W3 = resize_tensor(feass[i]).size()
                # print(B, C, H, W)
                # print("fea_rgb_CA[m],fea_rgb_SA[m],depth_hance_CA[m],depth_hance_SA[m]", fea_rgb_CA[m].shape,
                #       fea_rgb_SA[m].shape, depth_hance_CA[m].shape, depth_hance_SA[m].shape)
                rgbM = torch.bmm(x_rgb_CA_0, x_rgb_SA_i).view(B3, C3, H3, W3)
                depthM = torch.bmm(x_depth_CA_0, x_depth_SA_i).view(B3, C3, H3, W3)
                # print("rgbM,depthM", rgbM.shape, depthM.shape)

                rgb_smAR_i = x_depth * rgbM + x_depth
                depth_smAR_i = resize_tensor(x_rgb) * depthM + resize_tensor(x_rgb)
                # print("rgb_smAR_i,depth_smAR_i", rgb_smAR_i.shape, depth_smAR_i.shape)

                rgb_smAR_i = self.conv_rgb[i+1](rgb_smAR_i)
                depth_smAR_i = self.conv_rgb[i+1](depth_smAR_i)

                rgbd_smARx = torch.add(rgb_smAR_i, depth_smAR_i)
                rgbd_smARx_chen = rgb_smAR_i*depth_smAR_i

                # print ("rgb_smAR_i,depth_smAR_i,rgbd_smARx",rgb_smAR_i.shape,depth_smAR_i.shape,rgbd_smARx.shape)

            # x_decoder.append(x)
            #print('layer:', x.shape)

            fea = x.view(B, 7 * (2 ** i), 7 * (2 ** i), -1).permute(0, 3, 1, 2)
            #print('fea', fea.shape)
            sides.append(self.out[i](fea))

        x = self.norm(x)  # B L C  # [48, 3136, 96]

        return x, sides, x_decoder


if __name__ == "__main__":
    x_list = []
    x1 = torch.randn(1, 784, 192)
    x_list.append(x1)
    x2 = torch.randn(1, 196, 384)
    x_list.append(x2)
    x3 = torch.randn(1, 49, 768)
    x_list.append(x3)
    x4 = torch.randn(1, 49, 768)
    x_list.append(x4)

    model = TransformerDecoder_side()
    # print(model)
