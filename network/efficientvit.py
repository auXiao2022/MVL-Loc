# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values() #c->convolutional layer bn->batchnorm layer
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5 #eps->epsilon normalized weights=bn.weights / sd
        w = c.weight * w[:, None, None, None] #adds three singleton dimensions to the computed w tensor to broadcast kernel'shape
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5 #sd of activations of bn scales the normalized activations back to the original range + bias
        m = torch.nn.Conv2d(w.size(1) * self.c.groups,
                            w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups) # new convolutional layer with fused weights and bias
        m.weight.data.copy_(w) #fused weight
        m.bias.data.copy_(b)  #fused bias
        return m


class BN_Linear(torch.nn.Sequential): #bacth_normlization + fully connected layer
    def __init__(self, a, b, bias=True, std=0.02): # input_feature_size,output_feature_size
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std) # fn.weights using truncated normal distribution initialization
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5 #scaling factor
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5 #bias term
        w = l.weight * w[None, :] #Scales the weight parameter of input layer in second dimension,adjusting each neuron's connection to the inputs of layer
        if l.bias is None:   # feedforward operation
            b = b @ self.l.weight.T #matrix-multiplied by the transpose of the weight matrix
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias #reshaped b[:,None] then  back into a 1D tensor
        m = torch.nn.Linear(w.size(1), w.size(0)) #FF adjust tensor shape from (col1) to (col0)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution) #1x1 convolution dim->input output->4dim
        self.act = torch.nn.ReLU() #ReLU activation
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution) #3*3 stride2 add 1 zero-padding to both sides Groups mean that each input channel is connected only to its own set of hid_dim output channels, which can help capture more specific features from each input channel
        self.se = SqueezeExcite(hid_dim, .25) #SE enhance feature representation 0.25 * hid_dim(channels)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2) #half spatial resolution up-sampling

    def forward(self, x): #forward pass
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        # m could be either one layer random drop some features maps / (1-self.drop)
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)
# ed input features h hidden units

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5   #scaling factor for the dot product of queries and keys
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []    # query-key-value projection layers and  depthwise convolutional layers
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution)) #input size, output features
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution)) # stride 1 padding size //2, groups= output channel
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution)) # input d* heads, output dim

        points = list(itertools.product(range(resolution), range(resolution))) #Cartesian product
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets) #plug into the last one idx then ++
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))) # function filled with zeros then to tensor in Parameter
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))  #create Longtensor would not update

    @torch.no_grad()    #the computations within this function should not be tracked for gradient computation (which is useful to reduce memory consumption and speed up computation).
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W) 8,64,7,7
        B, C, H, W = x.shape #muti head attention
        trainingab = self.attention_biases[:, self.attention_bias_idxs]#4,49,49
        feats_in = x.chunk(len(self.qkvs), dim=1) #4,8,16,7,7
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)#8,48,7,7    #why 48 and h suppose to be 8?
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)#8,16,7,7  # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N 8,16,49\
            #print(q.device,k.device,v.device)
            #print(self.scale)
            qk = q.transpose(-2, -1) @ k # 计算q与k的内积，得到注意力矩阵 qk 8,49,49
            #print(qk.device)
            train_bias = (trainingab[i] if self.training else self.ab[i]).to(qk.device) #49,49
            #print(train_bias)
            attn = (qk * self.scale+train_bias) # 加上缩放因子和attention biases，得到最终的注意力矩阵
            attn = attn.softmax(dim=-1)  # BNN 进行softmax计算，得到归一化的注意力权重 #8,49,49
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW 8,16,7,7  计算加权后的v张量，并reshape回原来的形状，得到输出特征
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1)) #将所有的输出特征连接在一起，然后通过一个线性变换进行投影，得到最终的输出张量x。
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels, )

    def forward(self, x):#2,64,14,14  #why 2 ?
        H = W = self.resolution #14
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:#false
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1) #2,14,14,64
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution #0
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution #0
            padding = pad_b > 0 or pad_r > 0 #false

            if padding: #false
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r #14,14
            nH = pH // self.window_resolution #2
            nW = pW // self.window_resolution #2  #nH and nW are computed as the number of partitions in the height and width dimensions, respectively, based on self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2) #[8,64,7,7] treat as 4 images
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                                           C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution)) #kernel size 3, stride 1 padding 1
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution)) # hidden layer int(ed*2)

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):#[2,64,14,14]
        x = self.dw0(x) ##[2,64,14,14]
        x = self.ffn0(x)##[2,64,14,14]
        x = self.mixer(x)
        x = self.dw1(x)
        x = self.ffn1(x)
        return x


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],  #distillation transfer learing
                 distillation=False, ):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                         resolution=resolution // 2), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                         resolution=resolution // 4), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1,
                                                         resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                               Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],
                              resolution=resolution)),
                                               Residual(
                                                   FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):#[2,3,224,224]
        x = self.patch_embed(x)#2,64,14,14
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


EfficientViT_m0 = {
        'img_size': 256,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m4 = {
        'img_size': 256,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }
# EfficientViT_m4 = {
#         'img_size': 256,
#         'patch_size': 16,
#         'embed_dim': [192, 288, 256],
#         'depth': [1, 3, 12],
#         'num_heads': [3, 3, 4],
#         'window_size': [7, 7, 7],
#         'kernels': [7, 5, 3, 3],
#     }
EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

def EfficientViT_M0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m0):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model

def EfficientViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m3):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m4):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m5):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


if __name__ == "__main__":
    model_test = EfficientViT_M0()
    input_image = torch.zeros([2,3,256,256],dtype=torch.float)
    output = model_test(input_image)
    for feature in output:
        print(feature.shape)
