import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):#64,2048
        batch_size = x.size(0)#64
        out_channels = x.size(1) #2048

        g_x = self.g(x).view(batch_size, out_channels // 8, 1) #([64, 256, 1])

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)#([64, 256, 1])
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        """
        y = torch.nn.functional.scaled_dot_product_attention(phi_x, theta_x, g_x, attn_mask=None, dropout_p=0.0, is_causal=False)
        """
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z