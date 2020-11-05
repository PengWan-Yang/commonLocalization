import torch
from torch import nn
from torch.nn import functional as F
import math

class LayerNorm(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True, dropout= 0.2, out_channels=512):
        super(BasicBlock, self).__init__()
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Linear(512, 512)
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = nn.Linear(512, 512)

        if bn_layer:
            self.W = nn.Linear(512, out_channels)

        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = nn.Linear(512, 512)
        self.phi = nn.Linear(512, 512)

        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm()

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, query, support):
        '''
        :param query: (b, c, t, h, w)
        :return:
        '''

        batch_size = query.size(0)
        n_s = support.size(0)

        # round0
        g_x = self.g(support).view(n_s, -1)
        theta_x = self.theta(query).view(batch_size, -1)#BxCxHW
        phi_x = self.phi(support).view(n_s, -1)#BxCxHW
        phi_x = phi_x.permute(1, 0)
        f = torch.matmul(theta_x, phi_x)#BxHWxHW
        f = f/math.sqrt(self.in_channels)#rescale
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)#BxHWxC
        y = self.ln(y)#layer normalization in last dim
        y = y.permute(0, 1).contiguous()#BxCxHW
        tmp = F.relu(y)
        tmp = self.W(tmp)
        W_y = self.dropout(tmp)
        z = W_y + query
        return z

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, shot,  sub_sample=False):
        super(_NonLocalBlockND, self).__init__()
        self.shot = shot
        self.nor = nn.Sigmoid()
        self.in_channels = in_channels

        # pyramid
        self.round_1 = BasicBlock(in_channels=in_channels, sub_sample=sub_sample)
        self.round_2 = BasicBlock(in_channels=in_channels, sub_sample=sub_sample)
        self.round_3 = BasicBlock(in_channels=in_channels, sub_sample=sub_sample)

        # coexcitation
        self.round_co_1 = BasicBlock(in_channels=in_channels, sub_sample=sub_sample)
        self.round_co_2 = BasicBlock(in_channels=in_channels, sub_sample=sub_sample)
        # similarity
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, query, support):
        query_co = query
        support_co = support

        # coexcitation
        query_co = self.round_co_1(query, support)
        support_co = self.round_co_2(support, query)

        # pyramid
        z_1 = self.round_1(query_co, support_co)
        z_2 = self.round_2(z_1, support_co)
        query_fused = self.round_3(z_2, support_co)
        output = query_fused

        # similarity
        st = support.shape[0]  # support SXT
        t = int(st / self.shot)  # support T
        weight_vector = 0
        for i in range(self.shot):
            support_i = support[i * t:(i + 1) * t]
            support_i = torch.mean(support_i, dim=0, keepdim=True)
            # cosine
            cosine = self.cos(query, support_i.expand(query.shape[0], self.in_channels))
            cosine = torch.unsqueeze(cosine, 1)
            # l2_distance
            l2_distance = torch.norm(query - support_i.expand(query.shape[0], self.in_channels), p=2, dim=1)
            l2_score = self.nor(-l2_distance * 0.01)
            l2_score = torch.unsqueeze(l2_score, 1)
            weight_vector = weight_vector + (l2_score * cosine) / self.shot
        query_weighted = query_fused * weight_vector
        output = query_weighted
        return output



class fusion_modules(_NonLocalBlockND):
    def __init__(self, in_channels, shot,sub_sample=False):
        super(fusion_modules, self).__init__(in_channels=in_channels,
                                               sub_sample=sub_sample,
                                              shot=shot)


if __name__ == '__main__':
    pass

