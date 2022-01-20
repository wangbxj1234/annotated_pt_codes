from pointnet_util import  square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def index_points(points, idx):#从n个坐标中按照index提取s个坐标或者s*k个坐标，可以进行sampling 或者 sampling&grouping
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C] 
    """
    raw_size = idx.size()#这里是torch size 相当于 numpy的shape
    idx = idx.reshape(raw_size[0], -1) # B,S.  OR.   B,S*K.
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1))) #先把idx的第三维复制到=C,再用gather按索引提取对应点出来。
    return res.reshape(*raw_size, -1)#就是idx的shape（2维或者三维），再加上最后一维c，理论上这个-1也可以写成'points.size(-1)'吧。
    
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.fc_sub = nn.Sequential(
            nn.Linear(d_points, d_points),
            nn.ReLU(),
            nn.Linear(d_points, d_points),
      #      nn.BatchNorm1d(d_points),
       #     nn.ReLU()
        )        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)######b*n*k*f，local attention操作，没有sample，只有group！ （本来应该是b * n * n * f）
        
        x_in = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx) #q取所有点，localatten所以k和v按knn取点。
        #######
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  ### b x n x k x f
        #####
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        ####（做内积的对象是2个， 每个都是 b x n x k x f 
        x_r = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc) ########按照第3个维度做内积，又变成了 b * n * f。
        x_r = self.fc2(x_r) 
        res = self.fc_sub(x_in-x_r) + x_in
 #       res = self.fc2(res) + pre                             
        return res, attn
    