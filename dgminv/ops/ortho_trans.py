import torch
import torch.nn as nn
import numpy as np

class OrthoHH(nn.Module):
    ''' Reparameterize w by orthgonal layers'''
    def __init__(self, feature_dim, n_mat=1, seed=0):
        super().__init__()
        # self.n_mat = nbatch if n_mat == None else 1
        self.n_mat = n_mat
        self.feature_dim = feature_dim
        rs = np.random.RandomState(seed=seed)
        self.hd_vecs = nn.Parameter(torch.from_numpy(rs.randn(self.n_mat, self.feature_dim, self.feature_dim)).type(torch.float32))
        # self.hd_vecs = nn.Parameter(torch.zeros(self.n_mat, self.feature_dim, self.feature_dim))

    def get_hd_matrix(self, ind):
        # print(f'rot_dim = {self.rot_dim}')
        Q = torch.eye(self.feature_dim).to(self.hd_vecs.device)
        for i in range(self.feature_dim):
            v = self.hd_vecs[ind, i,:].reshape(-1,1)
            v = v / (v.norm() + 1e-6)
            Qi = torch.eye(self.feature_dim).to(self.hd_vecs.device) - 2.0 * torch.mm(v, v.T)
            Q = torch.mm(Q,Qi)
        return Q    

    def forward(self, x):
        ''' x: shape: [npatches, patchdim]'''
        nbatch = x.shape[0]
        th_patches_vec = x.view(nbatch, -1)

        if self.n_mat == 1:
            hd_m = self.get_hd_matrix(0)
            out = torch.mm(hd_m, th_patches_vec.T).T
        else:
            # v2
            assert self.n_mat == nbatch, "numbers of mat does not match batch size!!!"
            out_list = []
            for i in range (self.n_mat):
                hd_m = self.get_hd_matrix(i)
                out_list.append(torch.matmul(hd_m, th_patches_vec.T[:,i]).T.unsqueeze(0))
            print(len(out_list))
            print(out_list[-1].shape)
            out = torch.cat(out_list, dim=0)
            print(f'out shape = {out}')


        # with torch.no_grad():
        #     print(torch.mm(out.T, out)/nbatch)

        return out

class OrthoCP(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.w = nn.Parameter(torch.zeros(self.feature_dim, self.feature_dim))
        # self.w = nn.Parameter(torch.randn(self.feature_dim, self.feature_dim))
    
    def get_orthog_matrix(self):
        upper = torch.triu(self.w, diagonal=1)
        skew = upper - upper.T
        Q1 = torch.eye(self.feature_dim).to(skew.device) + skew
        Q2 = torch.inverse(torch.eye(self.feature_dim).to(skew.device) - skew)
        return torch.mm(Q1, Q2)
    
    def forward(self, x):
        ''' x: shape: [npatches, patchdim]'''
        Q = self.get_orthog_matrix()
        nbatch = x.shape[0]
        th_patches_vec = x.view(nbatch, -1)
        return torch.mm(Q, th_patches_vec.T).T

if __name__ == '__main__':
    ocp = OrthoCP(32)
    x = torch.randn(256, 32)
    print(ocp(x))
    Q = ocp.get_orthog_matrix()
    print(torch.mm(Q, Q.T))