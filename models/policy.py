import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from qpth.qp import QPFunction
from cvxopt import matrix, solvers


class BarrierPolicy(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device

        # system specific todo: remove
        self.obs_x = 40  # obstacle location
        self.obs_y = 15
        self.R = 6  # obstacle size
        self.p1 = 0
        self.p2 = 0

        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()
        self.fc32 = nn.Linear(nHidden22, nCls).double()

        # QP params.
        # from previous layers

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x * self.std + self.mean
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)

        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)

        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4 * nn.Sigmoid()(x32)  # ensure CBF parameters are positive

        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)

        return x

    def dCBF(self, x0, x31, x32, sgn, nBatch):

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)

        # system specific
        px = x0[:, 0].reshape(-1, 1)
        py = x0[:, 1].reshape(-1, 1)

        barrier = (px - self.obs_x) ** 2 + (py - self.obs_y) ** 2 - self.R ** 2
        assert barrier.shape == (nBatch, 1)
        barrier_grad = torch.cat([2 * (px - self.obs_x), 2 * (py - self.obs_y)], dim=1)
        assert barrier_grad.shape == (nBatch, self.nCls)

        fx = torch.zeros(self.nCls).unsqueeze(0).expand(nBatch, 1, self.nCls).double().to(self.device)
        assert fx.shape == (nBatch, 1, self.nCls)

        gx = torch.eye(self.nCls).unsqueeze(0).expand(nBatch, self.nCls, self.nCls).double().to(self.device)
        assert gx.shape == (nBatch, self.nCls, self.nCls)

        # compute lie derivative of the barrier function
        barrier = barrier.unsqueeze(1)
        barrier_grad = barrier_grad.unsqueeze(1)
        Lfbarrier = torch.bmm(barrier_grad, fx.view(nBatch, self.nCls, 1))
        assert Lfbarrier.shape == (nBatch, 1, 1)

        Lgbarrier = torch.bmm(barrier_grad, gx)
        assert Lgbarrier.shape == (nBatch, 1, self.nCls)

        #barrier_dot = 2 * (px - self.obs_x) * vx + 2 * (py - self.obs_y) * vy

        G = -Lgbarrier
        #G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        h = Lfbarrier + x32[:, 0] * barrier

        e = Variable(torch.Tensor()).to(self.device)

        if self.training or sgn == 1:
            x = QPFunction(verbose=0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0, 0]
            self.p2 = x32[0, 1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())

        return x

def solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

    return sol['x']