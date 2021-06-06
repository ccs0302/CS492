import torch
import torch.nn as NN
import numpy as np
from random import sample
import math_lib.se3 as se3
import math_lib.invmat as invmat

# a class to generate MLP network
class MLP(NN.Module):
    def __init__(self, nch_input, nch_layers):
        super().__init__()
        layers = []
        last = nch_input
        for i, outp in enumerate(nch_layers):
        	weights = NN.Conv1d(last, outp, 1)	        
        	layers.append(weights)
	        layers.append(NN.BatchNorm1d(outp, momentum=0.1))
        	layers.append(NN.ReLU())
        	last = outp        
        self.layers = NN.Sequential(*layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


# encoder network
class PointNet(NN.Module):
    def __init__(self):
        super().__init__()

        mlp_h1 = [64, 64]
        mlp_h2 = [64, 128, 1024]

        self.h1 = MLP(3, mlp_h1).layers
        self.h2 = MLP(mlp_h1[-1], mlp_h2).layers
        

    def forward(self, points):
        # feature extraction
        x = points.transpose(1, 2)  # [B, 3, N]
        x = self.h1(x)
        x = self.h2(x)  # [B, K, N]        
        r = NN.functional.max_pool1d(x, x.size(-1))       
        x = r.view(r.size(0), -1)
        return x



class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
#        self.num_points = 2048
#        self.bottleneck_size = 1024
        self.bn1 = NN.BatchNorm1d(1024)
        self.bn2 = NN.BatchNorm1d(1024 // 2)
        self.bn3 = NN.BatchNorm1d(1024 // 4)
        self.fc1 = NN.Linear(1024, 1024)
        self.fc2 = NN.Linear(1024, 1024 // 2)
        self.fc3 = NN.Linear(1024 // 2, 1024 // 4)
        self.fc4 = NN.Linear(1024 // 4, 2048 * 3)
        self.th = NN.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]        
        m = NN.ReLU() 
        x = m(self.bn2(self.fc2(x)))
        x = m(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, 2048).transpose(1, 2).contiguous()
        return x


# the neural network of feature-metric registration
class RegistrationRegression(torch.nn.Module):
    def __init__(self, ptnet, decoder=None, isTest=False):
        super().__init__()
        # network
        self.encoder = ptnet
        self.decoder = decoder

        # functions
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # initialization for dt: [w1, w2, w3, v1, v2, v3], 3 rotation angles and 3 translation
        delta = 1.0e-2  # step size for approx. Jacobian (default: 1.0e-2)
        dt_initial = torch.autograd.Variable(torch.Tensor([delta, delta, delta, delta, delta, delta]))
        self.dt = NN.Parameter(dt_initial.view(1, 6), requires_grad=True)

        # results
        self.last_err = None
        self.g_series = None  # for debug purpose
        self.prev_r = None
        self.g = None  # estimated transformation T
        self.isTest = isTest # whether it is testing

    # estimate T
    def estimate_t(self, p0, p1, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
        # normalization
        if p0_zero_mean:
            p0_m = p0.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a0 = a0.clone()
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0
        if p1_zero_mean:
            p1_m = p1.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a1 = a1.clone()
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        # use IC algorithm to estimate the transformation
        g0 = torch.eye(4).to(q0).view(1, 4, 4).expand(q0.size(0), 4, 4).contiguous()
        r, g, loss_ende = self.ic_algo(g0, q0, q1, maxiter, xtol, is_test=self.isTest)
        self.g = g

        # re-normalization
        if p0_zero_mean or p1_zero_mean:
            # output' = trans(p0_m) * output * trans(-p1_m)
            #        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
            #          [0, 1    ]   [0, 1 ]   [0,  1    ]
            est_g = self.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            self.g = est_g

            est_gs = self.g_series  # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            self.g_series = est_gs

        return r, loss_ende

    # Iverted Compositional algorithm
    def ic_algo(self, g0, p0, p1, maxiter, xtol, is_test=False):
        
        training = self.encoder.training
        # training = self.decoder.training
        batch_size = p0.size(0)

        self.last_err = None
        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        # generate the features
        f0 = self.encoder(p0)
        f1 = self.encoder(p1)

        # task 1
        loss_enco_deco = 0.0
        if not is_test:
            decoder_out_f0 = self.decoder(f0)
            decoder_out_f1 = self.decoder(f1)

            p0_dist1, p0_dist2 = self.chamfer_loss(p0.contiguous(), decoder_out_f0)  # loss function
            loss_net0 = (torch.mean(p0_dist1)) + (torch.mean(p0_dist2))
            p1_dist1, p1_dist2 = self.chamfer_loss(p1.contiguous(), decoder_out_f1)  # loss function
            loss_net1 = (torch.mean(p1_dist1)) + (torch.mean(p1_dist2))
            loss_enco_deco = loss_net0 + loss_net1

        self.encoder.eval()  # and fix them.

        # task 2
        f0 = self.encoder(p0)  # [B, N, 3] -> [B, K]
        # approx. J by finite difference
        dt = self.dt.to(p0).expand(batch_size, 6)  # convert to the type of p0. [B, 6]
        J = self.approx_Jac(p0, f0, dt)
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2)  # [B, 6, K]
            H = Jt.bmm(J)  # [B, 6, 6]
            # H = H + u_lamda * iDentity
            B = self.inverse(H)
            pinv = B.bmm(Jt)  # [B, 6, K]
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            print(err)
            f1 = self.encoder(p1)  # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, -1

        itr = 0
        r = None
        for itr in range(maxiter):
            p = self.transform(g.unsqueeze(1), p1)  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f1 = self.encoder(p)  # [B, N, 3] -> [B, K]
            r = f1 - f0  # [B,K]
            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0  # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()
            self.prev_r = r

        self.encoder.train(training)
        return r, g, loss_enco_deco

    # estimate Jacobian matrix
    def approx_Jac(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :])  # [6, 6]
            D = self.exp(-d)  # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  # [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1))  # x [B, 1, N, 3] -> [B, 6, N, 3]

        f0 = f0.unsqueeze(-1)  # [B, K, 1]
        f1 = self.encoder(p.view(-1, num_points, 3))
        f = f1.view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]

        df = f0 - f  # [B, K, 6]
        J = df / dt.unsqueeze(1)  # [B, K, 6]

        return J

    # update the transformation
    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    # calculate the chamfer loss
    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        diag_ind = torch.arange(0, num_points)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction='sum')

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)              
        loss = NN.MSELoss()
        output = loss(A, I) *16
        return output
        #output.backward()
        #output.backward() *16
        
        #return torch.nn.functional.mse_loss(A, I, reduction='mean') * 16


# main algorithm class
class Train:
    def __init__(self):
        self.dim_k = 1024
        self.num_points = 2048
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 0  # 0: unsupervised, 1: semi-supervised see. self.compute_loss()

    def create_model(self):
        # Encoder network: feature eatraction for every point. Nx1024
        ptnet = PointNet()
        # Decoder network: decode the feature into points
        decoder = Decoder()
        # Estimate the transformation T
        regression = RegistrationRegression(ptnet, decoder,isTest=False)
        return regression

    def compute_loss(self, solver, data, device):
        p0, p1, igt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source
        igt = igt.to(device)  # igt: p0 -> p1
        r, loss_ende = solver.estimate_t(p0, p1, self.max_iter)
        loss_r = solver.rsq(r)
        est_g = solver.g
        loss_g = solver.comp(est_g, igt)

        # unsupervised learning, set max_iter=0
        if self.max_iter == 0:
            return loss_ende

        # semi-supervised learning, set max_iter>0
        if self._loss_type == 0:
            loss = loss_ende
        elif self._loss_type == 1:
            loss = loss_ende + loss_g
        elif self._loss_type == 2:
            loss = loss_r + loss_g
        else:
            loss = loss_g
        return loss

    def train(self, model, trainloader, optimizer, device):
        model.train()

        Debug = True
        total_loss = 0
        if Debug:
            epe = 0
            count = 0
            count_mid = 9
        for i, data in enumerate(trainloader):
            loss = self.compute_loss(model, data, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            if Debug:
                epe += loss_item
                if count % 10 == 0:
                    print('i=%d, fmr_loss=%f ' % (i, float(epe) / (count_mid + 1)))
                    epe = 0.0
            count += 1
        ave_loss = float(total_loss) / count
        return ave_loss

    def validate(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                loss_net = self.compute_loss(model, data, device)
                vloss += loss_net.item()
                count += 1

        ave_vloss = float(vloss) / count
        return ave_vloss


class RegressionTest:
    def __init__(self, args):
        self.filename = args.outfile
        self.dim_k = args.dim_k
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        # Encoder network
        ptnet = PointNet()
        # Decoder network
        decoder = Decoder()
        # estimate the transformation T
        regration = RegistrationRegression(ptnet, decoder, isTest=True)
        return regration

   
   


