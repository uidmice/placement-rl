import torch

class StarNetwork:
    def __init__(self, delay, bw, speed):
        assert len(delay) == len(bw), 'Network delay and bw are for different numbers of devices'
        assert len(delay) == len(speed), 'Confusion on the number of devices in the network'

        self.n_devices = len(delay)

        self.delay = delay
        self.bw = bw
        self.dev_feature = torch.reshape(torch.tensor(speed), (self.n_devices,1))
        self.dev_feature = 1/self.dev_feature

        self.C1 = 0.2
        self.C2 = 30

        self.D = torch.zeros((self.n_devices, self.n_devices))
        self.R = torch.zeros((self.n_devices, self.n_devices))
        for i in range(self.n_devices):
            self.D[i,i] = 0
            self.R[i,i] = 0
            for j in range(i + 1, self.n_devices):
                self.D[i, j] = self.delay[i] + self.delay[j] + self.C2
                self.D[j, i] = self.D[i, j]
                self.R[i, j] = 1/self.bw[i] + 1/self.bw[j] + self.C1
                self.R[j, i] = self.R[i, j]

        self.D_r = (self.D - torch.mean(self.D))/torch.std(self.D)
        self.R_r = (self.R - torch.mean(self.R)) / torch.std(self.R)

        self.net_feature = torch.stack([self.D, self.R], dim=2)
        self.net_feature_norm = torch.stack([self.D_r, self.R_r], dim=2)
        self.dev_feature_norm = (self.dev_feature - torch.mean(self.dev_feature)) / torch.std(self.dev_feature)

    def get_node_feature(self, node):
        return self.dev_feature_norm[node]

    def get_node_feature_dim(self):
        return self.dev_feature_norm.shape[1]

    def get_edge_feature(self, node1, node2):
        return torch.squeeze(self.net_feature_norm[node1, node2])

    def get_edge_feature_dim(self):
        return self.net_feature_norm.shape[2]




