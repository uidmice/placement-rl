import torch

class StarNetwork:
    def __init__(self, delay, bw, speed):
        assert len(delay) == len(bw), 'Network delay and bw are for different numbers of devices'
        assert len(delay) == len(speed), 'Confusion on the number of devices in the network'

        self.n_devices = len(delay)

        self.delay = delay
        self.bw = bw
        self.comp_rate = 1 / torch.tensor(speed)

        self.comm_delay = torch.zeros((self.n_devices, self.n_devices))
        self.comm_rate = torch.zeros((self.n_devices, self.n_devices))
        for i in range(self.n_devices):
            self.comm_delay[i, i] = 0
            self.comm_rate[i, i] = 0
            for j in range(i + 1, self.n_devices):
                self.comm_delay[i, j] = self.delay[i] + self.delay[j]
                self.comm_delay[j, i] = self.comm_delay[i, j]
                self.comm_rate[i, j] = 1 / self.bw[i] + 1 / self.bw[j]
                self.comm_rate[j, i] = self.comm_rate[i, j]
        #
        # self.D_r = (self.D - torch.mean(self.D))/torch.std(self.D)
        # self.R_r = (self.R - torch.mean(self.R)) / torch.std(self.R)

        # self.net_feature_norm = torch.stack([self.D_r, self.R_r], dim=2)
        # self.dev_feature_norm = (self.dev_feature - torch.mean(self.dev_feature)) / torch.std(self.dev_feature)

    # def get_comp_rate(self, node):
    #     return self.comp_rate[node]
    #
    # # def get_node_feature_dim(self):
    # #     return self.comp_rate['rate'].shape[1]
    #
    # def get_edge_feature(self, node1, node2):
    #     return torch.squeeze(self.net_feature_norm[node1, node2])
    #
    # def get_edge_feature_dim(self):
    #     return self.net_feature_norm.shape[2]




class FullNetwork:
    def __init__(self, delay, comm_rate, speed, device_constraints: dict):
        self.n_devices = len(speed)
        self.comp_rate = 1 / torch.tensor(speed)
        self.comm_delay = torch.tensor(delay)
        self.comm_rate = torch.tensor(comm_rate)
        self.device_constraints = device_constraints


