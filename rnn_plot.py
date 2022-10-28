import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


'''
-pass in a folder with existing training data with an rnn.pk 
located in the same place as the training data the rnn used to train itself 
-names chart based on the parent folder of where the rnn pickle is located'''
def rnn_plot(folder):
    for (root, dirs, files) in os.walk(folder, topdown=True):
        rnnpk=None
        if "rnn_noise_0.pk" in files: rnnpk="rnn_noise_0.pk"
        elif "rnn_noise_0.2.pk" in files: rnnpk="rnn_noise_0.2.pk"
        if rnnpk!=None:
            f=open(root+'/'+"rnn_noise_0.pk",'rb')
            print("analysing"+root)
            results, _rewards, _runtime, _size = pickle.load(f) 
            f.close()
            torch.Tensor.ndim = property(lambda self: len(self.shape))
            episodes=[i for i in range(1,101)]
            res=[]
            for L in results:
                r=[abs(L[i]-L[i-1]) for i in range(1,100,2)]
                res.extend([r])
            avgs = np.array(res)
            avgrange=[i for i in range(50)]
            avgplot=np.average(avgs,axis=0)
            plt.xlabel('every 2 training episodes', size=13)
            plt.ylabel('average difference between previous and current ep', size=13)
            plt.scatter(avgrange,avgplot)
            plt.plot(avgrange,avgplot)
            plt.savefig(root+'/'+"rnn_graph.pdf")
            plt.clf()
            print(f"saved {root}'s graph")

rnn_plot("rnn_runs/multiple_networks/2022-06-15_22-33-56_giph")