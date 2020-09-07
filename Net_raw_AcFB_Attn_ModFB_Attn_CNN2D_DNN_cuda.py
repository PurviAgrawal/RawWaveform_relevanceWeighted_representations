import torch
import torch.nn as nn
import scipy
import scipy.signal
import scipy.io as sio
import numpy as np
import numpy
import torch.nn.functional as F

class Relevance_weights_acousticFB(torch.nn.Module):
    def __init__(self):
        super(Relevance_weights_acousticFB, self).__init__()
        self.ngf = 80
        self.fc1 = nn.Linear(101, 50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x is (B, 1, patch_len=t=101, ngf=f=80)
        batch_size = x.shape[0]
        x = x.permute(0,3,2,1)   # B,80,101,1
        x = x.reshape(batch_size * x.shape[1], -1) # Bx80, 101
        x = self.sigmoid(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(batch_size, -1)  # B, 80
        # print out.shape
        out = self.softmax(x)
        return out
        
class Relevance_weights_modFB(torch.nn.Module):
    def __init__(self):
        super(Relevance_weights_modFB, self).__init__()
        self.num_mod_filt = 40

        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))
        self.fc1 = nn.Linear(494, 100)  # 19*26=494
        self.fc2 = nn.Linear(100, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is (B, mod_filt=K=40, t'=19, f'=78)
        batch_size = x.shape[0]
        x = self.pool1(x)  # B,40,19,26
        x = x.reshape(batch_size * x.shape[1], 1, x.shape[2], x.shape[3]) # Bx40,1,19,26
        x = x.reshape(x.shape[0], -1)  # Bx40, 494
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, self.num_mod_filt) # B, 40
        # print out.shape
        out = self.softmax(x)
        return out
        
        
class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.ngf = 80
        self.patch_length = 101 # t
        self.initial_splice = 50
        self.final_splice = 10

        self.filt_h = 129  # k
        self.padding = 64
        self.win_length = 400 # s

        self.len_after_conv = 400
        self.hamming_window = torch.from_numpy(scipy.signal.hamming(self.win_length)).float().cuda()

        # Learnable Mean parameter (for Gaussian kernels)
        self.means = torch.nn.Parameter((torch.rand(self.ngf)*(-5) + 1.3).float().cuda()) # \mu
        t = range(-self.filt_h/2, self.filt_h/2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float().cuda() + 1

        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_conv, 1), stride=(1, 1))
        
        # Initializing the two relevance sub-networks
        self.relevance_filter_wts_acousticFB = Relevance_weights_acousticFB()
        self.relevance_filter_wts_modFB = Relevance_weights_modFB()

        # Modulation filtering parameters
        self.num_mod_filt = 40

        self.conv1 = nn.Conv2d(1, self.num_mod_filt, kernel_size=(5,5), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))
        self.conv2   = nn.Conv2d(self.num_mod_filt, self.num_mod_filt, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(1,3))

        self.bn_1 = torch.nn.BatchNorm2d(self.num_mod_filt, eps=1e-4, affine=True, track_running_stats=True)
        self.instance_norm = torch.nn.InstanceNorm2d(self.ngf, eps=1e-4)
        self.d1    = nn.Dropout(0.2)
        self.fc1   = nn.Linear(5440,2048)  # 40*17*8 = 5440
        self.d2    = nn.Dropout(0.2)
        self.fc2   = nn.Linear(2048,2048)
        self.fc3   = nn.Linear(2048,2048)
        self.d3    = nn.Dropout(0.2)
        self.fc4   = nn.Linear(2048,2048)
        self.fc5   = nn.Linear(2048,2040)
        #self.d4    = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Updating 1-D Gaussian kernels (acoustic FB) using updated Means Parameter
        means_sorted = torch.sort(self.means)[0]
        kernels = (torch.zeros([self.ngf, self.filt_h]).cuda())
        for i in range(self.ngf):
            kernels[i, :] = torch.cos(np.pi * torch.sigmoid(means_sorted[i]) * self.temp) * torch.exp(- (((self.temp)**2)/(2*(((1/(torch.sigmoid(means_sorted[i])+1e-3))*10)**2 + 1e-5))))

        kernels = (torch.reshape(kernels, (kernels.shape[0], 1, kernels.shape[1], 1)))

        # input x is (B, C=1, t=101, s=400)
        x = x * self.hamming_window
        x = x.permute(0,2,3,1) # B,t,s,1
        x = x.contiguous()
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size*self.patch_length, 1, self.win_length, 1)) # Bx101,1,s=400,1

        # Acoustic FB layer
        x = F.conv2d(x, kernels, padding = (self.padding, 0)) # 1-D Conv as acoustic filtering --> Bx101, f=80, s=400, 1
        x = torch.reshape(x, (batch_size, self.patch_length, self.ngf, self.len_after_conv)).permute(0,2,3,1)   # Bx101,80,400,1 --> B,101,80,400 --> B, 80, 400, 101
        x = torch.abs(x)
        x = torch.log(self.avg_pool_layer(x**2)+1e-3)  # B,f=80,1,t=101 - Square, avg pool and log
        x = x.permute(0, 2, 3, 1)  # Spectogram B, 1, t=101, f=80
        
        # Now x is (B, 1, t=101, f=80)
        # Calling acoustic FB relevance sub-network
        filter_wts_acousticFB = self.relevance_filter_wts_acousticFB(x)
        filter_wts_acousticFB = filter_wts_acousticFB.reshape(filter_wts_acousticFB.shape[0], filter_wts_acousticFB.shape[1], 1, 1) # B, 80, 1, 1

        x = filter_wts_acousticFB * x.permute(0,3,2,1) # B, 80, 101, 1 - Multiplied the acoustic FB relevance weights
        x = self.instance_norm(x)  # Instance Normalization

        batch_size = x.shape[0]

        x = x.permute(0,3,2,1)

        # Pruning to (B, 1, 21, f=80)
        x = x[:, :, self.initial_splice - self.final_splice : self.initial_splice + self.final_splice + 1, :] # Spec B, 1, (2*final_splice+1), 80

        # Modulation filtering layer (non-parametric)
        x = self.sigmoid(self.conv1(x))  # B, 40, 19, 78

        # Calling modulation filtering relevance sub-network
        filter_wts_modFB = self.relevance_filter_wts_modFB(x)
        filter_wts_modFB = filter_wts_modFB.reshape(filter_wts_modFB.shape[0], filter_wts_modFB.shape[1], 1, 1) # B, 40, 1, 1 - Modulation relevance weights

        x = filter_wts_modFB * x  # Multiplied the modulation FB relevance weights
        x = self.bn_1(x) # Batch normalization

        x = self.pool1(x)  # B,40, 19, 26
        x = self.sigmoid(self.conv2(x))  # B, 40, 17, 24
        x = self.pool2(x)  # B, 40, 17, 8
        x = torch.reshape(x, (batch_size, -1))  # 40*17*8 = 5440
        x = self.d1(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.d2(x)
        x = self.sigmoid(self.fc3(x))
        x = self.d3(x)
        x = self.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x
