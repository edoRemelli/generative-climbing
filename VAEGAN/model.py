import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import models_resnet as mr
from utils import idx2onehot

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.net = mr.EncoderRes(28, opt.n_z, opt.conditional, not opt.conditional, 10)


    def forward(self, x, c = None):
        mu, logvar = self.net(x, c)

        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        orig = 256
        self.c1 = nn.ConvTranspose2d(orig, orig//2, 2, stride = 2)  # b, 16, 5, 5
        self.relu = nn.ReLU(True)
        self.c2 = nn.ConvTranspose2d(orig//2, orig//4, 2, stride = 2)  # b, 8, 15, 15
        self.c3 = nn.ConvTranspose2d(orig//4, 1, 2, stride = 2, padding = 2)  # b, 1, 28, 28
        self.sig = nn.Sigmoid()
        self.conditional = opt.conditional
        if opt.conditional:
            self.fc = nn.Linear(opt.n_z + 10, orig*4*4)

        else:
            self.fc = nn.Linear(opt.n_z, orig*4*4)


    def forward(self, z, c = None):
        if self.conditional:
            c = idx2onehot(c, n=10)
            z = T.cat((z, c), dim=-1)
        z = self.fc(z)
        z = z.view([-1,256,4,4])
        x = self.c1(z)        
        x = self.relu(x)  
        x = self.c2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.sig(x)
        #x = x.view([-1,28*28])

        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        if not opt.conditional:
            final = 16
        else:
            final = 32
        self.conditional = opt.conditional
        self.convs = nn.Sequential(
            
            nn.Conv2d(1, final//4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final//4, final//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(final//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final//2, final, 4, 2, 2, bias=False),
            nn.BatchNorm2d(final),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(final, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, c = None):
        if self.conditional:
            c = idx2onehot(c, n=28)
            c = c.reshape(-1,1,  28, 1)
            x = T.cat((x, c), dim=-1)
        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()


class VAE(nn.Module):
    def __init__(self, Enc, Dec):
        super(VAE, self).__init__()
        self.E = Enc
        self.D = Dec

    def forward(self, x, y):
        z, mu, logvar = self.E(x,y)
        x = self.D(z,y)
        return x, mu, logvar, z


