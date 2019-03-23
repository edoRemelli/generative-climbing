import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import models_resnet as mr

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class MnistResNet(models.ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(models.resnet.BasicBlock, [1, 1, 1, 1], num_classes=10)
        self.conv1 = T.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
    def forward(self, x):
        return super(MnistResNet, self).forward(x)

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        #self.conv1 = nn.Conv2d(1,3,3,padding = 1)
        self.resnet = MnistResNet()
        self.resnet.avgpool = nn.AvgPool2d(4,1,0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.net = mr.EncoderRes(28, opt.n_z, False, True, 10)
        self.x_to_mu = nn.Linear(512,opt.n_z)
        self.x_to_logvar = nn.Linear(512, opt.n_z)

    def reparameterize(self, x):
        #mu = self.x_to_mu(x)
        #logvar = self.x_to_logvar(x)
        mu, logvar = self.net(x)
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, mu, logvar

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.resnet(x).squeeze()
        z, mu, logvar = self.reparameterize(x)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.c1 = nn.ConvTranspose2d(128, 64, 2, stride = 2)  # b, 16, 5, 5
        self.relu = nn.ReLU(True)
        self.c2 = nn.ConvTranspose2d(64, 32, 2, stride = 2)  # b, 8, 15, 15
        self.c3 = nn.ConvTranspose2d(32, 1, 2, stride = 2, padding = 2)  # b, 1, 28, 28
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(opt.n_z, 128*4*4)


    def forward(self, z):
        z = self.fc(z)
        z = z.view([-1,128,4,4])
        x = self.c1(z)        
        x = self.relu(x)  
        x = self.c2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.sig(x)
        #x = x.view([-1,28*28])

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        final = 16
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

    def forward(self, x):
        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()


class VAE(nn.Module):
    def __init__(self, Enc, Dec):
        super(VAE, self).__init__()
        self.E = Enc
        self.D = Dec

    def forward(self, x):
        z, mu, logvar = self.E(x)
        x = self.D(z)
        return x, mu, logvar, z


