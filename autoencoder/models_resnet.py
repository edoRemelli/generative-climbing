import torch
import torch.nn as nn

from utils import idx2onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, variational=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        self.variational = variational
        self.conditional = conditional

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, variational, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, variational, num_labels)

    def forward(self, x, c=None, testing=False):

        #if x.dim() > 2:
        #    x = x.view(-1, 28*28)

        batch_size = x.size(0)

        """If variational, we compute gaussian parameters"""
        if self.variational or self.conditional:

            if self.conditional:
                means, log_var = self.encoder(x, c)
            else:
                means, log_var = self.encoder(x)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            if testing:
                z = means
            else:
                z = eps * std + means

            recon_x = self.decoder(z, c)

            return recon_x, means, log_var, z
        """If not variational, we just encode z and reconstruct x from z"""
        z = self.encoder(x, c)
        recon_x = self.decoder(z, c)

        return recon_x, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, variational, num_labels):

        super().__init__()

        self.conditional = conditional
        self.variational = variational
        if self.conditional:
            layer_sizes[0] += num_labels

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.model = ResNet([1, 1, 1, 1], 4, start_planes=64, start_padding=2, latent_size=latent_size)

        if self.variational or self.conditional:
            self.linear_means = nn.Linear(128*7*7, latent_size)
            self.linear_log_var = nn.Linear(128*7*7, latent_size)
        else:
            self.linear_z = nn.Linear(128*7*7, latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            c = idx2onehot(c, n=28)
            c = c.reshape(-1,1,  28, 1)
            x = torch.cat((x, c), dim=-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.model(x)
        if not (self.variational or self.conditional):
            return self.linear_z(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, variational, num_labels):

        super().__init__()

        self.model = ResNet_inv(1, (128,7,7), [1, 1, 1, 1], latent_size=latent_size)

        self.conditional = conditional
        self.variational = variational
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size


    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.model(z)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        self.inplanes = inplanes
        self.planes = planes
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDecoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlockDecoder, self).__init__()

        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
            
        self.relu = nn.ReLU(inplace=True)

        if stride > 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes , kernel_size=2, stride=stride)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                layers,
                base_out_total,
                block = BasicBlock,
                latent_size = 2,
                start_planes=8,
                start_padding = 0):

        super(ResNet, self).__init__()
        self.base_out_total = base_out_total
        self.inplanes = start_planes

        
        self.m = nn.ConstantPad2d(start_padding,0)

        self.layer1 = self._make_layer(block, start_planes, layers[0], padding= start_padding)
        self.layer2 = self._make_layer(block, start_planes*2, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, start_planes*4, layers[2], stride=2)

        self.conv2 = nn.Conv2d(start_planes * 2, start_planes * 2, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(start_planes * 2)


    def _make_layer(self, block, planes, blocks, stride=1,padding = 0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride, padding= padding,
                    bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.view([-1,128*7*7])

        return x


class ResNet_inv(nn.Module):
    def __init__(self,
                o_planes,
                shape,
                layers,
                latent_size = 2,
                block = BasicBlockDecoder):

        super(ResNet_inv, self).__init__()
        self.inplanes = shape[0]
        self.shape = shape

        self.layer1 = self._make_layer(block, shape[0], layers[3], stride=1)
        self.layer2 = self._make_layer(block, shape[0] // 2, layers[2], stride=2)
        self.layer3 = self._make_layer(block, shape[0] // 4, layers[1], stride=2)

        self.fc = nn.Linear(latent_size, 128*7*7)

        self.c1 = nn.ConvTranspose2d(32, 1, 1, stride = 1)  # b, 16, 5, 5



    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion , kernel_size=2, stride=stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion))


        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view([-1,self.shape[0],self.shape[1],self.shape[2]])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.c1(x)
        x = nn.Sigmoid()(x)

        return x
