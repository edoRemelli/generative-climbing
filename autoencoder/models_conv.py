import torch
import torch.nn as nn

from utils import idx2onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, variational = False, num_labels=0):

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

    def forward(self, x, c=None, testing = False):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

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
        z = self.encoder(x,c)
        recon_x = self.decoder(z,c)
            
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

        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=3)  # b, 64, 10, 10
        self.relu =  nn.ReLU(True)
        self.pool = nn.MaxPool2d(2)  # b, 64, 5, 5
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # b, 64, 3, 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # b, 64, 3, 3
        #self.fc =  nn.Linear(128*4*4,128)
        

        if self.variational or self.conditional:
            self.linear_means = nn.Linear(128*4*4, latent_size)
            self.linear_log_var = nn.Linear(128*4*4, latent_size)
        else:
            self.linear_z = nn.Linear(128*4*4, latent_size)

    def forward(self, x, c=None):
        x = x.view([-1,1,28,28])
        

        if self.conditional:
            c = idx2onehot(c, n=28)
            c = c.reshape(-1,1,  28, 1)
            x = torch.cat((x, c), dim=-1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        
        x = x.view([-1,128*4*4])
        #x = self.fc(x)
        #print(x.shape)
        
        if not (self.variational or self.conditional):
            return self.linear_z(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, variational, num_labels):

        super().__init__()

        self.conditional = conditional

        
        self.c1 = nn.ConvTranspose2d(128, 64, 2, stride = 2)  # b, 16, 5, 5
        self.relu = nn.ReLU(True)
        self.c2 = nn.ConvTranspose2d(64, 32, 2, stride = 2)  # b, 8, 15, 15
        self.c3 = nn.ConvTranspose2d(32, 1, 2, stride = 2, padding = 2)  # b, 1, 28, 28
        self.sig = nn.Sigmoid()
        
        if self.conditional:
            self.fc = nn.Linear(latent_size + num_labels, 128*4*4)

        else:
            self.fc = nn.Linear(latent_size, 128*4*4)

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
