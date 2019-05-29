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
            x = x.view(-1, 3*18*11)

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

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        if self.variational or self.conditional:
            self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
            self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        else:
            self.linear_z = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            
            c = idx2onehot(c, n=13)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)
        
        if not (self.variational or self.conditional):
            return self.linear_z(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, variational, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        self.variational = variational
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=13)
            z = torch.cat((z, c), dim=-1)
            
        x = self.MLP(z)

        return x
