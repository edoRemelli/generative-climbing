import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import torch
import random
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models import VAE


def main(args):

    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()


    """Load the dataset"""
    dataset = MNIST(
        root='data', train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    """Define the loss"""
    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        #BCE = torch.nn.MSELoss()(recon_x.view(-1, 28*28), x.view(-1, 28*28))
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional, variational = args.variational,
        num_labels=10 if args.conditional else 0).to(device)

    """Define the optimizer"""
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):
            """Send data to GPU"""
            x, y = x.to(device), y.to(device)

            """CVAE or VAE generates data"""
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            elif args.variational:
                recon_x, mean, log_var, z = vae(x)
            else:
                recon_x, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                for j in range(args.latent_size):
                    tracker_epoch[id][str(j)] = z[i, j].item()
                tracker_epoch[id]['label'] = yi.item()
            
            """Compute loss"""
            if args.variational or args.conditional:                
                loss = loss_fn(recon_x, x, mean, log_var)
                """Compute KL divergence and binary crossentropy"""
                diverge = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
                #bce = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')/ x.size(0)
                bce = torch.nn.MSELoss()(recon_x.view(-1, 28*28), x.view(-1, 28*28))
                logs['KL divergence'].append(-diverge.item())
                logs['binary cross entropy'].append(bce.item())
                
            else:
                loss = torch.nn.MSELoss()(recon_x.view(-1, 28*28), x.view(-1, 28*28))
                #loss = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')/ x.size(0)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())
            

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                if args.variational or args.conditional:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f} KL {:f} BCE {:f}".format(
                        epoch, args.epochs, iteration, len(data_loader)-1, loss.item(), diverge.item(), bce.item()))
                else:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f} ".format(
                        epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
                
                """Create images from only latent variable"""
                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=10)
            

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).data.cpu().numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
                
                """Reconstruction of already existing images"""
                rnd_id = random.sample(range(1, len(dataset)), 5)
                x = [dataset[i][0] for i in rnd_id]
                x = torch.stack(x)
                x = x.to(device)
                
                c = [dataset[i][1] for i in rnd_id]
                c = torch.stack(c).view(5,1)
                c = c.to(device)
                
                
                for i in range(5):
                    if not args.conditional:
                        x = torch.cat((x,vae(x[i])[0].view(1,1,28, 28)),0)
                    else:
                        x = torch.cat((x,vae(x[i],c[i])[0].view(1,1,28, 28)),0)
                
                
                
                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(5):
                    plt.subplot(5, 2, 2*p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).data.cpu().numpy())
                    plt.axis('off')
                    
                    plt.subplot(5, 2, 2*p+2)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p+5].view(28, 28).data.cpu().numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "Reconstruction E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
        
        """Print the points of the latent space"""
        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')

        if args.representation == None:
            g = sns.lmplot(
            x='0', y='1', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)

        elif args.representation == "PCA":
            pca = PCA(n_components=2)
            feat_cols = [str(i) for i in range(args.latent_size)]
            pca_result = pca.fit_transform(df[feat_cols].values)
            df['pca-one'] = pca_result[:,0]
            df['pca-two'] = pca_result[:,1]
            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
            g = sns.lmplot(
            x='pca-one', y='pca-two', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)

        elif args.representation == "TSNE":     
            time_start = time.time()  
            feat_cols = [str(i) for i in range(args.latent_size)] 
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
            tsne_results = tsne.fit_transform(df[feat_cols].head(500).values)
            df_tsne = df.head(500).copy()
            df_tsne['x-tsne'] = tsne_results[:,0]
            df_tsne['y-tsne'] = tsne_results[:,1]

            print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

            g = sns.lmplot(
            x='x-tsne', y='y-tsne', hue='label', data=df_tsne.groupby('label').head(500),
            fit_reg=False, legend=True)
        
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)
        
        
    """Print curves for loss, KL divergence and BCE"""
    plt.clf()
    plt.plot(logs['loss'], label='Loss')
    plt.savefig(os.path.join(args.fig_root, str(ts),"loss_summary.png"),dpi=300)
    
    if args.variational or args.conditional:
        plt.clf()
        plt.plot(logs['KL divergence'], label='KL divergence')
        plt.savefig(os.path.join(args.fig_root, str(ts),"KL_summary.png"),dpi=300)
        
        plt.clf()
        plt.plot(logs['binary cross entropy'], label='Binary Cross Entropy')
        plt.savefig(os.path.join(args.fig_root, str(ts),"BCE_summary.png"),dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--representation", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--variational", action='store_true')

    args = parser.parse_args()

    main(args)
