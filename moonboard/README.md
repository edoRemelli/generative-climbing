### Generate dataset

Use File_creator.ipynb, can adjust the quality of the problems by increasing the required number of time a problem has been climbed.

Then the data is loaded in the Jupyter notebooks with load_moonbard.py

### Create a neural network to classify and for Inception score

Use NN.ipynb, different architecture : ResNet, CNN, Fullyconnected layers

### Generation

USe Generating.ipynb  for AE, VAE and CVAE generation.

Use GAN-generation.ipynb and cGAN-generation.ipynb for the GAN and the cGAN

Use vaegan.ipynb for vaegan (doesn't work at all for the moment)



### Edit models

All NN models are stored in models.py for the classical net, models_conv.py for the CNN and models_resnet.py for the resnet.

### Scrapper and data

scrapping.ipynb is used for scrapping the data on the website. problems2016.txt and problems2017.txt are the results

### Viewer

This folder contains tools to visualize the problems in a friendly way, never used or tested
