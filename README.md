# Automating problem-setting in climbing with deep generative models.

## Tentative Abstract for [ICSEHS 2019](https://waset.org/conference/2019/08/vancouver/ICSEHS/abstracts)

Route-setters combine technical craft and artistic skills to reproduce real rock moves on artifical climbing walls. 
In order to both keep the customers of climbing gyms challenged and to deal with the erosion of the holds, routes need to be changed very often. With indoor climbing recent boom, the demand for route-setters is rapidly increasing.

Can we use Machine Learning to automate route-setting and reduce this demand?

As a first step towards this ambitious goal we will consider a setting where the topology of both the wall and the climbing holds is fixed, such as MoonBoards and Tension Boards. Plotted against a grid of lettered and numbered coordinates, each board hold is rotated and set in a specific location. Climbers can share problems with fellow users around the globe via a dedicated app, and with thousands of boards available in climbing gyms around the world, a large quantity of graded routes (>50k) is available online. We propose to make use of this data to automate climbing routes generation by making use of state-of-the-art Machine Learning techniques. Our Generative Model will be able to, given a user specified grade, generate a climbing route of that difficulty. Moreover users will be able to get instaneous feedback on the difficulty of human crafted routes. We believe that this approach will make training with such boards more productive, and represent a fundamental step towards fully automated climbing route generation.

## Relevant reads: 

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

[Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)

[Machine Learning Methods for Climbing Route Classification](http://cs229.stanford.edu/proj2017/final-reports/5232206.pdf)

## Repo structure:

autoencoder/: tutorial on VAEs and CVAEs

## Weekly sync:

[link](https://docs.google.com/document/d/1B3Mo1C-zYg5-EoQoLhTrRd-cdgVfflhsij1-ZslY21g/edit?usp=sharing)
