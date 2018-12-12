# Grayscale-Colorization
# Columbia-COMS4995-StockPrediction
In this paper we present a model to automatically colorize black and white images with zero human interference. Our objective is not to match the ground truth, but rather to generate plausible color images that can aesthetically fool a real human observer. Our attempts include implementing a Conditional Deep Convolutional Generative Adversarial Network \cite{nazeri2018image} as well as using the technique of transfer learning to infuse global features into the network from a pre-trained model \cite{IizukaSIGGRAPH2016, Torrey_transferlearning}. Our model is trained on publicly available dataset such as CIFAR-10 and Places365.


## Dependencies
Environment:
- Python 2.7
- Ubuntu 16.04

Packages:
```
sudo apt-get install python-pip python-dev
sudo pip install Scrapy
sudo pip install tensorflow
sudo pip install numpy
sudo apt-get install libatlas-base-dev gfortran
sudo pip install scipy
sudo pip install -U scikit-learn
sudo pip install matplotlib
sudo pip install keras
sudo pip install pandas
```
