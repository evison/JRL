# Amazon Dataset RecSys Models

Includes reference implementations for:

* BMR: https://arxiv.org/pdf/1205.2618.pdf
* VBMR: https://arxiv.org/pdf/1510.01784.pdf

recommender systems trained on an Amazon Reviews dataset:

Dataset: http://jmcauley.ucsd.edu/data/amazon/

Original Code: https://sites.google.com/a/eng.ucsd.edu/ruining-he/ ( Ruining He)

## Requirements

Traditional gcc tool-chain setup, i.e. not a mac (LLVM).  * See Docker notes below *

* compiler support for openmp (ubuntu: `sudo apt install libomp-dev`)
* armidillo lib (ubuntu: `sudo apt-get install libarmadillo-dev`)


## Build 

To build the suite, run the `Makefile`, by typing `make` in the project directory. This will build a binary named `train`.


### Training

Run:

./train wholeData trainData testData imageFile latentFactor visualLatentFactor biasReg lambdaReg lambda2Reg epoch iteration outputPath

wholeData: Whole file path
trainData: Train file path
testData: Test file path
imageFile: Img feature path
latentFactor: Latent Feature Dim. (K)
visualLatentFactor: Visual Feature Dim. (K')
biasReg: biasReg (regularizer for bias terms)
lambdaReg: lambda  (regularizer for general terms)
lambda2Reg: lambda2 (regularizer for \"sparse\" terms)
epoch: #Epoch (number of epochs)
iteration: Max #iter 
outputPath: Corpus/Category name under \"Clothing Shoes & Jewelry\" (e.g. Women)\n

e.g. 

./train /root/share/cx/amazon_baseline_data/Beauty/bpr_whole.txt /root/share/cx/amazon_baseline_data/Beauty/bpr_train.txt /root/share/cx/amazon_baseline_data/Beauty/bpr_test.txt /root/share/cx/amazon_baseline_data/Beauty/image_features_Beauty.b 10 10 0.01 0.01 0.01 10 100 "Beauty"

Data format:

user item rating time word_number words

e.g.

A1Z59RFKN0M5QL 7806397051 1.0 1376611200 23 please dont rachett palette size picture colors sheer slides face wax dont expect  makeup stay put spend money good stuff 2 thumbs



