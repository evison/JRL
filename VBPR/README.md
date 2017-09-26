# Amazon Dataset RecSys Models

Includes reference implementations for:

* BMR: https://arxiv.org/pdf/1205.2618.pdf
* VBMR: https://arxiv.org/pdf/1510.01784.pdf
* TVBMP: https://arxiv.org/pdf/1602.01585.pdf

recommender systems trained on an Amazon Reviews dataset:

Dataset: http://jmcauley.ucsd.edu/data/amazon/

Original Code: https://sites.google.com/a/eng.ucsd.edu/ruining-he/ ( Ruining He)

## Requirements

Traditional gcc tool-chain setup, i.e. not a mac (LLVM).  * See Docker notes below *

* compiler support for openmp (ubuntu: `sudo apt install libomp-dev`)
* armidillo lib (ubuntu: `sudo apt-get install libarmadillo-dev`)


## Preprocessing

### Transform

All the models need the reviews dataset in a simplified format (CSV), use the `convert_to_simple.py` util to do this:

```
python convert_to_simple.py reviews_Clothing_Shoes_and_Jewelry.json.gz reviews_simple.gz
```

It is now possible to train w/ this output file which is all reviews across all categories of clothing. It is often useful to segment across categories: 

### Segmentation

It is useful for evaluation to segment the clothing dataset into:

* women
* men
* boys
* girls
* baby 

using the `getClothingSubReviews` script.

For input it takes the gzipped simplified output from above and a meta data file which has some category mappings:

 https://storage.googleapis.com/sharknado-recsys-public/productMeta_simple.txt.gz

The command takes the simplified dataset file and the metadata as input:

```
 ./getClothingSubReviews ../data/clothing_full.gz ../data/productMeta_simple.txt
```

 This will generate these segmented review files (which you need to gzip to use in the training program):

```
reviews_Baby.txt
reviews_Boys.txt
reviews_Girls.txt
reviews_Men.txt
reviews_Women.txt
```



## Build Suite

To build the suite, run the `Makefile`, by typing `make` in the project directory. This will build a binary named `train`.


### Training

To train all the models, pass in the following args:

1. Review file path — the simplified dataset
2. Img feature path — the image features
3. Latent Feature Dim. (K) — hyperparameter
4. Visual Feature Dim. (K') — hyperparameter
5. alpha (for WRMF only) — hyperparameter
6. biasReg (regularizer for bias terms) — hyperparameter
7. lambda  (regularizer for general terms) — hyperparameter
8. lambda2 (regularizer for \"sparse\" terms) — hyper-parameter
9. Epochs (number of epochs)
10. Max iterations
11. Corpus/Category name under \"Clothing Shoes & Jewelry\" (e.g. Women)

Although some parameters don't apply to some of the models, they are just passed in bulk.

To start the training routine:

```
./train simple_out.gz image_feat.gz 20 k2 alpha 10 10 lambda2 epoch 10 "Clothing/women"
```

Example Output:

```
{
  "corpus": "simple_out.gz",
  Loading votes from simple_out.gz, userMin = 5, itemMin = 0  ....

  Generating votes data
  "nUsers": 39387, "nItems": 23033, "nVotes": 278677

<<< BPR-MF__K_20_lambda_10.00_biasReg_10.00 >>>

Iter: 1, took 0.266870
Iter: 2, took 0.247799
Iter: 3, took 0.250260
Iter: 4, took 0.248863
Iter: 5, took 0.262415
[Valid AUC = 0.358993], Test AUC = 0.360745, Test Std = 0.305036
Iter: 6, took 0.245542
Iter: 7, took 0.245433
Iter: 8, took 0.236978
Iter: 9, took 0.236842
Iter: 10, took 0.234738
[Valid AUC = 0.610926], Test AUC = 0.611459, Test Std = 0.294283


 <<< BPR-MF >>> Test AUC = 0.611459, Test Std = 0.294283


 <<< BPR-MF >>> Cold Start: #Item = 11453, Test AUC = 0.554613, Test Std = 0.300705

Model saved to Clothing__BPR-MF__K_20_lambda_10.00_biasReg_10.00.txt.
}
```

### Computational Note

The training routine heavy make use of threading. In experimentation we have been able to utilize 15 cores during training. This is especially important during the expensive AUC calculation.

##Docker

To run the docker image:

```
docker run -v ~/Development/DSE/capstone/UpsDowns:/mnt/mac  -ti updowns /bin/bash
```

the `v` flag will mount the source repot to `/mnt/mac` in the container.


## Models

### BPR

On Amazon, regularization hyper-parameter lambda=10 works the best for BPR-MF, MM-MF and VBPR in most cases. 


### VBPR

```
../tools/getClothingSubImgFeatures _image_features_Clothing_Shoes_and_Jewelry.b productMeta_simple.txt.gz
```

```
./train data/reviews_Women.txt data/image_features_Women.b 10 10 na 1 1 1 na 10 "women"
```
