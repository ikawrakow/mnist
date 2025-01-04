# Recognizing hand-written digits the old fashioned way

### Why?

The other day @Nexesenex [was asking](https://github.com/ikawrakow/ik_llama.cpp/issues/133) to update [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) to the latest version of [llama.cpp](https://github.com/ggerganov/llama.cpp). Looking into the @Nexesenex' issue I noticed that there are now efforts to turn [ggml](https://github.com/ggerganov/ggml) into an actual machine learning rather than just an inferrence library, and that there is now a `ggml` [mnist example](https://github.com/ggerganov/ggml/tree/master/examples/mnist) showing how to train simple networks to recognize hand-written digits from the classic [MNIST database](https://yann.lecun.com/exdb/mnist/), which reminded me that many years ago I had experimented with `mnist`. After some searching in old backups I found the code and decided to put here the simple k-nearest-neigbor (kNN) approach I had developed.

### Usage

There are no external dependencies, so assuming `cmake, make` and a C++17 capable C++ compiler are installed, just
```
git clone git@github.com:ikawrakow/mnist.git
cd mnist
cmake .
make -j
./mnist_knn_vX
```
whete `X` is `1...4` for the 4 versions discussed below. This will "train" the model (no actual training is required for a kNN model, but some versions will add augmented data, see below) and predict that 10,000 `mnist` test images. For convenience, the `mnist` training and test datasets in the `data` subfolder of this repository (required `git-lfs`). This will produce output such as
```
> ./mnist_knn_v1
Predicting 10000 test images...done in 1498.61 ms -> 0.149861 ms per image

neighbors | error (%)
----------|-----------
     1    |  3.120
     2    |  3.120
     3    |  2.720
     4    |  2.580
     5    |  2.740
     6    |  2.790
     7    |  2.760
     8    |  2.690
     9    |  2.860
    10    |  2.740
    11    |  2.900
    12    |  2.840
    13    |  2.910
    14    |  2.980
    15    |  3.040
    16    |  3.050
    17    |  3.100
    18    |  3.070
    19    |  3.200
    20    |  3.160
```
that shows the error rate (fraction of mispredicted digits) as a function of the number of nearest neighbors used. I prefer to use the prediction error rather than the prediction accuracy as it better shows the performance of the algorithm (a prediction accuracy of 99% does not feel that much different from 98%, but when looking at error rate, 1% is 2 times between than 2%).   

### V1

All we need for a kNN model is a similarity (or distance) metric between pairs of images, and handling (sorting) of nearest neighbours. I'm using a very simple class for the latter, see https://github.com/ikawrakow/mnist/blob/2436864a03dcf5fffa77b022ec3915cddc0c3e34/knnHandler.h#L7 The similarity metric comes from my experience with dealing with medical images. It is a combination of the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and a very simple binary feature vector: for every pixel $j$ of the image compare the grey value $A_j$ of the pixel to the grey value of the 4 neghbouring pixels $A_{j,n}$, and set a bit when $A_j > A_{j, n}$. One than simply counts the number of bits that are set in both images being compared (a simple `popcount(a & b)` instruction per 8 pixels). All this is implemented in 38 LOC, see https://github.com/ikawrakow/mnist/blob/e1aa491b050a9bd8b9f7f12152bf73cfa5240a2d/mnist_knn_v1.cpp#L62

`mnist_knn_v1` needs 0.15 ms per prediction on my Ryzen-7950X CPU. It arrives at an error rate of 2.6-2.7% (see graph below, which shows prediction error as a function of number of nearest neighbours used). This is is not quite as good as the convolutional network in the `ggml` example, which runs in 0.06 ms/prediction on my CPU and arrives at an error rate of about 2%. 

![v1](https://github.com/user-attachments/assets/594ad6f8-1dc9-44ab-968b-37fa1c0c3145)

