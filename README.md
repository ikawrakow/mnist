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

### V2

How can we improve? Let's add some translations. Basically, take each training image and add shifted versions of it within `-smax...smax` (in x- and y-directions), where `smax` is a command line parameter to `mnist_knn_v2.cpp`. This indeed improves the prediction accuracy as can be seen in the following graph, which shows results for `smax = 1` (i.e., shifts of +/- 1 pixel). Prediction accuracy is now ~1.9%, so on par with the `ggml` `mnist` example. But this comes at the e4pxense of a much longer run time - 1.4 ms per prediction.

![v2](https://github.com/user-attachments/assets/1022b986-a015-4acc-bbbc-157f97157737)

### V3

Shifting the position of the digits is good, but we also know (see) that elastic deformations would be better. This is what happens in `mnist_knn_v3.cpp`. We generate randomly deformed versions of the training images [here](https://github.com/ikawrakow/mnist/blob/8ad1aa11bdbda0b7e1279965e6e9a06b2d970ef6/imageUtils.cpp#L89). We create random vector deformation fields by
* Sample random vectors in `(-1...1, -1...1)`
* Filter it with a very wide Gaussian filter
* Multiply the resulting filtered deformation vectors with a suitable constant factor $\alpha$

With some experimentation $\sigma = 6, \alpha = 38$ seem to work quite well. With this and 40 deformed versions of the training data we can bring the error down to 1.3-1.4% at the expense of increasing run time even further to about 6.3 ms/prediction.
![v3](https://github.com/user-attachments/assets/8711a269-f3ad-40fb-9884-319bfb1e11ac)

### V4

We have learned that adding translated and deformed versions of the training data is useful for improving prediction accuracy, but this increases the run time significantly. All we need to do now is to a) Combine translations and elastic deformations b) Find a way to skip unlikely training examples via checks that are much cheaper to compute than a full similarity (distance) calculation. This is what is done in `mnist_knn_v4.cpp`. The following two tricks are used to accomplish a) and b)
* After for-/background thresholding, consider the pixels in the foreground as a point cloud and compute moments $M_{kl} = 1/N \sum (x_i - x_0)^k (y_i - y_0)^l$, where $x_0$ and $y_0$ are the coordinates of the image midpoint, $x_i, y_i$ the coordinates of the $i$'th foreground pixel (all measured in pixels), and $N$ is the number of foreground pixels. In practice it is sufficient to use $M_{20}, M_{02}$ and $M_{11}$. If we pre-compute these for all training samples, we have a computationally very cheap check: we skip all training samples where the $L_2$ Euclidean distance between the moments of the image being predicted and the moments of the training sample is greater than some threshold.
* We use the x- and y-projections $P(x) = \sum_y A(x, y)$ and $P(y) = \sum_x A(x, y)$ where $A(x, y)$ is the grey value at position $(x, y)$. This allows us to a) qucikly compute a similarity between a test image and a training sample ($28\dot2$ multiply-adds instead of $28^2$ for a full similarity computation), and b) quickly find a translation of the test images where the x- and y-projection best match the training sample. If the projection similarity after translation is greater than some threshold, the training image is skipped.

The resulting algorithm can be found in `mnist_knn_v4.cpp`. It is more complicated than the very simple `mnist_knn_v1-3.cpp`, but still resonably simple with less than 300 LOC. `mnist_knn_v4` has several command-line options that influence the prediction accuracy vs run time tradeoff. Usage is
```
./mnist_knn_v4 [num_neighbors] [n_add] [thresh] [speed] [beta] [nthread]
```
where
* `num_neighbors` is the number of nearest neighbors to use. This affects run-time/accuracy because there are checks from time to time if all `4 * num_neighbors` nearest neighbors predict the same label and, if so, the calculation is stopped and the corresponding label (digit) is returned as the predicted result. Default value is 5.
* `n_add` is the number odeformations to add per training sample. This has the largest effect on accuracy and speed (see below for some examples)
* `thresh` is the threshold used for the projection similarity. Default values is 0.25, one can get slightly higher accuracy by increasing it to 0.33 or even 0.5 at the expense of a slower prediction
* `speed` can take values of 0, 1 or 2. If 0, the all-same-label check is never performed (see 1st bullet). If 1, the check is done only once after completing all non-deformed training samples (`n_train`). If 2, the check is done after every `n_train` samples. The difference in speed between the 3 options is quite significant with a relatively minor impact on prediction accuracy.
* `beta` is a parameter that determines the relative importance of the Pearson correlation coefficient and the binary feature similarity described above. It should be left at its default value of 1.
* `nthread` is the number of threads to use. If missing, `std::thread::hardware_concurrency()` is used. On systems with hyper-threading enabled or systems with a mix of performance and efficiency cores one may want to specify it explicitely to see if performance can be improved.

Here are some examples of 
| num_neighbors | n_add | thresh | speed | prediction time (ms) | prediction error (%) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 0.25 | 2 | 0.05 | 1.91 |
| 2 | 5 | 0.25 | 2 | 0.08 | 1.31 |
| 2 | 20 | 0.25 | 2 | 0.12 | 1.01 |
| 2 | 200 | 0.25 | 2 | 0.42 | 0.80 |
| 5 | 200 | 0.33 | 2 | 1.10 | 0.69 |
| 8 | 500 | 0.33 | 2 | 3.06 | 0.64 |
| 8 | 500 | 0.33 | 1 | 17.39 | 0.61 |

An error rate of 0.61% is nearly SOTA for kNN. As far as I can tell, [this paper](https://pubmed.ncbi.nlm.nih.gov/17568145) is the only reporting a lower error rate (0.54%). It cannot compete with modern neural networks, but it does beat the `ggml` `mnist` example by a huge margin, and it was only surpassed by a neural network around 2010 or some such. The graph below summarizes the results for the 4 kNN versions.


![v4](https://github.com/user-attachments/assets/7cc164e4-f563-40f2-b6b8-28044f1f2e7f)


