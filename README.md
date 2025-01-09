# Recognizing hand-written digits the old fashioned way

### TL;DR

This repository contains [k-nearest-neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (kNN) and [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) algorithms for recognizing hand-written digits from the [MNIST database](https://yann.lecun.com/exdb/mnist/).

kNN and SVM and not a neural network in the year 2025? Seriously?

I know. But the other day @Nexesenex [was asking](https://github.com/ikawrakow/ik_llama.cpp/issues/133) to update [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) to the latest version of [llama.cpp](https://github.com/ggerganov/llama.cpp). Looking into the @Nexesenex' issue I noticed that there are now efforts to turn [ggml](https://github.com/ggerganov/ggml) into an actual machine learning rather than just an inference library, and that there is now a `ggml` [mnist example](https://github.com/ggerganov/ggml/tree/master/examples/mnist) showing how to train simple networks to recognize hand-written digits from the classic [MNIST database](https://yann.lecun.com/exdb/mnist/), which reminded me that many years ago I had experimented with `mnist`. After some searching in old backups I found the code and decided to put here kNN and SVM approaches I had developed. The best of the kNN algorithms arrives at a prediction error of 0.61%. The SVM algorithm reaches 0.38%. As far as I can tell, these results are nearly SOTA or SOTA for the respective algorithm type (I haven't done any extended literature search, so these claims are based on what I see on the Wikipedia [mnist entry](https://en.wikipedia.org/wiki/MNIST_database), so I may be wrong).   

### Usage

There are no external dependencies, so assuming `cmake, make` and a C++17 capable C++ compiler are installed, just
```
git clone git@github.com:ikawrakow/mnist.git
cd mnist
cmake .
make -j
```
To run one of the kNN algorithms,
```
./mnist_knn_vX
```
where `X` is `1...4` for the 4 versions discussed below. This will "train" the model (no actual training is required for a kNN model, but some versions will add augmented data, see below) and predict that 10,000 `mnist` test images. For convenience, the `mnist` training and test datasets in the `data` sub-folder of this repository (required `git-lfs`). This will produce output such as
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

The SVM algorithm must first be trained. For a quick example
```
> ./mnist_svm_train 200 0 100
Pattern::apply: nimage=60000 np=24 Nx=28 Ny=28 Nx1=24 Ny1=24 Nx2=12 Ny2=12
Pattern::apply: nimage=10000 np=24 Nx=28 Ny=28 Nx1=24 Ny1=24 Nx2=12 Ny2=12
Pattern::apply: nimage=1440000 np=20 Nx=12 Ny=12 Nx1=10 Ny1=10 Nx2=5 Ny2=5
Pattern::apply: nimage=240000 np=20 Nx=12 Ny=12 Nx1=10 Ny1=10 Nx2=5 Ny2=5
12000 points per image
Using chunk=128
  Iteration  10: F=2.367736(2.325747) sump2=0.00041989, Ngood=58684,58684,9793  time=0.452434 s
  Iteration  20: F=1.129476(0.974787) sump2=0.00154688, Ngood=59498,59498,9885  time=0.747391 s
  Iteration  40: F=0.762469(0.461445) sump2=0.00301025, Ngood=59847,59847,9917  time=1.33499 s
  Iteration  60: F=0.737481(0.411337) sump2=0.00326144, Ngood=59877,59877,9923  time=2.00638 s
  Iteration  80: F=0.735779(0.407476) sump2=0.00328304, Ngood=59881,59881,9922  time=2.76658 s
  Iteration 100: F=0.735688(0.407481) sump2=0.00328207, Ngood=59880,59880,9922  time=3.60235 s
  Iteration 120: F=0.735685(0.407511) sump2=0.00328174, Ngood=59880,59880,9922  time=4.4851 s
Terminating due to too small change (9.75132e-05) in the last 50 iterations
Total time: 4990.98 ms, I-time: 9.299 ms, F-time: 124.97 ms, S-time: 1672.43 ms, U-time: 229.702 ms, V-time: 1673.31 ms, P-time: 1052.07 ms, C-time: 228.644 ms
Training: ngood = 59880 (0.998)  59880 (0.998)
0  9922
1  9988
2  9995
3  9999
4  10000
Wrote training results to test.dat
```
we get 0.9922 accuracy (0.78% prediction error) after 5 seconds of training. To run prediction with the just trained model written to `test.dat`
```
./mnist_svm_predict test.dat
============================== Dataset test.dat
Pattern::apply: nimage=10000 np=24 Nx=28 Ny=28 Nx1=24 Ny1=24 Nx2=12 Ny2=12
Pattern::apply: nimage=240000 np=20 Nx=12 Ny=12 Nx1=10 Ny1=10 Nx2=5 Ny2=5
# 12000 points per image
Predicted 10000 images in 103.289 ms -> 10.3289 us per image

0  9922
1  9988
2  9995
3  9999
4  10000
Confidence levels:
0.25:  9882 out of 9937 (0.994465)
0.50:  9836 out of 9874 (0.996152)
0.75:  9780 out of 9807 (0.997247)
1.00:  9682 out of 9699 (0.998247)
1.25:  9582 out of 9592 (0.998957)
1.50:  9425 out of 9430 (0.99947)
2.00:  9010 out of 9010 (1)
```
See below for more details

## kNN

### V1

All we need for a kNN model is a similarity (or distance) metric between pairs of images, and handling (sorting) of nearest neighbors. I'm using a very simple class for the latter, see https://github.com/ikawrakow/mnist/blob/2436864a03dcf5fffa77b022ec3915cddc0c3e34/knnHandler.h#L7 The similarity metric comes from my experience with dealing with medical images. It is a combination of the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and a very simple binary feature vector: for every pixel $j$ of the image compare the grey value $A_j$ of the pixel to the grey value of the 4 neighboring pixels $A_{j,n}$, and set a bit when $A_j > A_{j, n}$. One than simply counts the number of bits that are set in both images being compared (a simple `popcount(a & b)` instruction per 8 pixels). All this is implemented in 38 LOC, see https://github.com/ikawrakow/mnist/blob/e1aa491b050a9bd8b9f7f12152bf73cfa5240a2d/mnist_knn_v1.cpp#L62

`mnist_knn_v1` needs 0.15 ms per prediction on my Ryzen-7950X CPU. It arrives at an error rate of 2.6-2.7% (see graph below, which shows prediction error as a function of number of nearest neighbors used). This is is not quite as good as the convolutional network in the `ggml` example, which runs in 0.06 ms/prediction on my CPU and arrives at an error rate of about 2%. 

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
* We use the x- and y-projections $P(x) = \sum_y A(x, y)$ and $P(y) = \sum_x A(x, y)$ where $A(x, y)$ is the grey value at position $(x, y)$. This allows us to a) quickly compute a similarity between a test image and a training sample ($28\dot2$ multiply-adds instead of $28^2$ for a full similarity computation), and b) quickly find a translation of the test images where the x- and y-projection best match the training sample. If the projection similarity after translation is greater than some threshold, the training image is skipped.

The resulting algorithm can be found in `mnist_knn_v4.cpp`. It is more complicated than the very simple `mnist_knn_v1-3.cpp`, but still reasonably simple with less than 300 LOC. `mnist_knn_v4` has several command-line options that influence the prediction accuracy vs run time tradeoff. Usage is
```
./mnist_knn_v4 [num_neighbors] [n_add] [thresh] [speed] [beta] [nthread]
```
where
* `num_neighbors` is the number of nearest neighbors to use. This affects run-time/accuracy because there are checks from time to time if all `4 * num_neighbors` nearest neighbors predict the same label and, if so, the calculation is stopped and the corresponding label (digit) is returned as the predicted result. Default value is 5.
* `n_add` is the number of deformations to add per training sample. This has the largest effect on accuracy and speed (see below for some examples)
* `thresh` is the threshold used for the projection similarity. Default values is 0.25, one can get slightly higher accuracy by increasing it to 0.33 or even 0.5 at the expense of a slower prediction
* `speed` can take values of 0, 1 or 2. If 0, the all-same-label check is never performed (see 1st bullet). If 1, the check is done only once after completing all non-deformed training samples (`n_train`). If 2, the check is done after every `n_train` samples. The difference in speed between the 3 options is quite significant with a relatively minor impact on prediction accuracy.
* `beta` is a parameter that determines the relative importance of the Pearson correlation coefficient and the binary feature similarity described above. It should be left at its default value of 1.
* `nthread` is the number of threads to use. If missing, `std::thread::hardware_concurrency()` is used. On systems with hyper-threading enabled or systems with a mix of performance and efficiency cores one may want to specify it explicitly to see if performance can be improved.

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

## SVM

### Algorithm description

To train an SVM algorithm that recognizes a given digit $l$, one looks for a plane $B^l_j$ in an $N$ dimensional "image feature" space such that $y_{li} \sum B^l_j A_{ij} > 0$, where $A_{ij}$ are the "features" of image $i$, and $y_{li} = +1$ when the image is digit $l$ or $y_{il} = -1$ otherwise. The simplest possible set of "features" in the context of `mnist` would simply the $28^2 = 784$ image grey values. One does not get very far with this, so here we prepare a feature set for an image by
1. Let $G_j$ be the grey values of an image (`uint8_t` in the range `0...255` for `mnist`)
2. Let $\Delta_1$ and $\Delta_2$ be offsets relative to an image pixel ($\Delta_1 \neq \Delta_2$)
3. Let $O(j, \Delta_1, \Delta_2) = G_{j+\Delta_1} - G_{j+\Delta_2}$, if $G_{j+\Delta_1} \geq G_{j+\Delta_2}, O(j, \Delta_1, \Delta_2) = 0$ otherwise
4. Applying $O(j, \Delta_1, \Delta_2)$ to every pixel $j$ in a given image creates a new image. We say that we applied "pattern" $(\Delta_1, \Delta_2)$ to the image. This is somewhat like applying a convolutional kernel in a CNN, except that our kernels are not "learned" but consists of truncated differences between pixels at pre-determined offsets 
1. Apply $N_1$ different "patterns" to the original image
2. Downsample the resulting $N_1$ images by averaging (typically using `2x2` windows)
3. Apply $N_2$ different "patterns" to the $N_1$ downsampled images. We now have $N_1 \cdot N_2$ images
4. Downsample the $N_1 \cdot N_2$ images using averaging
5. Possibly apply $N_3$ different "patterns" to the $N_1 \cdot N_2$ downsampled images
6. ...   

Depending on $N_1, N_2, ...$, we end up with a given number of values $A_{ij}$ that are the "features" of image $i$. This is implemented in `svmPattern.h` and `svmPattern.cpp`.

To determine the plane coefficients $B^l_j$ the following optimization objective, which we will minimize, is used 

$$F = \sum_{l=0}^9~~\sum_{i=1}^{N_t} \left(1 - y_{li} V_{li} \right)^2 \Theta\left(1 - y_{li} V_{li}\right) + \lambda \sum_{l=0}^9~~\sum_{j=1}^N \left(B^l_j\right)^2$$

where

$$V_{li} = \sum_{j=1}^N B^l_j A_{ij}$$

Here $N_t$ is the number of training examples, $N$ is the number of features, $\Theta$ is the Heaviside step function, and we have added a [Tikhonov regularization](https://en.wikipedia.org/wiki/Ridge_regression) term proportional to $\lambda$ to (hopefully) reduce overfitting. There are $10 \cdot N$ free parameters. This is a slight departure from a classic SVM as we are aiming for a gap of $\pm 1$, which I think works slightly better for this particular dataset.

[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is used to minimize $F$, see `bfgs.h` and `bfgs.cpp`.

### Data augmentation

As with kNN, it is useful to add augmented data to the training set. Unlike kNN, where elastic deformations are used, here Affine transformations of the original `mnist` dataset work better. Random rotation angles in $\pm 12.5$ degrees, sheering in $\pm 0.6$, and translations within $\pm 1.4$ pixels are sampled to compose the Affine transformations.

### Patterns

There are 4 patterns provided
* Type "0" results in 12,000 features
* Type "1" has 16,000 features
* Type "2" has 30,720 features
* Type "3" has 38,400 features
* Type "4" has 57,600 features

We will focus on Type "0" and Type "4", the others are there just for experimentations.

### Training

The training code takes several command line parameters:
```
./mnist_svm_train num_iterations n_add lambda type output_file random_sequence n_history max_translation
```
where
* `num_iterations` is the maximum number of iterations to run. It is set by default to 200, but one should typically use more. If convergence is reached, the iteration will be terminated.
* `n_add` is the number of augmented images to add per original `mnist` image. Note, however, that this is very hacky: if `n_add > 0`, `n_add` elastically deformed images will be added. If `n_add < 0` (which one should always use), `-n_add` Affine-transformed images will be added. If `n_add > 100`, then `n_add - 100` elastic deformations **and** `n_add - 99` Affine transformations will be added. A big caveat is that I have not bothered implementing batching, so all augmented images are added, and the features are computed and held in RAM. This puts a limit on how much augmented data one can add (depending on RAM available). E.g., with 9 added Affine transformations,one has 600,000 images, so 7.2 GB of RAM are needed for Type "0" pattern and 34.6 GB for the Type "4" pattern.
* `lambda` is the Tikhonov regularization parameter. There is no precise science behind this, so determining good values is a matter of experimentation. Typically `lambda` should be in the range `10...100`
* `type` is the pattern type, so 0...4
* `output_file` is the file name to use to store the trained model. If missing, `test.dat` is used
* `random_sequence` is the random number sequence to use (for the random number generator used to generate random deformations and Affine transformations). Should be left at zero, unless one wants to experiment with training multiple models using different sets of added augmented data, and then to combine the results somehow
* `n_history` is the number of past iterations to keep in the L-BFGS history. 100 is a good value for this
* `max_translation` is the maximum translation contained in Affine transformations expressed as a fraction of the image size. 0.05 is the default value (so 1.4 pixels).

To very quickly train a model:
```
./mnist_svm_train 200 0 100 fast_model.dat"
```
No augmented data will be added. Run time is about 5 seconds on my Ryzen-7950X CPU and results in a model with a prediction error of 0.78%. Because we did not add any additional data, it is necessary to use a larger `lambda` (100 in this case) to avoid overfitting.

To train a small, but quite accurate model:
```
./mnist_svm_train 400 -19 10 small_model.dat 0 100 0.075
```
This will add 19 Affine transformations, so we have 1.2 million training samples for 12,000 free parameters. Hence, `lambda` can be relatively small (10 in this case). This runs in 112 seconds and produces a model with an error rate of 0.5%.

To train the most accurate model (128+ GB of RAM required):
```
./mnist_svm_train 400 -29 50 large_model.dat 0 100
```
This runs in about 428 seconds on a Ryzen-5975WX and produces a model with an error rate of 0.38%.

### Prediction

```
./mnist_svm_predict model_file
```
This will predict the 10,000 test images from the `mnist` database and will print some statistics about the results.
Prediction time is about 10.5 us/image for the small model (12,000 features), and about 45 us/image for the large model (57,600 features). 
