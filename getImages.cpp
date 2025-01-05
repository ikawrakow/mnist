#include "getImages.h"
#include <cstdio>
#include <fstream>

constexpr auto kTrainLabels = "data/train-labels-idx1-ubyte";
constexpr auto kTestLabels = "data/t10k-labels-idx1-ubyte";
constexpr auto kTrainImages = "data/train-images-idx3-ubyte";
constexpr auto kTestImages = "data/t10k-images-idx3-ubyte";

namespace {
std::vector<uint8_t> getLabels(int nimage, const char *fname) {
    std::vector<uint8_t> result;
    std::ifstream in(fname,std::ios::binary);
    if( !in ) {
        printf("Failed to open %s\n",fname);
        return result;
    }
    int dum;
    in.read((char *)&dum,sizeof(int));
    in.read((char *)&dum,sizeof(int));
    result.resize(nimage);
    in.read((char *)result.data(),nimage);
    if( in.fail() ) {
        printf("Failed reading labels from %s\n",fname); result.clear();
    }
    return result;
}
}

std::vector<uint8_t> getImages(int nimage, const char *fname) {
    std::vector<uint8_t> result;
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        printf("Failed to open %s\n",fname);
        return result;
    }
    int dum;
    in.read((char *)&dum,sizeof(int));
    in.read((char *)&dum,sizeof(int));
    in.read((char *)&dum,sizeof(int));
    in.read((char *)&dum,sizeof(int));
    result.resize(nimage*kNx*kNy);
    in.read((char *)result.data(),nimage*kNx*kNy);
    if (in.fail()) {
        printf("Failed reading images from %s\n",fname); result.clear();
    }
    return result;
}

std::vector<uint8_t> getTrainingImages() { return getImages(kNtrain,kTrainImages); }

std::vector<uint8_t> getTestImages() { return getImages(kNtest,kTestImages); }

std::vector<uint8_t> getTraningLabels() { return getLabels(kNtrain,kTrainLabels); }

std::vector<uint8_t> getTestLabels() { return getLabels(kNtest,kTestLabels); }

