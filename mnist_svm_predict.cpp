#include "getImages.h"
#include "imageUtils.h"
#include "svmPattern.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>

//#ifdef __AVX2__
//#include <immintrin.h>
//
//inline int hsum_i32_8(const __m256i a) {
//    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
//    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
//    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
//    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
//    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
//}
//#endif

using Float = float;

static bool loadModel(const char* fname, std::vector<Pattern>& patterns,
        std::vector<Float>& scales, std::vector<std::vector<int16_t>>& allP) {
    std::ifstream in(fname,std::ios::binary);
    if (!in) {
        printf("%s: failed top open %s\n",__func__,fname); return false;
    }
    int npattern;
    in.read((char *)&npattern,sizeof(int));
    patterns.resize(npattern);
    for (auto & p : patterns) {
        int np = p.points_.size();
        in.read((char *)&np,sizeof(int));
        p.points_.resize(np);
        in.read((char *)p.points_.data(),np*sizeof(int));
        in.read((char *)&p.margin_,sizeof(p.margin_));
        in.read((char *)&p.down_,sizeof(p.down_));
    }
    if ((int)allP.size() != kNlabels) allP.resize(kNlabels);
    if ((int)scales.size() != kNlabels) scales.resize(kNlabels);
    int npoint;
    in.read((char *)&npoint,sizeof(int));
    std::vector<float> aux(npoint);
    int l = 0;
    for (auto & p : allP) {
        p.resize(npoint);
        in.read((char *)aux.data(),p.size()*sizeof(Float));
        Float max = 0;
        for (int j = 0; j < npoint; ++j) {
            Float ax = std::abs(aux[j]);
            max = std::max(max, ax);
        }
        scales[l] = max/32766;
        Float is = scales[l] ? 1/scales[l] : 0;
        for (int j = 0; j < npoint; ++j) p[j] = toNearestInt(is*aux[j]);
        ++l;
    }
    return in.good() ? true : false;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s modelFile\n",argv[0]); return 1;
    }

    std::vector<float> confLevels = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f};

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread);

    printf("\n============================== Dataset %s\n",argv[1]);
    std::vector<Pattern> patterns;
    std::vector<Float> scales;
    std::vector<std::vector<int16_t>> allP(kNlabels);
    if (!loadModel(argv[1],patterns,scales,allP)) return 1;

    auto tim1 = std::chrono::steady_clock::now();

    int Nx = kNx, Ny = kNy;
    for (const auto& p : patterns) {
        int Nx1, Ny1;
        testImages = Pattern::apply(Nx,Ny,testImages,p,Nx1,Ny1);
        Nx = Nx1; Ny = Ny1;
    }
    int npoint = testImages.size()/testLabels.size();
    printf("# %d points per image\n",npoint);
    std::atomic<int> counter(0);
    std::vector<float> V(kNlabels*kNtest);
    auto predict = [&counter, &scales, &allP, &testImages, &V, npoint]() {
        int chunk = 64;
        while (true) {
            int first = counter.fetch_add(chunk);
            if (first >= kNtest) break;
            int n = first + chunk <= kNtest ? chunk : kNtest - first;
            for (int l=0; l<kNlabels; ++l) {
                auto B = testImages.data() + first*npoint;
                auto v = V.data() + kNlabels*first + l;
                auto& p = allP[l];
                for (int i=0; i<n; ++i) {
                    int s = 0;
//#ifdef __AVX2__
//                    __m256i sum = _mm256_setzero_si256();
//                    for(int j=0; j<npoint/16; ++j) {
//                        auto vp = _mm256_loadu_si256((const __m256i *)p.data() + j);
//                        auto vb = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)B + j));
//                        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(vp, vb));
//                    }
//                    s = hsum_i32_8(sum);
//                    for (int j = 16*(npoint/16); j < npoint; ++j) s += p[j]*B[j];
//#else
                    for(int j=0; j<npoint; ++j) s += p[j]*B[j];
//#endif
                    *v = scales[l]*s;
                    v += kNlabels; B += npoint;
                }
            }
        }
    };
    for (auto& w : workers) w = std::thread(predict);
    for (auto& w : workers) w.join();

    std::vector<int> countOK(kNtest,0);
    std::vector<std::pair<float,int>> X(kNlabels);

    std::vector<int> countGood(confLevels.size(),0), count(confLevels.size(),0);
    int ncheck = 5;
    std::vector<int> Ngood(ncheck,0);
    for (int i=0; i<kNtest; ++i) {
        for (int l=0; l<kNlabels; ++l) {
            X[l] = {V[kNlabels*i+l],l};
        }
        std::sort(X.begin(),X.end());
        if (X[kNlabels-1].second == testLabels[i]) ++countOK[i];
        auto delta = X[kNlabels-1].first - X[kNlabels-2].first;
        for(int k=0; k<(int)confLevels.size(); ++k) {
            if (delta > confLevels[k]) {
                ++count[k];
                if (X[kNlabels-1].second == testLabels[i]) ++countGood[k];
            }
        }
        for(int k=kNlabels-1; k>=0; --k) {
            if (X[k].second == testLabels[i]) {
                int n = kNlabels - 1 - k;
                for (int j=n; j<ncheck; ++j) ++Ngood[j];
                break;
            }
        }
    }
    auto tim2 = std::chrono::steady_clock::now();
    auto time = 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count();
    printf("Predicted %d images in %g ms -> %g us per image\n\n", kNtest, time, 1e3*time/kNtest);
    for(int n=0; n<ncheck; ++n) printf("%d  %d\n",n,Ngood[n]);
    printf("Confidence levels:\n");
    for(int k=0; k<(int)confLevels.size(); ++k) {
        printf("%4.2f:  %d out of %d (%g)\n",confLevels[k],countGood[k],count[k],(1.*countGood[k])/count[k]);
    }


    return 0;
}
