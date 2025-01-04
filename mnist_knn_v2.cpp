#include "getImages.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>

constexpr int kMargin = 6;
constexpr int kNpoints = 4;
constexpr int kPattern[kNpoints] = {2 + 2*kNx, 2 - 2*kNx, -2 + 2*kNx, -2 - 2*kNx };
constexpr int kNbits  = (kNx - 2*kMargin)*(kNy - 2*kMargin)*kNpoints;
constexpr int kNu32 = (kNbits + 31)/32;

#ifdef _MSC_VER
#include <intrin.h>
inline int popcount(uint32_t x) { return __popcnt(x); }
#else
constexpr int popcount(uint32_t x) { return __builtin_popcount(x); }
#endif

class NNHandler {
public:
    NNHandler(int nmax) : nmax_(nmax), nhave_(0) { data_.resize(nmax_); }
    inline void add(std::pair<float,int> a) {
        if (nhave_ < nmax_) {
            data_[nhave_++] = a;
            if (nhave_ == nmax_) std::sort(data_.begin(),data_.end());
            return;
        }
        if (a.first >= data_.back().first) return;
        auto i = findIndex(a);
        for (int k=nmax_-1; k>i; --k) data_[k] = data_[k-1];
        data_[i] = a;
    }
    void reset() { nhave_ = 0; }
    int predict(int n) const {
        if (nhave_ < n) return -1;
        float X[10] = {};
        for (int i=0; i<n; ++i) X[data_[i].second] += 1/(data_[i].first + 0.001f);
        auto best = X[0]; int lbest = 0;
        for (int l=1; l<10; ++l) if (X[l] > best) { best = X[l]; lbest = l; }
        return lbest;
    }
private:
    std::vector<std::pair<float,int>> data_;
    int nmax_;
    int nhave_;
    inline int findIndex(const std::pair<float,int> &x) const {
        if (x.first <= data_.front().first) return 0;
        int ml = 0, mu = nmax_-1;
        while (mu-ml > 1) {
            int mav = (ml+mu)/2;
            if (x.first < data_[mav].first) mu = mav; else ml = mav;
        }
        return mu;
    }
};

struct Image {
    const uint8_t * data;
    int sum = 0;
    int sum2 = 0;
    uint32_t bits[kNu32];
    Image(const uint8_t * A) : data(A) {
        sum = sum2 = 0;
        for (int j = 0; j < kSize; ++j) {
            int a = A[j];
            sum += a; sum2 += a*a;
        }
        uint32_t u = 0;
        int bit = 0, l = 0;
        constexpr int kShift = 1;
        for (int y = kMargin; y < kNy - kMargin; ++y) for (int x = kMargin; x < kNx - kMargin; ++x) {
            int j = x + y*kNx;
            auto a = A[j] >> kShift;
            for (int k = 0; k < kNpoints; ++k) {
                if (a > (A[j + kPattern[k]] >> kShift)) u |= (1u << bit);
                if (++bit == 32) {
                    bits[l++] = u;
                    bit = 0; u = 0;
                }
            }
        }
        if (bit > 0) bits[l] = u;
    }
};

static std::vector<Image> prepareTrainingData(int nimage, const uint8_t * allData) {
    std::vector<Image> result;
    result.reserve(nimage);
    for (int i = 0; i < nimage; ++i) result.emplace_back(allData + i*kSize);
    return result;
}

static inline float computeCC(const Image& a, const Image& b) {
    int non = 0;
    for (int i = 0; i < kNu32; ++i) non += popcount(a.bits[i] & b.bits[i]);
    float ccb = 1 - 1.f*non/kNbits;
    int sumab = 0;
    for (int j = 0; j < kSize; ++j) sumab += int(a.data[j])*int(b.data[j]);
    float norm = kSize;
    float denom = (norm*a.sum2 - a.sum*a.sum)*(norm*b.sum2 - b.sum*b.sum);
    if (denom <= 0) {
        printf("Oops: denom = %g\n", denom); exit(1);
    }
    float cc = denom > 0 ? 1.f - (norm*sumab - a.sum*b.sum)/sqrt(denom) : 2.f;
    return 0.125f*cc + ccb;
}

static void transform(int Nx, int Ny, const uint8_t *A, int kx, int ky, uint8_t *B) {
    for (int y=0; y<Ny; ++y) {
        int y1 = y + ky;
        if (y1 >= 0 && y1 < Ny) {
            for (int x=0; x<Nx; ++x) {
                int x1 = x + kx;
                B[x+y*kNx] = x1 >= 0 && x1 < Nx ? A[x1+y1*Nx] : 0;
            }
        }
        else for (int x=0; x<Nx; ++x) B[x+y*Nx] = 0;
    }
}

static void addShifted(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int smax) {
    if (smax <= 0) return;
    if (labels.size() != images.size()/kSize) {
        printf("Oops: inconsistent data\n"); exit(1);
    }
    int nimage = labels.size();
    int nadd = (2*smax+1)*(2*smax+1) - 1;
    images.resize(images.size() + kSize*nadd*nimage);
    labels.resize(labels.size()*(nadd + 1));
    auto A = images.data();
    auto B = images.data() + nimage*kSize;
    auto L = labels.data() + nimage;
    for (int i = 0; i < nimage; ++i) {
        for (int ky = -smax; ky <= smax; ++ky) for (int kx = -smax; kx <= smax; ++kx) {
            if (kx == 0 && ky == 0) continue;
            transform(kNx, kNy, A, kx, ky, B);
            B += kSize;
            *L++ = labels[i];
        }
        A += kSize;
    }
}

int main(int argc, char **argv) {

    int nneighb = argc > 1 ? atoi(argv[1]) : 5;
    int smax = argc > 2 ? atoi(argv[2]) : 1;

    auto labels = getTraningLabels();
    if (labels.size() != kNtrain) return 1;

    auto images = getTrainingImages();
    if (images.size() != kNtrain*kNx*kNy) return 1;

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    for (auto& a : images) a >>= 2;
    for (auto& a : testImages) a >>= 2;

    addShifted(images, labels, smax);

    auto train = prepareTrainingData(labels.size(), images.data());

    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread-1);

    std::vector<std::vector<int>> predicted(4*nneighb);
    for (auto & p : predicted) p.resize(kNtest);
    auto processChunk = [&train, &labels, &testImages, &predicted, nneighb] (int first, int last) {
         NNHandler nnhandler(4*nneighb);
         auto B = testImages.data() + first*kSize;
         for (int i=first; i<last; ++i) {
             nnhandler.reset();
             Image b(B);
             for (int j = 0; j < int(train.size()); ++j) {
                 auto cc = computeCC(b, train[j]);
                 nnhandler.add({cc, labels[j]});
             }
             for (int n=1; n<=4*nneighb; ++n) predicted[n-1][i] = nnhandler.predict(n);
             B += kSize;
         }
    };

    printf("Predicting %d test images...", kNtest);
    fflush(stdout);
    auto tim1 = std::chrono::steady_clock::now();
    std::atomic<int> counter(0);
    auto compute = [&counter, &processChunk]() {
        constexpr int chunk = 64;
        while(1) {
            uint32_t first = counter.fetch_add(chunk);
            if (first >= kNtest) break;
            uint32_t last = first + chunk;
            if (last > kNtest) last = kNtest;
            processChunk(first, last);
        }
    };
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
    auto tim2 = std::chrono::steady_clock::now();
    auto time = 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count();
    printf("done in %g ms -> %g ms per image\n\n", time, time/kNtest);

    printf("neighbors | error (%c)\n", '%');
    printf("----------|-----------\n");
    for (int n=1; n<=4*nneighb; ++n) {
        auto& p = predicted[n-1];
        int ngood = 0;
        for (uint32_t i=0; i<kNtest; ++i) if (p[i] == testLabels[i]) ++ngood;
        float err = 100.f*(kNtest - ngood)/kNtest;
        printf("   %3d    |  %.3f\n", n, err);
    }

    return 0;
}
