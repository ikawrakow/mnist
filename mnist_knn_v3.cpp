#include "getImages.h"
#include "imageUtils.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>

constexpr int kMargin = 6;
constexpr int kNpoints = 4;
constexpr int kPattern[kNpoints] = {2 + 2*kNx, 2 - 2*kNx, -2 + 2*kNx, -2 - 2*kNx };
constexpr int kNbits  = (kNx - 2*kMargin)*(kNy - 2*kMargin)*kNpoints;
constexpr int kNbytes = (kNbits + 31)/32;

constexpr double kSigma = 6.;
constexpr double kAlpha = 38.;

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
    bool allSame() const {
        if (nhave_ < nmax_) return false;
        int l = data_[0].second;
        for (int i=1; i<nmax_; ++i) if (data_[i].second != l) return false;
        return true;
    }
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
    uint32_t bits[kNbytes];
    Image(const uint8_t * A) : data(A) {
        sum = sum2 = 0;
        for (int j = 0; j < kSize; ++j) {
            int a = A[j];
            sum += a; sum2 += a*a;
        }
        uint32_t u = 0;
        int bit = 0, l = 0;
        constexpr int kShift = 3;
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
    for (int i = 0; i < kNbytes; ++i) non += popcount(a.bits[i] & b.bits[i]);
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

int main(int argc, char **argv) {

    int nneighb = argc > 1 ? atoi(argv[1]) : 5;
    int nadd = argc > 2 ? atoi(argv[2]) : 10;

    auto labels = getTraningLabels();
    if (labels.size() != kNtrain) return 1;

    auto images = getTrainingImages();
    if (images.size() != kNtrain*kNx*kNy) return 1;

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    if (nadd > 0) {
        addElasticDeformationsSameT(images, labels, nadd, kSigma, kAlpha, 0, nullptr);
    }

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
                 if (i == kNtrain && nnhandler.allSame()) break;
             }
             for (int n=1; n<=4*nneighb; ++n) predicted[n-1][i] = nnhandler.predict(n);
             B += kSize;
         }
    };

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

    printf("Ngood:\n");
    for (int n=1; n<=4*nneighb; ++n) {
        auto& p = predicted[n-1];
        int ngood = 0;
        for (uint32_t i=0; i<kNtest; ++i) if (p[i] == testLabels[i]) ++ngood;
        printf("%d  %d\n",n,ngood);
    }
    return 0;
}
