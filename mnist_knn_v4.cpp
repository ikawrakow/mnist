#include "getImages.h"
#include "imageUtils.h"
#include "knnHandler.h"

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <limits>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
inline int popcount(uint32_t x) { return __popcnt(x); }
#else
constexpr int popcount(uint32_t x) { return __builtin_popcount(x); }
#endif

constexpr int kMargin = 6;

static inline void transform(int Nx, int Ny, const uint8_t *A, int kx, int ky, uint8_t *B) {
    int ymin = 0, ymax = kNy;
    if (ky < 0) { std::memset(B, 0, -ky*Nx*sizeof(uint8_t)); ymin = -ky; }
    else if (ky > 0) { std::memset(B + (Ny-1-ky)*Nx, 0, ky*Nx*sizeof(uint8_t)); ymax = Ny-ky; }
    for (int y=ymin; y<ymax; ++y) {
        int y1 = y + ky;
        auto By = B + y*Nx;
        for (int x=0; x<Nx; ++x) {
            int x1 = x + kx;
            By[x] = x1 >= 0 && x1 < Nx ? A[x1+y1*Nx] : 0;
        }
    }
}

static inline void computeMatrix(const uint8_t *A, float &m0, float &m1, float &m2, int &suma, int &suma2) {
    constexpr int kThresh = 64; //128;
    int sx = kNx/2, sy = kNy/2;
    int M0 = 0, M1 = 0, M2 = 0, s = 0, j = 0, sa = 0, sa2 = 0;
    for (int y=0; y<kNy; ++y) for (int x=0; x<kNx; ++x) {
        int a = A[j++]; sa += a; sa2 += a*a;
        if (a > kThresh) {
            ++s; M0 += (x-sx)*(x-sx); M1 += (x-sx)*(y-sy); M2 += (y-sy)*(y-sy);
        }
    }
    float norm = s > 0 ? 1.f/s : 0.f;
    m0 = norm*M0;
    m1 = norm*M1;
    m2 = norm*M2;
    suma = sa; suma2 = sa2;
}

static inline void computeSums(const uint8_t *A, int &suma, int &suma2) {
    int sa = 0, sa2 = 0;
    for (int j = 0; j < kSize; ++j) {
        int a = A[j]; sa += a; sa2 += a*a;
    }
    suma = sa; suma2 = sa2;
}

const std::vector<int>& getPattern() {
    static std::vector<int> pattern{1+kNx, -1+kNx, 1-kNx, -1-kNx};
    return pattern;
}

static void prepareOther(const std::vector<int> &pattern, const uint8_t  *A, uint32_t *B) {
    constexpr int kShift = 1;
    uint32_t u = 0, m = 1;
    for (int y=kMargin; y<kNy-kMargin; ++y) for (int x=kMargin; x<kNx-kMargin; ++x) {
        int j = x + y*kNx; uint8_t a = A[j] >> kShift;
        for (auto dj : pattern) {
            uint8_t a1 = A[j+dj] >> kShift;
            if (a > a1) u |= m;
            m <<= 1;
            if (!m) {
                *B++ = u; u = 0; m = 1;
            }
        }
    }
    if (m > 1) *B++ = u;
}

static inline float computeOther(int n, const uint32_t *A, const uint32_t *B) {
    int m = 0;
    for (int j=0; j<n; ++j) {
        auto a = A[j] & B[j];
        m += popcount(a);
    }
    return 1 - (1.f*m)/(32*n);
}

static void computeProjection(const uint8_t *A, uint16_t *B, int &sumxb, int &sumxb2, int &sumyb, int &sumyb2) {
    for (int j=0; j<kNx+kNy; ++j) B[j] = 0;
    for (int y=0; y<kNy; ++y) for (int x=0; x<kNx; ++x) {
        uint16_t a = *A++;
        B[x] += a; B[kNx+y] += a;
    }
    int sxb = 0, sxb2 = 0, syb = 0, syb2 = 0;
    for (int j=0; j<kNx; ++j) {
        int b = B[j]; sxb += b; sxb2 += b*b;
    }
    for (int j=0; j<kNy; ++j) {
        int b = B[kNx+j]; syb += b; syb2 += b*b;
    }
    sumxb = sxb; sumxb2 = sxb2;
    sumyb = syb; sumyb2 = syb2;
}

static inline float computeProjectionCC(int N, const uint16_t *A, const uint16_t *B, int sxa, int sxa2, int sxb, int sxb2) {
    int sxab = 0;
    for (int j=0; j<N; ++j) {
        int a = A[j], b = B[j]; sxab += a*b;
    }
    float norm = N;
    float ccx = 1 - (norm*sxab - 1.f*sxa*sxb)/sqrt((norm*sxa2 - 1.f*sxa*sxa)*(norm*sxb2 - 1.f*sxb*sxb));
    return ccx;
}

static inline float computeCC(const uint8_t *A, const uint8_t *B, int sa, int sa2, int sb, int sb2) {
    int sab = 0;
    for (int j=0; j<kSize; ++j) {
        int a = A[j], b = B[j]; sab += a*b;
    }
    float norm = kSize;
    float cc = 1 - (norm*sab - 1.f*sa*sb)/sqrt((norm*sa2 - 1.f*sa*sa)*(norm*sb2 - 1.f*sb*sb));
    return cc;
}

int main(int argc, char **argv) {

    int nneighb = argc > 1 ? atoi(argv[1]) : 5;
    int nadd = argc > 2 ? atoi(argv[2]) : 5;
    float thresh = argc > 3 ? atof(argv[3]) : 0.25f;
    int speed = argc > 4 ? atoi(argv[4]) : 1;
    float beta = argc > 5 ? atof(argv[5]) : 1.f;
    int nthread = argc > 6 ? atoi(argv[6]) : std::thread::hardware_concurrency();

    auto labels = getTraningLabels();
    if (labels.size() != kNtrain) return 1;

    auto images = getTrainingImages();
    if (images.size() != kNtrain*kNx*kNy) return 1;

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    for (auto& a : images) a >>= 1;
    for (auto& a : testImages) a >>= 1;

    addElasticDeformationsSameT(images,labels,nadd);
    int ntrain = labels.size();

    std::vector<std::thread> workers(nthread-1);

    printf("Creating pattern with 4 points. margin=%d\n", kMargin);
    auto& pattern = getPattern();
    int nbit = (kNx-2*kMargin)*(kNy-2*kMargin)*pattern.size();
    int osize = (nbit + 31)/32;
    printf("nbit = %d, osize = %d\n",nbit,osize);
    std::vector<uint32_t> otrain(((int64_t)ntrain)*osize);

    std::vector<float> allM(3*ntrain);
    std::vector<int> allSums(2*ntrain);
    std::vector<int> sumPA((int64_t)ntrain * 4);
    std::vector<uint16_t> projA((int64_t)ntrain * (kNx+kNy));
    std::atomic<int> counter(0); int chunk = 128;
    auto prepare = [&counter, &images, &otrain, &pattern, &allM, &allSums, &projA, &sumPA, ntrain, osize, chunk]() {
        while(1) {
            int first = counter.fetch_add(chunk);
            if (first >= ntrain) break;
            int last = first + chunk < ntrain ? first + chunk : ntrain;
            auto A = images.data() + (int64_t)first * kSize;
            auto oA = otrain.data() + (int64_t)first * osize;
            auto M = allM.data() + 3*first;
            auto S = allSums.data() + 2*first;
            auto P = projA.data() + (kNx+kNy)*uint64_t(first);
            auto Sp = sumPA.data() + 4*first;
            for (int i=first; i<last; ++i) {
                prepareOther(pattern,A,oA);
                computeMatrix(A,M[0],M[1],M[2],S[0],S[1]);
                computeProjection(A,P,Sp[0],Sp[1],Sp[2],Sp[3]);
                A += kSize; oA += osize; M += 3; S += 2;
                P += kNx+kNy; Sp += 4;
            }
        }
    };

    {
        auto t1 = std::chrono::steady_clock::now();
        for (auto & w : workers) w = std::thread(prepare);
        prepare();
        for (auto & w : workers) w.join();
        auto t2 = std::chrono::steady_clock::now();
        printf("It took %g ms to prepare the training data\n",1e-3*std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());
    }

    std::vector<std::vector<int>> predicted(4*nneighb);
    for (auto & p : predicted) p.resize(kNtest);
    auto processChunk = [&allM, &allSums, &images, &labels, &testImages, &predicted, &projA, &sumPA,
         &pattern, &otrain, thresh, nneighb, ntrain, osize, beta, speed](int first, int last) {
         constexpr int smax = 2;
         std::vector<uint8_t> shiftedB((2*smax+1)*(2*smax+1)*kSize);
         std::vector<uint32_t> otherB((2*smax+1)*(2*smax+1)*osize);
         std::vector<std::vector<uint16_t>> projBx(2*smax+1);
         std::vector<std::vector<uint16_t>> projBy(2*smax+1);
         for (auto & p : projBx) p.resize(kNx);
         for (auto & p : projBy) p.resize(kNy);
         std::vector<int> auxPx(2*(2*smax+1));
         std::vector<int> auxPy(2*(2*smax+1));
         std::vector<int> auxS(2*(2*smax+1)*(2*smax+1));
         std::vector<uint16_t> projB(kNx+kNy);
         NNHandler nnhandler(4*nneighb);
         auto B = testImages.data() + first*kSize;
         for (int i=first; i<last; ++i) {
             nnhandler.reset();
             int sxb, sxb2, syb, syb2, sb, sb2; float m0, m1, m2;
             computeProjection(B,projB.data(),sxb,sxb2,syb,syb2);
             for (int ky=-smax; ky<=smax; ++ky) for (int kx=-smax; kx<=smax; ++kx) {
                 int k = smax+kx + (smax+ky)*(2*smax+1);
                 transform(kNx,kNy,B,kx,ky,shiftedB.data()+k*kSize);
                 computeSums(shiftedB.data()+k*kSize,auxS[2*k],auxS[2*k+1]);
                 prepareOther(pattern,shiftedB.data()+k*kSize,otherB.data()+k*osize);
             }
             computeMatrix(B,m0,m1,m2,sb,sb2);
             for (int kx=-smax; kx<=smax; ++kx) {
                 auto & p = projBx[kx+smax];
                 int ssb=0, ssb2=0;
                 for (int x=0; x<kNx; ++x) {
                     int x1 = x + kx;
                     auto b = x1 >= 0 && x1 < kNx ? projB[x1] : 0;
                     p[x] = b; ssb += b; ssb2 += b*b;
                 }
                 auxPx[2*(kx+smax)+0] = ssb;
                 auxPx[2*(kx+smax)+1] = ssb2;
             }
             for (int ky=-smax; ky<=smax; ++ky) {
                 auto & p = projBy[ky+smax];
                 int ssb=0, ssb2=0;
                 for (int y=0; y<kNy; ++y) {
                     int y1 = y + ky;
                     auto b = y1 >= 0 && y1 < kNy ? projB[kNx+y1] : 0;
                     p[y] = b; ssb += b; ssb2 += b*b;
                 }
                 auxPy[2*(ky+smax)+0] = ssb;
                 auxPy[2*(ky+smax)+1] = ssb2;
             }
             auto Pa = projA.data(); auto Sp = sumPA.data();
             auto A = images.data(); auto S = allSums.data(); auto M = allM.data();
             for (int t=0; t<ntrain; ++t) {
                 if (speed == 1 && t == kNtrain && nnhandler.allSame(4*nneighb)) break;
                 float d2 = (M[0]-m0)*(M[0]-m0)+(M[1]-m1)*(M[1]-m1)+(M[2]-m2)*(M[2]-m2);
                 if (d2 < 75) {
                     float bestccpx = std::numeric_limits<float>::max(); int bestkx = 0;
                     float bestccpy = std::numeric_limits<float>::max(); int bestky = 0;
                     for (int ks=-smax; ks<=smax; ++ks) {
                         auto ccx = computeProjectionCC(kNx,Pa,    projBx[ks+smax].data(),Sp[0],Sp[1],auxPx[2*(ks+smax)],auxPx[2*(ks+smax)+1]);
                         auto ccy = computeProjectionCC(kNy,Pa+kNx,projBy[ks+smax].data(),Sp[2],Sp[3],auxPy[2*(ks+smax)],auxPy[2*(ks+smax)+1]);
                         if (ccx < bestccpx) { bestccpx = ccx; bestkx = ks; }
                         if (ccy < bestccpy) { bestccpy = ccy; bestky = ks; }
                     }
                     auto ccp = bestccpx + bestccpy;
                     if (ccp < thresh) {
                         int k = smax+bestkx + (smax+bestky)*(2*smax+1);
                         auto cc = computeCC(A,shiftedB.data()+k*kSize,S[0],S[1],auxS[2*k],auxS[2*k+1]);
                         auto otherA = otrain.data() + (int64_t)t * osize;
                         float ccb = computeOther(osize,otherA,otherB.data()+k*osize);
                         nnhandler.add({beta*cc+ccb,labels[t]});
                     }
                 }
                 Pa += kNx+kNy; Sp += 4;
                 A += kSize; S += 2; M += 3;
                 if (speed >= 2 && (t + 1)%kNtrain == 0 && nnhandler.allSame(4*nneighb)) break;
             }
             for (int n=1; n<=4*nneighb; ++n) predicted[n-1][i] = nnhandler.predict(n);
             B += kSize;
         }
    };

    printf("Predicting %d test images...", kNtest);
    fflush(stdout);
    auto tim1 = std::chrono::steady_clock::now();

    counter = 0; chunk = 64;
    auto compute = [&counter, &processChunk, chunk]() {
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
