#include "getImages.h"
#include "imageUtils.h"
#include "knnHandler.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <array>
#include <fstream>

constexpr int kMargin = 6;

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

static void computeMatrix(const uint8_t *A, float &m0, float &m1, float &m2, int &suma, int &suma2) {
    constexpr int kThresh = 128;
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

static void computeSums(const uint8_t *A, int &suma, int &suma2) {
    int j = 0, sa = 0, sa2 = 0;
    for (int y=0; y<kNy; ++y) for (int x=0; x<kNx; ++x) {
        int a = A[j++]; sa += a; sa2 += a*a;
    }
    suma = sa; suma2 = sa2;
}

std::vector<int> getPattern() {
    return std::vector<int>{1+kNx, -1+kNx, 1-kNx, -1-kNx };
    //return std::vector<int>{2+2*kNx, -2+2*kNx, 2-2*kNx, -2-2*kNx };
    //return std::vector<int>{1+kNx, -1+kNx, 1-kNx, -1-kNx, 2, -2, 2*kNx, -2*kNx };
}

static void prepareOther(const std::vector<int> &pattern, const uint8_t  *A, uint32_t *B) {
    uint32_t u = 0; int bit = 0;
    for (int y=kMargin; y<kNy-kMargin; ++y) for (int x=kMargin; x<kNx-kMargin; ++x) {
        int j = x + y*kNx; int a = A[j] >> 2;
        for (auto dj : pattern) {
            int a1 = A[j+dj] >> 2;
            if (a > a1) u |= (1u << bit);
            if (++bit == 32) {
                *B++ = u; u = 0; bit = 0;
            }
        }
    }
    if (bit > 0) *B++ = u;
}

static inline float computeOther(int n, const uint32_t *A, const uint32_t *B) {
    int m = 0;
    for (int j=0; j<n; ++j) {
        auto a = A[j] & B[j];
        m += __builtin_popcount(a);
    }
    return 1 - (1.f*m)/(32*n);
    //for (int j=0; j<n; ++j) {
    //    auto a = A[j] ^ B[j];
    //    m += __builtin_popcount(a);
    //}
    //return 1.f*m/(32*n);
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
    float thresh = argc > 2 ? atof(argv[2]) : 0.25f;
    float sigma = argc > 3 ? atof(argv[3]) : 6;
    float alpha = argc > 4 ? atof(argv[4]) : 38;
    int nadd = argc > 5 ? atoi(argv[5]) : 5;
    float beta = argc > 6 ? atof(argv[6]) : 1.f;
    int nthread = argc > 7 ? atoi(argv[7]) : std::thread::hardware_concurrency();

    auto labels = getTraningLabels();
    if (labels.size() != kNtrain) return 1;

    auto images = getTrainingImages();
    if (images.size() != kNtrain*kNx*kNy) return 1;

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    if (nadd > 0) addElasticDeformationsSameT(images,labels,nadd,sigma,alpha,0,nullptr);
    int ntrain = labels.size();

    std::vector<std::thread> workers(nthread-1);

    printf("Creating pattern with 4 points. margin=%d\n", kMargin);
    auto pattern = getPattern();
    int nbit = (kNx-2*kMargin)*(kNy-2*kMargin)*pattern.size();
    int osize = (nbit + 31)/32;
    printf("nbit = %d, osize = %d\n",nbit,osize);
    std::vector<uint32_t> otrain(((int64_t)ntrain)*osize);

    std::vector<float> allM(3*ntrain);
    std::vector<int> allSums(2*ntrain);
    std::atomic<int> counter(0); int chunk = 128;
    auto prepare = [&counter, &images, &otrain, &pattern, &allM, &allSums, ntrain, osize, chunk]() {
        while(1) {
            int first = counter.fetch_add(chunk);
            if (first >= ntrain) break;
            int last = first + chunk < ntrain ? first + chunk : ntrain;
            auto A = images.data() + (int64_t)first * kSize;
            auto oA = otrain.data() + (int64_t)first * osize;
            auto M = allM.data() + 3*first;
            auto S = allSums.data() + 2*first;
            for (int i=first; i<last; ++i) {
                prepareOther(pattern,A,oA);
                computeMatrix(A,M[0],M[1],M[2],S[0],S[1]);
                A += kSize; oA += osize; M += 3; S += 2;
            }
        }
    };

    {
        auto t1 = std::chrono::steady_clock::now();
        for (auto & w : workers) w = std::thread(prepare);
        prepare();
        for (auto & w : workers) w.join();
        auto t2 = std::chrono::steady_clock::now();
        printf("It took %g ms to prepare the data\n",1e-3*std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());
    }

    std::vector<int> sumPA((int64_t)ntrain * 4);
    std::vector<uint16_t> projA((int64_t)ntrain * (kNx+kNy));
    auto A = images.data(); auto P = projA.data(); auto S = sumPA.data();
    for (int i=0; i<ntrain; ++i) {
        computeProjection(A,P,S[0],S[1],S[2],S[3]);
        A += kSize; P += kNx+kNy; S += 4;
    }
    int smax = 2;

    std::vector<std::vector<int>> predicted(4*nneighb);
    for (auto & p : predicted) p.resize(kNtest);
    auto processChunk = [&allM, &allSums, &images, &labels, &testImages, &predicted, &projA, &sumPA,
         &pattern, &otrain, thresh, nneighb, ntrain, osize, beta, smax](int first, int last) {
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
                 //computeMatrix(shiftedB.data()+k*kSize,m0,m1,m2,auxS[2*k],auxS[2*k+1]);
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
                 auxPx[2*(kx+smax)] = ssb;
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
                 auxPy[2*(ky+smax)] = ssb;
                 auxPy[2*(ky+smax)+1] = ssb2;
             }
             auto Pa = projA.data(); auto Sp = sumPA.data();
             auto A = images.data(); auto S = allSums.data(); auto M = allM.data();
             for (int t=0; t<ntrain; ++t) {
                 if (t == kNtrain && nnhandler.allSame(4*nneighb)) break;
                 float d2 = (M[0]-m0)*(M[0]-m0)+(M[1]-m1)*(M[1]-m1)+(M[2]-m2)*(M[2]-m2);
                 if (d2 < 75) { //60) {
                     float bestccpx = 1e30; int bestkx = 0;
                     for (int kx=-smax; kx<=smax; ++kx) {
                         auto cc = computeProjectionCC(kNx,Pa,projBx[kx+smax].data(),Sp[0],Sp[1],auxPx[2*(kx+smax)],auxPx[2*(kx+smax)+1]);
                         if (cc < bestccpx) { bestccpx = cc; bestkx = kx; }
                     }
                     float bestccpy = 1e30; int bestky = 0;
                     for (int ky=-smax; ky<=smax; ++ky) {
                         auto cc = computeProjectionCC(kNy,Pa+kNx,projBy[ky+smax].data(),Sp[2],Sp[3],auxPy[2*(ky+smax)],auxPy[2*(ky+smax)+1]);
                         if (cc < bestccpy) { bestccpy = cc; bestky = ky; }
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
             }
             for (int n=1; n<=4*nneighb; ++n) predicted[n-1][i] = nnhandler.predict(n);
             B += kSize;
         }
    };

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

    printf("Ngood:\n");
    for (int n=1; n<=4*nneighb; ++n) {
        auto& p = predicted[n-1];
        int ngood = 0;
        for (uint32_t i=0; i<kNtest; ++i) if (p[i] == testLabels[i]) ++ngood;
        printf("%d  %d\n",n,ngood);
    }
    return 0;
}
