#include "svmPattern.h"
#include "getImages.h"

#include <cstdio>
#include <atomic>
#include <thread>

std::vector<uint8_t> Pattern::apply(int Nx, int Ny, const std::vector<uint8_t>& images, const Pattern& pattern,
        int& Nxf, int& Nyf) {
    int64_t nimage = images.size()/(Nx*Ny);
    int np = pattern.points_.size()/2;
    int margin = pattern.margin_; if( margin < 0 ) margin = -margin;
    int Nx1 = Nx - 2*margin, Ny1 = Ny - 2*margin;
    int Nx2 = Nx1, Ny2 = Ny1;
    if (pattern.down_) {
        if (pattern.down_ > 0) {
            Nx2 = (Nx2 - 2 + pattern.down_)/pattern.down_; Ny2 = (Ny2 - 2 + pattern.down_)/pattern.down_;
        }
        else { Nx2 = Nx1 + Ny1; Ny2 = 1; }
    }
    std::vector<uint8_t> result(Nx2*Ny2*np*nimage);
    printf("Pattern::%s: nimage=%d np=%d Nx=%d Ny=%d Nx1=%d Ny1=%d Nx2=%d Ny2=%d\n",__func__,(int)nimage,np,Nx,Ny,Nx1,Ny1,Nx2,Ny2);
    fflush(stdout);
    std::atomic<int64_t> counter(0); int chunk = 64;
    auto compute = [&counter, &images, &pattern, &result, Nx, Ny, Nx1, Ny1, Nx2, Ny2, np, nimage, chunk]() {
        std::vector<uint8_t> aux(Nx1*Ny1);
        std::vector<int> aux1(Nx1*Ny1);
        while(1) {
            int64_t first = counter.fetch_add(chunk);
            if( first >= nimage ) break;
            int n = first + chunk < nimage ? chunk : nimage - first;
            auto A = images.data() + first * Nx*Ny;
            auto R = result.data() + first * Nx2*Ny2*np;
            for(int i=0; i<n; ++i) {
                pattern.apply(Nx,Ny,A,R,aux.data(),aux1.data());
                A += Nx*Ny; R += Nx2*Ny2*np;
            }
        }
    };
    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread-1);
    for(auto& w : workers) w = std::thread(compute);
    compute();
    for(auto& w : workers) w.join();
    Nxf = Nx2; Nyf = Ny2;
    return result;
}

namespace {
// 12,000 points. 0.9950 with 19 affine, lambda = 10, nhist=100 (converges after 220 iterations)
std::vector<Pattern> getPatternsV0() {
    std::vector<Pattern> result;
    Pattern p1;
    p1.margin_ = 2; p1.down_ = 2;
    p1.points_ = {0,1+kNx,0,1-kNx,0,-1+kNx,0,-1-kNx,0,-1,0,1,0,kNx,0,-kNx,0,2,0,-2,0,2*kNx,0,-2*kNx,
                  0,2+2*kNx,0,2-2*kNx,0,-2+2*kNx,0,-2-2*kNx,
                  0,1+2*kNx,0,1-2*kNx,0,-1+2*kNx,0,-1-2*kNx,
                  0,2+1*kNx,0,2-1*kNx,0,-2+1*kNx,0,-2-1*kNx};
    Pattern p2;
    int Nx1 = (kNx - 2*p1.margin_)/2;
    p2.margin_ = 1; p2.down_ = 2;
    p2.points_ = {1,-1,Nx1,-Nx1,1+Nx1,-1-Nx1,1-Nx1,-1+Nx1,1,-1+Nx1,1,-1-Nx1,-1,1+Nx1,-1,1-Nx1,Nx1,-Nx1+1,Nx1,-Nx1-1,-Nx1,Nx1+1,-Nx1,Nx1-1,
                  0,1, 0,-1, 0,Nx1, 0,-Nx1, 0,1+Nx1, 0,1-Nx1, 0,-1+Nx1, 0,-1-Nx1};
    result.emplace_back(std::move(p1));
    result.emplace_back(std::move(p2));
    return result;
}

// 16,000 points. 0.9952 with new affine and 9 extra, lambda = 20.
std::vector<Pattern> getPatternsV1() {
    std::vector<Pattern> result;
    Pattern p1;
    p1.margin_ = 2; p1.down_ = 2;
    p1.points_ = {0,1+kNx,0,1-kNx,0,-1+kNx,0,-1-kNx,0,-1,0,1,0,kNx,0,-kNx,
                  kNx,1, 1,-kNx, -kNx,-1, -1,kNx,
                  -1-kNx,1-kNx, 1-kNx,1+kNx, 1+kNx,-1+kNx, -1+kNx,-1-kNx,
                  0,2,0,-2,0,2*kNx,0,-2*kNx,
                  0,2+2*kNx,0,2-2*kNx,0,-2+2*kNx,0,-2-2*kNx,
                  0,1+2*kNx,0,1-2*kNx,0,-1+2*kNx,0,-1-2*kNx,
                  0,2+1*kNx,0,2-1*kNx,0,-2+1*kNx,0,-2-1*kNx};
    Pattern p2;
    int Nx1 = (kNx - 2*p1.margin_)/2;
    p2.margin_ = 1; p2.down_ = 2;
    p2.points_ = {1,-1,Nx1,-Nx1,1+Nx1,-1-Nx1,1-Nx1,-1+Nx1,1,-1+Nx1,1,-1-Nx1,-1,1+Nx1,-1,1-Nx1,Nx1,-Nx1+1,Nx1,-Nx1-1,-Nx1,Nx1+1,-Nx1,Nx1-1,
                  0,1, 0,-1, 0,Nx1, 0,-Nx1, 0,1+Nx1, 0,1-Nx1, 0,-1+Nx1, 0,-1-Nx1};
    result.emplace_back(std::move(p1));
    result.emplace_back(std::move(p2));
    return result;
}

// 30720 points, 9963 with lambda=10, 29 extra new affine
std::vector<Pattern> getPatternsV2() {
    std::vector<Pattern> result;
    int Nx = kNx;
    Pattern p1; p1.margin_ = 1; p1.down_ = 2;
    p1.points_ = {0,1+Nx, 0,1-Nx, 0,-1+Nx, 0,-1-Nx, 1,-1, Nx,-Nx,
                  -1-Nx,1-Nx, 1-Nx,1+Nx, 1+Nx,-1+Nx, -1+Nx,-1-Nx,
                  1+Nx,-1-Nx, 1-Nx,-1+Nx,
                  -1,1+Nx, -1,1-Nx, -Nx,-1+Nx, -Nx,1+Nx};

    Nx = (Nx - 2*p1.margin_)/2;
    Pattern p2; p2.margin_ = 1; p2.down_ = 1;
    p2.points_ = {0,1+Nx, 0,1-Nx, 0,-1+Nx, 0,-1-Nx, 1,-1, Nx,-Nx,
                  -1-Nx,1-Nx, 1-Nx,1+Nx, 1+Nx,-1+Nx, -1+Nx,-1-Nx,
                  1+Nx,-1-Nx, 1-Nx,-1+Nx};

    Nx = Nx - 2*p2.margin_ - 1;
    Pattern p3; p3.margin_ = 1; p3.down_ = 2;
    p3.points_ = {0,1+Nx, 0,1-Nx, 0,-1+Nx, 0,-1-Nx, 1,-1, Nx,-Nx,
                  -1-Nx,1-Nx, 1-Nx,1+Nx, 1+Nx,-1+Nx, -1+Nx,-1-Nx,
                  1+Nx,-1-Nx, 1-Nx,-1+Nx,
                  -1,1+Nx, -1,1-Nx, -Nx,-1+Nx, -Nx,1+Nx};

    Nx = (Nx - 2*p3.margin_)/2;
    Pattern p4; p4.margin_ = 1; p4.down_ = 2;
    p4.points_ = {0,1+Nx, 0,1-Nx, 0,-1+Nx, 0,-1-Nx, 1,-1, Nx,-Nx,
                  -1-Nx,1-Nx, 1-Nx,1+Nx, 1+Nx,-1+Nx, -1+Nx,-1-Nx};

    result.emplace_back(std::move(p1));
    result.emplace_back(std::move(p2));
    result.emplace_back(std::move(p3));
    result.emplace_back(std::move(p4));

    return result;
}

// same as V4, but last level is downsampled with -1
// 38400 points, 9963 with z5 using lambda = 20 and 29 new affine
std::vector<Pattern> getPatternsV3() {
    std::vector<Pattern> result;
    Pattern p1;
    p1.margin_ = 2; p1.down_ = 2;
    p1.points_ = {0,1+kNx, 0,1-kNx, 0,-1+kNx, 0,-1-kNx, 0,-1, 0,1, 0,kNx, 0,-kNx, //8
                  kNx,1, 1,-kNx, -kNx,-1, -1,kNx,                                 //4
                  -1-kNx,1-kNx, 1-kNx,1+kNx, 1+kNx,-1+kNx, -1+kNx,-1-kNx,         //4
                  0,2, 0,-2, 0,2*kNx, 0,-2*kNx,                                   //4
                  0,2+2*kNx, 0,2-2*kNx, 0,-2+2*kNx, 0,-2-2*kNx,                   //4
                  0,1+2*kNx, 0,1-2*kNx, 0,-1+2*kNx, 0,-1-2*kNx,                   //4
                  0,2+1*kNx, 0,2-1*kNx, 0,-2+1*kNx, 0,-2-1*kNx};                  //4
    Pattern p2;
    int Nx1 = (kNx - 2*p1.margin_)/2;
    p2.margin_ = 1; p2.down_ = 2;
    p2.points_ = {1,-1, Nx1,-Nx1, 1+Nx1,-1-Nx1, 1-Nx1,-1+Nx1,
                  1,-1+Nx1,1,-1-Nx1, -1,1+Nx1,-1,1-Nx1,
                  Nx1,-Nx1+1, Nx1,-Nx1-1, -Nx1,Nx1+1,-Nx1,Nx1-1,
                  0,1, 0,-1, 0,Nx1, 0,-Nx1, 0,1+Nx1, 0,1-Nx1, 0,-1+Nx1, 0,-1-Nx1};
    int Nx2 = (Nx1 - 2*p2.margin_)/2;
    Pattern p3; p3.margin_ = 1; p3.down_ = -1;
    p3.points_ = {0,1+Nx2,  0,1-Nx2,  0,-1+Nx2,  0,-1-Nx2,  1,-1,  Nx2,-Nx2,
                  Nx2,1,  1,-Nx2,  -Nx2,-1, -1,Nx2};
    result.emplace_back(std::move(p1));
    result.emplace_back(std::move(p2));
    result.emplace_back(std::move(p3));
    return result;
}

// 57,600 points. Achieves 0.9964!!! trained with 29 extra new affine, lambda = 50, 600 iterations.
std::vector<Pattern> getPatternsV4() {
    std::vector<Pattern> result;
    Pattern p1;
    p1.margin_ = 2; p1.down_ = 2;
    p1.points_ = {0,1+kNx, 0,1-kNx, 0,-1+kNx, 0,-1-kNx, 0,-1, 0,1, 0,kNx, 0,-kNx, //8
                  kNx,1, 1,-kNx, -kNx,-1, -1,kNx,                                 //4
                  -1-kNx,1-kNx, 1-kNx,1+kNx, 1+kNx,-1+kNx, -1+kNx,-1-kNx,         //4
                  0,2, 0,-2, 0,2*kNx, 0,-2*kNx,                                   //4
                  0,2+2*kNx, 0,2-2*kNx, 0,-2+2*kNx, 0,-2-2*kNx,                   //4
                  0,1+2*kNx, 0,1-2*kNx, 0,-1+2*kNx, 0,-1-2*kNx,                   //4
                  0,2+1*kNx, 0,2-1*kNx, 0,-2+1*kNx, 0,-2-1*kNx};                  //4
    Pattern p2;
    int Nx1 = (kNx - 2*p1.margin_)/2;
    p2.margin_ = 1; p2.down_ = 2;
    p2.points_ = {1,-1, Nx1,-Nx1, 1+Nx1,-1-Nx1, 1-Nx1,-1+Nx1,
                  1,-1+Nx1,1,-1-Nx1, -1,1+Nx1,-1,1-Nx1,
                  Nx1,-Nx1+1, Nx1,-Nx1-1, -Nx1,Nx1+1,-Nx1,Nx1-1,
                  0,1, 0,-1, 0,Nx1, 0,-Nx1, 0,1+Nx1, 0,1-Nx1, 0,-1+Nx1, 0,-1-Nx1};
    int Nx2 = (Nx1 - 2*p2.margin_)/2;
    Pattern p3; p3.margin_ = 1; p3.down_ = 0;
    p3.points_ = {0,1+Nx2,  0,1-Nx2,  0,-1+Nx2,  0,-1-Nx2,  1,-1,  Nx2,-Nx2,
                  Nx2,1,  1,-Nx2,  -Nx2,-1, -1,Nx2};
    result.emplace_back(std::move(p1));
    result.emplace_back(std::move(p2));
    result.emplace_back(std::move(p3));
    return result;
}
}

std::vector<Pattern> Pattern::getPatterns(int type) {
    return type == 0 ? getPatternsV0() :
           type == 1 ? getPatternsV1() :
           type == 2 ? getPatternsV2() :
           type == 3 ? getPatternsV3() : getPatternsV4();
}

