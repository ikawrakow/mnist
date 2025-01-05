#include "getImages.h"
#include "imageUtils.h"

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

struct Pattern {
    std::vector<int> points_;
    int margin_;
    int down_;
};

void applyPattern(int Nx, int Ny, const Pattern &pattern, const uint8_t *A, uint8_t *R, uint8_t *aux, int *aux1) {
    int type = 0, margin = pattern.margin_;
    if( margin < 0 ) { type = 1; margin = -margin; }
    int Nx1 = Nx - 2*margin, Ny1 = Ny - 2*margin;
    int Nx2 = Nx1, Ny2 = Ny1;
    if( pattern.down_ ) {
        if( pattern.down_ > 0 ) {
            Nx2 = (Nx2 - 2 + pattern.down_)/pattern.down_; Ny2 = (Ny2 - 2 + pattern.down_)/pattern.down_;
        }
        else { Nx2 = Nx1 + Ny1; Ny2 = 1; }
    }
    int np = pattern.points_.size()/2;
    for(int ip=0; ip<np; ++ip) {
        auto j1 = pattern.points_[2*ip], j2 = pattern.points_[2*ip+1];
        auto B = pattern.down_ ? aux : R;
        int j = 0;
        if( type == 0 ) {
            for(int y=margin; y<Ny-margin; ++y) for(int x=margin; x<Nx-margin; ++x) {
                int jj = x + y*Nx;
                int a1 = A[jj+j1], a2 = A[jj+j2];
                B[j++] = a1 > a2 ? a1 - a2 : 0;
            }
        }
        else {
            for(int y=margin; y<Ny-margin; ++y) for(int x=margin; x<Nx-margin; ++x) {
                int jj = x + y*Nx;
                int a1 = A[jj], a2 = A[jj+j1], a3 = A[jj+j2]; a2 = (a2 + a3)/2;
                B[j++] = a1 > a2 ? a1 - a2 : 0;
            }
        }
        if( pattern.down_ ) {
            if( pattern.down_ > 0 ) {
                j = 0;
                for(int y=0; y<Ny1-1; y+=pattern.down_) for(int x=0; x<Nx1-1; x+=pattern.down_) {
                    int s = 0; for(int ky=y; ky<y+2; ++ky) for(int kx=x; kx<x+2; ++kx) s += aux[kx+ky*Nx1];
                    R[j++] = s >> 2;
                }
            }
            else {
                for(j=0; j<Nx2*Ny2; ++j) aux1[j] = 0;
                for(int y=0; y<Ny1; ++y) for(int x=0; x<Nx1; ++x) {
                    int a = aux[x+y*Nx1];
                    aux1[x] += a; aux1[Nx1+y] += a;
                }
                for(j=0; j<Nx1; ++j) R[j] = toNearestInt((1.f*aux1[j])/Ny1);
                for(j=Nx1; j<Nx1+Ny1; ++j) R[j] = toNearestInt((1.f*aux1[j])/Nx1);
            }
        }
        R += Nx2*Ny2;
    }
}

std::vector<uint8_t> applyPattern(int Nx, int Ny, const std::vector<uint8_t> &images, const Pattern &pattern,
        int &Nxf, int &Nyf) {
    int64_t nimage = images.size()/(Nx*Ny);
    int np = pattern.points_.size()/2;
    int margin = pattern.margin_; if( margin < 0 ) margin = -margin;
    int Nx1 = Nx - 2*margin, Ny1 = Ny - 2*margin;
    int Nx2 = Nx1, Ny2 = Ny1;
    if( pattern.down_ ) {
        if( pattern.down_ > 0 ) {
            Nx2 = (Nx2 - 2 + pattern.down_)/pattern.down_; Ny2 = (Ny2 - 2 + pattern.down_)/pattern.down_;
        }
        else { Nx2 = Nx1 + Ny1; Ny2 = 1; }
    }
    std::vector<uint8_t> result(Nx2*Ny2*np*nimage);
    printf("%s: nimage=%d np=%d Nx=%d Ny=%d Nx1=%d Ny1=%d Nx2=%d Ny2=%d\n",__func__,(int)nimage,np,Nx,Ny,Nx1,Ny1,Nx2,Ny2);
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
                applyPattern(Nx,Ny,pattern,A,R,aux.data(),aux1.data());
                A += Nx*Ny; R += Nx2*Ny2*np;
            }
        }
    };
    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread-1);
    for(auto & w : workers) w = std::thread(compute);
    compute();
    for(auto & w : workers) w.join();
    Nxf = Nx2; Nyf = Ny2;
    return result;
}

bool getPattern(const char *fname, std::vector<Pattern> &patterns, std::vector<std::vector<double>> &allP) {
    std::ifstream in(fname,std::ios::binary);
    if( !in ) {
        printf("%s: failed top open %s\n",__func__,fname); return false;
    }
    int npattern;
    in.read((char *)&npattern,sizeof(int));
    patterns.resize(npattern);
    for(auto & p : patterns) {
        int np = p.points_.size();
        in.read((char *)&np,sizeof(int));
        p.points_.resize(np);
        in.read((char *)p.points_.data(),np*sizeof(int));
        in.read((char *)&p.margin_,sizeof(p.margin_));
        in.read((char *)&p.down_,sizeof(p.down_));
    }
    if( (int)allP.size() != 10 ) allP.resize(10);
    int npoint;
    in.read((char *)&npoint,sizeof(int));
    for(auto & p : allP) {
        p.resize(npoint);
        in.read((char *)p.data(),p.size()*sizeof(double));
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
    std::vector<std::vector<double>> allP(10);
    if (!getPattern(argv[1],patterns,allP)) return 1;

    auto tim1 = std::chrono::steady_clock::now();

    int Nx = kNx, Ny = kNy;
    for(const auto & p : patterns) {
        int Nx1, Ny1;
        testImages = applyPattern(Nx,Ny,testImages,p,Nx1,Ny1);
        Nx = Nx1; Ny = Ny1;
    }
    int npoint = testImages.size()/testLabels.size();
    printf("# %d points per image\n",npoint);
    std::atomic<int> counter(0);
    std::vector<float> V(10*kNtest);
    auto predict = [&counter, &allP, &testImages, &V, npoint]() {
        int chunk = 64;
        while (true) {
            int first = counter.fetch_add(chunk);
            if (first >= kNtest) break;
            int n = first + chunk <= kNtest ? chunk : kNtest - first;
            for (int l=0; l<10; ++l) {
                auto B = testImages.data() + first*npoint;
                auto v = V.data() + 10*first + l;
                auto& p = allP[l];
                for (int i=0; i<n; ++i) {
                    double s = 0; for(int j=0; j<npoint; ++j) s += p[j]*B[j];
                    *v = s; v += 10; B += npoint;
                }
            }
        }
    };
    for (auto& w : workers) w = std::thread(predict);
    for (auto& w : workers) w.join();

    std::vector<int> countOK(kNtest,0);
    std::vector<std::pair<float,int>> X(10);

    std::vector<int> countGood(confLevels.size(),0), count(confLevels.size(),0);
    int ncheck = 5;
    std::vector<int> Ngood(ncheck,0);
    for (int i=0; i<kNtest; ++i) {
        for (int l=0; l<10; ++l) {
            X[l] = {V[10*i+l],l};
        }
        std::sort(X.begin(),X.end());
        if (X[9].second == testLabels[i]) ++countOK[i];
        auto delta = X[9].first - X[8].first;
        for(int k=0; k<(int)confLevels.size(); ++k) {
            if (delta > confLevels[k]) {
                ++count[k];
                if (X[9].second == testLabels[i]) ++countGood[k];
            }
        }
        for(int k=9; k>=0; --k) {
            if (X[k].second == testLabels[i]) {
                int n = 9 - k;
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
