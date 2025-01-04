#include "imageUtils.h"
#include "getImages.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <fstream>
#include <random>
#include <chrono>

namespace {
class GaussianBlur {
public:
    GaussianBlur(float sigma, int Nx, int Ny);
    void blurImage(const short *inImage, float *outImage) { blurImageT(inImage, outImage); }
    void blurImage(const float *inImage, float *outImage) { blurImageT(inImage, outImage); }
    void blurImage(const double *inImage, float *outImage){ blurImageT(inImage, outImage); }

private:

    std::vector<float>  _kernel;  //!< The Gaussian kernel
    std::vector<float>  _wtx;     //!< For normalizing edge voxels
    std::vector<float>  _wty;     //!< For normalizing edge voxels
    std::vector<float>  _tmp;     //!< Temporary storage
    int                 _nk;      //!< Kernel size
    int                 _Nx;      //!< Number of voxels in x
    int                 _Ny;      //!< Number of voxels in y

    void setWeights(int Nx, std::vector<float> &wtx);
    template<class T> void blurImageT(const T* inImage, float *outImage);
};

GaussianBlur::GaussianBlur(float sigma, int Nx, int Ny) : _Nx(Nx), _Ny(Ny) {
    int nk = round(2.7*sigma);
    if (nk < 1) nk = 1;
    _nk = nk;
    _kernel.resize(2*nk+1);
    float sig2i = 0.5f/(sigma*sigma);
    float sum = 0;
    for (int i=0; i<=2*nk; ++i) {
        _kernel[i] = exp(-sig2i*(i-nk)*(i-nk)); sum += _kernel[i];
    }
    sum = 1/sum;
    for (int i=0; i<=2*nk; ++i) _kernel[i] *= sum;
    setWeights(_Nx, _wtx);
    setWeights(_Ny, _wty);
    _tmp.resize(_Nx);
}

template<class T> void GaussianBlur::blurImageT(const T* inImage, float *outImage) {
    float *B = outImage;
    for (int iy=0; iy<_Ny; ++iy) {
        int iymin = iy - _nk; if (iymin < 0) iymin = 0;
        int iymax = iy + _nk; if (iymax > _Ny-1) iymax = _Ny - 1;
        float wy = _wty[iymin]*_kernel[_nk+iymin-iy]; const T *I = &inImage[iymin*_Nx];
        for (int ix=0; ix<_Nx; ++ix) _tmp[ix] = wy*I[ix];
        for (int iy1=iymin+1; iy1<=iymax; ++iy1) {
            wy = _wty[iy1]*_kernel[_nk+iy1-iy]; I = &inImage[iy1*_Nx];
            for (int ix=0; ix<_Nx; ++ix) _tmp[ix] += wy*I[ix];
        }
        for (int ix=0; ix<_Nx; ++ix) {
            int ixmin = ix - _nk; if (ixmin < 0) ixmin = 0;
            int ixmax = ix + _nk; if (ixmax > _Nx-1) ixmax = _Nx - 1;
            float sum = 0; for (int ix1=ixmin; ix1<=ixmax; ++ix1) sum += _kernel[_nk + ix1 - ix]*_tmp[ix1];
            *B++ = _wtx[ix]*sum;
        }
    }
}

void GaussianBlur::setWeights(int Nx, std::vector<float> &wtx) {
    wtx.resize(Nx);
    for (int ix=0; ix<Nx; ++ix) {
        int ixmin = std::max(0, ix - _nk);
        int ixmax = std::min(Nx-1, ix + _nk);
        if (ixmax - ixmin == 2*_nk) wtx[ix] = 1;
        else {
            float sum = 0;
            for (int i=ixmin; i<=ixmax; ++i) sum += _kernel[_nk+i-ix];
            wtx[ix] = 1/sum;
        }
    }
}

}

void addElasticDeformationsSameT(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd, double sigGauss, double alpha,
        int seq, const char* fname) {
    std::mt19937 rndm(1234+seq);
    float rnorm = 2.f/4294967296.f;
    int nimage = labels.size();
    uint64_t nnew = (nAdd+1)*nimage;
    uint64_t ntot = nnew*kSize;
    printf("Adding %d same elastic deformations with sigmaGauss=%g alpha=%g. New number of images is %g\n",nAdd,sigGauss,alpha,1.*nnew);
    labels.resize(nnew);
    images.resize(ntot);
    int margin = toNearestInt(2*sigGauss);
    int Nx = kNx + 2*margin, Ny = kNy + 2*margin;
    std::vector<float> ux(Nx*Ny), uy(Nx*Ny), uxb(nAdd*Nx*Ny), uyb(nAdd*Nx*Ny);
    GaussianBlur gauss(sigGauss,Nx,Ny);
    for (int it=0; it<nAdd; ++it) {
        for (int i=0; i<Nx*Ny; ++i) { ux[i] = rnorm*rndm()-1; uy[i] = rnorm*rndm()-1; }
        gauss.blurImage(ux.data(), &uxb[it*Nx*Ny]);
        gauss.blurImage(uy.data(), &uyb[it*Nx*Ny]);
        for (int i=0; i<Nx*Ny; ++i) { uxb[it*Nx*Ny+i] *= alpha; uyb[it*Nx*Ny+i] *= alpha; }
    }
    std::atomic<int> counter(0);
    auto compute = [&counter, &images, &labels, &uxb, &uyb, nimage, nAdd, Nx, Ny, margin]() {
        int chunk = 64;
        while (true) {
            int first = counter.fetch_add(chunk);
            if (first >= nimage) break;
            int last = std::min(nimage, first + chunk);
            for (int i=first; i<last; ++i) {
                const uint8_t* A = &images[i*kSize];
                uint64_t start1 = nimage + i*nAdd;
                uint64_t start2 = start1*kSize;
                uint8_t* B = &images[start2], *L = &labels[start1];
                const float *ux = uxb.data(), *uy = uyb.data();
                for (int it=0; it<nAdd; ++it) {
                    for (int y=0; y<kNy; ++y) for (int x=0; x<kNx; ++x) {
                        float x1 = ux[x+margin+(y+margin)*Nx] + x;
                        float y1 = uy[x+margin+(y+margin)*Nx] + y;
                        if (x1 >= 0 && x1 < kNx && y1 >= 0 && y1 < kNy) {
                            int ix1 = (int)x1, iy1 = (int)y1;
                            x1 -= ix1; y1 -= iy1;
                            int ix2 = ix1 + 1; float x2 = 1 - x1; if (ix2 >= kNx) { ix2 = kNx-1; x1 = 1; x2 = 0; }
                            int iy2 = iy1 + 1; float y2 = 1 - y1; if (iy2 >= kNy) { iy2 = kNy-1; y1 = 1; y2 = 0; }
                            float b = y1*(x1*A[ix2+iy2*kNx] + x2*A[ix1+iy2*kNx])+y2*(x1*A[ix2+iy1*kNx] + x2*A[ix1+iy1*kNx]);
                            if (b < 0 || b > 255.49f) {
                                printf("Huh? b=%g. x=%d y=%d x1=%g y1=%g x2=%g y2=%g ix1=%d ix2=%d iy1=%d iy2=%d\n",b,x,y,x1,y1,x2,y2,ix1,ix2,iy1,iy2);
                                exit(1);
                            }
                            B[x+y*kNx] = toNearestInt(b);
                        }
                        else B[x+y*kNx] = 0;
                    }
                    B += kSize; *L++ = labels[i];
                    ux += Nx*Ny; uy += Nx*Ny;
                }
            }
        }
    };
    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread);
    auto tim1 = std::chrono::steady_clock::now();
    for (auto& w : workers) w = std::thread(compute);
    for (auto& w : workers) w.join();
    auto tim2 = std::chrono::steady_clock::now();
    printf("%s: it took %g seconds to add %d extra impages per input image\n", __func__,
            1e-6*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count(), nAdd);
    if (fname) {
        std::ofstream out(fname,std::ios::binary);
        out.write((char *)&images[nimage*kSize],ntot-nimage*kSize);
        printf("Wrote %d distorted images to %s\n",nAdd*nimage,fname);
    }
}

void addElasticDeformationsSameT(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd) {
    if (nAdd < 1) return;
    addElasticDeformationsSameT(images, labels, nAdd, 6., 38., 0, nullptr);
}

