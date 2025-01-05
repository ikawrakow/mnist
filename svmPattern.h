#pragma once

#include "imageUtils.h"

#include <vector>
#include <cstdint>

struct Pattern {
    std::vector<int> points_;
    int margin_;
    int down_;
    inline void apply(int Nx, int Ny, const uint8_t* A, uint8_t* R, uint8_t* aux, int* aux1) const {
        int type = 0, margin = margin_;
        if (margin < 0) { type = 1; margin = -margin; }
        int Nx1 = Nx - 2*margin, Ny1 = Ny - 2*margin;
        int Nx2 = Nx1, Ny2 = Ny1;
        if (down_) {
            if (down_ > 0) {
                Nx2 = (Nx2 - 2 + down_)/down_; Ny2 = (Ny2 - 2 + down_)/down_;
            }
            else { Nx2 = Nx1 + Ny1; Ny2 = 1; }
        }
        int np = points_.size()/2;
        for (int ip=0; ip<np; ++ip) {
            auto j1 = points_[2*ip], j2 = points_[2*ip+1];
            auto B = down_ ? aux : R;
            int j = 0;
            if (type == 0) {
                for (int y=margin; y<Ny-margin; ++y) for (int x=margin; x<Nx-margin; ++x) {
                    int jj = x + y*Nx;
                    int a1 = A[jj+j1], a2 = A[jj+j2];
                    B[j++] = a1 > a2 ? a1 - a2 : 0;
                }
            }
            else {
                for (int y=margin; y<Ny-margin; ++y) for (int x=margin; x<Nx-margin; ++x) {
                    int jj = x + y*Nx;
                    int a1 = A[jj], a2 = A[jj+j1], a3 = A[jj+j2]; a2 = (a2 + a3)/2;
                    B[j++] = a1 > a2 ? a1 - a2 : 0;
                }
            }
            if (down_) {
                if (down_ > 0) {
                    j = 0;
                    for (int y=0; y<Ny1-1; y+=down_) for (int x=0; x<Nx1-1; x+=down_) {
                        int s = 0; for(int ky=y; ky<y+2; ++ky) for(int kx=x; kx<x+2; ++kx) s += aux[kx+ky*Nx1];
                        R[j++] = s >> 2;
                    }
                }
                else {
                    for (j=0; j<Nx2*Ny2; ++j) aux1[j] = 0;
                    for (int y=0; y<Ny1; ++y) for (int x=0; x<Nx1; ++x) {
                        int a = aux[x+y*Nx1];
                        aux1[x] += a; aux1[Nx1+y] += a;
                    }
                    for (j=0;   j<Nx1;     ++j) R[j] = toNearestInt((1.f*aux1[j])/Ny1);
                    for (j=Nx1; j<Nx1+Ny1; ++j) R[j] = toNearestInt((1.f*aux1[j])/Nx1);
                }
            }
            R += Nx2*Ny2;
        }
    }
    static std::vector<uint8_t> apply(int Nx, int Ny, const std::vector<uint8_t>& images, const Pattern &pattern, int& Nxf, int& Nyf);
    static std::vector<Pattern> getPatterns(int type);
};

