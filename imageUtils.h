#pragma once

#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>

void addElasticDeformationsSameT(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd, double sigGauss, double alpha, int seq, const char* fname);

void addElasticDeformationsSameT(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd);

void addElasticDeformations(int nx, int ny, std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd,
        double sigGauss, double alpha, int seq, const char* fname);

void addElasticDeformations(std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAdd);

void addAffineTransformations(int Nx, int Ny, std::vector<uint8_t>& images, std::vector<uint8_t>& labels, int nAffine,
        double phiRangle, double shearRange, double zoomRange, double shiftXrange, double shiftYrange,
        int rng_seq, bool recenter = false);

static inline int toNearestInt(float fval) noexcept {
    assert(fval <= 4194303.f);
    constexpr float kSnapper=3<<22;
    auto val = fval + kSnapper;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline int toNearestInt(double dval) noexcept {
    return toNearestInt(float(dval));
}

