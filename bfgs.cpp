#include "bfgs.h"
#include <cmath>

BFGSHistory::BFGSHistory(int np, int nh) : F_(0), np_(np), nh_(nh), index_(0), nhave_(0), first_(true) {
    lastp_.resize(np_,0);
    lastdp_.resize(np_,0);
    deltaX_.resize(nh_);
    deltaG_.resize(nh_);
    rhoi_.resize(nh_);
    alpha_.resize(nh_);
    for (int h=0; h<nh_; ++h) deltaX_[h].resize(np_);
    for (int h=0; h<nh_; ++h) deltaG_[h].resize(np_);
}

double BFGSHistory::setSearchDirection(double F, const double *p, const double *dp, double *du) {
    return setSearchDirectionT<double>(F,p,dp,du);
}

double BFGSHistory::setSearchDirection(double F, const float *p, const float *dp, float *du) {
    return setSearchDirectionT<float>(F,p,dp,du);
}

template<typename T> double BFGSHistory::setSearchDirectionT(double F, const T *p, const T *dp, T *du) {
    if (!nh_) {
        for (int j=0; j<np_; ++j) du[j] = -dp[j];
        return 1;
    }
    int index = index_;
    if (index >= nh_) index -= nh_;
    double dxdg = 0, dgnorm = 0, gnorm = 0;
    if (!first_) {
        for (int j=0; j<np_; ++j) {
            double deltaX = p[j] - lastp_[j], deltaG = dp[j] - lastdp_[j];
            deltaX_[index][j] = deltaX;
            deltaG_[index][j] = deltaG;
            gnorm += dp[j]*dp[j];
            dxdg += deltaX*deltaG; dgnorm += deltaG*deltaG;
            du[j] = -dp[j];
            lastp_[j] = p[j];
            lastdp_[j] = dp[j];
        }
    }
    else {
        for (int j=0; j<np_; ++j) {
            gnorm += dp[j]*dp[j];
            lastp_[j] = p[j];
            lastdp_[j] = dp[j];
        }
        gnorm = -1/sqrt(gnorm);
        for (int j=0; j<np_; ++j) du[j] = gnorm*dp[j];
        F_ = F;
        first_ = false;
        return 0;
    }
    if (nhave_ < nh_) ++nhave_;
    rhoi_[index] = 1/dxdg;
    index_ = ++index;
    int h = index;
    for (int hh=0; hh<nhave_; ++hh) {
        h = (h + nh_ - 1) % nh_;
        double alpha = 0;
        for (int j=0; j<np_; ++j) alpha += du[j]*deltaX_[h][j];
        alpha *= rhoi_[h];
        alpha_[h] = alpha;
        for (int j=0; j<np_; ++j) du[j] -= alpha*deltaG_[h][j];
    }
    double s = dxdg/dgnorm;
    for (int j=0; j<np_; ++j) du[j] *= s;
    for (int hh=0; hh<nhave_; ++hh) {
        double beta = 0;
        for (int j=0; j<np_; ++j) beta += deltaG_[h][j]*du[j];
        beta = alpha_[h] - rhoi_[h]*beta;
        for (int j=0; j<np_; ++j) du[j] += beta*deltaX_[h][j];
        h = (h + 1) % nh_;
    }
    double pg = 0, pnorm = 0; dgnorm = 0;
    for (int j=0; j<np_; ++j) {
        dgnorm += dp[j]*dp[j];
        pg += du[j]*dp[j];
        pnorm += du[j]*du[j];
    }
    pnorm = sqrt(pnorm);
    double scale = pg > 0 ? -1./pnorm : 1./pnorm;
    for (int j=0; j<np_; ++j) du[j] *= scale;
    pg *= scale;
    double deltaF = F - F_;
    F_ = F;
    return 8*deltaF/pg;
}

