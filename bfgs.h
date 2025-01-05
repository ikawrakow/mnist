#pragma once

#include <vector>

class BFGSHistory {
public:
    BFGSHistory(int np, int nh);
    double setSearchDirection(double F, const double *p, const double *dp, double *du);
    double setSearchDirection(double F, const float *p, const float *dp, float *du);
private:
    template <typename T> double setSearchDirectionT(double F, const T *p, const T *dp, T *du);
    std::vector<double>  lastp_;
    std::vector<double>  lastdp_;
    std::vector<std::vector<double>> deltaX_;
    std::vector<std::vector<double>> deltaG_;
    std::vector<double>  rhoi_;
    std::vector<double>  alpha_;
    double               F_;
    int                  np_;
    int                  nh_;
    int                  index_;
    int                  nhave_;
    bool                 first_;
};

