#pragma once

#include <vector>
#include <utility>
#include <algorithm>

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
        auto i = findIndex(a.first);
        for (int k=nmax_-1; k>i; --k) data_[k] = data_[k-1];
        data_[i] = a;
    }
    void reset() { nhave_ = 0; }
    int nHave() const { return nhave_; }
    bool allSame() const {
        if (nhave_ < nmax_) return false;
        int l = data_[0].second;
        for (int i=1; i<nmax_; ++i) if (data_[i].second != l) return false;
        return true;
    }
    bool allSame(int n) const {
        if (nhave_ < n) return false;
        int l = data_[0].second;
        for (int i=1; i<n; ++i) if (data_[i].second != l) return false;
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
    bool haveVote(int l) const {
        for (int i=0; i<nhave_; ++i) if (data_[i].second == l) return true;
        return false;
    }
    void getStoredLabels(int *use) const {
        for (int l=0; l<10; ++l) use[l] = 0;
        for (int i=0; i<nhave_; ++i) use[data_[i].second] = 1;
    }
    const std::vector<std::pair<float,int>>& getData() const { return data_; }
private:
    std::vector<std::pair<float,int>> data_;
    int nmax_;
    int nhave_;
    inline int findIndex(float x) const {
        if (x <= data_.front().first) return 0;
        auto bounds = std::make_pair(0, nmax_-1);
        while (bounds.second - bounds.first > 1) {
            int mav = (bounds.second + bounds.first)/2;
            bounds = x < data_[mav].first ? std::make_pair(bounds.first, mav) : std::make_pair(mav, bounds.second);
        }
        return bounds.second;
        //int ml = 0, mu = nmax_-1;
        //while (mu-ml > 1) {
        //    int mav = (ml+mu)/2;
        //    if (x < data_[mav].first) mu = mav; else ml = mav;
        //}
        //return mu;
    }
};

