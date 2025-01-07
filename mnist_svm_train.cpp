#include "getImages.h"
#include "imageUtils.h"
#include "svmPattern.h"
#include "bfgs.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>

template <typename Float>
double computeFdF(Float norm, const std::vector<uint8_t>& labels, const std::vector<Float>& V,
        std::vector<Float>& dfdv, std::vector<Float>& d2fdv2, std::vector<Float>& F) {
    auto t1 = std::chrono::steady_clock::now();
    int nimage = labels.size();
    for (int j=0; j<kNlabels*nimage; ++j) dfdv[j] = d2fdv2[j] = 0;
    for (auto& f : F) f = 0;
    Float n = 2*norm;
    for (int l=0; l<kNlabels; ++l) {
        Float f = 0;
        for (int i=0; i<nimage; ++i) {
            auto y = labels[i] == l ? 1 : -1;
            Float d = 1 - V[kNlabels*i+l]*y;
            if (d > 0) {
                f += d*d;
                dfdv[kNlabels*i+l] -= n*d*y;
                d2fdv2[kNlabels*i+l] += n;
            }
        }
        F[l] = norm*f;
    }
    auto t2 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
}

template <typename Float>
double computeP(const std::vector<uint8_t>& dImage, const std::vector<Float>& dfdv, std::vector<Float>& dp,
        int npoint, int nimage, int chunk, std::vector<std::thread>& workers) {
    auto t1 = std::chrono::steady_clock::now();
    std::atomic<int> counter(0);
    auto compute = [&dImage,&dfdv,&dp,&counter,npoint,nimage,chunk]() {
        std::vector<Float> tmp(kNlabels*chunk);
        while (true) {
            int first = counter.fetch_add(chunk);
            if (first >= npoint) break;
            auto A = dImage.data() + first;
            int n = first + chunk < npoint ? chunk : npoint - first;
            for (int j=0; j<kNlabels*n; ++j) tmp[j] = 0;
            for (int i=0; i<nimage; ++i) {
                auto t = tmp.data();
                for (int l=0; l<kNlabels; ++l) {
                    Float d = dfdv[kNlabels*i+l];
                    if (d) for(int j=0; j<n; ++j) t[j] += d*A[j];
                    t += n;
                }
                A += npoint;
            }
            auto p = &dp[first];
            for (int l=0; l<kNlabels; ++l) for(int j=0; j<n; ++j) p[j+l*npoint] += tmp[n*l+j];
        }
    };
    for (auto& w : workers) w = std::thread(compute);
    for (auto& w : workers) w.join();
    auto t2 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
}

template <typename Float>
double computeVfast(const std::vector<uint8_t>& dImage, std::vector<Float>& dv, std::vector<Float>& du,
        std::vector<int16_t>& ui, int npoint, int nimage,std::vector<std::thread> &workers) {
    constexpr Float kUmax = 32766;
    Float scale[kNlabels], scaleI[kNlabels];
    auto t1 = std::chrono::steady_clock::now();
    auto u = du.data(); auto iu = ui.data();
    for (int l=0; l<kNlabels; ++l) {
        Float um = 0;
        for (int j=0; j<npoint; ++j) {
            Float au = std::abs(u[j]);
            um = std::max(um, au);
        }
        scale[l] = kUmax/um; scaleI[l] = um/kUmax;
        for (int j=0; j<npoint; ++j) {
            iu[j] = toNearestInt(u[j]*scale[l]);
            u[j] = scaleI[l]*iu[j];
        }
        u += npoint; iu += npoint;
    }
    int nthread = 1 + workers.size();
    int chunk1 = nimage/(8*nthread);
    int chunk = 16; while (chunk < chunk1) chunk *= 2;
    std::atomic<int> counter(0);
    auto compute = [&dImage,&dv,&ui,&counter,&scaleI,npoint,nimage,chunk]() {
        int n128 = npoint/128;
        int nn = npoint - 128*n128;
        while (true) {
            int first = counter.fetch_add(chunk);
            if (first >= nimage) break;
            int last = first + chunk; if (last > nimage) last = nimage;
            auto A = &dImage[(int64_t)first * (int64_t)npoint];
            for (int i=first; i<last; ++i) {
                auto u = ui.data();
                for (int l=0; l<kNlabels; ++l) {
                    int64_t s = 0; auto a = A;
                    for (int i128=0; i128<n128; ++i128) {
                        int si = 0; for (int k=0; k<128; ++k) si += (int)u[k] * a[k];
                        s += si; a += 128; u += 128;
                    }
                    int si = 0; for (int k=0; k<nn; ++k) si += (int)u[k] * a[k];
                    s += si;
                    a += nn; u += nn;
                    dv[kNlabels*i+l] = scaleI[l]*s;
                }
                A += npoint;
            }
        }
    };
    for (auto& w : workers) w = std::thread(compute);
    for (auto& w : workers) w.join();
    auto t2 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
}

template <typename Float>
void writeResults(const char *ofile, int npoint, const std::vector<Pattern> &patterns, const std::vector<Float> &P, bool tell = false) {
    std::ofstream out(ofile,std::ios::binary);
    int npattern = patterns.size();
    out.write((char *)&npattern,sizeof(int));
    for (auto& p : patterns) {
        int np = p.points_.size();
        out.write((char *)&np,sizeof(int));
        out.write((char *)p.points_.data(),np*sizeof(int));
        out.write((char *)&p.margin_,sizeof(p.margin_));
        out.write((char *)&p.down_,sizeof(p.down_));
    }
    out.write((char *)&npoint,sizeof(int));
    out.write((char *)P.data(),P.size()*sizeof(Float));
    if (tell) {
        printf("Wrote training results to %s\n",ofile);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {

    using Float = float;

    int iarg = 1;
    int    niter  = argc > iarg ? atoi(argv[iarg++]) : 200;
    int    nadd   = argc > iarg ? atoi(argv[iarg++]) : 0;
    Float  lambda = argc > iarg ? atof(argv[iarg++]) : 10.;
    int    type   = argc > iarg ? atoi(argv[iarg++]) : 0;
    auto   ofile  = argc > iarg ? argv[iarg++]       : "test.dat";
    int    rseq   = argc > iarg ? atoi(argv[iarg++]) : 0;
    int    nhist  = argc > iarg ? atoi(argv[iarg++]) : 100;
    float  max_t  = argc > iarg ? atof(argv[iarg++]) : 0.05f;

    auto labels = getTraningLabels();
    if (labels.size() != kNtrain) return 1;

    auto images = getTrainingImages();
    if (images.size() != kNtrain*kNx*kNy) return 1;

    auto testLabels = getTestLabels();
    if (testLabels.size() != kNtest) return 1;

    auto testImages = getTestImages();
    if (testImages.size() != kNtest*kSize) return 1;

    if (nadd > 0) {
        if (nadd < 100) addElasticDeformations(images,labels,nadd);
        else {
            nadd -= 100;
            auto images1 = images; auto labels1 = labels;
            addElasticDeformations(images,labels,nadd);
            addAffineTransformations(kNx,kNy,images1,labels1,nadd+1,12.,0.6,0.1,max_t,max_t,0,false);
            auto curSize = images.size();
            images.resize(images.size() + images1.size() - kNtrain*kSize);
            std::copy(images1.begin() + kNtrain*kSize, images1.end(), images.begin() + curSize);
            curSize = labels.size();
            labels.resize(labels.size() + labels1.size() - kNtrain);
            std::copy(labels1.begin() + kNtrain, labels1.end(), labels.begin() + curSize);
        }
    }
    else if (nadd < 0) {
        addAffineTransformations(kNx,kNy,images,labels,-nadd,12.,0.6,0.1,max_t,max_t,rseq,false);
    }
    int ntrain = labels.size();

    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread);

    const auto patterns = Pattern::getPatterns(type);
    int Nx = kNx, Ny = kNy;
    for (auto& p : patterns) {
        int Nx1, Ny1;
        images = Pattern::apply(Nx,Ny,images,p,Nx1,Ny1);
        testImages = Pattern::apply(Nx,Ny,testImages,p,Nx1,Ny1);
        Nx = Nx1; Ny = Ny1;
    }
    int npoint = images.size()/labels.size();
    printf("%d points per image\n",npoint);
    fflush(stdout);

    std::vector<Float> V(kNlabels*ntrain,0), dV(kNlabels*ntrain,0), dfdv(kNlabels*ntrain), d2fdv2(kNlabels*ntrain),
                       P(kNlabels*npoint,0), dP(kNlabels*npoint,0), du(kNlabels*npoint), sump2(kNlabels), F(kNlabels);
    std::vector<int16_t> ui(kNlabels*npoint);

    const Float norm = 1.*kNlabels/ntrain;

    int chunk1 = npoint/(4*nthread);
    int chunk = 8; while (chunk < chunk1) chunk *= 2;
    printf("Using chunk=%d\n",chunk);
    fflush(stdout);

    std::vector<Float> Fvalues(niter);
    std::vector<BFGSHistory> bfgs;
    bfgs.reserve(kNlabels);
    for (int l=0; l<kNlabels; ++l) bfgs.emplace_back(npoint,nhist);

    auto computeF = [ntrain,&sump2,lambda,norm,&V,&dV,&labels](int l, Float s1, Float s2, Float t) -> Float {
        Float Fn = lambda*(sump2[l] + 2*t*s1 + t*t*s2);
        Float F1 = 0;
        for (int i=0; i<ntrain; ++i) {
            auto y = labels[i] == l ? 1 : -1;
            auto v = V[kNlabels*i+l] + t*dV[kNlabels*i+l];
            Float d = 1 - v*y;
            if (d > 0) F1 += d*d;
        }
        return Fn + norm*F1;
    };

    Float Fold = std::numeric_limits<Float>::max();
    double totTime = 0, totItime = 0, totFtime = 0, totStime = 0, totUtime, totPtime = 0, totVtime = 0, totCtime = 0;

    std::vector<Float> bestP(kNlabels*npoint,0);
    int bestNgood = 0;
    int nconv = 0;

    auto setSearchDirection = [&bfgs, &F, &P, &dP, &du, npoint] () {
        auto t1 = std::chrono::steady_clock::now();
        for (int l=0; l<kNlabels; ++l) bfgs[l].setSearchDirection(F[l],P.data()+npoint*l,dP.data()+npoint*l,du.data()+npoint*l);
        auto t2 = std::chrono::steady_clock::now();
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    };

    Float totF, totP2;
    bool allConverged;
    auto takeStep = [&computeF, &F, &totF, &totP2, &allConverged, &P, &du, &sump2, &V, &dV, &dfdv, &d2fdv2, lambda, ntrain, npoint] () {
        auto tim1 = std::chrono::steady_clock::now();
        totF = totP2 = 0;
        allConverged = true;
        auto p = P.data(); auto dU = du.data();
        for (int l=0; l<kNlabels; ++l) {
            Float F1 = F[l] + lambda*sump2[l];
            Float sum1 = 0, sum2 = 0;
            for (int i=0; i<ntrain; ++i) { sum1 -= dV[kNlabels*i+l]*dfdv[kNlabels*i+l]; sum2 += dV[kNlabels*i+l]*dV[kNlabels*i+l]*d2fdv2[kNlabels*i+l]; }
            sum1 *= 0.5; sum2 *= 0.5;
            Float s1 = 0, s2 = 0;
            if (lambda > 0) {
                for (int j=0; j<npoint; ++j) { s1 += p[j]*dU[j]; s2 += dU[j]*dU[j]; }
                sum1 -= lambda*s1; sum2 += lambda*s2;
            }
            if (fabs(sum1) < 1e-13 || !sum2) {
                printf("Converged for l = %d (sums are %g,%g)\n",l,sum1,sum2);
                fflush(stdout);
                continue;
            }
            allConverged = false;
            Float t = sum1/sum2;
            Float Fn = computeF(l,s1,s2,t);
            Float Fe = F1 - sum1*t;
            if (Fn > Fe + 1e-6) {
                Float tn = sum1*t*t/(Fn - F1 + 2*t*sum1);
                auto Fn1 = computeF(l,s1,s2,tn);
                if (Fn1 < Fn) { Fn = Fn1; t = tn; }
                if (Fn > F1 + 1e-4) printf("Oops: l=%d F[l]=%g F1=%g Fn=%g Fn1=%g Fe=%g t=%g, %g\n",l,F[l],F1,Fn,Fn1,Fe,t,tn);
            }
            for (int j=0; j<npoint; ++j) { p[j] += t*dU[j]; totP2 += p[j]*p[j]; }
            for (int i=0; i<ntrain; ++i) V[kNlabels*i+l] += t*dV[kNlabels*i+l];
            totF += Fn;
            p += npoint; dU += npoint;
        }
        auto tim2 = std::chrono::steady_clock::now();
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count();
    };

    auto initIteration = [&P, &dP, &sump2, npoint, lambda] () {
        auto tim1 = std::chrono::steady_clock::now();
        auto p = P.data(); auto dp = dP.data();
        for (int l=0; l<kNlabels; ++l) {
            Float s = 0;
            for (int j=0; j<npoint; ++j) { dp[j] = 2*lambda*p[j]; s += p[j]*p[j]; }
            sump2[l] = s;
            p += npoint; dp += npoint;
        }
        auto tim2 = std::chrono::steady_clock::now();
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count();
    };

    auto predictTest = [&testImages, testLabels, &P, &workers, npoint] () {
        std::vector<int> R(workers.size());
        std::atomic<int> counter(0);
        auto compute = [&counter, &testImages, &testLabels, &P, &R, npoint] (int it) {
            constexpr int kChunk = 64;
            int ngood = 0;
            while (true) {
                int first = counter.fetch_add(kChunk);
                if (first >= kNtest) {
                    R[it] = ngood; break;
                }
                int last = std::min(first + kChunk, (int)kNtest);
                auto B = testImages.data() + uint64_t(first)*npoint;
                for (int i = first; i < last; ++i) {
                    auto p = P.data();
                    Float best = -std::numeric_limits<Float>::max(); int lbest = -1;
                    for (int l=0; l<kNlabels; ++l) {
                        Float s = 0; for (int j=0; j<npoint; ++j) s += p[j]*B[j];
                        if (s > best) { best = s; lbest = l; }
                        p += npoint;
                    }
                    if (lbest == testLabels[i]) ++ngood;
                    B += npoint;
                }
            }
        };
        int it = 0;
        for (auto& w : workers) w = std::thread(compute, it++);
        for (auto& w : workers) w.join();
        int ngood = 0;
        for (auto n : R) ngood += n;
        return ngood;
    };

    auto tStart = std::chrono::steady_clock::now();

    auto checkResults = [&predictTest, &P, &V, &labels, &bestP, &bestNgood, &patterns, &totF, &totP2, &workers,
         npoint, ntrain, lambda, tStart, ofile] (int iter) {
        auto tim1 = std::chrono::steady_clock::now();
        int ngoodTest = predictTest();
        int ngood = 0, ngood1 = 0;
        for (int i=0; i<ntrain; ++i) {
            auto best = V[kNlabels*i]; int lbest = 0;
            for (int l=1; l<kNlabels; ++l) if (V[kNlabels*i+l] > best) { best = V[kNlabels*i+l]; lbest = l; }
            if (lbest == labels[i]) {
                ++ngood;
                if (i < kNtrain) ++ngood1;
            }
        }
        if (ngoodTest >= bestNgood) {
            bestNgood = ngoodTest;
            bestP = P;
            writeResults(ofile,npoint,patterns,bestP);
        }
        auto tNow = std::chrono::steady_clock::now();
        auto time = 1e-6*std::chrono::duration_cast<std::chrono::microseconds>(tNow-tStart).count();
        printf("  Iteration %d: F=%g(%g) sump2=%g, Ngood=%d,%d,%d  time=%g s\n",
                iter+1,totF,totF-lambda*totP2,totP2,ngood,ngood1,ngoodTest,time);
        fflush(stdout);
        auto tim2 = std::chrono::steady_clock::now();
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count();
    };

    auto checkIfConverged = [&totF, &totP2, &Fold, &Fvalues, &nconv, lambda] (int iter) {
        if (Fold/totF-1 < 1e-10) {
            ++nconv;
            if (nconv > 3) {
                printf("Converged at iteration %d with F = %g (%g) (change is %g)\n",
                        iter,totF,totF-lambda*totP2,Fold/totF-1); return true;
            }
        }
        else nconv = 0;
        Fvalues[iter] = totF;
        if (iter >= 50 && Fvalues[iter-50]/totF - 1 < 1e-4) {
            printf("Terminating due to too small change (%g) in the last 50 iterations\n",Fvalues[iter-50]/totF - 1);
            return true;
        }
        Fold = totF;
        return false;
    };

    for (int iter=0; iter<niter; ++iter) {

        totItime += initIteration();
        totFtime += computeFdF(norm,labels,V,dfdv,d2fdv2,F);
        totPtime += computeP(images,dfdv,dP,npoint,ntrain,chunk,workers);
        totStime += setSearchDirection();
        totVtime += computeVfast(images,dV,du,ui,npoint,ntrain,workers);
        totUtime += takeStep();
        if ((iter+1)%20 == 0) {
            totCtime += checkResults(iter);
        }
        if (allConverged) break;
        if (checkIfConverged(iter)) break;
    }

    auto tEnd = std::chrono::steady_clock::now();
    auto time = 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tEnd-tStart).count();
    printf("Total time: %g ms, I-time: %g ms, F-time: %g ms, S-time: %g ms, U-time: %g ms, V-time: %g ms, P-time: %g ms, C-time: %g ms\n",time,
            1e-3*totItime,1e-3*totFtime,1e-3*totStime,1e-3*totUtime,1e-3*totVtime,1e-3*totPtime, 1e-3*totCtime);
    fflush(stdout);

    std::vector<std::pair<float,int>> X(kNlabels);
    int ngood = 0, ngood1 = 0;
    for (int i=0; i<ntrain; ++i) {
        auto best = V[kNlabels*i]; int lbest = 0;
        for (int l=1; l<kNlabels; ++l) if (V[kNlabels*i+l] > best) { best = V[kNlabels*i+l]; lbest = l; }
        if (lbest == labels[i]) {
            ++ngood;
            if (i < kNtrain) ++ngood1;
        }
    }
    printf("Training: ngood = %d (%g)  %d (%g)\n",ngood,(1.*ngood)/ntrain,ngood1,(1.*ngood1)/kNtrain);
    fflush(stdout);

    int ncheck = 5;
    std::vector<int> Ngood(ncheck,0);
    std::vector<std::pair<float,int>> predicted(2*kNtest);
    for (int i=0; i<kNtest; ++i) {
        auto B = testImages.data() + i*npoint;
        auto p = P.data();
        for (int l=0; l<kNlabels; ++l) {
            Float s = 0; for(int j=0; j<npoint; ++j) s += p[j]*B[j];
            X[l] = {s,l};
            p += npoint;
        }
        std::sort(X.begin(),X.end());
        predicted[2*i+0] = X[kNlabels-1];
        predicted[2*i+1] = X[kNlabels-2];
        for (int k=kNlabels-1; k>=0; --k) {
            if (X[k].second == testLabels[i]) {
                int n = kNlabels - 1 - k;
                for (int j=n; j<ncheck; ++j) ++Ngood[j];
                break;
            }
        }
    }
    for (int n=0; n<ncheck; ++n) printf("%d  %d\n",n,Ngood[n]);

    writeResults(ofile,npoint,patterns,bestP,true);

    return 0;
}
