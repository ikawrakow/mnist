#include "getImages.h"
#include "imageUtils.h"
#include "svmPattern.h"
#include "bfgs.h"

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

constexpr int kThresh = 128;

void computeFdF(double norm, const std::vector<uint8_t> &labels, const std::vector<double> &V,
        std::vector<double> &dfdv, std::vector<double> &d2fdv2, std::vector<double> &F) {
    int nimage = labels.size();
    for(int j=0; j<10*nimage; ++j) dfdv[j] = d2fdv2[j] = 0;
    for(auto & f : F) f = 0;
    double n = 2*norm;
    for(int l=0; l<10; ++l) {
        double f = 0;
        for(int i=0; i<nimage; ++i) {
            auto y = labels[i] == l ? 1 : -1;
            double d = 1 - V[10*i+l]*y;
            if( d > 0 ) {
                f += d*d;
                dfdv[10*i+l] -= n*d*y;
                d2fdv2[10*i+l] += n;
            }
        }
        F[l] = norm*f;
    }
}

double computeP(const std::vector<uint8_t> &dImage, const std::vector<double> &dfdv, std::vector<double> &dp,
        int npoint, int nimage, int chunk, std::vector<std::thread> &workers) {
    auto t1 = std::chrono::steady_clock::now();
    std::atomic<int> counter(0);
    auto compute = [&dImage,&dfdv,&dp,&counter,npoint,nimage,chunk]() {
        std::vector<double> tmp(10*chunk);
        while(1) {
            int first = counter.fetch_add(chunk);
            if( first >= npoint ) break;
            const uint8_t *A = dImage.data() + first;
            int n = first + chunk < npoint ? chunk : npoint - first;
            for(int j=0; j<10*n; ++j) tmp[j] = 0;
            for(int i=0; i<nimage; ++i) {
                auto t = tmp.data();
                for(int l=0; l<10; ++l) {
                    double d = dfdv[10*i+l];
                    if( d ) for(int j=0; j<n; ++j) t[j] += d*A[j];
                    t += n;
                }
                A += npoint;
            }
            double *p = &dp[first];
            for(int l=0; l<10; ++l) for(int j=0; j<n; ++j) p[j+l*npoint] += tmp[n*l+j];
        }
    };
    for(auto &w : workers) w = std::thread(compute);
    for(auto &w : workers) w.join();
    auto t2 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
}

double computeVfast(const std::vector<uint8_t> &dImage, std::vector<double> &dv, std::vector<double> &du,
        std::vector<int16_t> &ui, int npoint, int nimage,std::vector<std::thread> &workers) {
    static std::vector<double> scale(10), scaleI(10);
    auto t1 = std::chrono::steady_clock::now();
    auto u = du.data(); auto iu = ui.data();
    for(int l=0; l<10; ++l) {
        double um = 0; for(int j=0; j<npoint; ++j) if( fabs(u[j]) > um ) um = fabs(u[j]);
        scale[l] = 32766./um; scaleI[l] = um/32766.;
        for(int j=0; j<npoint; ++j) {
            iu[j] = toNearestInt(u[j]*scale[l]);
            u[j] = scaleI[l]*iu[j];
        }
        u += npoint; iu += npoint;
    }
    int nthread = 1 + workers.size();
    int chunk1 = nimage/(8*nthread);
    int chunk = 16; while( chunk < chunk1 ) chunk *= 2;
    std::atomic<int> counter(0);
    auto compute = [&dImage,&dv,&ui,&counter,npoint,nimage,chunk]() {
        //int chunk = 2048; //256;
        int n128 = npoint/128;
        int nn = npoint - 128*n128;
        while(1) {
            int first = counter.fetch_add(chunk);
            if( first >= nimage ) break;
            int last = first + chunk; if( last > nimage ) last = nimage;
            const uint8_t *A = &dImage[(int64_t)first * (int64_t)npoint];
            for(int i=first; i<last; ++i) {
                auto u = ui.data();
                for(int l=0; l<10; ++l) {
                    int64_t s = 0; auto a = A;
                    for(int i128=0; i128<n128; ++i128) {
                        int si = 0; for(int k=0; k<128; ++k) si += (int)u[k] * a[k];
                        s += si; a += 128; u += 128;
                    }
                    for(int k=0; k<nn; ++k) s += (int)u[k] * a[k];
                    a += nn; u += nn;
                    dv[10*i+l] = scaleI[l]*s;
                }
                A += npoint;
            }
        }
    };
    for(auto &w : workers) w = std::thread(compute);
    for(auto &w : workers) w.join();
    auto t2 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
}

void writeResults(const char *ofile, int npoint, const std::vector<Pattern> &patterns, const std::vector<double> &P) {
    std::ofstream out(ofile,std::ios::binary);
    int npattern = patterns.size();
    out.write((char *)&npattern,sizeof(int));
    for(const auto & p : patterns) {
        int np = p.points_.size();
        out.write((char *)&np,sizeof(int));
        out.write((char *)p.points_.data(),np*sizeof(int));
        out.write((char *)&p.margin_,sizeof(p.margin_));
        out.write((char *)&p.down_,sizeof(p.down_));
    }
    out.write((char *)&npoint,sizeof(int));
    out.write((char *)P.data(),P.size()*sizeof(double));
    printf("Wrote training results to %s\n",ofile);
    fflush(stdout);
}

int main(int argc, char **argv) {

    int iarg = 1;
    int    niter  = argc > iarg ? atoi(argv[iarg++]) : 200;
    int    nadd   = argc > iarg ? atoi(argv[iarg++]) : 0;
    double lambda = argc > iarg ? atof(argv[iarg++]) : 10.;
    int    nhist  = argc > iarg ? atoi(argv[iarg++]) : 10;
    int    rseq   = argc > iarg ? atoi(argv[iarg++]) : 0;
    int    type   = argc > iarg ? atoi(argv[iarg++]) : 0;
    auto   ofile  = argc > iarg ? argv[iarg++] : "test.dat";

    auto labels = getTraningLabels();
    if( labels.size() != kNtrain ) return 1;

    auto images = getTrainingImages();
    if( images.size() != kNtrain*kNx*kNy ) return 1;

    auto testLabels = getTestLabels();
    if( testLabels.size() != kNtest ) return 1;

    auto testImages = getTestImages();
    if( testImages.size() != kNtest*kSize ) return 1;

    if (nadd > 0) {
        if (nadd < 100) addElasticDeformations(images,labels,nadd);
        else {
            nadd -= 100;
            auto images1 = images; auto labels1 = labels;
            addElasticDeformations(images,labels,nadd);
            addAffineTransformations(kNx,kNy,images1,labels1,nadd+1,12.,0.6,0.1,0.05,0.05,0,false);
            auto curSize = images.size();
            images.resize(images.size() + images1.size() - kNtrain*kSize);
            std::copy(images1.begin() + kNtrain*kSize, images1.end(), images.begin() + curSize);
            curSize = labels.size();
            labels.resize(labels.size() + labels1.size() - kNtrain);
            std::copy(labels1.begin() + kNtrain, labels1.end(), labels.begin() + curSize);
        }
    }
    else if (nadd < 0) {
        addAffineTransformations(kNx,kNy,images,labels,-nadd,12.,0.6,0.1,0.05,0.05,0,false);
    }
    int ntrain = labels.size();

    int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(nthread);

    auto patterns = Pattern::getPatterns(type);
    int Nx = kNx, Ny = kNy;
    for(const auto& p : patterns) {
        int Nx1, Ny1;
        images = Pattern::apply(Nx,Ny,images,p,Nx1,Ny1);
        testImages = Pattern::apply(Nx,Ny,testImages,p,Nx1,Ny1);
        Nx = Nx1; Ny = Ny1;
    }
    int npoint = images.size()/labels.size();
    printf("%d points per image\n",npoint);
    fflush(stdout);

    std::vector<double> V(10*ntrain,0), dV(10*ntrain,0), dfdv(10*ntrain), d2fdv2(10*ntrain),
                        P(10*npoint,0), dP(10*npoint,0), du(10*npoint), sump2(10), F(10);
    std::vector<int16_t> ui(10*npoint);

    double norm = 10./ntrain;

    int chunk1 = npoint/(4*nthread);
    int chunk = 8; while( chunk < chunk1 ) chunk *= 2;
    printf("Using chunk=%d\n",chunk);
    fflush(stdout);

    std::vector<double> Fvalues(niter);
    std::vector<BFGSHistory> bfgs;
    bfgs.reserve(10);
    //for (int l=0; l<10; ++l) bfgs.emplace_back(BFGSHistory(npoint,nhist));
    for (int l=0; l<10; ++l) bfgs.emplace_back(npoint,nhist);

    auto computeF = [ntrain,&sump2,lambda,norm,&V,&dV,&labels](int l, double s1, double s2, double t) -> double {
        double Fn = lambda*(sump2[l] + 2*t*s1 + t*t*s2);
        double F1 = 0;
        for(int i=0; i<ntrain; ++i) {
            auto y = labels[i] == l ? 1 : -1;
            auto v = V[10*i+l] + t*dV[10*i+l];
            double d = 1 - v*y;
            if( d > 0 ) F1 += d*d;
        }
        return Fn + norm*F1;
    };

    double Fold = 1e100; double totTime = 0, totPtime = 0, totVtime = 0;

    std::vector<double> bestP(10*npoint,0);
    int bestNgood = 0;
    int nconv = 0;

    auto tStart = std::chrono::steady_clock::now();

    for (int iter=0; iter<niter; ++iter) {

        auto p = P.data(); auto dp = dP.data();
        for (int l=0; l<10; ++l) {
            double s = 0;
            for (int j=0; j<npoint; ++j) { dp[j] = 2*lambda*p[j]; s += p[j]*p[j]; }
            sump2[l] = s;
            p += npoint; dp += npoint;
        }
        computeFdF(norm,labels,V,dfdv,d2fdv2,F);
        totPtime += computeP(images,dfdv,dP,npoint,ntrain,chunk,workers);
        for (int l=0; l<10; ++l) bfgs[l].setSearchDirection(F[l],P.data()+npoint*l,dP.data()+npoint*l,du.data()+npoint*l);
        totVtime += computeVfast(images,dV,du,ui,npoint,ntrain,workers);
        double totF = 0, totP2 = 0;
        bool allConverged = true;
        p = P.data(); auto dU = du.data();
        for (int l=0; l<10; ++l) {
            double F1 = F[l] + lambda*sump2[l];
            double sum1 = 0, sum2 = 0;
            for (int i=0; i<ntrain; ++i) { sum1 -= dV[10*i+l]*dfdv[10*i+l]; sum2 += dV[10*i+l]*dV[10*i+l]*d2fdv2[10*i+l]; }
            sum1 *= 0.5; sum2 *= 0.5;
            double s1 = 0, s2 = 0;
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
            double t = sum1/sum2;
            double Fn = computeF(l,s1,s2,t);
            double Fe = F1 - sum1*t;
            if (Fn > Fe + 1e-6) {
                double tn = sum1*t*t/(Fn - F1 + 2*t*sum1);
                auto Fn1 = computeF(l,s1,s2,tn);
                if (Fn1 < Fn) { Fn = Fn1; t = tn; }
                if (Fn > F1) printf("Oops: l=%d F[l]=%g Fn=%g Fe=%g t=%g, %g\n",l,F[l],Fn,Fe,t,tn);
            }
            for (int j=0; j<npoint; ++j) { p[j] += t*dU[j]; totP2 += p[j]*p[j]; }
            for (int i=0; i<ntrain; ++i) V[10*i+l] += t*dV[10*i+l];
            totF += Fn;
            p += npoint; dU += npoint;
        }
        bool out = (iter+1)%20 == 0 ? true : false;
        if (out) {
            int ngoodTest = 0;
            auto B = testImages.data();
            for (int i=0; i<kNtest; ++i) {
                double best = -1e300; int lbest = -1;
                auto p = P.data();
                for (int l=0; l<10; ++l) {
                    double s = 0; for (int j=0; j<npoint; ++j) s += p[j]*B[j];
                    if (s > best) { best = s; lbest = l; }
                    p += npoint;
                }
                if (lbest == testLabels[i]) ++ngoodTest;
                B += npoint;
            }
            int ngood = 0, ngood1 = 0;
            for (int i=0; i<ntrain; ++i) {
                auto best = V[10*i]; int lbest = 0;
                for (int l=1; l<10; ++l) if (V[10*i+l] > best) { best = V[10*i+l]; lbest = l; }
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
        }
        if (allConverged ) break;
        if (Fold/totF-1 < 1e-10) {
            ++nconv;
            if (nconv > 3) {
                printf("Converged at iteration %d with F = %g (%g) (change is %g)\n",
                        iter,totF,totF-lambda*totP2,Fold/totF-1); break;
            }
        }
        else nconv = 0;
        Fvalues[iter] = totF;
        if (iter >= 50 && Fvalues[iter-50]/totF - 1 < 1e-4) {
            printf("Terminating due to too small change (%g) in the last 50 iterations\n",Fvalues[iter-50]/totF - 1);
            break;
        }
        Fold = totF;
    }

    auto tEnd = std::chrono::steady_clock::now();
    auto time = 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tEnd-tStart).count();
    printf("Total time: %g ms, V-time: %g ms, P-time: %g ms\n",time,1e-3*totVtime,1e-3*totPtime);
    fflush(stdout);

    std::vector<std::pair<float,int>> X(10);
    int ngood = 0, ngood1 = 0;
    for (int i=0; i<ntrain; ++i) {
        auto best = V[10*i]; int lbest = 0;
        for (int l=1; l<10; ++l) if (V[10*i+l] > best) { best = V[10*i+l]; lbest = l; }
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
        for (int l=0; l<10; ++l) {
            double s = 0; for(int j=0; j<npoint; ++j) s += p[j]*B[j];
            X[l] = {s,l};
            p += npoint;
        }
        std::sort(X.begin(),X.end());
        predicted[2*i+0] = X[9];
        predicted[2*i+1] = X[8];
        for (int k=9; k>=0; --k) {
            if (X[k].second == testLabels[i]) {
                int n = 9 - k;
                for (int j=n; j<ncheck; ++j) ++Ngood[j];
                break;
            }
        }
    }
    for (int n=0; n<ncheck; ++n) printf("%d  %d\n",n,Ngood[n]);

    writeResults(ofile,npoint,patterns,bestP);

    return 0;
}
