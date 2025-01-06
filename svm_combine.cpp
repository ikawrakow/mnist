#include "svmPattern.h"
#include "getImages.h"

#include <vector>
#include <cstdint>
#include <fstream>
#include <cstdio>

using Float = float;

static bool loadModel(const char* fname, std::vector<Pattern>& patterns, std::vector<std::vector<Float>>& allP) {
    std::ifstream in(fname,std::ios::binary);
    if (!in) {
        printf("%s: failed top open %s\n",__func__,fname); return false;
    }
    int npattern;
    in.read((char *)&npattern,sizeof(int));
    patterns.resize(npattern);
    for (auto & p : patterns) {
        int np = p.points_.size();
        in.read((char *)&np,sizeof(int));
        p.points_.resize(np);
        in.read((char *)p.points_.data(),np*sizeof(int));
        in.read((char *)&p.margin_,sizeof(p.margin_));
        in.read((char *)&p.down_,sizeof(p.down_));
    }
    if ((int)allP.size() != kNlabels) allP.resize(kNlabels);
    int npoint;
    in.read((char *)&npoint,sizeof(int));
    for (auto & p : allP) {
        p.resize(npoint);
        in.read((char *)p.data(),p.size()*sizeof(Float));
    }
    return in.good() ? true : false;
}

template <typename Float>
void writeResults(const char *ofile, int npoint, const std::vector<Pattern> &patterns, const std::vector<std::vector<Float>>& allP, bool tell = false) {
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
    for (auto& P : allP) out.write((char *)P.data(),P.size()*sizeof(Float));
    if (tell) {
        printf("Wrote training results to %s\n",ofile);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s model1 model2 ... modelN\n", argv[0]); return 1;
    }

    std::vector<Pattern> patterns;
    std::vector<std::vector<Float>> allP;
    int nmodel = 0;
    for (int iarg = 1; iarg < argc; ++iarg) {
        if (iarg == 1) {
            if (!loadModel(argv[iarg], patterns, allP)) return 1;
            ++nmodel;
        } else {
            std::vector<Pattern> patterns_j;
            std::vector<std::vector<Float>> allP_j;
            if (!loadModel(argv[iarg], patterns_j, allP_j)) return 1;
            if (allP_j.size() != allP.size()) {
                printf("Oops: allP_j.size = %zu vs allP.size = %zu for %s\n", allP_j.size(), allP_j.size(), argv[iarg]);
                return 1;
            }
            for (int i = 0; i < int(allP.size()); ++i) {
                if (allP_j[i].size() != allP[i].size()) {
                    printf("Oops: allP_j[%d].size = %zu vs allP[%d].size = %zu for %s\n", i, allP_j.size(), i, allP_j.size(), argv[iarg]);
                    return 1;
                }
                for (int j = 0; j < int(allP[i].size()); ++j) allP[i][j] += allP_j[i][j];
            }
            ++nmodel;
        }
    }
    printf("Loaded %d models, %d patterns, %d points\n", nmodel, int(patterns.size()), int(allP.front().size()));
    Float norm = 1.f/nmodel;
    for (auto& P : allP) for (auto& p : P) p *= norm;
    writeResults("combined.dat", allP.front().size(), patterns, allP, true);

    return 0;

}
