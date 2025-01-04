#pragma once

#include <vector>
#include <cstdint>

constexpr uint32_t kNtrain = 60000;
constexpr uint32_t kNtest  = 10000;
constexpr int kNx = 28;
constexpr int kNy = 28;
constexpr int kSize = kNx*kNy;

std::vector<uint8_t> getTraningLabels();

std::vector<uint8_t> getTestLabels();

std::vector<uint8_t> getTrainingImages();

std::vector<uint8_t> getTestImages();

std::vector<uint8_t> getImages(int nimage, const char *fname);
