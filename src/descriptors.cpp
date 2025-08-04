//
// Created by fausto on 8/4/25.
//

#include "descriptors.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

namespace descriptors {
    std::vector<float> get_hog(const cv::Mat &img, const HOGParams &params) {
        const cv::HOGDescriptor hog(
            params.winSize,
            params.blockSize,
            params.blockStride,
            params.cellSize,
            params.nbins
        );

        std::vector<float> features;
        hog.compute(img, features);
        return features;
    }


    std::vector<float> get_lbp(const cv::Mat &img) {
        cv::Mat gray;
        if (img.channels() != 1)
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else
            img.copyTo(gray);

        std::vector<int> features;
        features.resize(gray.rows * gray.cols);


        for (int j = 0; j < gray.rows; j++) {
            if (j == 0 || j == gray.rows - 1) {
                continue;
            }

            const auto *previous = gray.ptr<const uchar>(j - 1);
            const auto *current = gray.ptr<const uchar>(j);
            const auto *next = gray.ptr<const uchar>(j + 1);

            auto *output = &features[j * gray.cols];

            for (int i = 0; i < gray.cols; i++) {
                if (i == 0 || i == gray.cols - 1) {
                    output++;
                    continue;
                }

                *output = previous[i - 1] > current[i] ? 1 : 0;
                *output |= previous[i] > current[i] ? 2 : 0;
                *output |= previous[i + 1] > current[i] ? 4 : 0;

                *output |= current[i - 1] > current[i] ? 8 : 0;
                *output |= current[i + 1] > current[i] ? 16 : 0;

                *output |= next[i - 1] > current[i] ? 32 : 0;
                *output |= next[i] > current[i] ? 64 : 0;
                *output |= next[i + 1] > current[i] ? 128 : 0;

                output++;
            }
        }

        std::vector<float> featuresf;
        featuresf.reserve(features.size());
        for (const auto &i: features) {
            featuresf.emplace_back(i * 1.0 / 256);
        }
        return featuresf;
    }
}
