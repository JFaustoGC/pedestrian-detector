//
// Created by fausto on 8/4/25.
//

#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <vector>
#include <opencv2/core.hpp>

namespace descriptors {
    struct HOGParams {
        cv::Size winSize;
        cv::Size blockSize;
        cv::Size blockStride;
        cv::Size cellSize;
        int nbins;
    };

    std::vector<float> get_hog(const cv::Mat &img, const HOGParams &params);

    std::vector<float> get_lbp(const cv::Mat &img);
}


#endif //DESCRIPTORS_H
