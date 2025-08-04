//
// Created by fausto on 8/4/25.
//

#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <vector>
#include <opencv2/core.hpp>

namespace descriptors {
    std::vector<float> get_hog(const cv::Mat &img);
    std::vector<float> get_lbp(const cv::Mat &img);
}


#endif //DESCRIPTORS_H
