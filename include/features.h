//
// Created by fausto on 8/4/25.
//

#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <opencv2/core.hpp>

namespace features {
    std::vector<float> get_hog(const cv::Mat &img);
    std::vector<float> get_lbp(const cv::Mat &img);
}


#endif //FEATURES_H
