//
// Created by fausto on 8/4/25.
//

#ifndef EVALUATION_H
#define EVALUATION_H
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "data.h"

namespace eval {
    std::tuple<double, double, double, double, std::vector<ml::Data>> test_svm(
        const std::vector<ml::Data> &data,
        const ml::TrainingMatrices &matrices,
        const cv::Ptr<cv::ml::SVM> &svm,
        ml::FeatureType type
    ) ;


    constexpr int POS_LABEL = 1;
    constexpr int NEG_LABEL = -1;
}


#endif //EVALUATION_H
