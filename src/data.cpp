//
// Created by fausto on 8/4/25.
//

#include "data.h"

#include <iostream>
#include <random>
#include <opencv2/core/mat.hpp>

namespace ml {



    std::vector<float> extract_features(const Data &sample, FeatureType type) {
        std::vector<float> features;
        if (type == FeatureType::HOG || type == FeatureType::BOTH) {
            features.insert(features.end(), sample.hog_features.begin(), sample.hog_features.end());
        }

        if (type == FeatureType::LBP || type == FeatureType::BOTH) {
            features.insert(features.end(), sample.lbp_features.begin(), sample.lbp_features.end());
        }

        return features;
    }

    TrainingMatrices generate_training_matrices(const std::vector<Data> &data, FeatureType type) {
        int descriptor_size = static_cast<int>(extract_features(data[0], type).size());
        cv::Mat training_features(static_cast<int>(data.size()), descriptor_size, CV_32F);
        cv::Mat training_labels(static_cast<int>(data.size()), 1, CV_32S);

        for (size_t i = 0; i < data.size(); ++i) {
            auto features = extract_features(data[i], type);
            for (size_t j = 0; j < features.size(); ++j) {
                training_features.at<float>(static_cast<int>(i), static_cast<int>(j)) = features[j];
            }
            training_labels.at<int>(static_cast<int>(i)) = data[i].label;
        }

        return {training_features, training_labels, descriptor_size};
    }

    SplitData split_vector(const std::vector<Data> &data, const int percent) {
        if (percent <= 0 || percent >= 100) {
            throw std::invalid_argument("Percent must be between 1 and 99.");
        }

        std::vector<Data> shuffled = data;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(shuffled.begin(), shuffled.end(), gen);

        const auto split_idx = shuffled.size() * percent / 100;

        const std::vector first(shuffled.begin(), shuffled.begin() + split_idx);
        const std::vector second(shuffled.begin() + split_idx, shuffled.end());

        return {first, second};
    }

    std::vector<std::vector<Data>> split_vector_parts(const std::vector<Data> &data, const int parts) {
        if (parts <= 0) {
            throw std::invalid_argument("Parts must be a positive integer.");
        }

        std::vector<std::vector<Data>> splits;
        const size_t total_size = data.size();
        const size_t base_size = total_size / parts;
        const size_t remainder = total_size % parts;

        size_t start = 0;
        for (int i = 0; i < parts; ++i) {
            size_t end = start + base_size + (i < remainder ? 1 : 0);
            splits.emplace_back(data.begin() + start, data.begin() + end);
            start = end;
        }

        return splits;
    }


}
