//
// Created by fausto on 8/4/25.
//

#ifndef DATA_H
#define DATA_H
#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>

namespace ml {
    struct Data {
        std::vector<float> hog_features;
        std::vector<float> lbp_features;
        int label;
        std::string dataset;
    };

    struct SplitData {
        std::vector<Data> train;
        std::vector<Data> test;
    };

    struct TrainingMatrices {
        cv::Mat features;
        cv::Mat labels;
        int descriptor_size;
    };

    struct Metrics {
        double accuracy_sum = 0;
        double precision_sum = 0;
        double recall_sum = 0;
        double f1_sum = 0;
        int runs = 0;

        void accumulate(double acc, double prec, double rec, double f1) {
            accuracy_sum += acc;
            precision_sum += prec;
            recall_sum += rec;
            f1_sum += f1;
            runs++;
        }

        void print_avg(const std::string &label) const {
            std::cout << "=== Average Metrics for " << label << " over " << runs << " runs ===\n";
            std::cout << "Accuracy:  " << accuracy_sum / runs << "%\n";
            std::cout << "Precision: " << precision_sum / runs << "%\n";
            std::cout << "Recall:    " << recall_sum / runs << "%\n";
            std::cout << "F1 Score:  " << f1_sum / runs << "%\n\n";
        }
    };
    enum class FeatureType { HOG, LBP, BOTH };

    std::vector<float> extract_features(const Data &sample, FeatureType type);

    TrainingMatrices generate_training_matrices(const std::vector<Data> &, FeatureType);

    SplitData split_vector(const std::vector<Data> &data, int percent);

    std::vector<std::vector<Data>> split_vector_parts(const std::vector<Data> &data, const int parts);
}


#endif //DATA_H
