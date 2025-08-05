//
// Created by fausto on 8/4/25.
//

#include "evaluation.h"

namespace eval {
    std::tuple<double, double, double, double, std::vector<ml::Data>> test_svm(
        const std::vector<ml::Data> &data,
        const ml::TrainingMatrices &training_matrices,
        const cv::Ptr<cv::ml::SVM> &svm,
        ml::FeatureType type
    ) {
        int TP = 0, TN = 0, FP = 0, FN = 0;
        std::vector<ml::Data> false_positives;

        for (const auto &sample : data) {
            auto features = extract_features(sample, type);
            cv::Mat input(1, training_matrices.descriptor_size, CV_32F);
            for (size_t j = 0; j < features.size(); ++j) {
                input.at<float>(0, static_cast<int>(j)) = features[j];
            }

            int prediction = static_cast<int>(svm->predict(input));
            int label = sample.label;

            if (prediction == POS_LABEL && label == POS_LABEL) {
                TP++;
            } else if (prediction == NEG_LABEL && label == NEG_LABEL) {
                TN++;
            } else if (prediction == POS_LABEL && label == NEG_LABEL) {
                FP++;
                false_positives.push_back(sample);  // ← Store the false positive
            } else if (prediction == NEG_LABEL && label == POS_LABEL) {
                FN++;
            }
        }

        int total = TP + TN + FP + FN;
        double accuracy = 100.0 * (TP + TN) / total;
        double precision = (TP + FP == 0) ? 0 : 100.0 * TP / (TP + FP);
        double recall = (TP + FN == 0) ? 0 : 100.0 * TP / (TP + FN);
        double f1 = (precision + recall == 0) ? 0 : 2.0 * precision * recall / (precision + recall);

        return {accuracy, precision, recall, f1, false_positives};
    }
}
