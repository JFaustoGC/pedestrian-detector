#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <utility>
#include <opencv2/ml.hpp>
#include "descriptors.h"
#include <io.h>
constexpr int WIDTH = 64;
constexpr int HEIGHT = 128;

constexpr int POS_LABEL = 1;
constexpr int NEG_LABEL = -1;


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


SplitData split_vector(const std::vector<Data> &data, int percent) {
    if (percent <= 0 || percent >= 100) {
        throw std::invalid_argument("Percent must be between 1 and 99.");
    }

    // std::vector<Data> shuffled = data;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::shuffle(shuffled.begin(), shuffled.end(), gen);

    // Just use the input vector directly
    const auto split_idx = data.size() * percent / 100;

    std::vector<Data> first(data.begin(), data.begin() + split_idx);
    std::vector<Data> second(data.begin() + split_idx, data.end());

    return {first, second};
}


enum class FeatureType {
    HOG,
    LBP,
    BOTH
};

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


std::tuple<double, double, double, double> test_svm(
    const std::vector<Data> &data,
    const TrainingMatrices &training_matrices,
    const cv::Ptr<cv::ml::SVM> &svm,
    FeatureType type
) {
    int TP = 0, TN = 0, FP = 0, FN = 0;

    for (const auto &sample: data) {
        auto features = extract_features(sample, type);
        cv::Mat input(1, training_matrices.descriptor_size, CV_32F);
        for (size_t j = 0; j < features.size(); ++j) {
            input.at<float>(0, static_cast<int>(j)) = features[j];
        }

        int prediction = static_cast<int>(svm->predict(input));
        int label = sample.label;
        if (prediction == POS_LABEL && label == POS_LABEL) TP++;
        else if (prediction == NEG_LABEL && label == NEG_LABEL) TN++;
        else if (prediction == POS_LABEL && label == NEG_LABEL) FP++;
        else if (prediction == NEG_LABEL && label == POS_LABEL) FN++;
    }

    int total = TP + TN + FP + FN;
    double accuracy = 100.0 * (TP + TN) / total;
    double precision = (TP + FP == 0) ? 0 : 100.0 * TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : 100.0 * TP / (TP + FN);
    double f1 = (precision + recall == 0) ? 0 : 2.0 * precision * recall / (precision + recall);

    return {accuracy, precision, recall, f1};
}

std::pair<std::vector<Data>, std::vector<Data> > get_data() {
    std::vector<Data> posData, negData;
    const std::string p_folder = "../pedestrian_dataset";
    const std::string n_folder = "../para_imagenesNegativas";
    const std::string f_prefix = "FudanPed";
    const std::string p_prefix = "PennPed";


    const auto fudan_images = io::load_positive(p_folder, f_prefix);
    const auto penn_images = io::load_positive(p_folder, p_prefix);
    const auto neg_images = io::load_negative(n_folder, {64, 128}, 1000, 1000 / 6);

    for (const auto &fudan_image: fudan_images) {
        Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(fudan_image);
        d.lbp_features = descriptors::get_lbp(fudan_image);
        d.dataset = "fudan";
        posData.push_back(d);
    }

    for (const auto &penn_image: penn_images) {
        Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(penn_image);
        d.lbp_features = descriptors::get_lbp(penn_image);
        d.dataset = "penn";
        posData.push_back(d);
    }

    for (const auto &neg_image: neg_images) {
        Data d;
        d.label = -1;
        d.hog_features = descriptors::get_hog(neg_image);
        d.lbp_features = descriptors::get_lbp(neg_image);
        d.dataset = "neg";
        negData.push_back(d);
    }
    return std::make_pair(posData, negData);
}

int main(int argc, char **argv) {
    constexpr int runs = 5;
    std::map<FeatureType, Metrics> results;

    for (int i = 0; i < runs; ++i) {
        auto [posData, negData] = get_data();
        auto [trainPos, testPos] = split_vector(posData, 80);
        auto [trainNeg, testNeg] = split_vector(negData, 80);

        std::vector<Data> trainData, testData;
        trainData.insert(trainData.end(), trainPos.begin(), trainPos.end());
        trainData.insert(trainData.end(), trainNeg.begin(), trainNeg.end());
        testData.insert(testData.end(), testPos.begin(), testPos.end());
        testData.insert(testData.end(), testNeg.begin(), testNeg.end());

        for (FeatureType type: {FeatureType::HOG, FeatureType::LBP, FeatureType::BOTH}) {
            auto training_matrices = generate_training_matrices(trainData, type);

            cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
            svm->setKernel(cv::ml::SVM::LINEAR);
            svm->setType(cv::ml::SVM::C_SVC);
            svm->train(training_matrices.features, cv::ml::ROW_SAMPLE, training_matrices.labels);

            auto [acc, prec, rec, f1] = test_svm(testData, training_matrices, svm, type);
            results[type].accumulate(acc, prec, rec, f1);
        }
    }

    for (const auto &[type, metrics]: results) {
        std::string label = (type == FeatureType::HOG ? "HOG" : type == FeatureType::LBP ? "LBP" : "BOTH");
        metrics.print_avg(label);
    }


    return 0;
}
