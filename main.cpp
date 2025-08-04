#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <utility>
#include <opencv2/ml.hpp>
#include "descriptors.h"
#include <io.h>
#include "data.h"
#include "evaluation.h"

constexpr int WIDTH = 64;
constexpr int HEIGHT = 128;




std::pair<std::vector<ml::Data>, std::vector<ml::Data> > get_data() {
    std::vector<ml::Data> posData, negData;
    const std::string p_folder = "../pedestrian_dataset";
    const std::string n_folder = "../para_imagenesNegativas";
    const std::string f_prefix = "FudanPed";
    const std::string p_prefix = "PennPed";


    const auto fudan_images = io::load_positive(p_folder, f_prefix);
    const auto penn_images = io::load_positive(p_folder, p_prefix);
    const auto neg_images = io::load_negative(n_folder, {64, 128}, 1000, 1000 / 6);

    for (const auto &fudan_image: fudan_images) {
        ml::Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(fudan_image);
        d.lbp_features = descriptors::get_lbp(fudan_image);
        d.dataset = "fudan";
        posData.push_back(d);
    }

    for (const auto &penn_image: penn_images) {
        ml::Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(penn_image);
        d.lbp_features = descriptors::get_lbp(penn_image);
        d.dataset = "penn";
        posData.push_back(d);
    }

    for (const auto &neg_image: neg_images) {
        ml::Data d;
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
    std::map<ml::FeatureType, ml::Metrics> results;

    for (int i = 0; i < runs; ++i) {
        auto [posData, negData] = get_data();
        auto [trainPos, testPos] = split_vector(posData, 80);
        auto [trainNeg, testNeg] = split_vector(negData, 80);

        std::vector<ml::Data> trainData, testData;
        trainData.insert(trainData.end(), trainPos.begin(), trainPos.end());
        trainData.insert(trainData.end(), trainNeg.begin(), trainNeg.end());
        testData.insert(testData.end(), testPos.begin(), testPos.end());
        testData.insert(testData.end(), testNeg.begin(), testNeg.end());

        for (ml::FeatureType type: {ml::FeatureType::HOG, ml::FeatureType::LBP, ml::FeatureType::BOTH}) {
            auto training_matrices = generate_training_matrices(trainData, type);

            cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
            svm->setKernel(cv::ml::SVM::LINEAR);
            svm->setType(cv::ml::SVM::C_SVC);
            svm->train(training_matrices.features, cv::ml::ROW_SAMPLE, training_matrices.labels);

            auto [acc, prec, rec, f1] = eval::test_svm(testData, training_matrices, svm, type);
            results[type].accumulate(acc, prec, rec, f1);
        }
    }

    for (const auto &[type, metrics]: results) {
        std::string label = (type == ml::FeatureType::HOG ? "HOG" : type == ml::FeatureType::LBP ? "LBP" : "BOTH");
        metrics.print_avg(label);
    }


    return 0;
}
