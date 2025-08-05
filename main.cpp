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


std::vector<ml::Data> load_positive(const descriptors::HOGParams &params) {
    const std::string folder = "../pedestrian_dataset";
    const std::string fudan_prefix = "FudanPed";
    const std::string penn_prefix = "PennPed";

    std::vector<ml::Data> positives;

    const auto fudan_images = io::load_positive(folder, fudan_prefix);
    const auto penn_images = io::load_positive(folder, penn_prefix);

    for (const auto &img: fudan_images) {
        ml::Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(img, params);
        d.lbp_features = descriptors::get_lbp(img);
        d.dataset = "fudan";
        positives.push_back(std::move(d));
    }

    for (const auto &img: penn_images) {
        ml::Data d;
        d.label = +1;
        d.hog_features = descriptors::get_hog(img, params);
        d.lbp_features = descriptors::get_lbp(img);
        d.dataset = "penn";
        positives.push_back(std::move(d));
    }

    return positives;
}

std::vector<ml::Data> load_negative(const descriptors::HOGParams &params, int count) {
    const std::string folder = "../para_imagenesNegativas";

    std::vector<ml::Data> negatives;

    const auto neg_images = io::load_negative(folder, {64, 128}, count);

    for (const auto &img: neg_images) {
        ml::Data d;
        d.label = -1;
        d.hog_features = descriptors::get_hog(img, params);
        d.lbp_features = descriptors::get_lbp(img);
        d.dataset = "neg";
        negatives.push_back(std::move(d));
    }

    return negatives;
}

std::pair<std::vector<ml::Data>, std::vector<ml::Data> >
get_data(const descriptors::HOGParams &params, const int proportion = 1) {
    auto posData = load_positive(params);
    const int neg_count = static_cast<int>(posData.size() * proportion);
    auto negData = load_negative(params, neg_count);
    return {std::move(posData), std::move(negData)};
}


std::vector<ml::Data> merge_data(const std::vector<ml::Data> &a, const std::vector<ml::Data> &b) {
    std::vector<ml::Data> result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

int main(int argc, char **argv) {
    constexpr int runs = 5;
    std::vector<descriptors::HOGParams> param_grid;

    for (int nbins: {6, 9, 12}) {
        for (int block = 16; block <= 24; block += 8) {
            for (int stride = 8; stride <= 16; stride += 8) {
                descriptors::HOGParams params{
                    {64, 128},
                    {block, block},
                    {stride, stride},
                    {8, 8},
                    nbins
                };
                param_grid.push_back(params);
            }
        }
    }

    auto bootstrap = true;

    for (const auto &hogParams: param_grid) {
        std::cout << "Testing HOG with nbins=" << hogParams.nbins
                << ", blockSize=" << hogParams.blockSize
                << ", blockStride=" << hogParams.blockStride << "\n";

        std::map<ml::FeatureType, ml::Metrics> results;

        for (int i = 0; i < runs; ++i) {
            const int proportion = bootstrap ? 1 : 3;
            auto [posData, negData] = get_data(hogParams, proportion);

            auto [trainPos, testPos] = split_vector(posData, 80);

            std::vector<ml::Data> trainNeg, testNeg, miningSet1, miningSet2;

            if (bootstrap) {
                // Step 1: Split negData into 33%, 33%, 33%
                auto [negBase, remaining1] = split_vector(negData, 33);
                auto [negMining1, negMining2] = split_vector(remaining1, 50); // halves of 66%

                trainNeg = negBase; // standard negatives
                miningSet1 = negMining1;
                miningSet2 = negMining2;
            } else {
                auto [negTrain, negTest] = split_vector(negData, 80);
                trainNeg = negTrain;
                testNeg = negTest;
            }

            auto trainData = merge_data(trainPos, trainNeg);
            auto [testNegAlt, testNegRest] = split_vector(negData, 80); // for test when bootstrap
            auto testData = merge_data(testPos, bootstrap ? testNegAlt : testNeg);

            for (ml::FeatureType type: {ml::FeatureType::HOG, ml::FeatureType::LBP, ml::FeatureType::BOTH}) {
                auto training_matrices = generate_training_matrices(trainData, type);

                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
                svm->setKernel(cv::ml::SVM::LINEAR);
                svm->setType(cv::ml::SVM::C_SVC);
                svm->train(training_matrices.features, cv::ml::ROW_SAMPLE, training_matrices.labels);

                if (bootstrap) {
                    // First mining pass
                    auto [acc1, prec1, rec1, f11, false_positives_1] = eval::test_svm(
                        miningSet1, training_matrices, svm, type);
                    trainData.insert(trainData.end(), false_positives_1.begin(), false_positives_1.end());

                    training_matrices = generate_training_matrices(trainData, type);
                    svm = cv::ml::SVM::create();
                    svm->setKernel(cv::ml::SVM::LINEAR);
                    svm->setType(cv::ml::SVM::C_SVC);
                    svm->train(training_matrices.features, cv::ml::ROW_SAMPLE, training_matrices.labels);

                    // Second mining pass
                    auto [acc2, prec2, rec2, f12, false_positives_2] = eval::test_svm(
                        miningSet2, training_matrices, svm, type);
                    trainData.insert(trainData.end(), false_positives_2.begin(), false_positives_2.end());

                    training_matrices = generate_training_matrices(trainData, type);
                    svm = cv::ml::SVM::create();
                    svm->setKernel(cv::ml::SVM::LINEAR);
                    svm->setType(cv::ml::SVM::C_SVC);
                    svm->train(training_matrices.features, cv::ml::ROW_SAMPLE, training_matrices.labels);
                }

                auto [acc, prec, rec, f1, false_positives] = eval::test_svm(testData, training_matrices, svm, type);
                results[type].accumulate(acc, prec, rec, f1);
            }
        }

        for (const auto &[type, metrics]: results) {
            std::string label = (type == ml::FeatureType::HOG ? "HOG" : type == ml::FeatureType::LBP ? "LBP" : "BOTH");
            std::cout << "Params: nbins=" << hogParams.nbins
                    << " block=" << hogParams.blockSize
                    << " stride=" << hogParams.blockStride << "\n";
            metrics.print_avg(label);
        }
    }


    return 0;
}
