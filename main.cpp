#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/objdetect.hpp>
#include <random>
#include <utility>
#include <opencv2/ml.hpp>

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


std::vector<cv::Rect> get_bounding_boxes(const std::string &annotation_file) {
    std::vector<cv::Rect> boxes;
    std::ifstream file(annotation_file);
    std::string line;

    while (std::getline(file, line)) {
        int x1, y1, x2, y2;
        char dummy;
        std::stringstream ss(line);
        ss >> dummy >> x1 >> dummy >> y1 >> dummy >> dummy >> dummy >> x2 >> dummy >> y2 >> dummy;
        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }

    return boxes;
}

std::vector<cv::Mat> extract_boxes(const cv::Mat &img, const std::vector<cv::Rect> &boxes) {
    std::vector<cv::Mat> result;

    for (const auto &box: boxes) {
        if ((box & cv::Rect(0, 0, img.cols, img.rows)) != box) continue;

        cv::Mat pedestrian = img(box);
        result.push_back(pedestrian);
    }

    return result;
}

std::vector<cv::Mat> get_images(std::vector<std::pair<std::string, std::string> > const &files) {
    std::vector<cv::Mat> images;
    for (const auto &[image_filename, boxes_filename]: files) {
        cv::Mat og_img = cv::imread(image_filename);
        std::vector<cv::Rect> boxes = get_bounding_boxes(boxes_filename);
        std::vector<cv::Mat> img_objs = extract_boxes(og_img, boxes);
        for (const auto &img_obj: img_objs) {
            images.emplace_back(img_obj);
        }
    }
    return images;
}

std::vector<cv::Mat> process_images(const std::vector<cv::Mat> &input_images, const cv::Size &target_size = {64, 128}) {
    std::vector<cv::Mat> processed;

    for (const auto &img: input_images) {
        if (img.empty()) continue;
        cv::Mat resized;
        cv::resize(img, resized, target_size);

        processed.push_back(resized);

        cv::Mat flipped;
        cv::flip(resized, flipped, 1);
        processed.push_back(flipped);
    }

    return processed;
}


std::vector<std::pair<std::string, std::string> > match_image_annotation_pairs(
    const std::string &folder, const std::string &filename_prefix) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::string> > files;

    for (const auto &entry: fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        if (std::string filename = entry.path().filename().string();
            entry.path().extension() == ".png" && filename.rfind(filename_prefix, 0) == 0) {
            std::string base_name = entry.path().stem().string();
            std::string img_path = entry.path().string();

            if (std::string txt_path = (fs::path(folder) / (base_name + ".txt")).string(); fs::exists(txt_path)) {
                files.emplace_back(img_path, txt_path);
            } else {
                std::cerr << "Warning: Missing annotation file for image: " << img_path << '\n';
            }
        }
    }

    return files;
}

std::vector<cv::Mat> load_positive(
    const std::string &folder, const std::string &filename_prefix,
    const cv::Size &target_size = {64, 128}) {
    const auto files = match_image_annotation_pairs(folder, filename_prefix);
    const auto cropped_images = get_images(files);
    return process_images(cropped_images, target_size);
}


void generate_random_annotations(const std::string &annotation_file, int img_width,
                                 int img_height, int count = 5,
                                 int min_w = 48, int max_w = 96,
                                 int min_h = 96, int max_h = 160) {
    std::ofstream file(annotation_file);
    if (!file.is_open()) {
        std::cerr << "Could not create annotation file: " << annotation_file << '\n';
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution width_dist(min_w, max_w);
    std::uniform_int_distribution height_dist(min_h, max_h);

    int written = 0;
    while (written < count) {
        int box_w = width_dist(gen);
        int box_h = height_dist(gen);

        if (box_w > img_width || box_h > img_height) continue;

        std::uniform_int_distribution x_dist(0, img_width - box_w);
        std::uniform_int_distribution y_dist(0, img_height - box_h);

        int x1 = x_dist(gen);
        int y1 = y_dist(gen);
        int x2 = x1 + box_w;
        int y2 = y1 + box_h;

        file << "(" << x1 << ", " << y1 << ") - (" << x2 << ", " << y2 << ")\n";
        ++written;
    }
}

std::vector<cv::Mat> load_negative(const std::string &folder,
                                   const cv::Size &target_size = {64, 128},
                                   const int count = 1000,
                                   const int per_image = 5) {
    namespace fs = std::filesystem;
    std::vector<cv::Mat> negatives;

    for (const auto &entry: fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".png") continue;

        std::string img_path = entry.path().string();
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::string base = entry.path().stem().string();
        std::string annotation_file = (fs::path(folder) / (base + ".txt")).string();

        generate_random_annotations(annotation_file, img.cols, img.rows, per_image);

        auto boxes = get_bounding_boxes(annotation_file);
        auto cropped = extract_boxes(img, boxes);

        for (auto &patch: cropped) {
            if (patch.empty()) continue;
            cv::Mat resized;
            cv::resize(patch, resized, target_size);
            negatives.push_back(resized);
            if (static_cast<int>(negatives.size()) >= count) return negatives;
        }
    }

    return negatives;
}


std::vector<float> get_hog_features(const cv::Mat &image, const cv::Size winSize = {64, 128},
                                    const cv::Size blockSize = {16, 16}, const cv::Size blockStride = {8, 8},
                                    const cv::Size cellSize = {8, 8}, const int nbins = 9) {
    const cv::HOGDescriptor hog(
        winSize, blockSize, blockStride, cellSize, nbins
    );

    std::vector<float> features;
    hog.compute(image, features);
    return features;
}

std::vector<int> lbp(const cv::Mat &temp) {
    cv::Mat gray;
    if (temp.channels() != 1)
        cv::cvtColor(temp, gray, cv::COLOR_BGR2GRAY);
    else
        temp.copyTo(gray);

    std::vector<int> features;
    features.resize(gray.rows * gray.cols);


    for (int j = 0; j < gray.rows; j++) {
        if (j == 0 || j == gray.rows - 1) {
            continue;
        }

        const auto *previous = gray.ptr<const uchar>(j - 1);
        const auto *current = gray.ptr<const uchar>(j);
        const auto *next = gray.ptr<const uchar>(j + 1);

        auto *output = &features[j * gray.cols];

        for (int i = 0; i < gray.cols; i++) {
            if (i == 0 || i == gray.cols - 1) {
                output++;
                continue;
            }

            *output = previous[i - 1] > current[i] ? 1 : 0;
            *output |= previous[i] > current[i] ? 2 : 0;
            *output |= previous[i + 1] > current[i] ? 4 : 0;

            *output |= current[i - 1] > current[i] ? 8 : 0;
            *output |= current[i + 1] > current[i] ? 16 : 0;

            *output |= next[i - 1] > current[i] ? 32 : 0;
            *output |= next[i] > current[i] ? 64 : 0;
            *output |= next[i + 1] > current[i] ? 128 : 0;

            output++;
        }
    }

    return features;
}

std::vector<float> get_lbp_features(const cv::Mat &image) {
    std::vector<float> features;
    const auto intValue = lbp(image);
    features.reserve(intValue.size());
    for (const auto &i: intValue) {
        features.emplace_back(i * 1.0 / 256);
    }
    return features;
}


SplitData split_vector(const std::vector<Data> &data, int percent) {
    if (percent <= 0 || percent >= 100) {
        throw std::invalid_argument("Percent must be between 1 and 99.");
    }

    std::vector<Data> shuffled = data;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), gen);

    const auto split_idx = shuffled.size() * percent / 100;

    std::vector first(shuffled.begin(), shuffled.begin() + split_idx);
    std::vector second(shuffled.begin() + split_idx, shuffled.end());

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


    const auto fudan_images = load_positive(p_folder, f_prefix);
    const auto penn_images = load_positive(p_folder, p_prefix);
    const auto neg_images = load_negative(n_folder, {64, 128}, 1000, 1000 / 6);

    for (const auto &fudan_image: fudan_images) {
        Data d;
        d.label = +1;
        d.hog_features = get_hog_features(fudan_image);
        d.lbp_features = get_lbp_features(fudan_image);
        d.dataset = "fudan";
        posData.push_back(d);
    }

    for (const auto &penn_image: penn_images) {
        Data d;
        d.label = +1;
        d.hog_features = get_hog_features(penn_image);
        d.lbp_features = get_lbp_features(penn_image);
        d.dataset = "penn";
        posData.push_back(d);
    }

    for (const auto &neg_image: neg_images) {
        Data d;
        d.label = -1;
        d.hog_features = get_hog_features(neg_image);
        d.lbp_features = get_lbp_features(neg_image);
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
