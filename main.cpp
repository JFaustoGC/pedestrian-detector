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

    for (const auto &entry : fs::directory_iterator(folder)) {
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

        for (auto &patch : cropped) {
            if (patch.empty()) continue;
            cv::Mat resized;
            cv::resize(patch, resized, target_size);
            negatives.push_back(resized);
            if (static_cast<int>(negatives.size()) >= count) return negatives;
        }
    }

    return negatives;
}



void lbp(const cv::Mat &image, cv::Mat &result) {
    assert(image.channels() == 1); // input image must be gray scale

    result.create(image.size(), CV_8U); // allocate if necessary

    for (int j = 1; j < image.rows - 1; j++) {
        // for all rows (except first and last)

        const uchar *previous = image.ptr<const uchar>(j - 1); // previous row
        const uchar *current = image.ptr<const uchar>(j); // current row
        const uchar *next = image.ptr<const uchar>(j + 1); // next row

        uchar *output = result.ptr<uchar>(j); // output row

        for (int i = 1; i < image.cols - 1; i++) {
            // compose local binary pattern
            *output = previous[i - 1] > current[i] ? 1 : 0;
            *output |= previous[i] > current[i] ? 2 : 0;
            *output |= previous[i + 1] > current[i] ? 4 : 0;

            *output |= current[i - 1] > current[i] ? 8 : 0;
            *output |= current[i + 1] > current[i] ? 16 : 0;

            *output |= next[i - 1] > current[i] ? 32 : 0;
            *output |= next[i] > current[i] ? 64 : 0;
            *output |= next[i + 1] > current[i] ? 128 : 0;

            output++; // next pixel
        }
    }

    // Set the unprocess pixels to 0
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols - 1).setTo(cv::Scalar(0));
}


int main() {
    const std::string p_folder = "../pedestrian_dataset";
    const std::string n_folder = "../para_imagenesNegativas";
    const std::string f_prefix = "FudanPed";
    const std::string p_prefix = "PennPed";

    const auto fudan_images = load_positive(p_folder, f_prefix);
    const auto penn_images = load_positive(p_folder, p_prefix);
    const auto neg_images = load_negative(n_folder, {64, 128}, 1000, 5);

    for (const auto &img: neg_images) {
        cv::imshow("neg", img);
        cv::waitKey(0);
    }

    return 0;
}
