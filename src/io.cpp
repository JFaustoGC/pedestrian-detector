//
// Created by fausto on 8/4/25.
//

#include "io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.h"

namespace io {
    std::vector<cv::Mat> load_positive(const std::string &folder, const std::string &prefix, const cv::Size &size) {
        const auto files = utils::match_image_annotation_pairs(folder, prefix);
        const auto cropped_images = utils::get_images(files);
        return utils::process_images(cropped_images, size);
    }

    std::vector<cv::Mat> load_negative(const std::string &folder, const cv::Size &size, int count, int per_image) {
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
            auto cropped = utils::extract_boxes(img, boxes);

            for (auto &patch: cropped) {
                if (patch.empty()) continue;
                cv::Mat resized;
                cv::resize(patch, resized, size);
                negatives.push_back(resized);
                if (static_cast<int>(negatives.size()) >= count) return negatives;
            }
        }

        return negatives;
    }

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

    void generate_random_annotations(const std::string &annotation_file, int img_width, int img_height, int count,
                                     int min_w, int max_w, int min_h, int max_h) {
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
}
