//
// Created by fausto on 8/4/25.
//

#include "utils.h"

#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "io.h"

namespace utils {
    std::vector<cv::Mat> process_images(const std::vector<cv::Mat> &images, const cv::Size &target_size) {
        std::vector<cv::Mat> processed;

        for (const auto &img: images) {
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

    std::vector<cv::Mat> get_images(const std::vector<std::pair<std::string, std::string> > &files) {
        std::vector<cv::Mat> images;
        for (const auto &[image_filename, boxes_filename]: files) {
            cv::Mat og_img = cv::imread(image_filename);
            std::vector<cv::Rect> boxes = io::get_bounding_boxes(boxes_filename);
            std::vector<cv::Mat> img_objs = extract_boxes(og_img, boxes);
            for (const auto &img_obj: img_objs) {
                images.emplace_back(img_obj);
            }
        }
        return images;
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
}
