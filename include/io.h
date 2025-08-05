//
// Created by fausto on 8/4/25.
//

#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

namespace io {
    std::vector<cv::Mat> load_positive(const std::string &folder, const std::string &prefix,
                                       const cv::Size &size = {64, 128});

    std::vector<cv::Mat> load_negative(const std::string &folder, const cv::Size &size, int count);

    std::vector<cv::Rect> get_bounding_boxes(const std::string &annotation_file);

    void generate_random_annotations(const std::string &annotation_file, int img_width,
                                     int img_height, int count = 5,
                                     int min_w = 48, int max_w = 96,
                                     int min_h = 96, int max_h = 160);
}

#endif //IO_H
