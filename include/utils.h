//
// Created by fausto on 8/4/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

namespace utils {
    std::vector<cv::Mat> process_images(const std::vector<cv::Mat>&, const cv::Size &);
    std::vector<std::pair<std::string, std::string>> match_image_annotation_pairs(const std::string&, const std::string&);
    std::vector<cv::Mat> get_images(const std::vector<std::pair<std::string, std::string>>& files);
    std::vector<cv::Mat> extract_boxes(const cv::Mat&, const std::vector<cv::Rect>& boxes);
}


#endif //UTILS_H
