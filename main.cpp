#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/objdetect.hpp>
#include <random>

class DataHandlerBase {
public:
    static std::vector<cv::Rect> get_bounding_boxes(const std::string &annotation_file) {
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

    static std::vector<cv::Mat> extract_boxes(const cv::Mat &img, const std::vector<cv::Rect> &boxes) {
        std::vector<cv::Mat> result;

        for (const auto &box: boxes) {
            if ((box & cv::Rect(0, 0, img.cols, img.rows)) != box) continue;

            cv::Mat pedestrian = img(box);
            cv::resize(pedestrian, pedestrian, cv::Size(64, 128));
            result.push_back(pedestrian);
        }

        return result;
    }

    virtual  std::vector<cv::Mat> get_images(const std::string &folder = "../pedestrian_dataset",
                                                   const std::string &filename_prefix = "FudanPed",
                                                   int count = 60,
                                                   int boxes_per_image = 5) {
    }
};

class NegativeDataHandler : public DataHandlerBase {
    static void generate_random_annotations_if_missing(const std::string &annotation_file, int img_width,
                                                       int img_height, int count = 5) {
        constexpr int box_width = 64;
        constexpr int box_height = 128;

        if (std::filesystem::exists(annotation_file)) return;

        std::ofstream file(annotation_file);
        if (!file.is_open()) {
            std::cerr << "Could not create annotation file: " << annotation_file << '\n';
            return;
        }
        // // Create annotation file if missing
        // generate_random_annotations_if_missing(txt_path.string(), img.cols, img.rows, boxes_per_image);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist_x(0, img_width - box_width);
        std::uniform_int_distribution<> dist_y(0, img_height - box_height);

        for (int i = 0; i < count; ++i) {
            int x1 = dist_x(gen);
            int y1 = dist_y(gen);
            int x2 = x1 + box_width;
            int y2 = y1 + box_height;

            file << "(" << x1 << ", " << y1 << ") - (" << x2 << ", " << y2 << ")\n";
        }
    }
};

class PositiveDataHandler : public DataHandlerBase {
public:
    PositiveDataHandler() = default;


     std::vector<cv::Mat> get_images(const std::string &folder = "../pedestrian_dataset",
                                           const std::string &filename_prefix = "FudanPed",
                                           int count = 60,
                                           int boxes_per_image = 5) override {
        std::vector<cv::Mat> images;
        std::filesystem::path base_path(folder);

        for (int i = 1; i <= count; ++i) {
            std::ostringstream img_name_ss;
            img_name_ss << filename_prefix << std::setw(5) << std::setfill('0') << i << ".png";
            std::filesystem::path img_path = base_path / img_name_ss.str();

            cv::Mat img = cv::imread(img_path.string());
            if (img.empty()) continue;

            std::ostringstream txt_name_ss;
            txt_name_ss << filename_prefix << std::setw(5) << std::setfill('0') << i << ".txt";
            std::filesystem::path txt_path = base_path / txt_name_ss.str();


            auto boxes = get_bounding_boxes(txt_path.string());
            auto pedestrians = extract_boxes(img, boxes);
            images.insert(images.end(), pedestrians.begin(), pedestrians.end());
        }

        return images;
    }
};

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
    const auto images = PositiveDataHandler::get_images();
    for (auto &img: images) {
        cv::HOGDescriptor hog;
        hog.winSize = cv::Size(64, 128);
        std::vector<float> descriptors;
        hog.compute(img, descriptors);

        cv::Mat descriptor;
        cv::Mat descriptor_lbp;
        cv::Mat image_gray;
        cv::cvtColor(img, image_gray, cv::COLOR_BGR2GRAY);
        lbp(image_gray, descriptor_lbp);
        cv::normalize(descriptor_lbp, descriptor_lbp, 0, 255, cv::NORM_MINMAX);
        cv::Mat result;
        //cv::hconcat(descriptor_lbp, descriptor, result);
        //cv::imshow("result", result);
    }
    return 0;
}
