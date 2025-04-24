#pragma once
#include <opencv2/opencv.hpp>

static constexpr float white_value(int depth)
{
    switch (depth)
    {
        case CV_8U:
            return 255;
        case CV_8S:
            return 127;
        case CV_16U:
            return 65535;
        case CV_16S:
            return 32767;
        case CV_32S:
            return 2147483647;
        case CV_32F:
        case CV_64F:
            return 1;
    }

    return -1;
}

// Select biggest component if mask consists of multiple components
static int extract_largest_component(cv::Mat& mask)
{
    int height = mask.rows;
    int width = mask.cols;

    std::vector<int> sizes;

    for (int v = 0; v != height; ++v)
    {
        for (int u = 0; u != width; ++u)
        {
            const uchar& val = mask.at<uchar>(v, u);

            if (val == 255)
            {
                cv::Mat tmp;
                cv::floodFill(mask, tmp, cv::Point(u, v), sizes.size() + 1);
                sizes.emplace_back(1);
            }
            else if (val > 0 && val <= sizes.size())
            {
                sizes[val - 1]++;
            }
        }
    }

    // Find biggest component
    int max_component = std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end())) + 1;

    for (int v = 0; v != height; ++v)
    {
        for (int u = 0; u != width; ++u)
        {
            uchar& val = mask.at<uchar>(v, u);
            val = (val == max_component) ? 255 : 0;
        }
    }

    return sizes.size();
}

static std::pair<cv::Mat, cv::Mat> load_images(const std::string& path_to_normals, const std::string& path_to_mask = "")
{
    cv::Mat raw_normals = cv::imread(path_to_normals, cv::IMREAD_UNCHANGED);
    cv::Mat normals;
    raw_normals.convertTo(normals, CV_32F, 1. / white_value(raw_normals.depth()));
    cvtColor(normals, normals, cv::COLOR_BGRA2BGR);

    int height = normals.rows;
    int width = normals.cols;

    cv::Mat mask;

    if (path_to_mask != "")
    {
        std::cout << "Loading Mask from File\n";
        cv::Mat raw_mask = cv::imread(path_to_mask, cv::IMREAD_GRAYSCALE);
        raw_mask.convertTo(mask, CV_8UC1, white_value(CV_8UC1) / white_value(raw_mask.depth()));
    }
    else if (raw_normals.channels() == 4) // Has alpha channel
    {
        std::cout << "Extracting Mask from Alpha Channel\n";
        std::vector<cv::Mat> channels;
        cv::split(raw_normals, channels);
        channels[3].convertTo(mask, CV_8UC1, white_value(CV_8UC1) / white_value(channels[3].depth()));
    }
    else
    {
        std::cout << "Using Default (All Foreground) Mask\n";
        mask = cv::Mat(height, width, CV_8UC1);
        mask.setTo(cv::Scalar(255));
    }

    // Thresholding
    for (int v = 0; v != height; ++v)
    {
        for (int u = 0; u != width; ++u)
        {
            if (mask.at<uchar>(v, u) > 127)
            {
                mask.at<uchar>(v, u) = 255;
            }
            else
            {
                mask.at<uchar>(v, u) = 0;
            }
        }
    }

    extract_largest_component(mask);

    return std::make_pair(normals, mask);
}