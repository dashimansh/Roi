#include "Detector.h"
#include "Config.h"
#include <iostream>

Detector::Detector()
{
    ColorLower1 = cv::Scalar(
        RED_LOWER1_H, RED_LOWER1_S,
        RED_LOWER1_V);
    ColorUpper1 = cv::Scalar(
        RED_UPPER1_H, RED_UPPER1_S,
        RED_UPPER1_V);
    ColorLower2 = cv::Scalar(
        RED_LOWER2_H, RED_LOWER2_S,
        RED_LOWER2_V);
    ColorUpper2 = cv::Scalar(
        RED_UPPER2_H, RED_UPPER2_S,
        RED_UPPER2_V);
    bHasTwoRanges = true;
}

void Detector::SetColorRange(
    const cv::Scalar& Lower1,
    const cv::Scalar& Upper1,
    const cv::Scalar& Lower2,
    const cv::Scalar& Upper2)
{
    ColorLower1 = Lower1;
    ColorUpper1 = Upper1;
    if (Lower2[0] >= 0)
    {
        ColorLower2 = Lower2;
        ColorUpper2 = Upper2;
        bHasTwoRanges = true;
    }
    else bHasTwoRanges = false;
}

void Detector::BuildMask(
    const cv::Mat& Frame)
{
    cv::cvtColor(Frame, HSVFrame,
        cv::COLOR_BGR2HSV);
    cv::inRange(HSVFrame,
        ColorLower1, ColorUpper1, Mask);
    if (bHasTwoRanges)
    {
        cv::inRange(HSVFrame,
            ColorLower2, ColorUpper2,
            Mask2);
        cv::bitwise_or(Mask, Mask2, Mask);
    }
    cv::erode(Mask, Mask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(Mask, Mask,
        cv::Mat(), cv::Point(-1,-1), 3);
}

DetectionResult Detector::FindLargestBlob()
{
    DetectionResult Result;
    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(Mask, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty()) return Result;

    double LargestArea = 0;
    int    LargestIdx  = -1;
    for (int i = 0;
        i < (int)Contours.size(); i++)
    {
        double Area =
            cv::contourArea(Contours[i]);
        if (Area > LargestArea)
        {
            LargestArea = Area;
            LargestIdx  = i;
        }
    }

    if (LargestArea < MIN_OBJECT_AREA)
        return Result;

    Result.bFound      = true;
    Result.Area        = LargestArea;
    Result.BoundingBox =
        cv::boundingRect(
            Contours[LargestIdx]);
    Result.Center = cv::Point(
        Result.BoundingBox.x +
        Result.BoundingBox.width  / 2,
        Result.BoundingBox.y +
        Result.BoundingBox.height / 2);

    float BoxArea = (float)(
        Result.BoundingBox.width *
        Result.BoundingBox.height);
    Result.Confidence = std::min(1.f,
        (float)(LargestArea / BoxArea));
    return Result;
}

DetectionResult Detector::Detect(
    const cv::Mat& Frame)
{
    if (Frame.empty())
        return DetectionResult();
    BuildMask(Frame);
    return FindLargestBlob();
}

// ============================================
// AUTO ROI FROM CLICK POINT
// Automatically creates bounding box
// around clicked object based on
// color similarity expansion
// ============================================
cv::Rect Detector::GetAutoROI(
    const cv::Mat& Frame,
    const cv::Point& ClickPoint,
    int DefaultSize)
{
    // Check click point is valid
    if (ClickPoint.x < 0 ||
        ClickPoint.y < 0 ||
        ClickPoint.x >= Frame.cols ||
        ClickPoint.y >= Frame.rows)
    {
        return cv::Rect(
            ClickPoint.x - DefaultSize / 2,
            ClickPoint.y - DefaultSize / 2,
            DefaultSize, DefaultSize) &
            cv::Rect(0, 0,
                Frame.cols, Frame.rows);
    }

    // Get color at click point
    cv::Vec3b ClickColor =
        Frame.at<cv::Vec3b>(
            ClickPoint.y, ClickPoint.x);

    // Convert to HSV for better
    // color comparison
    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Vec3b ClickHSV =
        HSV.at<cv::Vec3b>(
            ClickPoint.y, ClickPoint.x);

    // Color tolerance for expansion
    int HTol = 15; // Hue tolerance
    int STol = 60; // Saturation tolerance
    int VTol = 60; // Value tolerance

    // Create similarity mask using
    // flood fill approach
    cv::Mat SimilarMask =
        cv::Mat::zeros(
            Frame.rows, Frame.cols,
            CV_8UC1);

    // Search expanding rectangle
    // around click point
    int SearchRadius = DefaultSize * 2;
    int X1 = std::max(0,
        ClickPoint.x - SearchRadius);
    int Y1 = std::max(0,
        ClickPoint.y - SearchRadius);
    int X2 = std::min(Frame.cols - 1,
        ClickPoint.x + SearchRadius);
    int Y2 = std::min(Frame.rows - 1,
        ClickPoint.y + SearchRadius);

    // Find pixels with similar color
    // to clicked point
    for (int Y = Y1; Y <= Y2; Y++)
    {
        for (int X = X1; X <= X2; X++)
        {
            cv::Vec3b PixelHSV =
                HSV.at<cv::Vec3b>(Y, X);

            int HDiff = std::abs(
                (int)PixelHSV[0] -
                (int)ClickHSV[0]);
            int SDiff = std::abs(
                (int)PixelHSV[1] -
                (int)ClickHSV[1]);
            int VDiff = std::abs(
                (int)PixelHSV[2] -
                (int)ClickHSV[2]);

            // Handle hue wrap around
            // (red spans 0 and 180)
            if (HDiff > 90)
                HDiff = 180 - HDiff;

            if (HDiff <= HTol &&
                SDiff <= STol &&
                VDiff <= VTol)
            {
                SimilarMask.at<uchar>(
                    Y, X) = 255;
            }
        }
    }

    // Clean up the mask
    cv::erode(SimilarMask, SimilarMask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(SimilarMask, SimilarMask,
        cv::Mat(), cv::Point(-1,-1), 4);

    // Find contours in similar region
    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(SimilarMask,
        Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (!Contours.empty())
    {
        // Find contour containing
        // click point
        for (auto& C : Contours)
        {
            cv::Rect Box =
                cv::boundingRect(C);

            if (Box.contains(ClickPoint)
                && Box.area() > 400)
            {
                // Add small padding
                int Pad = 10;
                Box.x -= Pad;
                Box.y -= Pad;
                Box.width  += Pad * 2;
                Box.height += Pad * 2;

                Box &= cv::Rect(0, 0,
                    Frame.cols,
                    Frame.rows);

                std::cout
                    << "Auto ROI: "
                    << Box.width
                    << "x"
                    << Box.height
                    << " at ("
                    << Box.x << ","
                    << Box.y << ")\n";

                return Box;
            }
        }
    }

    // Fallback — use default size box
    // centered on click point
    std::cout
        << "Using default ROI size\n";

    int Half = DefaultSize / 2;
    cv::Rect DefaultBox(
        ClickPoint.x - Half,
        ClickPoint.y - Half,
        DefaultSize, DefaultSize);

    DefaultBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    return DefaultBox;
}

void Detector::AdaptColorRange(
    const cv::Mat& Frame,
    const cv::Rect& ObjectBox)
{
    cv::Rect SafeBox = ObjectBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0) return;

    cv::Mat ObjectHSV;
    cv::cvtColor(Frame(SafeBox),
        ObjectHSV, cv::COLOR_BGR2HSV);

    cv::Scalar Mean, StdDev;
    cv::meanStdDev(ObjectHSV,
        Mean, StdDev);

    float Margin = 15.f;
    cv::Scalar NewLower(
        std::max(0.0, Mean[0] - Margin),
        std::max(0.0, Mean[1] - 40.0),
        std::max(0.0, Mean[2] - 40.0));
    cv::Scalar NewUpper(
        std::min(180.0, Mean[0] + Margin),
        255.0, 255.0);

    for (int i = 0; i < 3; i++)
    {
        ColorLower1[i] =
            ColorLower1[i] *
            (1 - AdaptRate) +
            NewLower[i] * AdaptRate;
        ColorUpper1[i] =
            ColorUpper1[i] *
            (1 - AdaptRate) +
            NewUpper[i] * AdaptRate;
    }
    bHasTwoRanges = false;
}

cv::Rect Detector::SelectROI(
    cv::Mat& Frame)
{
    cv::Rect ROI = cv::selectROI(
        "Select Object - press Enter",
        Frame, false, false);
    cv::destroyWindow(
        "Select Object - press Enter");
    return ROI;
}

void Detector::Draw(
    cv::Mat& Frame,
    const DetectionResult& Result)
{
    if (!Result.bFound) return;
    cv::rectangle(Frame,
        Result.BoundingBox,
        cv::Scalar(0, 165, 255), 2);
    cv::putText(Frame,
        "DETECTED " +
        std::to_string(
            (int)(Result.Confidence * 100))
        + "%",
        cv::Point(
            Result.BoundingBox.x,
            Result.BoundingBox.y - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 165, 255), 2);
}
