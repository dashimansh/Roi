#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>
#include "Config.h"
#include "Detector.h"
#include "Tracker.h"
#include "FrameStabilizer.h"
#include "ThreadedCamera.h"

// ============================================
// Mouse drawing state
// Handles draw ROI while video plays
// ============================================
struct MouseDrawData
{
    std::atomic<bool> bDrawing{ false };
    std::atomic<bool> bROIReady{ false };
    cv::Point         StartPoint{ 0, 0 };
    cv::Point         EndPoint{ 0, 0 };
    cv::Rect          DrawnROI;
    std::mutex        Mutex;
};

MouseDrawData GMouse;

// ============================================
// Mouse callback
// Left press → start drawing
// Left drag → update rectangle
// Left release → ROI ready!
// ============================================
void OnMouse(
    int Event,
    int X, int Y,
    int Flags,
    void* UserData)
{
    // Clamp to valid range
    X = std::max(0, X);
    Y = std::max(0, Y);

    if (Event ==
        cv::EVENT_LBUTTONDOWN)
    {
        // Start drawing
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.StartPoint =
            cv::Point(X, Y);
        GMouse.EndPoint =
            cv::Point(X, Y);
        GMouse.bDrawing  = true;
        GMouse.bROIReady = false;

        std::cout
            << "Drawing ROI...\n";
    }
    else if (Event ==
        cv::EVENT_MOUSEMOVE &&
        GMouse.bDrawing.load())
    {
        // Update end point while dragging
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.EndPoint =
            cv::Point(X, Y);
    }
    else if (Event ==
        cv::EVENT_LBUTTONUP &&
        GMouse.bDrawing.load())
    {
        // Finish drawing
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.EndPoint =
            cv::Point(X, Y);
        GMouse.bDrawing  = false;

        // Calculate ROI from
        // start and end points
        int RX = std::min(
            GMouse.StartPoint.x,
            GMouse.EndPoint.x);
        int RY = std::min(
            GMouse.StartPoint.y,
            GMouse.EndPoint.y);
        int RW = std::abs(
            GMouse.EndPoint.x -
            GMouse.StartPoint.x);
        int RH = std::abs(
            GMouse.EndPoint.y -
            GMouse.StartPoint.y);

        // Only accept if box
        // is large enough
        if (RW > 10 && RH > 10)
        {
            GMouse.DrawnROI =
                cv::Rect(RX, RY, RW, RH);
            GMouse.bROIReady = true;

            std::cout
                << "ROI drawn: "
                << RW << "x" << RH
                << " at ("
                << RX << ","
                << RY << ")\n";
        }
        else
        {
            std::cout
                << "ROI too small!"
                << " Draw bigger box.\n";
        }
    }
}

class FPSCounter
{
public:
    void Tick()
    {
        auto N =
            std::chrono::steady_clock::now();
        double E =
            std::chrono::duration<double>(
                N - Last).count();
        Last = N;
        FPS  = 1.0 / E;
    }
    double Get() const { return FPS; }

private:
    std::chrono::steady_clock::time_point
        Last =
        std::chrono::steady_clock::now();
    double FPS = 0.0;
};

void DrawHUD(
    cv::Mat& Frame,
    double FPS,
    ETrackState State,
    int Lost)
{
    cv::rectangle(Frame,
        cv::Rect(0, 0, Frame.cols, 75),
        cv::Scalar(0, 0, 0), -1);

    cv::putText(Frame,
        "FPS: " + std::to_string((int)FPS),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 255), 2);

    std::string S;
    cv::Scalar  SC;

    switch (State)
    {
    case ETrackState::Idle:
        S  = "DRAW BOX ON OBJECT";
        SC = cv::Scalar(0, 255, 255);
        break;
    case ETrackState::Tracking:
        S  = "TRACKING";
        SC = cv::Scalar(0, 255, 0);
        break;
    case ETrackState::Occluded:
        S  = "OCCLUDED [" +
            std::to_string(Lost) +
            "/" +
            std::to_string(MAX_LOST_FRAMES)
            + "]";
        SC = cv::Scalar(0, 165, 255);
        break;
    case ETrackState::Lost:
        S  = "SEARCHING...";
        SC = cv::Scalar(0, 0, 255);
        break;
    }

    cv::putText(Frame,
        "STATE: " + S,
        cv::Point(160, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55, SC, 2);

    cv::putText(Frame,
        "DRAG=Draw ROI  R=Reset  "
        "1=CSRT 2=KCF  Q=Quit",
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(180, 180, 180), 1);
}

// Draw live rectangle while dragging
void DrawLiveROI(cv::Mat& Frame)
{
    if (!GMouse.bDrawing.load())
        return;

    std::lock_guard<std::mutex>
        Lock(GMouse.Mutex);

    cv::Point P1 = GMouse.StartPoint;
    cv::Point P2 = GMouse.EndPoint;

    // Draw animated rectangle
    // while user is dragging
    cv::rectangle(Frame,
        P1, P2,
        cv::Scalar(0, 255, 255), 2);

    // Draw corner markers
    int CornerLen = 10;

    // Top left corner
    cv::line(Frame,
        P1,
        cv::Point(P1.x + CornerLen, P1.y),
        cv::Scalar(0, 255, 0), 3);
    cv::line(Frame,
        P1,
        cv::Point(P1.x, P1.y + CornerLen),
        cv::Scalar(0, 255, 0), 3);

    // Bottom right corner
    cv::line(Frame,
        P2,
        cv::Point(P2.x - CornerLen, P2.y),
        cv::Scalar(0, 255, 0), 3);
    cv::line(Frame,
        P2,
        cv::Point(P2.x, P2.y - CornerLen),
        cv::Scalar(0, 255, 0), 3);

    // Show size text
    int W = std::abs(P2.x - P1.x);
    int H = std::abs(P2.y - P1.y);
    cv::putText(Frame,
        std::to_string(W) + "x" +
        std::to_string(H),
        cv::Point(
            std::min(P1.x, P2.x),
            std::min(P1.y, P2.y) - 5),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(0, 255, 255), 1);
}

int main()
{
    std::cout
        << "Object Tracker\n"
        << "==============\n"
        << "DRAG mouse to draw ROI!\n"
        << "R = Reset\n"
        << "1 = CSRT\n"
        << "2 = KCF\n"
        << "3 = CamShift\n"
        << "Q = Quit\n\n";

    // ================================================
    // MODE 1 = Webcam
    // MODE 2 = Video file
    // MODE 3 = IP camera
    // ================================================
    const int MODE = 2;

    const int CAMERA_INDEX = 0;

    // !! CHANGE TO YOUR VIDEO PATH !!
    const std::string VIDEO_PATH =
        "C:/Users/Victus/Downloads/video.mp4";

    const std::string IP_URL =
        "rtsp://admin:admin123@192.168.1.64:554/stream";
    // ================================================

    ThreadedCamera  Camera;
    Detector        Det;
    ObjectTracker   Tracker;
    FrameStabilizer Stab;
    FPSCounter      FPS;

    if (MODE == 1)
    {
        std::cout << "Opening webcam\n";
        if (!Camera.Open(CAMERA_INDEX))
        {
            std::cerr
                << "Failed to open webcam!\n";
            return -1;
        }
        std::cout << "Webcam opened!\n";
    }
    else if (MODE == 2)
    {
        std::cout
            << "Opening video: "
            << VIDEO_PATH << "\n";
        if (!Camera.OpenFile(VIDEO_PATH))
        {
            std::cerr
                << "Failed! Check path.\n";
            return -1;
        }
        std::cout << "Video opened!\n";
    }
    else if (MODE == 3)
    {
        std::cout
            << "Connecting to: "
            << IP_URL << "\n";
        if (!Camera.OpenFile(IP_URL))
        {
            std::cerr
                << "Failed! Check URL.\n";
            return -1;
        }
        std::cout << "IP Camera connected!\n";
    }

    cv::Mat Frame;
    int  FrameCount = 0;
    bool bStabilize = false;

    // Create window
    cv::namedWindow(
        WINDOW_NAME,
        cv::WINDOW_NORMAL);
    cv::resizeWindow(
        WINDOW_NAME, 960, 540);

    // Set mouse callback
    cv::setMouseCallback(
        WINDOW_NAME,
        OnMouse, nullptr);

    std::cout
        << "\nVideo playing!\n"
        << "DRAG mouse on object "
        << "to draw ROI and track!\n\n";

    while (true)
    {
        if (!Camera.GetLatestFrame(Frame))
            continue;

        FrameCount++;
        FPS.Tick();

        cv::Mat PF = Frame.clone();

        if (bStabilize)
            PF = Stab.Stabilize(Frame);

        // ============================================
        // CHECK IF ROI WAS DRAWN
        // ============================================
        if (GMouse.bROIReady.load())
        {
            cv::Rect DrawnROI;
            {
                std::lock_guard<std::mutex>
                    Lock(GMouse.Mutex);
                DrawnROI = GMouse.DrawnROI;
                GMouse.bROIReady = false;
            }

            // Clamp ROI to frame size
            DrawnROI &= cv::Rect(0, 0,
                PF.cols, PF.rows);

            if (DrawnROI.area() > 100)
            {
                // Initialize tracker
                // with drawn ROI
                Tracker.Init(PF,
                    DrawnROI,
                    ETrackerType::CSRT);

                std::cout
                    << "Tracking started!\n";
            }
        }

        // Update tracker
        if (Tracker.GetState() !=
            ETrackState::Idle)
            Tracker.Update(PF);

        // Draw tracking result
        Tracker.Draw(PF);

        // Draw live ROI rectangle
        // while user is dragging
        DrawLiveROI(PF);

        // Draw HUD
        DrawHUD(PF, FPS.Get(),
            Tracker.GetState(),
            Tracker.GetLostCount());

        // Show instruction when idle
        if (Tracker.GetState() ==
            ETrackState::Idle &&
            !GMouse.bDrawing.load())
        {
            cv::putText(PF,
                "Drag mouse to select object",
                cv::Point(
                    PF.cols / 2 - 180,
                    PF.rows / 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 255),
                2);
        }

        cv::imshow(WINDOW_NAME, PF);

        char Key = (char)cv::waitKey(1);

        if (Key == 'r' || Key == 'R')
        {
            Tracker.Reset();
            Stab.Reset();
            std::cout << "Reset!\n";
        }
        else if (Key == 't' || Key == 'T')
        {
            bStabilize = !bStabilize;
            std::cout << "Stabilize: "
                << (bStabilize ?
                    "ON" : "OFF") << "\n";
        }
        else if (Key == '1')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CSRT);
                std::cout << "CSRT\n";
            }
        }
        else if (Key == '2')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::KCF);
                std::cout << "KCF\n";
            }
        }
        else if (Key == '3')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CAMSHIFT);
                std::cout << "CamShift\n";
            }
        }
        else if (Key == 'q' ||
            Key == 'Q' || Key == 27)
            break;
    }

    Camera.Stop();
    cv::destroyAllWindows();
    return 0;
}
