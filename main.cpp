#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <atomic>
#include "Config.h"
#include "Detector.h"
#include "Tracker.h"
#include "FrameStabilizer.h"
#include "ThreadedCamera.h"

// ============================================
// Global mouse click data
// Shared between mouse callback
// and main loop
// ============================================
struct MouseData
{
    std::atomic<bool> bClicked{ false };
    cv::Point         ClickPoint{ 0, 0 };
    std::mutex        Mutex;
};

MouseData GMouse;

// ============================================
// Mouse callback — called when user
// clicks on the video window
// ============================================
void OnMouse(
    int Event,
    int X, int Y,
    int Flags,
    void* UserData)
{
    // Only handle left button click
    if (Event ==
        cv::EVENT_LBUTTONDOWN)
    {
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.ClickPoint = cv::Point(X, Y);
        GMouse.bClicked   = true;

        std::cout
            << "Clicked at: ("
            << X << ", " << Y << ")\n";
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
    int Lost,
    bool bWaitingForClick)
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

    if (bWaitingForClick)
    {
        S  = "CLICK ON OBJECT TO TRACK";
        SC = cv::Scalar(0, 255, 255);
    }
    else
    {
        switch (State)
        {
        case ETrackState::Idle:
            S  = "IDLE - CLICK OBJECT";
            SC = cv::Scalar(200, 200, 200);
            break;
        case ETrackState::Tracking:
            S  = "TRACKING";
            SC = cv::Scalar(0, 255, 0);
            break;
        case ETrackState::Occluded:
            S  = "OCCLUDED [" +
                std::to_string(Lost) +
                "/" +
                std::to_string(
                    MAX_LOST_FRAMES)
                + "]";
            SC = cv::Scalar(0, 165, 255);
            break;
        case ETrackState::Lost:
            S  = "SEARCHING...";
            SC = cv::Scalar(0, 0, 255);
            break;
        }
    }

    cv::putText(Frame,
        "STATE: " + S,
        cv::Point(160, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55, SC, 2);

    cv::putText(Frame,
        "CLICK=Select  R=Reset  "
        "1=CSRT 2=KCF  Q=Quit",
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(180, 180, 180), 1);
}

// Draw crosshair at click point
// while waiting for ROI
void DrawCrosshair(
    cv::Mat& Frame,
    const cv::Point& Pt)
{
    cv::line(Frame,
        cv::Point(Pt.x - 20, Pt.y),
        cv::Point(Pt.x + 20, Pt.y),
        cv::Scalar(0, 255, 255), 2);
    cv::line(Frame,
        cv::Point(Pt.x, Pt.y - 20),
        cv::Point(Pt.x, Pt.y + 20),
        cv::Scalar(0, 255, 255), 2);
    cv::circle(Frame, Pt, 5,
        cv::Scalar(0, 255, 255), -1);
}

int main()
{
    std::cout
        << "Object Tracker\n"
        << "==============\n"
        << "CLICK on object to track!\n"
        << "R = Reset\n"
        << "1 = CSRT (most accurate)\n"
        << "2 = KCF\n"
        << "3 = CamShift\n"
        << "Q = Quit\n\n";

    // ================================================
    // MODE 1 = Webcam
    // MODE 2 = Video file
    // MODE 3 = IP camera
    // ================================================
    const int MODE = 1;

    const int CAMERA_INDEX = 0;

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
    int  FrameCount  = 0;
    bool bStabilize  = false;

    // Create window
    cv::namedWindow(
        WINDOW_NAME,
        cv::WINDOW_NORMAL);
    cv::resizeWindow(
        WINDOW_NAME, 960, 540);

    // Set mouse callback on window
    // This enables click to select!
    cv::setMouseCallback(
        WINDOW_NAME,
        OnMouse, nullptr);

    std::cout
        << "\nVideo is playing!\n"
        << "CLICK on any object "
        << "to start tracking it!\n\n";

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
        // CHECK FOR MOUSE CLICK
        // Video keeps playing while
        // checking for clicks!
        // ============================================
        if (GMouse.bClicked.load())
        {
            cv::Point ClickPt;
            {
                std::lock_guard<std::mutex>
                    Lock(GMouse.Mutex);
                ClickPt = GMouse.ClickPoint;
                GMouse.bClicked = false;
            }

            std::cout
                << "Auto detecting ROI "
                << "around click point...\n";

            // Automatically create ROI
            // around clicked object
            // Video does NOT pause!
            cv::Rect AutoROI =
                Det.GetAutoROI(
                    PF, ClickPt, 80);

            if (AutoROI.area() > 100)
            {
                // Initialize tracker
                // with auto detected ROI
                Tracker.Init(PF, AutoROI,
                    ETrackerType::CSRT);

                std::cout
                    << "Tracking started!\n"
                    << "ROI: "
                    << AutoROI.width
                    << "x"
                    << AutoROI.height
                    << "\n";
            }
            else
            {
                std::cout
                    << "Could not detect "
                    << "object at click "
                    << "point. Try again!\n";
            }
        }

        // Update tracker
        if (Tracker.GetState() !=
            ETrackState::Idle)
            Tracker.Update(PF);

        // Draw tracking box
        Tracker.Draw(PF);

        // Draw HUD
        DrawHUD(PF, FPS.Get(),
            Tracker.GetState(),
            Tracker.GetLostCount(),
            false);

        // Show click instruction
        // when idle
        if (Tracker.GetState() ==
            ETrackState::Idle)
        {
            cv::putText(PF,
                "CLICK on object to track",
                cv::Point(
                    PF.cols / 2 - 150,
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
