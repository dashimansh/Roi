#include "ThreadedCamera.h"
#include "Config.h"
#include <iostream>
#include <thread>
#include <chrono>

ThreadedCamera::ThreadedCamera() {}

ThreadedCamera::~ThreadedCamera()
{
    Stop();
}

bool ThreadedCamera::Open(int ID)
{
    Cap.open(ID, cv::CAP_DSHOW);
    if (!Cap.isOpened())
    {
        // Try without DSHOW
        Cap.open(ID);
        if (!Cap.isOpened())
        {
            std::cerr
                << "Cannot open camera\n";
            return false;
        }
    }
    SetupCamera();
    return true;
}

bool ThreadedCamera::OpenFile(
    const std::string& Path)
{
    Cap.open(Path);
    if (!Cap.isOpened()) return false;

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    // If FPS not detected use 30
    if (FPS <= 0 || FPS > 120)
        FPS = 30;

    std::cout << "Opened: "
        << Width << "x" << Height
        << " @ " << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
    return true;
}

void ThreadedCamera::SetupCamera()
{
    Cap.set(cv::CAP_PROP_BUFFERSIZE,
        CAMERA_BUFFER_SIZE);
    Cap.set(cv::CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH);
    Cap.set(cv::CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT);
    Cap.set(cv::CAP_PROP_FPS,
        CAMERA_FPS);

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    // If FPS not detected use 30
    if (FPS <= 0 || FPS > 120)
        FPS = 30;

    std::cout << "Camera: "
        << Width << "x" << Height
        << " @ " << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
}

void ThreadedCamera::CaptureLoop()
{
    while (bRunning)
    {
        // Record time before capture
        auto FrameStart =
            std::chrono::steady_clock::now();

        cv::Mat Frame;
        Cap >> Frame;

        if (Frame.empty())
        {
            // Loop video back to start
            Cap.set(
                cv::CAP_PROP_POS_FRAMES,
                0);
            continue;
        }

        {
            std::lock_guard<std::mutex>
                Lock(FrameMutex);
            LatestFrame = Frame.clone();
            bNewFrame   = true;
        }

        // Calculate delay to maintain
        // correct playback speed
        if (FPS > 0)
        {
            // Time per frame in ms
            int TargetMS =
                (int)(1000.0 / FPS);

            // How long capture took
            auto FrameEnd =
                std::chrono::steady_clock
                ::now();
            int ElapsedMS =
                (int)std::chrono::duration
                <double, std::milli>(
                    FrameEnd - FrameStart)
                .count();

            // Sleep remaining time
            int SleepMS =
                TargetMS - ElapsedMS;

            if (SleepMS > 0)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(
                        SleepMS));
            }
        }
    }
}

bool ThreadedCamera::GetLatestFrame(
    cv::Mat& Out)
{
    if (!bNewFrame) return false;
    std::lock_guard<std::mutex>
        Lock(FrameMutex);
    Out       = LatestFrame.clone();
    bNewFrame = false;
    return true;
}

void ThreadedCamera::Stop()
{
    bRunning = false;
    if (CamThread.joinable())
        CamThread.join();
    Cap.release();
}
