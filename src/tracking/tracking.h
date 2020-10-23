//
// Created by sgunnam on 10/4/20.
//

#ifndef TRACKING_TRACKING_H
#define TRACKING_TRACKING_H
#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

#include <librealsense2/rs.hpp>

// TODO
// Two parameters to consider while tracking
// 1) Motion of the camera
// 2) Motion of objects within the scene
// Simple visual tests first, later on find datasets with ground truth
// Size of the bounding box doesn't change
// The tracking object can rotate about itself a lot (how to model orientation changes of the object)
// Optimization: Model mask for template matching step,
// i.e Having custom contour masks to track and detect instead of rectangles that introduce error in correlation


// this should be in camera_helper.cpp(doesn't exist)
class RSCam
{
  // Declare depth colorizer for pretty visualization of depth data
  rs2::colorizer color_map;
  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;

public:
  RSCam()
  {
    // Start streaming with default recommended configuration
    pipe.start();
  }

  ~RSCam()
  {
    pipe.stop();
  }

  void GetCurrentFrame(cv::Mat &image)
  {
    rs2::frameset data = pipe.wait_for_frames();// Wait for next set of frames from the camera
    rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
    rs2::frame color_frame = data.get_color_frame();

    // Query frame size (width and height)
    const int w = color_frame.as<rs2::video_frame>().get_width();
    const int h = color_frame.as<rs2::video_frame>().get_height();

    // Create OpenCV cv::Matrix of size (w,h) from the colorized depth data
    image = cv::Mat(cv::Size(w, h), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
  }
};


struct FilterParams
{
  int gaussianKernelWidth = 25;
  double epsilonForApproxPolyDp = 2e-6;
  double maxContourAreaToDetect = 5000;
  bool enableConvexHull = false;
};

// Parameters that effect the result
// size of the object to be tracked
enum TrackingAlgorithm { SSD, CONTOUR};

struct TrackingParams{
  std::string videoFilePath = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.avi";
  std::string trackerInitialLocation = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.txt";
  TrackingAlgorithm trackingType = TrackingAlgorithm::CONTOUR;
  FilterParams filterParams;
};

struct State{
  cv::Point2d topLeftPt = {320.8751, 175.1776};
  cv::Point2d sizeOfRect = {103.5404, 129.0504};
  cv::Point2d velocity = {0.01, 0.01};
  cv::Point2d searchWin = {100, 100};
};

struct States{
  State CurrState;
  State NextStatePredicted;
  State CurrStateCorrected;
};

class Tracking
{
  TrackingParams trackingParams_;
  States states_;
  //TODO remove patch_to_track_
  cv::Mat patch_to_track_;

public:
  explicit Tracking(TrackingParams& trackingParams);
private:
  void UpdateTracker(const cv::Mat& currFrame,const cv::Mat& prevFrame, States& states ) const;
  void ReadVideo() const;

  void FilterAndErode(cv::Mat& im) const;
};


#endif//TRACKING_TRACKING_H
