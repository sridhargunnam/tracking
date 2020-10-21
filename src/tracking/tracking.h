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


// Parameters that effect the result
// size of the object to be tracked
enum TrackingAlgos{ SSD, CONTOUR};
struct TrackingParams{
  std::string video_file_path           = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.avi";
  std::string tracker_initial_location  = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.txt";
  TrackingAlgos tracking_type = TrackingAlgos::CONTOUR;
};

struct MyMorphParams
{
  int gaussian_kernel_width = 25;
};

struct State{
  cv::Point2d top_left_pt = {320.8751, 175.1776};
  cv::Point2d size_of_rect = {103.5404, 129.0504};
  cv::Point2d velocity = {0.01, 0.01};
  cv::Point2d search_win = {100, 100};
};

struct States{
  State CurrState;
  State NextStatePredicted;
  State CorrectedState;
};

class Tracking
{
  TrackingParams trackingParams_;
  States states_;
  cv::Mat patch_to_track_;

  MyMorphParams morphParams_;
public:
  explicit Tracking(TrackingParams& trackingParams);
private:
  void updateTracker(const cv::Mat& currFrame,const cv::Mat& prevFrame, States& states ) const;
  void ReadVideo() const;

  void Morph(cv::Mat& im);
};


#endif//TRACKING_TRACKING_H
