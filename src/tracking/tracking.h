//
// Created by sgunnam on 10/4/20.
//

#ifndef TRACKING_TRACKING_H
#define TRACKING_TRACKING_H
#include <iostream>
#include <random>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI


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

struct FilterParams
{
  int gaussianKernelWidth = 25;
  double epsilonForApproxPolyDp = 2e-6;
  double maxContourAreaToDetect = 10000;
  bool enableConvexHull = false;
};

// Parameters that effect the result
// size of the object to be tracked
enum TrackingAlgorithm { SSD, COLOR_CONTOUR, DEPTH_AND_COLOR_CONTOUR };
enum CameraType {REALSENSE, LAPTOP};
struct TrackingParams{
  CameraType cameraType = CameraType::REALSENSE;
  std::string videoFilePath = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.avi";
  std::string trackerInitialLocation = "/home/sgunnam/CLionProjects/tracking/ps6/input/noisy_debate.txt";
  TrackingAlgorithm trackingType = TrackingAlgorithm::DEPTH_AND_COLOR_CONTOUR;
  FilterParams filterParams;
};

struct StateKF{
  cv::Point2d centroid_measured = {0,0};
  cv::Point2d centroid_tracked = {0,0};
};

struct StatesKF{
  StateKF PrevStateDetected;
  StateKF CurrStateDetected;
  StateKF NextStatePredicted;
  StateKF CurrStateCorrected;
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



// this should be in camera_helper.cpp(doesn't exist)
class MyCam
{
  // Declare depth colorizer for pretty visualization of depth data
  rs2::colorizer color_map;
  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;

  TrackingParams tracking_params_;
  cv::VideoCapture video_capture_;
public:
  MyCam(TrackingParams &tracking_params) : tracking_params_(tracking_params)
  {
    if(tracking_params_.cameraType == CameraType::LAPTOP){
      cv::Mat image;
      // Realsense camera(435i), RGB camera id is 4 when used from open cv, whereas on a laptop this ID would be 0
      video_capture_ = cv::VideoCapture(4);
      //video_capture_.open(0,cv::CAP_V4L);
    } else {
      //Create a configuration for configuring the pipeline with a non default profile
      rs2::config cfg;

      //Add desired streams to configuration
      cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
      cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
      // Start streaming with default recommended configuration
      pipe.start(cfg);
    }
  }

  ~MyCam()
  {
    if(tracking_params_.cameraType == CameraType::LAPTOP){
      video_capture_.release();
    } else {
      pipe.stop();
    }
  }

  void GetCurrentFrame(cv::Mat &image)
  {
    if(tracking_params_.cameraType == CameraType::LAPTOP) {
      video_capture_ >> image;
    } else {
      rs2::frameset data = pipe.wait_for_frames();// Wait for next set of frames from the camera
      rs2::frame depth_frame = data.get_depth_frame().apply_filter(color_map);
      rs2::frame color_frame = data.get_color_frame();

      // Query frame size (width and height)
      const int w = color_frame.as<rs2::video_frame>().get_width();
      const int h = color_frame.as<rs2::video_frame>().get_height();

      // Create OpenCV cv::Matrix of size (w,h) from the colorized depth data
      image = cv::Mat(cv::Size(w, h), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
    }
  }

  void GetDepthFrame(cv::Mat &image){
    rs2::frameset data = pipe.wait_for_frames();// Wait for next set of frames from the camera
    rs2::frame depth_frame = data.get_depth_frame();

    // Query frame size (width and height)
    const int w = depth_frame.as<rs2::video_frame>().get_width();
    const int h = depth_frame.as<rs2::video_frame>().get_height();
    //std::cout << (void *)depth_frame.get_data() << std::endl;

    image = cv::Mat(cv::Size(w,h), CV_16U, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
    //cv::rgbd::DepthCleaner* depthc = new cv::rgbd::DepthCleaner(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

    //cv::Mat cleanedDepth(cv::Size(w, h), CV_16U);
    //depthc->operator()(image, cleanedDepth);

  }
};



class Tracking
{
  TrackingParams trackingParams_;
  States states_;

  StatesKF statesKf_;
  //TODO remove patch_to_track_
  cv::Mat patch_to_track_;

public:
  explicit Tracking(TrackingParams& trackingParams);
private:
  void UpdateTracker(const cv::Mat& currFrame,const cv::Mat& prevFrame, States& states ) const;
  void ReadVideo() const;

  void FilterAndErode(cv::Mat& im) const;
  void CleanDepthMap(cv::Mat& im) const;

};


#endif//TRACKING_TRACKING_H
