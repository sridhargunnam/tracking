//
// Created by sgunnam on 10/4/20.
//

#ifndef TRACKING_TRACKING_H
#define TRACKING_TRACKING_H
#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

//#include "ShowMultipleImages.h"

// TODO
// Two parameters to consider while tracking
// 1) Motion of the camera
// 2) Motion of objects within the scene
// Simple approach first i.e model one motion at a time. Do think about how to combine then later from the beginning itself
// It's just it's easier to test them separately
// Simple visual tests first, later on find datasets with ground truth
// "ssd" assumption
// Size of the bounding box doesn't change
// The tracking object can rotate about itself a lot
// Error in prediction is random noise
// Optimization: Model mask for template matching step, i.e Having custom contour masks to track and detect instead of rectangles that introduce error in correlation

struct TrackingParams{
  std::string video_file_path           = "/home/sgunnam/CLionProjects/tracking/ps6/input/pres_debate.avi";
  std::string tracker_initial_location  = "/home/sgunnam/CLionProjects/tracking/ps6/input/pres_debate.txt";
  std::string tracking_type = "ssd";
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
public:
  explicit Tracking(TrackingParams& trackingParams);
private:
  void updateTracker(const cv::Mat& currFrame,const cv::Mat& prevFrame, States& states ) const;
  void ReadVideo() const;
};


#endif//TRACKING_TRACKING_H
