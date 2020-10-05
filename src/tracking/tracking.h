//
// Created by sgunnam on 10/4/20.
//

#ifndef TRACKING_TRACKING_H
#define TRACKING_TRACKING_H
#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
struct TrackingParams{
  std::string video_file_path           = "/home/sgunnam/CLionProjects/tracking/ps6/input/pres_debate.avi";
  std::string tracker_initial_location  = "/home/sgunnam/CLionProjects/tracking/ps6/input/pres_debate.txt";
  std::string tracking_type = "ssd";
};
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
// Optimization: Having custom contour masks to track and detect instead of rectangles.
struct State{
  cv::Point2d top_left_pt = {320.8751, 175.1776};
  cv::Point2d size_of_rect = {103.5404, 129.0504};
  cv::Point2d velocity = {0.01, 0.01};
  cv::Point2d search_wind = {500, 500};
};

struct States{
  State CurrentState;
  State NextStatePredicted;
  State CorrectedState;
};

class Tracking
{
  TrackingParams trackingParams_;
  States states_;
  cv::Mat patch_to_track_;
public:
  explicit Tracking(TrackingParams& trackingParams): trackingParams_(trackingParams){
    cv::Mat currFrame;
    cv::Mat prevFrame;
    cv::VideoCapture cap(trackingParams_.video_file_path);
    cv::Point2d p1;
    cv::Point2d p2;
    cap >> currFrame;

    cv::Rect2d trackerROI(
      states_.CurrentState.top_left_pt.x,
      states_.CurrentState.top_left_pt.y,
      states_.CurrentState.size_of_rect.x,
      states_.CurrentState.size_of_rect.y
    );
    patch_to_track_ = currFrame(trackerROI);

    while(true) {
      prevFrame = std::move(currFrame);
      cap >> currFrame;
      updateTracker(currFrame, prevFrame, states_);
      p1 = states_.CurrentState.top_left_pt;
      p2 = states_.CurrentState.top_left_pt + states_.CurrentState.size_of_rect;
      cv::rectangle(currFrame, p2, p1, cv::Scalar(255, 0, 0), 5, 8, 0);
      cv::imshow("Tracking window", currFrame);
      char c = static_cast<char>(cv::waitKey(5000));
      if (c == 27)
        break;
    }
  }

private:
  void updateTracker([[maybe_unused]]const cv::Mat& currFrame,[[maybe_unused]]const cv::Mat& prevFrame, States& states ){
    cv::Rect2d trackerROI(
                    states.CurrentState.top_left_pt.x,
                    states.CurrentState.top_left_pt.y,
                    states.CurrentState.size_of_rect.x,
                    states.CurrentState.size_of_rect.y
                    );
    cv::Mat patch_to_track = patch_to_track_.clone(); //prevFrame(trackerROI);
//    cv::imshow("ROI", patch_to_track);
//    cv::waitKey(0);
    cv::Point2i search_window_top_left = {
      static_cast<int>(states.CurrentState.top_left_pt.x - states.CurrentState.search_wind.x),
      static_cast<int>(states.CurrentState.top_left_pt.y - states.CurrentState.search_wind.y)
    };


    cv::Point2i window_size{
      static_cast<int>( states.CurrentState.size_of_rect.x + 2 * states.CurrentState.search_wind.x),
      static_cast<int>( states.CurrentState.size_of_rect.y + 2 * states.CurrentState.search_wind.y)
    };
    cv::Rect2i search_rect = {
      search_window_top_left.x,
      search_window_top_left.y,
      window_size.x,
      window_size.y
    };
    cv::Mat searchImg = currFrame(search_rect);
//    cv::imshow("search Image", searchImg);
//    cv::waitKey(0);
    cv::Mat result;
    cv::matchTemplate(searchImg, patch_to_track, result, cv::TM_CCOEFF_NORMED);
    normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
    matchLoc = minLoc;
    states.CurrentState.top_left_pt.x = states.CurrentState.top_left_pt.x + matchLoc.x;
    states.CurrentState.top_left_pt.y = states.CurrentState.top_left_pt.y + matchLoc.y;
//    cv::rectangle(
//      currFrame,
//      cv::Point(
//        static_cast<int>(states.CurrentState.top_left_pt.x) + matchLoc.x ,
//        static_cast<int>(states.CurrentState.top_left_pt.y) + matchLoc.y
//        ),
//      cv::Point(
//        static_cast<int>(states.CurrentState.top_left_pt.x) + static_cast<int>(states.CurrentState.size_of_rect.x) + matchLoc.x  ,
//        static_cast<int>(states.CurrentState.top_left_pt.y) + static_cast<int>(states.CurrentState.size_of_rect.y) + matchLoc.y
//        ),
//      cv::Scalar(255,0,0),
//      10, 8, 0 );

//    cv::imshow("Tracked result ", currFrame);
//    cv::waitKey(0);
//    std::cout << "result = \n" << result << "\n";
  }
  void ReadVideo() const;
};


#endif//TRACKING_TRACKING_H
