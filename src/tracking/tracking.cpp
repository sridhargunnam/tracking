//
// Created by sgunnam on 10/4/20.
//

#include "tracking.h"

Tracking::Tracking(TrackingParams& trackingParams): trackingParams_(trackingParams)
{
  cv::Mat currFrame;
  cv::Mat prevFrame;
  cv::VideoCapture cap(trackingParams_.video_file_path);
  cv::Point2d p1;
  cv::Point2d p2;
  cap >> currFrame;
  while (true) {
    currFrame.copyTo(prevFrame);
    cap >> currFrame;
    updateTracker(currFrame, prevFrame, states_);
    p1 = states_.CurrState.top_left_pt;
    p2 = states_.CurrState.top_left_pt + states_.CurrState.size_of_rect;
    cv::rectangle(currFrame, p2, p1, cv::Scalar(255, 0, 0), 5, 8, 0);
    cv::imshow("Tracking window", currFrame);
    char c = static_cast<char>(cv::waitKey(27));
    if (c == 27)
      break;
  }
}

void Tracking::updateTracker(const cv::Mat& currFrame,const cv::Mat& prevFrame, States& states ) const{
  cv::Rect2d trackerROI(
    states.CurrState.top_left_pt.x,
    states.CurrState.top_left_pt.y,
    states.CurrState.size_of_rect.x,
    states.CurrState.size_of_rect.y
  );
  cv::Rect2d searchROI(
    std::max( static_cast<int>(states_.CurrState.top_left_pt.x   - 1*states_.CurrState.search_win.x) , 0),
    std::max( static_cast<int>(states_.CurrState.top_left_pt.y   - 1*states_.CurrState.search_win.y) , 0),
    std::min( static_cast<int>(states_.CurrState.size_of_rect.x  + 2*states_.CurrState.search_win.x) , currFrame.rows-1),
    std::min( static_cast<int>(states_.CurrState.size_of_rect.y  + 2*states_.CurrState.search_win.y) , currFrame.cols-1)
  );
  cv::Mat patch_track_win = prevFrame(trackerROI);
  cv::Mat patch_search_win = currFrame(searchROI);
  //std::vector<cv::Mat> showImagesList{patch_track_win, patch_search_win};
  //ShowMultipleImagesTracking("patch and search window",  showImagesList);
  cv::Mat result;
  cv::matchTemplate(patch_search_win, patch_track_win, result, cv::TM_CCORR_NORMED);
  normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::Point matchLoc;
  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
  matchLoc = maxLoc;
  states.CurrState.top_left_pt.x = std::min(
    (std::max(static_cast<int>(states_.CurrState.top_left_pt.x - 1*states_.CurrState.search_win.x), 0) + matchLoc.x ),
    (currFrame.rows -1));
  states.CurrState.top_left_pt.y = std::min(
    (std::max(static_cast<int>(states_.CurrState.top_left_pt.y - 1*states_.CurrState.search_win.y), 0) + matchLoc.y ),
    (currFrame.cols -1));
/*
  cv::Mat debug_img ;
  currFrame.copyTo(debug_img);
  cv::rectangle(
    debug_img,
    cv::Point(
      static_cast<int>(states.CurrState.top_left_pt.x),
      static_cast<int>(states.CurrState.top_left_pt.y)
    ),
    cv::Point(
      static_cast<int>(states.CurrState.top_left_pt.x + states.CurrState.search_win.x),
      static_cast<int>(states.CurrState.top_left_pt.y + states.CurrState.search_win.y)
    ),
    cv::Scalar(255,0,0),
    10, 8, 0 );

  std::vector<cv::Mat> showImagesList{ patch_track_win, patch_search_win, debug_img};
  ShowMultipleImagesTracking("patch for track & search, debug images",  showImagesList);
  */
}

void Tracking::ReadVideo() const{
  cv::VideoCapture cap(trackingParams_.video_file_path);
  if(!cap.isOpened()){
    std::cout << "Error opening the file " << trackingParams_.video_file_path << "\n";
    return;
  }

  cv::namedWindow("Tracking window",cv::WINDOW_AUTOSIZE);
  while(true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
      break;
    cv::imshow("Tracking window", frame);

    char c = static_cast<char>(cv::waitKey(1000));
    if (c == 27)
      break;
  }
  cap.release();
  cv::destroyAllWindows();
}
