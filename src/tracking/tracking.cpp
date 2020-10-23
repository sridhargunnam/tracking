//
// Created by sgunnam on 10/4/20.
//

#include "tracking.h"

Tracking::Tracking(TrackingParams &trackingParams) : trackingParams_(trackingParams)
{
  //TODO Implement a strategy method for different tracking algorithms instead of if else conditionals
  if (trackingParams.trackingType == TrackingAlgorithm::SSD) {
    cv::Mat currFrame;
    cv::Mat prevFrame;
    cv::VideoCapture cap(trackingParams_.videoFilePath);
    cv::Point2d p1;
    cv::Point2d p2;
    cap >> currFrame;
    while (true) {
      currFrame.copyTo(prevFrame);
      cap >> currFrame;
      UpdateTracker(currFrame, prevFrame, states_);
      p1 = states_.CurrState.topLeftPt;
      p2 = states_.CurrState.topLeftPt + states_.CurrState.sizeOfRect;
      cv::rectangle(currFrame, p2, p1, cv::Scalar(255, 0, 0), 5, 8, 0);
      cv::imshow("Tracking window", currFrame);
      char c = static_cast<char>(cv::waitKey(27));
      if (c == 27)
        break;
    }
  }
  else if (trackingParams.trackingType == TrackingAlgorithm::CONTOUR) {

    auto rsCam_ = RSCam();
    //create Background Subtract objects
    cv::Ptr<cv::BackgroundSubtractor> pBackSub{ cv::createBackgroundSubtractorKNN() };

    cv::Mat frame, fgMask;
    while (true) {
      rsCam_.GetCurrentFrame(frame);
      //TODO : frame_orig only used for debugging but affects performance now, should be refactored to include only for debug mode
      cv::Mat frame_orig = frame.clone();
      FilterAndErode(frame);
      if (frame.empty())
        break;

      //update the background model
      pBackSub->apply(frame, fgMask);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(fgMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

      std::vector<std::vector<cv::Point>> filteredContours;
      std::vector<cv::Point> approxCurve;
      for (auto i = 0ul; i < contours.size(); i++) {
        approxPolyDP(contours[i], approxCurve, arcLength(contours[i], true) * trackingParams.filterParams.epsilonForApproxPolyDp, true);
        if (fabs(contourArea(approxCurve)) > trackingParams.filterParams.maxContourAreaToDetect) {
          if(trackingParams.filterParams.enableConvexHull) {
            cv::Mat hullPoints;
            cv::convexHull(contours[i], hullPoints);
            filteredContours.push_back(hullPoints);
          } else
            filteredContours.push_back(contours[i]);
        }
      }

      cv::drawContours(frame_orig, filteredContours, -1, cv::Scalar(0, 0, 255), 3);

      //show the current frame and the fg masks
      imshow("Frame", frame);
      imshow("FG Mask", fgMask);
      imshow("Contours", frame_orig);

      //get the input from the keyboard
      int keyboard = cv::waitKey(30);
      if (keyboard == 'q' || keyboard == 27)
        break;
    }
  }
}

// Based on opencv squares.cpp sample
void Tracking::FilterAndErode(cv::Mat &im) const
{
  cv::Size gaussian_kernel = cv::Size(trackingParams_.filterParams.gaussianKernelWidth, trackingParams_.filterParams.gaussianKernelWidth);
  cv::GaussianBlur(im, im, gaussian_kernel, 0, 0);
  cv::dilate(im, im, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
}


void Tracking::UpdateTracker(const cv::Mat &currFrame, const cv::Mat &prevFrame, States &states) const
{
  cv::Rect2d trackerROI(
    states.CurrState.topLeftPt.x,
    states.CurrState.topLeftPt.y,
    states.CurrState.sizeOfRect.x,
    states.CurrState.sizeOfRect.y);
  cv::Rect2d searchROI(
    std::max(static_cast<int>(states_.CurrState.topLeftPt.x - 1 * states_.CurrState.searchWin.x), 0),
    std::max(static_cast<int>(states_.CurrState.topLeftPt.y - 1 * states_.CurrState.searchWin.y), 0),
    std::min(static_cast<int>(states_.CurrState.sizeOfRect.x + 2 * states_.CurrState.searchWin.x), currFrame.rows - 1),
    std::min(static_cast<int>(states_.CurrState.sizeOfRect.y + 2 * states_.CurrState.searchWin.y), currFrame.cols - 1));
  cv::Mat patch_track_win = prevFrame(trackerROI);
  cv::Mat patch_search_win = currFrame(searchROI);
  //std::vector<cv::Mat> showImagesList{patch_track_win, patch_search_win};
  //ShowMultipleImagesTracking("patch and search window",  showImagesList);
  cv::Mat result;
  cv::matchTemplate(patch_search_win, patch_track_win, result, cv::TM_CCORR_NORMED);
  normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::Point matchLoc;
  minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
  matchLoc = maxLoc;
  states.CurrState.topLeftPt.x = std::min(
    (std::max(static_cast<int>(states_.CurrState.topLeftPt.x - 1 * states_.CurrState.searchWin.x), 0) + matchLoc.x),
    (currFrame.rows - 1));
  states.CurrState.topLeftPt.y = std::min(
    (std::max(static_cast<int>(states_.CurrState.topLeftPt.y - 1 * states_.CurrState.searchWin.y), 0) + matchLoc.y),
    (currFrame.cols - 1));
  /*
  cv::Mat debug_img ;
  currFrame.copyTo(debug_img);
  cv::rectangle(
    debug_img,
    cv::Point(
      static_cast<int>(states.CurrState.topLeftPt.x),
      static_cast<int>(states.CurrState.topLeftPt.y)
    ),
    cv::Point(
      static_cast<int>(states.CurrState.topLeftPt.x + states.CurrState.searchWin.x),
      static_cast<int>(states.CurrState.topLeftPt.y + states.CurrState.searchWin.y)
    ),
    cv::Scalar(255,0,0),
    10, 8, 0 );

  std::vector<cv::Mat> showImagesList{ patch_track_win, patch_search_win, debug_img};
  ShowMultipleImagesTracking("patch for track & search, debug images",  showImagesList);
  */
}

void Tracking::ReadVideo() const
{
  cv::VideoCapture cap(trackingParams_.videoFilePath);
  if (!cap.isOpened()) {
    std::cout << "Error opening the file " << trackingParams_.videoFilePath << "\n";
    return;
  }

  cv::namedWindow("Tracking window", cv::WINDOW_AUTOSIZE);
  while (true) {
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
