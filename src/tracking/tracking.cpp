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
    auto cam = MyCam(trackingParams);

    cv::Ptr<cv::BackgroundSubtractor> pBackSub{ cv::createBackgroundSubtractorKNN(1, 100.0, true) };

    // TODO refactor Kalman related states
    cv::KalmanFilter kalmanFilter_(2,1,0);
    cv::Mat kalmanState_(2,1, CV_32F);
    cv::Mat processNoise_(2,1,CV_32F);
    cv::Mat measurement_ = cv::Mat::zeros(2,1,CV_32F);
    {
      {
        //cv::Mat frame, fgMask;
        //for(int i=0; i<0; i++) {
        //  cam.GetCurrentFrame(frame);
        //  if (frame.empty()) {
        //    std::cout << "Couldn't read frame at init\n";
        //  }
        //  FilterAndErode(frame);
        //  //update the background model
        //  pBackSub->apply(frame, fgMask, 0.99);
        //}
        //cv::Mat frame_orig = frame.clone();
        //std::vector<std::vector<cv::Point>> contours;
        //cv::findContours(fgMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        //std::sort(contours.begin(), contours.end(), [](auto& lhs, auto& rhs)
        //{
        //       return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs) ) ) ;
        //});
        //auto all_moments = cv::moments(contours[0], true);
        //statesKf_.CurrStateDetected.centroid_measured = cv::Point2d {
        //  (all_moments.m10/ all_moments.m00),
        //  (all_moments.m01 / all_moments.m00)};
        //kalmanFilter_.statePre.at<double>(0,0) = statesKf_.CurrStateDetected.centroid_measured.x;
        //kalmanFilter_.statePre.at<double>(1,0) = statesKf_.CurrStateDetected.centroid_measured.y;
        kalmanFilter_.statePre.at<double>(0,0) = 320;
        kalmanFilter_.statePre.at<double>(1,0) = 240;
        //cv::Rect rect = cv::boundingRect(contours[0]);
        //cv::rectangle(frame_orig, rect, cv::Scalar(0,0,255), 3);

        //show the current frame and the fg masks
        //imshow("FrameI", frame);
        //imshow("FG MaskI", fgMask);
        //imshow("ContoursI", frame_orig);
        //cv::waitKey(0);
      }
      //randn( kalmanState_, cv::Scalar::all(0), cv::Scalar::all(0.1) );
      kalmanFilter_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
      setIdentity(kalmanFilter_.measurementMatrix);
      setIdentity(kalmanFilter_.processNoiseCov, cv::Scalar::all(1e-5));
      setIdentity(kalmanFilter_.measurementNoiseCov, cv::Scalar::all(1e-1));
      setIdentity(kalmanFilter_.errorCovPost, cv::Scalar::all(1));
      // TODO may need to fix statePost initialization
      randn(kalmanFilter_.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    }
    cv::Mat frame, fgMask;
    cv::Rect rect_measured;
    std::vector<std::vector<cv::Point>> maxAreaContours;
    cam.GetCurrentFrame(frame);
    while (true) {
      cam.GetCurrentFrame(frame);
      if (frame.empty())
        break;
      cv::Mat frame_orig = frame.clone();
      if(cv::theRNG().uniform(0,4) != 0) {
        FilterAndErode(frame);
        pBackSub->apply(frame, fgMask, 0.99);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        //std::vector<cv::Point2d> filteredCentroids;
        //std::vector<cv::Point> approxCurve;

        if (!contours.empty()) {
          std::sort(contours.begin(), contours.end(), [](auto &lhs, auto &rhs) {
            return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs)));
          });

          if (trackingParams.filterParams.enableConvexHull) {
            cv::Mat hullPoints;
            cv::convexHull(contours[0], hullPoints);
            maxAreaContours.push_back(hullPoints);
          } else {
            maxAreaContours.push_back(contours[0]);
          }
          auto all_moments = cv::moments(maxAreaContours[0], true);
          measurement_.at<double>(0,0) = (all_moments.m10 / all_moments.m00);
          measurement_.at<double>(1,0) = (all_moments.m01 / all_moments.m00);

          cv::Rect rect_detect = cv::boundingRect(maxAreaContours[0]);
          rect_detect.height = 50;
          rect_detect.width  = 50;
          cv::rectangle(frame_orig, rect_detect, cv::Scalar(0, 255, 0), 3);
          rect_measured = rect_detect;
          //measurement_ = kalmanFilter_.measurementMatrix*kalmanState_;
          std::cout << "measurement = " << measurement_ << std::endl;
          if(!measurement_.empty())
            kalmanFilter_.correct(measurement_);
          cv::rectangle(frame_orig, rect_measured, cv::Scalar(0,0,255), 3);
          //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
        }
      } else {
        kalmanState_ = kalmanFilter_.predict();
        cv::Rect rect_tracking(kalmanState_.at<int>(0), kalmanState_.at<int>(1), 50, 50);
        cv::rectangle(frame_orig, rect_measured, cv::Scalar(255,0,0), 3);
      }

      //cv::rectangle(frame_orig, rect_measured, cv::Scalar(0,0,255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
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
