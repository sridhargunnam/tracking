//
// Created by sgunnam on 10/4/20.
//

#include "tracking.h"

Tracking::Tracking(DetectionType detection_type) : detection_type_(detection_type)
{
  switch (detection_type) {
  case DetectionType::DEPTH_AND_COLOR_CONTOUR: {

    auto cam = CameraModule(CameraType::REALSENSE_VISION_AND_DEPTH);
    //Create a depth cleaner instance
    cv::rgbd::DepthCleaner depthc(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

    cv::Ptr<cv::BackgroundSubtractor> pBackSub{ cv::createBackgroundSubtractorKNN(1, 100.0, true) };
    //cv::Ptr<cv::BackgroundSubtractor> pBackSubD{ cv::createBackgroundSubtractorKNN(1, 100.0, true) };
    cv::Ptr<cv::BackgroundSubtractor> pBackSubD{ cv::createBackgroundSubtractorMOG2(1, 100.0, false) };

    cv::KalmanFilter kalmanFilter_(4, 2, 0);
    cv::Mat kalmanState_(2, 1, CV_32F);
    cv::Mat processNoise_(2, 1, CV_32F);
    cv::Mat measurement_ = cv::Mat::zeros(2, 1, CV_32F);
    cv::Mat measurementPrev_ = cv::Mat::zeros(2, 1, CV_32F);


    // TODO refactor Kalman related states
    cv::KalmanFilter kalmanFilter_D(6, 3, 0);
    cv::Mat kalmanState_D(3, 1, CV_32F);
    cv::Mat processNoise_D(3, 1, CV_32F);
    cv::Mat measurement_D = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat measurementPrev_D = cv::Mat::zeros(3, 1, CV_32F);

    {
      // Vision
      {
        kalmanFilter_.statePre.at<double>(0) = 320;
        kalmanFilter_.statePre.at<double>(1) = 240;
        kalmanFilter_.statePre.at<double>(2) = 0;
        kalmanFilter_.statePre.at<double>(3) = 0;
      }
      setIdentity(kalmanFilter_.measurementMatrix);
      kalmanFilter_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
      // processNoise needs empirical estimation, it can't be just some random value. eg. for 1e-5 , ti stops tracking when there is no motion in the scene.
      setIdentity(kalmanFilter_.processNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_.measurementNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_.errorCovPost, cv::Scalar::all(1));

      // Depth
      {
        kalmanFilter_D.statePre.at<double>(0) = 320;
        kalmanFilter_D.statePre.at<double>(1) = 240;
        kalmanFilter_D.statePre.at<double>(2) = 500;
        kalmanFilter_D.statePre.at<double>(3) = 0;
        kalmanFilter_D.statePre.at<double>(4) = 0;
        kalmanFilter_D.statePre.at<double>(5) = 0;
      }
      //randn( kalmanState_, cv::Scalar::all(0), cv::Scalar::all(0.1) );
      // x  y  z  dx dy dz
      // 1  0  0  1  0  0
      // 0  1  0  0  1  0
      // 0  0  1  0  0  1
      // 0  0  0  1  0  0
      // 0  0  0  0  1  0
      // 0  0  0  0  0  1

      // States
      // x y z dx dx dz
      setIdentity(kalmanFilter_D.measurementMatrix);
      kalmanFilter_D.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
      // processNoise needs empirical estimation, it can't be just some random value. eg. for 1e-5 , ti stops tracking when there is no motion in the scene.
      setIdentity(kalmanFilter_D.processNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_D.measurementNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_D.errorCovPost, cv::Scalar::all(1));
      // TODO may need to fix statePost initialization
      //randn(kalmanFilter_.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    }
    cv::Mat frame, fgMask, frame_orig;
    cv::Rect rect_measured;
    std::vector<std::vector<cv::Point>> maxAreaContours;
    cv::Rect rect_tracking;
    cam.GetCurrentFrame(frame);
    int cnt = 0;
    bool run_once = true;

    // depth related
    cv::Mat frameD, fgMaskD, frame_origD;
    cv::Mat im8u;
    cv::Rect rect_measuredD;
    std::vector<std::vector<cv::Point>> maxAreaContoursD;
    cv::Rect rect_trackingD;
    cam.GetDepthFrame(frameD);
    std::vector<std::vector<cv::Point>> contoursD;

    while (true) {
      std::cout << "At index " << cnt << " : -------------------------------------------------------------- " << std::endl;
      ++cnt;
      maxAreaContours.clear();
      maxAreaContoursD.clear();
      contoursD.clear();
      //rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
      //rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
      cam.GetCurrentFrame(frame);
      cam.GetDepthFrame(frameD);
      if (frame.empty() || frameD.empty()) {
        std::cout << "input frame is empty\n";
        break;
      }
      frame_orig = frame.clone();
      frame_origD = frameD.clone();
      frameD.convertTo(im8u, CV_8UC1, 0.1);

      cv::Rect rect_detect;
      cv::Rect rect_detectD;

      if (run_once || (cv::theRNG().uniform(0, 100) < 100)) {
        run_once = false;
        // Color frame processing
        FilterAndErode(frame);
        pBackSub->apply(frame, fgMask, 0.99);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        // Depth frame processing
        CleanDepthMap(frameD);
        pBackSubD->apply(frameD, fgMaskD, 0.99);
        cv::findContours(fgMaskD, contoursD, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        if (!contoursD.empty() && !contours.empty()) {
          std::sort(contoursD.begin(), contoursD.end(), [](auto &lhs, auto &rhs) {
            return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs)));
          });
          std::sort(contours.begin(), contours.end(), [](auto &lhs, auto &rhs) {
            return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs)));
          });

          maxAreaContours.push_back(contours[0]);
          maxAreaContoursD.push_back(contoursD[0]);
          //std::cout << " cv::contourArea(maxAreaContours[0]) " << cv::contourArea(maxAreaContours[0]) << "\n";
          //std::cout << " cv::contourArea(maxAreaContoursD[0]) " << cv::contourArea(maxAreaContoursD[0]) << "\n";

          if ((fabs(cv::contourArea(maxAreaContours[0])) > 100) && (fabs(cv::contourArea(maxAreaContoursD[0])) > 1000)) {

            auto all_moments = cv::moments(maxAreaContours[0], true);
            measurement_.at<float>(0, 0) = static_cast<float>((all_moments.m10 / all_moments.m00));
            measurement_.at<float>(1, 0) = static_cast<float>((all_moments.m01 / all_moments.m00));
            rect_detect = cv::boundingRect(maxAreaContours[0]);
            //std::cout << "Vision all_moments.m00= " << all_moments.m00 << " ;";
            measurement_.copyTo(measurementPrev_);
            //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);

            auto all_momentsD = cv::moments(maxAreaContoursD[0], true);
            measurement_D.at<float>(0, 0) = static_cast<float>((all_momentsD.m10 / all_momentsD.m00));
            measurement_D.at<float>(1, 0) = static_cast<float>((all_momentsD.m01 / all_momentsD.m00));
            int xpos = static_cast<int>(measurement_D.at<float>(0, 0));
            int ypos = static_cast<int>(measurement_D.at<float>(1, 0));
            measurement_D.at<float>(2, 0) = static_cast<float>(frame_origD.at<int>(xpos, ypos));
            rect_detectD = cv::boundingRect(maxAreaContoursD[0]);
            measurement_D.copyTo(measurementPrev_D);
            std::cout << "Vision measurement  x = " << measurement_.at<float>(0, 0) << " ";
            std::cout << "Vision measurement  y = " << measurement_.at<float>(1, 0) << std::endl;
            std::cout << "Depth  measurementD x = " << measurement_D.at<float>(0, 0) << " ";
            std::cout << "Depth  measurementD y = " << measurement_D.at<float>(1, 0) << std::endl;
            std::cout << "Difference          x = " << measurement_D.at<float>(0, 0) - measurement_.at<float>(0, 0) << " ";
            std::cout << "Difference          y = " << measurement_D.at<float>(1, 0) - measurement_.at<float>(1, 0) << std::endl;
          } else {
            //std::cout << "No motion: kalman state = " << kalmanState_ << std::endl;
            measurementPrev_.copyTo(measurement_);
            measurementPrev_D.copyTo(measurement_D);
          }
        }
        cv::drawContours(frame_orig, maxAreaContours, 0, cv::Scalar(255, 0, 0), 3);
        cv::drawContours(im8u, maxAreaContoursD, 0, cv::Scalar(255, 0, 0), 3);
        //std::cout << "measure error point = " << measurement_ << std::endl;

        kalmanFilter_.correct(measurement_);
        kalmanState_ = kalmanFilter_.predict();

        kalmanFilter_D.correct(measurement_D);
        kalmanState_D = kalmanFilter_D.predict();
        //rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, rect_detect.width, rect_detect.height);
        cv::rectangle(frame_orig, rect_detect, cv::Scalar(0, 255, 0), 3);
        cv::rectangle(im8u, rect_detectD, cv::Scalar(0, 255, 0), 3);
        //std::cout << "kalman state = " << kalmanState_ << std::endl;
      } else {
        kalmanState_ = kalmanFilter_.predict();
        kalmanState_D = kalmanFilter_D.predict();
        //rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, 100, 100);
        //      cv::rectangle(frame_orig, rect_tracking, cv::Scalar(0, 0, 255), 3);
      }

      rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, 100, 100);
      cv::rectangle(frame_orig, rect_tracking, cv::Scalar(0, 0, 255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //cv::rectangle(frame_orig, rect_measured, cv::Scalar(0,0,255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //show the current frame and the fg masks
      imshow("Vision Frame", frame);
      imshow("Vision FG Mask", fgMask);
      imshow("Vision Contours", frame_orig);


      rect_trackingD = cv::Rect(static_cast<int>(kalmanState_D.at<float>(0)) - rect_detectD.width / 2, static_cast<int>(kalmanState_D.at<float>(1)) - rect_detectD.height / 2, 100, 100);
      cv::rectangle(im8u, rect_trackingD, cv::Scalar(0, 0, 255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //cv::rectangle(frame_orig, rect_measured, cv::Scalar(0,0,255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //show the current frame and the fg masks
      imshow("Depth Frame", frameD);
      imshow("Depth fgMaskD", fgMaskD);
      imshow("Depth Contours", im8u);

      //get the input from the keyboard
      int keyboard = cv::waitKey(30);
      if (keyboard == 'q' || keyboard == 27)
        break;
    }
    break;
  }

  case DetectionType::COLOR_CONTOUR: {
    auto cam = CameraModule(CameraType::REALSENSE_VISION);

    //    while(true){
    //      cv::Mat depth_frame;
    //      cam.GetDepthFrame(depth_frame);
    //      cv::imshow("depth_frame", depth_frame);
    //      cv::waitKey(0);
    //    }

    cv::Ptr<cv::BackgroundSubtractor> pBackSub{ cv::createBackgroundSubtractorKNN(1, 100.0, true) };

    // TODO refactor Kalman related states
    cv::KalmanFilter kalmanFilter_(4, 2, 0);
    cv::Mat kalmanState_(2, 1, CV_32F);
    cv::Mat processNoise_(2, 1, CV_32F);
    cv::Mat measurement_ = cv::Mat::zeros(2, 1, CV_32F);
    cv::Mat measurementPrev_ = cv::Mat::zeros(2, 1, CV_32F);
    {
      {
        kalmanFilter_.statePre.at<double>(0) = 320;
        kalmanFilter_.statePre.at<double>(1) = 240;
        kalmanFilter_.statePre.at<double>(2) = 0;
        kalmanFilter_.statePre.at<double>(3) = 0;
      }
      //randn( kalmanState_, cv::Scalar::all(0), cv::Scalar::all(0.1) );
      // x y dx dy
      // 1 0 1 0
      // 0 1 0 1
      // 0 0 1 0
      // 0 0 0 1
      setIdentity(kalmanFilter_.measurementMatrix);
      kalmanFilter_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
      // processNoise needs empirical estimation, it can't be just some random value. eg. for 1e-5 , ti stops tracking when there is no motion in the scene.
      setIdentity(kalmanFilter_.processNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_.measurementNoiseCov, cv::Scalar::all(1e-3));
      setIdentity(kalmanFilter_.errorCovPost, cv::Scalar::all(1));
      // TODO may need to fix statePost initialization
      //randn(kalmanFilter_.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    }
    cv::Mat frame, fgMask, frame_orig;
    cv::Rect rect_measured;
    std::vector<std::vector<cv::Point>> maxAreaContours;
    cam.GetCurrentFrame(frame);
    int cnt = 0;
    bool run_once = true;

    cv::Rect rect_tracking;
    while (true) {
      std::cout << "At index " << cnt << " : " << std::endl;
      ++cnt;
      maxAreaContours.clear();
      cam.GetCurrentFrame(frame);
      if (frame.empty())
        break;
      frame_orig = frame.clone();
      cv::Rect rect_detect;

      if (run_once || (cv::theRNG().uniform(0, 100) < 100)) {
        run_once = false;
        FilterAndErode(frame);
        pBackSub->apply(frame, fgMask, 0.99);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
          std::sort(contours.begin(), contours.end(), [](auto &lhs, auto &rhs) {
            return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs)));
          });


          maxAreaContours.push_back(contours[0]);

          if (fabs(cv::contourArea(maxAreaContours[0])) > 100) {
            auto all_moments = cv::moments(maxAreaContours[0], true);
            measurement_.at<float>(0, 0) = static_cast<float>((all_moments.m10 / all_moments.m00));
            measurement_.at<float>(1, 0) = static_cast<float>((all_moments.m01 / all_moments.m00));
            rect_detect = cv::boundingRect(maxAreaContours[0]);
            //rect_detect.height = 50;
            //rect_detect.width  = 50;
            //cv::rectangle(frame_orig, rect_detect, cv::Scalar(0, 255, 0), 3);
            //measurement_ = kalmanFilter_.measurementMatrix*kalmanState_;
            std::cout << "all_moments.m00= " << all_moments.m00 << " ;";
            std::cout << "measurement 00 = " << measurement_.at<float>(0, 0) << " ";
            std::cout << "measurement 01 = " << measurement_.at<float>(0, 1) << std::endl;
            measurement_.copyTo(measurementPrev_);
            //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
          } else {
            std::cout << "No motion: kalman state = " << kalmanState_ << std::endl;
            measurementPrev_.copyTo(measurement_);
          }
        }
        kalmanFilter_.correct(measurement_);
        kalmanState_ = kalmanFilter_.predict();
        //rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, rect_detect.width, rect_detect.height);
        cv::rectangle(frame_orig, rect_detect, cv::Scalar(0, 255, 0), 3);
        std::cout << "kalman state = " << kalmanState_ << std::endl;
      } else {
        kalmanState_ = kalmanFilter_.predict();
        //rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, 100, 100);
        //      cv::rectangle(frame_orig, rect_tracking, cv::Scalar(0, 0, 255), 3);
      }
      rect_tracking = cv::Rect(static_cast<int>(kalmanState_.at<float>(0)) - rect_detect.width / 2, static_cast<int>(kalmanState_.at<float>(1)) - rect_detect.height / 2, 100, 100);
      cv::rectangle(frame_orig, rect_tracking, cv::Scalar(0, 0, 255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //cv::rectangle(frame_orig, rect_measured, cv::Scalar(0,0,255), 3);
      //cv::drawContours(frame_orig, maxAreaContours, -1, cv::Scalar(255, 0, 0), 3);
      //show the current frame and the fg masks
      imshow("Frame", frame);
      imshow("FG Mask", fgMask);
      imshow("Contours", frame_orig);

      cv::Mat depth_frame;
      cam.GetDepthFrame(depth_frame);
      cv::imshow("depth_frame", depth_frame);
      cv::waitKey(0);

      //get the input from the keyboard
      int keyboard = cv::waitKey(30);
      if (keyboard == 'q' || keyboard == 27)
        break;
    }
    break;
  }
  }
}


// Based on opencv squares.cpp sample
void Tracking::FilterAndErode(cv::Mat &img) const
{
  cv::Size gaussian_kernel = cv::Size(25,25);
  cv::GaussianBlur(img, img, gaussian_kernel, 0, 0);
  cv::dilate(img, img, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
}

// Clean depth, fill holes
void Tracking::CleanDepthMap(cv::Mat &img) const
{
  cv::rgbd::DepthCleaner depthc(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);
  const int w = img.rows;
  const int h = img.cols;
  cv::Mat cleanedDepth(cv::Size(w, h), CV_16U);
  depthc.operator()(img, cleanedDepth);
  cv::Mat cleanedDepth8U(cv::Size(w, h), CV_8UC1);
  cleanedDepth.convertTo(cleanedDepth8U, CV_8UC1);
  //int kernel_size = 7;
  //cv::Size gaussian_kernel = cv::Size(kernel_size, kernel_size);
  //cv::GaussianBlur(cleanedDepth8U, cleanedDepth8U, gaussian_kernel, 0, 0);
  cv::adaptiveThreshold(cleanedDepth8U, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 7, 0 );
  //cv::dilate(img, img, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
  //cv::erode(img,img, cv::Mat(),cv::Point(-1,-1));
  //cv::Mat kernel_mat = cv::Mat::ones(30,30, CV_8U);
  //cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel_mat);
}
