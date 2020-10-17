//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

int main()
{
  //
  // Declare depth colorizer for pretty visualization of depth data
  rs2::colorizer color_map;

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;
  // Start streaming with default recommended configuration
  pipe.start();

  using namespace cv;
  int i=0;
  /*
  cv::namedWindow("Background substraction", WINDOW_AUTOSIZE);
  while (i<=2) {
    rs2::frameset data = pipe.wait_for_frames();// Wait for next set of frames from the camera
    rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
    rs2::frame color_frame = data.get_color_frame();

    // Query frame size (width and height)
    const int w = color_frame.as<rs2::video_frame>().get_width();
    const int h = color_frame.as<rs2::video_frame>().get_height();

    // Create OpenCV matrix of size (w,h) from the colorized depth data
    Mat image(Size(w, h), CV_8UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);
    std::string filename = "/home/sgunnam/CLionProjects/tracking/data/background_substraction/inputs/im";
    filename += std::to_string(i) + ".png";
    cv::imwrite(filename, image);
    imshow("Background substraction", image);
    waitKey(1000);
    i++;
  }
  destroyAllWindows();
  */

  //Algorithm
  // 1a) Smooth the image
  // 1b) Do edge detection
  // 1b) Get rid weak edges, fill edges
  // 2) Difference and threshold
  std::string filename0 = "/home/sgunnam/CLionProjects/tracking/data/background_substraction/inputs/realsense_viewer/im0_Color.png";
  std::string filename1 = "/home/sgunnam/CLionProjects/tracking/data/background_substraction/inputs/realsense_viewer/im1_Color.png";
  cv::Mat im0 = cv::imread(filename0);
  cv::Mat im1 = cv::imread(filename1);
  {
    Mat prvs;
    cvtColor(im0, prvs, COLOR_BGR2GRAY);
    while(true){
      Mat next;
      if (im1.empty())
        break;
      cvtColor(im1, next, COLOR_BGR2GRAY);
      Mat flow(prvs.size(), CV_32FC2);
      calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      // visualization
      Mat flow_parts[2];
      split(flow, flow_parts);
      Mat magnitude, angle, magn_norm;
      cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
      normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
      angle *= ((1.f / 360.f) * (180.f / 255.f));
      //build hsv image
      Mat _hsv[3], hsv, hsv8, bgr;
      _hsv[0] = angle;
      _hsv[1] = Mat::ones(angle.size(), CV_32F);
      _hsv[2] = magn_norm;
      merge(_hsv, 3, hsv);
      hsv.convertTo(hsv8, CV_8U, 255.0);
      cvtColor(hsv8, bgr, COLOR_HSV2BGR);
      imshow("frame2", bgr);
      int keyboard = waitKey(30000);
    }
  }
  /*
  {
    cv::Mat im0gray, im1gray;
    cv::cvtColor(im0, im0gray, COLOR_BGR2GRAY);
    cv::cvtColor(im1, im1gray, COLOR_BGR2GRAY);
    cv::Mat im0blur, im1blur;
    int gaussian_kernel_width = 25;
    cv::Size gaussian_kernel = cv::Size(gaussian_kernel_width, gaussian_kernel_width);
    cv::GaussianBlur(im1gray, im1blur, gaussian_kernel, 0, 0);
    cv::GaussianBlur(im0gray, im0blur, gaussian_kernel, 0, 0);

    //  cv::Mat im0Thres, im1Thres;
    //  cv::threshold(im0gray, im0Thres, 127, 255, 0);
    //  cv::threshold(im1gray, im1Thres, 127, 255, 0);

    //  mask = cv2.erode(mask, None, iterations=2)
    //  mask = cv2.dilate(mask, None, iterations=2)
    //cv::erode(im0blur, im0blur, cv::Mat() , cv::Point (-1, -1),5 , 1, 1);
    cv::dilate(im1blur, im0blur, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
    cv::imshow("erode", im0blur);
    waitKey(30000);
    cv::Mat im0edge, im1edge;
    double th0 = 5;
    double th1 = 30;

    Canny(im0blur, im0edge, th0, th1, 3, false);
    Canny(im1blur, im1edge, th0, th1, 3, false);

    cv::imshow("edges", im0edge);
    waitKey(30000);
    std::vector<std::vector<cv::Point>> im0Contours, im1Contours;
    cv::findContours(im0edge, im0Contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(im0, im0Contours, -1, cv::Scalar(0, 0, 255), 3);
    cv::imshow("Countours", im0);
    waitKey(30000);
  }
  */
//  std::cout << "Testing Tracking\n";
//  TrackingParams trackingParams;
//  Tracking tracking{ trackingParams };
  return 0;
}