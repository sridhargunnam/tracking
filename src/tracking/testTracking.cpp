//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"
// TODO
// Morph the bg-sub mask image to get countors
// Then define state space - get centroid of countour, velocity, orientation(may need different tracking, measurement model)

//Assumptions
/*
 * 1) Only foreground moving
 * 2) Tracking foreground
 */
void getCurrentFrame(const  rs2::pipeline& pipe, rs2::colorizer& color_map, cv::Mat& image)
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


int main()
{
  //
  // Declare depth colorizer for pretty visualization of depth data
  rs2::colorizer color_map;

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;
  // Start streaming with default recommended configuration
  pipe.start();
  int i = 0;

  cv::namedWindow("Background substraction", cv::WINDOW_AUTOSIZE);

    using namespace cv;
    using namespace std;

  {
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
      pBackSub = createBackgroundSubtractorMOG2();
      //pBackSub = createBackgroundSubtractorKNN();

    Mat frame, fgMask;
    while (true) {
      getCurrentFrame( pipe,  color_map, frame);
      //capture >> frame;
      if (frame.empty())
        break;
      //update the background model
      pBackSub->apply(frame, fgMask);
      //get the frame number and write it on the current frame
      rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                cv::Scalar(255,255,255), -1);
      //stringstream ss;
      //ss << capture.get(CAP_PROP_POS_FRAMES);
      //string frameNumberString = ss.str();
//      putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
//              FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
      //show the current frame and the fg masks
      imshow("Frame", frame);
      imshow("FG Mask", fgMask);
      //get the input from the keyboard
      int keyboard = waitKey(30);
      if (keyboard == 'q' || keyboard == 27)
        break;
    }
    }

    // Optical flow - LK features tracking
    {
      // Create some random colors
      vector<Scalar> colors;
      RNG rng;
      for (int i = 0; i < 100; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
      }
      Mat old_frame, old_gray;
      vector<Point2f> p0, p1;
      // Take first frame and find corners in it
      getCurrentFrame( pipe,  color_map, old_frame);
      //capture >> old_frame;
      cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
      goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
      // Create a mask image for drawing purposes
      Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
      while (true) {
        Mat frame, frame_gray;
        getCurrentFrame( pipe,  color_map, frame);
        //capture >> frame;
        if (frame.empty())
          break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++) {
          // Select good points
          if (status[i] == 1) {
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask, p1[i], p0[i], colors[i], 2);
            circle(frame, p1[i], 5, colors[i], -1);
          }
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
          break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
      }
    }



  // Our initial approach - morphology based, didn't work yet
  /*
    {//Algorithm
      // 1a) Smooth the image
      // 1b) Do edge detection
      // 1b) Get rid weak edges, fill edges
      // 2) Difference and threshold
      std::string filename0 = "/home/sgunnam/CLionProjects/tracking/data/background_substraction/inputs/realsense_viewer/im0_Color.png";
      std::string filename1 = "/home/sgunnam/CLionProjects/tracking/data/background_substraction/inputs/realsense_viewer/im1_Color.png";
      cv::Mat im0 = cv::imread(filename0);
      cv::Mat im1 = cv::imread(filename1);

      cv::Mat prvs;
      cv::cvtColor(im0, prvs, COLOR_BGR2GRAY);
      cv::Mat next;
      cv::cvtColor(im1, next, COLOR_BGR2GRAY);
      cv::Mat flow(prvs.size(), CV_32FC2);
      cv::calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      // visualization
      cv::Mat flow_parts[2];
      cv::split(flow, flow_parts);
      cv::Mat magnitude, angle, magn_norm;
      cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
      cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
      angle *= ((1.f / 360.f) * (180.f / 255.f));
      //build hsv image
      cv::Mat _hsv[3], hsv, hsv8, bgr;
      _hsv[0] = angle;
      _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
      _hsv[2] = magn_norm;
      cv::merge(_hsv, 3, hsv);
      hsv.convertTo(hsv8, CV_8U, 255.0);
      cv::cvtColor(hsv8, bgr, COLOR_HSV2BGR);
      cv::imshow("frame2", bgr);
      waitKey(30000);

      cv::Mat imTh;
      bgr = bgr * 255;
      cv::erode(bgr, bgr, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
      cv::dilate(bgr, bgr, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
      bgr.convertTo(bgr, CV_8U, 255.0);
      cv::threshold(bgr, imTh, 200, 255, 0);
      cv::imshow("flow th", imTh);
      waitKey(30000);
    }
  */
  // Our approach 2 - morphology based
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

    cv::erode(im0blur, im0blur, cv::Mat() , cv::Point (-1, -1),5 , 1, 1);
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

