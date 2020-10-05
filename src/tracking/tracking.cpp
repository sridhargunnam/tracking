//
// Created by sgunnam on 10/4/20.
//

#include "tracking.h"

void Tracking::ReadVideo() const{
  cv::VideoCapture cap(trackingParams_.video_file_path);
  if(!cap.isOpened()){
    std::cout << "Error opening the file " << trackingParams_.video_file_path << "\n";
    return;
  }

  cv::namedWindow("Tracking window",cv::WINDOW_AUTOSIZE);
  while(1) {
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