//
// Created by srigun on 11/23/20.
//

#ifndef TRACKING_SRC_TRACKING_SENSORMODULE_H_
#define TRACKING_SRC_TRACKING_SENSORMODULE_H_

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI


#include <librealsense2/rs.hpp>
enum class CameraType {REALSENSE_VISION, REALSENSE_DEPTH, REALSENSE_VISION_AND_DEPTH, LAPTOP};

class SensorModule
{
  
};


class CameraModule
{
  CameraType camera_type_;
  rs2::colorizer color_map;
  rs2::pipeline pipe;
  cv::VideoCapture video_capture_;
public:
  explicit CameraModule(CameraType camera_type) : camera_type_(camera_type)
  {
    switch (camera_type) {
    case CameraType::LAPTOP: {
      cv::Mat image;
      // Realsense camera(435i), RGB camera id is 4 when used from open cv, whereas on a laptop this ID would be 0
      video_capture_ = cv::VideoCapture(4);
      //video_capture_.open(0,cv::CAP_V4L);
      break;
    }
    case CameraType::REALSENSE_VISION: {
      rs2::config cfg;
      cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
      pipe.start(cfg);
      break;
    }

    case CameraType::REALSENSE_DEPTH: {
      rs2::config cfg;
      cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
      pipe.start(cfg);
      break;
    }

    case CameraType::REALSENSE_VISION_AND_DEPTH: {
      rs2::config cfg;
      cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
      cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
      pipe.start(cfg);
      break;
    }
    }
  }

  ~CameraModule()
  {
    if(camera_type_ == CameraType::LAPTOP){
      video_capture_.release();
    } else {
      pipe.stop();
    }
  }

  void GetCurrentFrame(cv::Mat &image)
  {
    if(camera_type_ == CameraType::LAPTOP) {
      video_capture_ >> image;
    } else {
      rs2::frameset data = pipe.wait_for_frames();// Wait for next set of frames from the camera
      rs2::frame color_frame = data.get_color_frame();

      const int w = color_frame.as<rs2::video_frame>().get_width();
      const int h = color_frame.as<rs2::video_frame>().get_height();
      image = cv::Mat(cv::Size(w, h), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
    }
  }

  void GetDepthFrame(cv::Mat &image){
    rs2::frameset data = pipe.wait_for_frames();
    rs2::frame depth_frame = data.get_depth_frame();

    const int w = depth_frame.as<rs2::video_frame>().get_width();
    const int h = depth_frame.as<rs2::video_frame>().get_height();

    image = cv::Mat(cv::Size(w,h), CV_16U, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
    //cv::rgbd::DepthCleaner* depthc = new cv::rgbd::DepthCleaner(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);
    //cv::Mat cleanedDepth(cv::Size(w, h), CV_16U);
    //depthc->operator()(image, cleanedDepth);
  }
};

#endif//TRACKING_SRC_TRACKING_SENSORMODULE_H_
