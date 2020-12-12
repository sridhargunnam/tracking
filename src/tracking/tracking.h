//
// Created by sgunnam on 10/4/20.
//

#ifndef TRACKING_TRACKING_H
#define TRACKING_TRACKING_H
#include "SensorModule.h"

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/rgbd.hpp>
#include<opencv2/highgui/highgui_c.h>

#include <librealsense2/rs.hpp>

#include <iostream>
#include <random>



// Parameters that effect the result
// size of the object to be tracked
enum class DetectionType { COLOR_CONTOUR, DEPTH_AND_COLOR_CONTOUR };

struct ConfigurationParams{
  DetectionType detection_type_;
  CameraType camera_type_;
};

class Tracking
{
  ConfigurationParams configuration_params_;
public:
  explicit Tracking(ConfigurationParams);
private:

  void FilterAndErode(cv::Mat& im) const;
  void CleanDepthMap(cv::Mat& im) const;

};


#endif//TRACKING_TRACKING_H
