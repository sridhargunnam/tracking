//
// Created by sgunnam on 10/5/20.
//

#ifndef TRACKING_SHOWMULTIPLEIMAGES_H
#define TRACKING_SHOWMULTIPLEIMAGES_H
#include <vector>
#include <cstdio>
#include <cstdarg>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
// TODO get rid of different ShowMultipleImages files - do code reuse
void ShowMultipleImagesTracking(const std::string& title, std::vector<cv::Mat>& images_list){

  auto nImages = images_list.size();
  int size;
  int i;
  int m, n;
  int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
  int w, h;

// scale - How much we have to resize the image
  float scale;
  int max;

// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
  if(nImages <= 0) {
    printf("Number of arguments too small....\n");
    return;
  }
  else if(nImages > 14) {
    printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
    return;
  }
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
  else if (nImages == 1) {
    w = h = 1;
    size = 300;
  }
  else if (nImages == 2) {
    w = 2; h = 1;
    size = 300;
  }
  else if (nImages == 3 || nImages == 4) {
    w = 2; h = 2;
    size = 300;
  }
  else if (nImages == 5 || nImages == 6) {
    w = 3; h = 2;
    size = 200;
  }
  else if (nImages == 7 || nImages == 8) {
    w = 4; h = 2;
    size = 200;
  }
  else {
    w = 4; h = 3;
    size = 150;
  }

// Create a new 3 channel image
  Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);


// Loop for nImages number of arguments
  for (i = 0, m = 20, n = 20; i <  static_cast<int>(nImages); i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    Mat img = images_list.at( static_cast<unsigned long>(i));

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if(img.empty()) {
      printf("Invalid arguments");
      return;
    }

    // Find the width and height of the image
    x = img.cols;
    y = img.rows;

    // Find whether height or width is greater in order to resize the image
    max = (x > y)? x: y;

    // Find the scaling factor to resize the image
    scale =  ( static_cast<float>( max) / static_cast<float>(size));

    // Used to Align the images
    if( i % w == 0 && m!= 20) {
      m = 20;
      n+= 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    Rect ROI(m, n,  static_cast<int>( static_cast<float>(x)/scale ),
             static_cast<int>( static_cast<float>(y)/scale ));
    Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
  }

// Create a new window, and show the Single Big Image
  namedWindow( title, 1 );
  imshow( title, DispImage);
  waitKey();
}
#endif//TRACKING_SHOWMULTIPLEIMAGES_H
