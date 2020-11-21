
//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"
// https://www.udacity.com/wiki/cs373/kalman-filter-matrices
// VECTOR X, the State variables ( n = 2, contour's centroid_measured position(x,y), depth(z), velocity)
// Size = n x 1 = 2 x 1,
// Matrix F, the update matrix
// Size = n x n = 2 x 2,
// Z, the measurement vector ( m = 1, contour position, in future we could use depth as well to make it x, y, z
// Z = 1x1,
// H is the extraction matrix
// Size = m x n = 1 x 2
// P, the covariance of X
// Size = n x n = 2 x 2,
// R = the covariance of Z
// Size = m x m = 1 x 1
// U, The move vector
// Size = n x 1 = 2 x 1

// Requirements
// Should detect still objects in the foreground, and as they move it should track. If the object stops moving, still keep track of it.
// Occlusions, track it for up a threshold time limit, default = 2 sec
// TODO Add the depth sensor to kalman filter
//  - Fix kalman filter. Viz contours from frameD.
//      - Work more scientifically
//      - Capture an image and step through getCleanedDepthMap step by step.
//      - Complete GetDepth frame code
//      - Move the runDepthCleaner logic to separate function in Tracking.cpp, just like FilterAndErode
//  - Add 3 measurements (x,y,z) from depth map to kalman measure state
// TODO Refactor the code to cleanup, to reorg data structs
// TODO Create data structures that tracks all the filtered Contours. - How to model occlusions? missed detecting object in a certain frames.
// TODO Determine residual error, for each each of the contours.
// TODO Lighting affects foreground noise, enhance it using hist equalization(didn't work), AGC(need to try).
// Depth map hole filling - https://github.com/juniorxsound/ThreadedDepthCleaner



#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI

// STD
#include <string>
#include <thread>
#include <atomic>
#include <queue>

void runDepthCleaner();

int main()
{
    //runDepthCleaner();
    std::cout << "Testing Tracking\n";
    TrackingParams trackingParams;
    Tracking tracking{trackingParams};
    return 0;
}

using namespace cv;

#define SCALE_FACTOR 1

/*
* Class for enqueuing and dequeuing cv::Mats efficiently
* Thanks to this awesome post by PKLab
* http://pklab.net/index.php?id=394&lang=EN
*/
class QueuedMat{
public:

  Mat img; // Standard cv::Mat

  QueuedMat(){}; // Default constructor

  // Destructor (called by queue::pop)
  ~QueuedMat(){
    img.release();
  };

  // Copy constructor (called by queue::push)
  QueuedMat(const QueuedMat& src){
    src.img.copyTo(img);
  };
};

/*
* Awesome method for visualizing the 16bit unsigned depth data using a histogram, slighly modified (:
* Thanks to @Catree from https://stackoverflow.com/questions/42356562/realsense-opencv-depth-image-too-dark
*/
void make_depth_histogram(const Mat &depth, Mat &normalized_depth, int coloringMethod) {
  normalized_depth = Mat(depth.size(), CV_8U);
  int width = depth.cols, height = depth.rows;

  static uint32_t histogram[0x10000];
  memset(histogram, 0, sizeof(histogram));

  for(int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      ++histogram[depth.at<ushort>(i,j)];
    }
  }

  for(int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i-1]; // Build a cumulative histogram for the indices in [1,0xFFFF]

  for(int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (uint16_t d = depth.at<ushort>(i,j)) {
        int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
        normalized_depth.at<uchar>(i,j) = static_cast<uchar>(f);
      } else {
        normalized_depth.at<uchar>(i,j) = 0;
      }
    }
  }

  // Apply the colormap:
  applyColorMap(normalized_depth, normalized_depth, coloringMethod);
}


void runDepthCleaner() {

  //Create a depth cleaner instance
  rgbd::DepthCleaner depthc(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;

  //Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;

  //Add desired streams to configuration
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

  // Start streaming with default recommended configuration
  pipe.start(cfg);

  // openCV window
  const auto window_name_source = "Source Depth";
  namedWindow(window_name_source, WINDOW_AUTOSIZE);

  const auto window_name_filter = "Filtered Depth";
  namedWindow(window_name_filter, WINDOW_AUTOSIZE);

  // Atomic boolean to allow thread safe way to stop the thread
  //std::atomic_bool stopped(false);

  // Declaring two concurrent queues that will be used to push and pop frames from different threads
  std::queue<QueuedMat> filteredQueue;
  std::queue<QueuedMat> originalQueue;

  // The threaded processing thread function
  //std::thread processing_thread([&]() {
         while (true){

           rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
           rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
           if (!depth_frame) // Should not happen but if the pipeline is configured differently
             return;       //  it might not provide depth and we don't want to crash

           //Save a reference
           rs2::frame filtered = depth_frame;

           // Query frame size (width and height)
           const int w = depth_frame.as<rs2::video_frame>().get_width();
           const int h = depth_frame.as<rs2::video_frame>().get_height();

//           //Create queued mat containers
//           QueuedMat depthQueueMat;
//           QueuedMat cleanDepthQueueMat;

           // Create an openCV matrix from the raw depth (CV_16U holds a matrix of 16bit unsigned ints)
           Mat rawDepthMat(Size(w, h), CV_16U, (void*)depth_frame.get_data());

           // Create an openCV matrix for the DepthCleaner instance to write the output to
           Mat cleanedDepth(Size(w, h), CV_16U);
           //Run the RGBD depth cleaner instance
           depthc.operator()(rawDepthMat, cleanedDepth);
//           cv::Mat fgMask;
//           cv::Ptr<cv::BackgroundSubtractor> pBackSub{ cv::createBackgroundSubtractorKNN(1, 100.0, true) };
//           cv::Size gaussian_kernel = cv::Size(25, 25);
//           cv::GaussianBlur(cleanedDepth, cleanedDepth, gaussian_kernel, 0, 0);
//           cv::dilate(cleanedDepth, cleanedDepth, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
//           cvtColor(cleanedDepth, fgMask, CV_ADAPTIVE_THRESH_GAUSSIAN_C, 0);
//           imshow("cleanedDepth depth BEFORE ", cleanedDepth);
//           waitKey(0);
//           pBackSub->apply(cleanedDepth, fgMask, 0.99);
//           imshow("fgmask depth", fgMask);
//           waitKey(0);


           Mat im8u;
//           std::cout << "Min = " << min << std::endl;
//           std::cout << "Max = " << max << std::endl;

           cleanedDepth.convertTo(im8u, CV_8UC1, 0.1);
           imshow("im8u  depth", im8u);
           waitKey(0);
           std::cout << "Image 8 bit depth values" << im8u << std::endl;


           std::cout << "  cleanedDepth.at(250, 300) " << cleanedDepth.at<unsigned short>(250, 300) << std::endl;
           std::cout << "Cleaned depth channels " << cleanedDepth.channels() << std::endl;
//           imshow("cleanedDepth",cleanedDepth);
//           waitKey(0);
           Mat cleanedDepth8U(Size(w, h), CV_8UC1);
           cleanedDepth.convertTo(cleanedDepth8U, CV_8UC1);
           cv::Mat cleanedDepthMask;
           cv::adaptiveThreshold(cleanedDepth8U, cleanedDepthMask, 500, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0 );
           imshow("cleanedDepthMask",cleanedDepthMask);
           waitKey(0);
           std::vector<std::vector<cv::Point>> contours;
           cv::findContours(cleanedDepthMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
           if (!contours.empty()) {
             std::sort(contours.begin(), contours.end(), [](auto &lhs, auto &rhs) {
                    return fabs(cv::contourArea(lhs) > fabs(cv::contourArea(rhs)));
             });
             cv::drawContours(im8u, contours, 0, cv::Scalar(255, 0, 0), 3);
             imshow("contoured depth", im8u);
             waitKey(0);
           }
           /*
           const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
           Mat temp, temp2;

           // Downsize for performance, use a smaller version of depth image (defined in the SCALE_FACTOR macro)
           Mat small_depthf;
           resize(cleanedDepth8U, small_depthf, Size(), SCALE_FACTOR, SCALE_FACTOR);

           // Inpaint only the masked "unknown" pixels
           inpaint(small_depthf, (small_depthf == noDepth), temp, 5.0, INPAINT_TELEA);

           // Upscale to original size and replace inpainted regions in original depth image
           resize(temp, temp2, cleanedDepth.size());
           temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal
           cv::Mat cleanedDepthViz;
           make_depth_histogram(cleanedDepth, cleanedDepthViz, COLORMAP_JET);

           imshow("cleanedDepthViz", cleanedDepthViz);
           waitKey(0);
*/
         }
//  });

  Mat filteredDequeuedMat(Size(1280, 720), CV_16UC1);
  Mat originalDequeuedMat(Size(1280, 720), CV_8UC3);

  //Main thread function
  while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source) && cvGetWindowHandle(window_name_filter)){

    //If the frame queue is not empty pull a frame out and clean the queue
    while(!originalQueue.empty()){
      originalQueue.front().img.copyTo(originalDequeuedMat);
      originalQueue.pop();
    }


    while(!filteredQueue.empty()){
      filteredQueue.front().img.copyTo(filteredDequeuedMat);
      filteredQueue.pop();
    }

    Mat coloredCleanedDepth;
    Mat coloredOriginalDepth;

    make_depth_histogram(filteredDequeuedMat, coloredCleanedDepth, COLORMAP_JET);
    make_depth_histogram(originalDequeuedMat, coloredOriginalDepth, COLORMAP_JET);

    imshow(window_name_filter, coloredCleanedDepth);
    imshow(window_name_source, coloredOriginalDepth);
  }

  // Signal the processing thread to stop, and join
//  stopped = true;
  //processing_thread.join();
}
