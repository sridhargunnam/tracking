//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"
// https://www.udacity.com/wiki/cs373/kalman-filter-matrices
// VECTOR X, the State variables ( n = 2, contour's centroid position(x,y), depth(z), velocity)
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

// TODO Go through the kalman filter implementation in openCV
// TODO Visualize the Matrix value changes as well
// TODO Understand Kalman filter i.e measurement, predict, correct steps. Set up kalman filter based on the opencv example - https://docs.opencv.org/4.5.0/de/d70/samples_2cpp_2kalman_8cpp-example.html#a23
// TODO Create data structures that tracks all the filtered Contours. - How to model occlusions? missed detecting object in a certain frames.
// TODO Determine velocity, and define residual error, for each each of the contours.
// TODO Lighting affects foreground noise, enhance it using hist equalization(didn't work), AGC(need to try).
// Depth map hole filling - https://github.com/juniorxsound/ThreadedDepthCleaner

//void GetContourDepth(std::vector<std::vector<cv::Point2d>>, ){
//
//}

int main()
{
    std::cout << "Testing Tracking\n";
    TrackingParams trackingParams;
    Tracking tracking{trackingParams};
    return 0;
}

