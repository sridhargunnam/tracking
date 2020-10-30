//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"

// https://www.udacity.com/wiki/cs373/kalman-filter-matrices
X = 1x2, Z = 1x1, H = 2x1, P = 2x2, R = 1x1,
// TODO Understand Kalman filter i.e measurement, predict, correct steps. Set up kalman filter based on the opencv example - https://docs.opencv.org/4.5.0/de/d70/samples_2cpp_2kalman_8cpp-example.html#a23
// TODO Create data structures that tracks all the filtered Contours. - How to model occlusions? missed detecting object in a certain frames.
// TODO Determine velocity, and define residual error, for each each of the contours.
// TODO Lighting affects foreground noise, enhance it using hist equalization(didn't work), AGC(need to try).
int main()
{
    std::cout << "Testing Tracking\n";
    TrackingParams trackingParams;
    Tracking tracking{trackingParams};
    return 0;
}
