//
// Created by sgunnam on 10/4/20.
//
#include "tracking.h"

// TODO Create data structures that tracks all the filtered Contours.
// TODO Determine velocity, and define residual error, for each each of the contours.
// TODO Lighting affects foreground noise, enhance it using hist equalization(didn't work), AGC(need to try).
int main()
{
    std::cout << "Testing Tracking\n";
    TrackingParams trackingParams;
    Tracking tracking{trackingParams};
    return 0;
}
