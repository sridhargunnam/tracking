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
// TODO Refactor the code to cleanup, to reorg data structs
// TODO Create data structures that tracks all the filtered Contours. - How to model occlusions? missed detecting object in a certain frames.
// TODO Determine residual error, for each each of the contours.
// TODO Lighting affects foreground noise, enhance it using hist equalization(didn't work), AGC(need to try).
// Depth map hole filling - https://github.com/juniorxsound/ThreadedDepthCleaner


int main()
{
    std::cout << "Testing Tracking\n";
    TrackingParams trackingParams;
    Tracking tracking{trackingParams};
    return 0;
}

