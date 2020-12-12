// Requirements
// Should detect still objects in the foreground, and as they move it should track. If the object stops moving, still keep track of it.
// Nice to have: Occlusions, track it for up a threshold time limit, default = 2 sec

// TODO Refactor the code to cleanup, to reorg data structs
//      - Create data structures that tracks all the filtered Contours. - How to model occlusions? missed detecting object in a certain frames.
// TODO Add the depth sensor to kalman filter
//      - Depth cleaner is not doing it's job. Not filling holes. Therefore to reduce noise we increased maxAreaContour(Hack) to suppress the noise.
//      - Depth cleaner should not use background subtractor.
//          - Foreground will have smaller values than the background. So thresholding needs to be better. But it has lot of noise now.
//          - Because for small/no movements background subtractor will ignore the static parts.
//      - Visual to depth capture latency

// TODO Refactoring notes
// - Cleanup tracking.cpp, complete KalmanWrapper
// - Try registering sensor module with kalman?

#include "tracking.h"

#include "tracking_helper.h"

int main()
{
    //runDepthCleaner();
    std::cout << "Testing Tracking\n";
    Tracking tracking{ {DetectionType::DEPTH_AND_COLOR_CONTOUR, CameraType::REALSENSE_VISION_AND_DEPTH}};
    return 0;
}
