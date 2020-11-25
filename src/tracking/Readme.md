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

// Depth map hole filling - https://github.com/juniorxsound/ThreadedDepthCleaner
