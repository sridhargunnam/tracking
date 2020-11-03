# Generic test that uses conan libs
set(RS2_DEPENDENCIES realsense2)
add_library(libTracking tracking.cpp tracking.h ShowMultipleImages.h)
target_link_libraries(libTracking
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS}
        ${RS2_DEPENDENCIES})


# Generic test that uses conan libs
add_executable(testTracking testTracking.cpp)
target_include_directories(testTracking PRIVATE )
target_link_libraries(testTracking
        PRIVATE
        libTracking
        -lpthread)

#add_executable(testTracking2 testTracking.cpp tracking.cpp tracking.h)
#target_include_directories(testTracking2 PRIVATE )
#target_link_libraries(testTracking2
#        PRIVATE
#        libTracking)


# Generic test that uses conan libs
