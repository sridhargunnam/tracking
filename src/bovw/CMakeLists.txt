# Generic test that uses conan libs

add_library(libTracking ../tracking/tracking.cpp ../tracking/tracking.h)
target_link_libraries(libTracking
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})

# Generic test that uses conan libs
add_executable(testTracking ../tracking/testTracking.cpp)
target_include_directories(testTracking PRIVATE )
target_link_libraries(testTracking
        PRIVATE
        libTracking)

#add_executable(testTracking2 ../tracking/testTracking.cpp ../tracking/tracking.cpp ../tracking/tracking.h)
#target_include_directories(testTracking2 PRIVATE )
#target_link_libraries(testTracking2
#        PRIVATE
#        libTracking)