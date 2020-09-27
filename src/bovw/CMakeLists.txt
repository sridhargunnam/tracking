# Generic test that uses conan libs

add_library(bowStats Stats.cpp Stats.h)
target_link_libraries(bowStats
        PRIVATE project_options
        project_warnings
        ${OpenCV_LIBS})

add_library(libBoVW BoVW.cpp BoVW.h showMultipleImages.h)
target_link_libraries(libBoVW
        PRIVATE bowStats
        project_options
        project_warnings
        ${OpenCV_LIBS}
        gtest
        gmock)

# Generic test that uses conan libs
add_executable(testBoVW
        testBoVW.cpp)
target_include_directories(testBoVW PRIVATE experimental)
target_link_libraries(testBoVW
        PRIVATE
        libBoVW)