cmake_minimum_required(VERSION 3.10)

project(STDC-Seg_TensorRT)

set(CMAKE_BUILD_TYPE Release)

add_executable(STDC-Seg_TensorRT "STDC-Seg_TensorRT.cpp")

# Full path to where OpenCV .so files are
unset(PATH_TO_OPENCV_LIBS CACHE)
set(PATH_TO_OPENCV_LIBS "/usr/lib")

# PATH_TO_OPENCV_INCLUDES should be full path to dir containing opencv2 directory.
unset(PATH_TO_OPENCV_INCLUDES CACHE)
set(PATH_TO_OPENCV_INCLUDES "/home/integrity/OpenCv/x86_64_linux/include")

# Help CMake find cuda by giving it the path to the cuda compiler
unset(CMAKE_CUDA_COMPILER CACHE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")                

# Full path to TensorRT install dir                             
unset(PATH_TO_TENSORRT_INSTALL CACHE)
set(PATH_TO_TENSORRT_INSTALL "/usr/local/cuda/TensorRT-8.2.2.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.2.1/")

# Start fresh
set(LibrariesToLink "")
set(DirsToLookInForIncludes "")

# OpenCV
find_library(CV_VIDEOIO libopencv_videoio.so HINTS "${PATH_TO_OPENCV_LIBS}")
find_library(CV_IMGCODECS libopencv_imgcodecs.so HINTS "${PATH_TO_OPENCV_LIBS}")
find_library(CV_HIGHGUI libopencv_highgui.so HINTS "${PATH_TO_OPENCV_LIBS}")
find_library(CV_CORE libopencv_core.so HINTS "${PATH_TO_OPENCV_LIBS}")
find_library(CV_IMGPROC libopencv_imgproc.so HINTS "${PATH_TO_OPENCV_LIBS}")
find_library(CV_DNN libopencv_dnn.so HINTS "${PATH_TO_OPENCV_LIBS}")

set(LibrariesToLink ${LibrariesToLink}
                    ${CV_VIDEOIO}
                    ${CV_IMGCODECS}
                    ${CV_HIGHGUI}
                    ${CV_CORE}
                    ${CV_IMGPROC}
                    ${CV_DNN})

set(DirsToLookInForIncludes ${DirsToLookInForIncludes}
                            ${PATH_TO_OPENCV_INCLUDES})


# CUDA + TensorRT

# This call populates CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES and CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES, among others
enable_language(CUDA)

find_library(CUDART libcudart.so ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NVINFER libnvinfer.so HINTS "${PATH_TO_TENSORRT_INSTALL}/lib")

set(LibrariesToLink ${LibrariesToLink}
                    ${CUDART}
                    ${NVINFER})

set(DirsToLookInForIncludes ${DirsToLookInForIncludes} 
                            "${PATH_TO_TENSORRT_INSTALL}/include"
                            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Include logger
target_sources(STDC-Seg_TensorRT PRIVATE "logger.cpp"
                                         "Segmenter.cpp")

target_link_libraries(STDC-Seg_TensorRT ${LibrariesToLink})

target_include_directories(STDC-Seg_TensorRT PUBLIC ${DirsToLookInForIncludes})