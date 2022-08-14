cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)
# Project
project(dis_bert C CXX)
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
#onnxruntime providers

# set(ONNXRUNTIME_ROOT_PATH "/Users/wangqi/workspace/cpp-workspace/onnx-bert/onnxruntime")
# set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                            #  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                            #  ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
# set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)

include_directories(ONNXRUNTIME_ROOT_PATH)
#TODO: we should only need one of them.
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
link_libraries("${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib")

add_executable(dis_bert main.cpp)
# target_link_libraries(dis_bert "${ONNXRUNTIME_ROOTDIR}/lib")
# 
#rm