CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(deploy_use_tensorrt)

set(USE_CUDA_VERSION 9.0)
set(USE_CUDNN_VERSION 7.5)

set(DEPS_LIB_DIR ${PROJECT_SOURCE_DIR}/deps)

IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    IF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        SET(NEWCXX_ENABLED_FLAG "-std=c++11 -stdlib=libc++ -Wall -O2")
    ELSE()
        SET(NEWCXX_ENABLED_FLAG "-std=c++11 -Wall -Wno-mismatched-tags -O2")
    ENDIF()
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    SET(NEWCXX_ENABLED_FLAG "-std=c++11 -Wall")
ELSE()
    MESSAGE(FATAL_ERROR, "Only GCC and clang is supported for now.")
ENDIF()

SET(CMAKE_CXX_FLAGS ${NEWCXX_ENABLED_FLAG})
MESSAGE(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

SET(CUDA_TOOLKIT_ROOT_DIR ${DEPS_LIB_DIR}/cuda${USE_CUDA_VERSION})
SET(CUDNN_ROOT_DIR ${DEPS_LIB_DIR}/cudnn${USE_CUDNN_VERSION})
SET(TENSORRT_ROOT_DIR  ${DEPS_LIB_DIR}/tensorrt5.1.2)

SET(CUDNN_INCLUDE_DIR ${CUDNN_ROOT_DIR}/include)
SET(CUDNN_LIBRARY_PATH ${CUDNN_ROOT_DIR}/lib64)

SET(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)
SET(CUDA_LIBRARY_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

SET(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT_DIR}/include)
SET(TENSORRT_LIBRARY_PATH ${TENSORRT_ROOT_DIR}/lib)

INCLUDE_DIRECTORIES(
    ${DEPS_LIB_DIR}/tensorrt5.1.2/include
	${DEPS_LIB_DIR}/cudnn${USE_CUDA_VERSION}/include
	${DEPS_LIB_DIR}/cuda${USE_CUDA_VERSION}/include
)

INCLUDE_DIRECTORIES(include)
ADD_EXECUTABLE(test_googlenet src/sampleGoogleNet.cpp include/logger.cpp)
TARGET_LINK_LIBRARIES(test_googlenet PUBLIC
    ${TENSORRT_LIBRARY_PATH}/libnvinfer.so
    ${TENSORRT_LIBRARY_PATH}/libnvinfer_plugin.so
    ${TENSORRT_LIBRARY_PATH}/libnvcaffe_parser.so
    ${CUDA_LIBRARY_PATH}/libcublas.so
    ${CUDA_LIBRARY_PATH}/libcudart.so
    ${CUDA_LIBRARY_PATH}/libnvblas.so
    ${CUDA_LIBRARY_PATH}/libnvrtc.so
    ${CUDNN_LIBRARY_PATH}/libcudnn.so
)

ADD_EXECUTABLE(trtexec src/trtexec/trtexec.cpp include/logger.cpp)
TARGET_LINK_LIBRARIES(trtexec PUBLIC
    ${TENSORRT_LIBRARY_PATH}/libnvinfer.so
    ${TENSORRT_LIBRARY_PATH}/libnvinfer_plugin.so
    ${TENSORRT_LIBRARY_PATH}/libnvcaffe_parser.so
    ${TENSORRT_LIBRARY_PATH}/libnvonnxparser.so
    ${CUDA_LIBRARY_PATH}/libcublas.so
    ${CUDA_LIBRARY_PATH}/libcudart.so
    ${CUDA_LIBRARY_PATH}/libnvblas.so
    ${CUDA_LIBRARY_PATH}/libnvrtc.so
    ${CUDNN_LIBRARY_PATH}/libcudnn.so
)