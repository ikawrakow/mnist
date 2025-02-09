cmake_minimum_required(VERSION 3.14)
project("mnist" CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(ARCH_FLAGS "")
if (NOT MSVC)
    list(APPEND ARCH_FLAGS -march=native)
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(mnist_knn_v1 mnist_knn_v1.cpp getImages.h getImages.cpp)

add_executable(mnist_knn_v2 mnist_knn_v2.cpp getImages.h getImages.cpp)

add_executable(mnist_knn_v3 mnist_knn_v3.cpp getImages.h getImages.cpp imageUtils.h imageUtils.cpp)

add_executable(mnist_knn_v4 mnist_knn_v4.cpp getImages.h getImages.cpp imageUtils.h imageUtils.cpp)

add_executable(mnist_svm_train
    mnist_svm_train.cpp
    getImages.h
    getImages.cpp
    imageUtils.h
    imageUtils.cpp
    bfgs.h
    bfgs.cpp
    svmPattern.h
    svmPattern.cpp
)

add_executable(mnist_svm_predict
    mnist_svm_predict.cpp
    getImages.h
    getImages.cpp
    svmPattern.h
    svmPattern.cpp
)

add_executable(svm_combine svm_combine.cpp)
