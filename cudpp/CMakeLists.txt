# run cmake from top-level

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../src/filters)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../src/modules/base)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)

#link_libraries(filters)
#link_libraries(base)
#link_libraries(cv${OPENCV_VERSION} highgui${OPENCV_VERSION} cxcore${OPENCV_VERSION} cvaux${OPENCV_VERSION})

if (BUILD_CUDA_MODULES)

set(cudpp_srcs
  cudpp.cpp
  cudpp_maximal_launch.cpp
  cudpp_plan.cpp
  cudpp_plan_manager.cpp
)

cuda_compile(cudpp_srcs_cuda
compact_app.cu
scan_app.cu
scan_cta.cu
scan_kernel.cu
vector_kernel.cu	  
compact_kernel.cu
 )
endif()


add_library(cudpp SHARED ${cudpp_srcs} ${cudpp_srcs_cuda})
MAKE_MOD_EXTENSION(cudpp)


