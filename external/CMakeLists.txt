# project(Adaptive_Screen_Space_Meshing)

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#Python
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# OpenGL & Co.
#find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
#include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

#nvdiffrast
file(GLOB NVDIFFRAST_COMMON_CU "nvdiffrast/nvdiffrast/common/*cu" "nvdiffrast/nvdiffrast/common/cuda_raster/*cu" "nvdiffrast/nvdiffrast/common/cudaraster/impl/*cu")
file(GLOB NVDIFFRAST_COMMON_CXX "nvdiffrast/nvdiffrast/common/*cpp" "nvdiffrast/nvdiffrast/common/cuda_raster/*cpp" "nvdiffrast/nvdiffrast/common/cudaraster/impl/*cpp")
file(GLOB NVDIFFRAST_COMMON_HXX "nvdiffrast/nvdiffrast/common/*hpp" "nvdiffrast/nvdiffrast/common/cuda_raster/*hpp" "nvdiffrast/nvdiffrast/common/cudaraster/impl/*hpp")
file(GLOB NVDIFFRAST_LIBS "nvdiffrast/nvdiffrast/lib/*")
file(GLOB NVDIFFRAST_TORCH_CXX "nvdiffrast/nvdiffrast/torch/[ac-z]*cpp")
# set(NVDIFFRAST_TORCH_CXX "nvdiffrast/nvdiffrast/torch/torch_rasterize_gl.cpp" "nvdiffrast/nvdiffrast/torch/torch_bindings.cpp")

# This compiles nvdiffrast for torch
add_compile_definitions(NVDR_TORCH)

add_library(nvdiffrast ${NVDIFFRAST_COMMON_CU} ${NVDIFFRAST_COMMON_CXX} ${NVDIFFRAST_TORCH_CXX} ${NVDIFFRAST_LIBS} ${NVDIFFRAST_COMMON_HXX})
target_link_libraries(nvdiffrast PUBLIC "${TORCH_LIBRARIES}" "${Python_LIBRARIES}" OpenGL::GL OpenGL::EGL)
#target_link_directories(nvdiffrast PRIVATE ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_include_directories(nvdiffrast PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/nvdiffrast" ${Python_INCLUDE_DIRS})

#set(NVDIFFRAST_INCLUDE_DIRS ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

add_subdirectory(pmp-library)
# target_include_directories(pmp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pmp-library/src")

#find_package(pmp REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR} NO_DEFAULT_PATHS)