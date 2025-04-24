#Python
find_package(Python COMPONENTS Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)

# boost::timer
find_package(Boost 1.48 REQUIRED COMPONENTS timer)

include_directories(../external/pmp-library/external/eigen ../external/nvdiffrast)
include_directories(../external/pmp-library/src)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC pmp nvdiffrast OpenMP::OpenMP_CXX ${OpenCV_LIBS} "${TORCH_LIBRARIES}" "${Python_LIBRARIES}" ${Boost_LIBRARIES} ${Boost_TIMER_LIBRARY})
