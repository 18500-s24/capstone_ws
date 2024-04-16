cmake_minimum_required(VERSION 3.5)
project(kinect2_map)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(octomap REQUIRED)
find_package(std_msgs REQUIRED)

# Add executable
add_executable(kinect2_map_node src/kinect2_map_node.cpp)

# Include directories
include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Link dependencies
target_link_libraries(kinect2_map_node
  ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
)
ament_target_dependencies(kinect2_map_node
  rclcpp
  sensor_msgs
  octomap_msgs
  pcl_conversions
  octomap
  std_msgs)

# Install
install(TARGETS kinect2_map_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

# Install package.xml
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME})

ament_package()
