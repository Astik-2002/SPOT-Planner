cmake_minimum_required(VERSION 3.8)
project(rrt_path_finder)

# Set policy to suppress PCL_ROOT warning
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)  # Add MAVROS package
find_package(geometry_msgs REQUIRED)  # For MAVROS messages
find_package(tf2_eigen REQUIRED)

# Find libdwarf and libdw
find_path(LIBDWARF_INCLUDE_DIRS NAMES dwarf.h libdwarf.h PATHS /usr/include/libdwarf REQUIRED)
find_library(LIBDWARF_LIBRARIES NAMES dwarf libdwarf REQUIRED)
find_path(LIBDW_INCLUDE_DIRS NAMES libdw.h PATHS /usr/include/elfutils REQUIRED)
find_library(LIBDW_LIBRARIES NAMES dw REQUIRED)

# Set custom paths for custom_interface_gym
set(CUSTOM_INTERFACE_GYM_INSTALL_DIR "/home/astik/gym-pybullet-drones/ros2/install/custom_interface_gym")

# Add the custom_interface_gym install path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CUSTOM_INTERFACE_GYM_INSTALL_DIR}")

# Find custom_interface_gym - try multiple approaches
find_package(custom_interface_gym REQUIRED
  PATHS "${CUSTOM_INTERFACE_GYM_INSTALL_DIR}"
  NO_DEFAULT_PATH
)

if(NOT custom_interface_gym_FOUND)
  # Try alternative approach if not found
  find_package(custom_interface_gym REQUIRED)
endif()

# Verify the package was found
if(custom_interface_gym_FOUND)
  message(STATUS "Found custom_interface_gym: ${custom_interface_gym_DIR}")
else()
  message(FATAL_ERROR "Failed to find custom_interface_gym package")
endif()

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${LIBDWARF_INCLUDE_DIRS}
  ${LIBDW_INCLUDE_DIRS}
  ${custom_interface_gym_INCLUDE_DIRS}
)

# Common target link libraries
set(COMMON_LIBS
  ${PCL_LIBRARIES}
  ${LIBDWARF_LIBRARIES}
  ${LIBDW_LIBRARIES}
)

# Common dependencies
set(COMMON_DEPS
  rclcpp
  nav_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  PCL
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  std_msgs
  custom_interface_gym
)

# Create object libraries for kdtree implementations
add_library(kdtree_obj OBJECT src/kdtree.c)
target_compile_features(kdtree_obj PUBLIC c_std_99)

add_library(kdtree_dynamic_obj OBJECT src/kdtree_dynamic.c)
target_compile_features(kdtree_dynamic_obj PUBLIC c_std_99)

# Common sources (without kdtree implementations)
set(COMMON_SOURCES
  src/non_uniform_bspline.cpp
  src/ciri.cpp
  src/ciri_ellip.cpp
  src/ciri_sphere.cpp
  src/utils/banded_system.cpp
  src/utils/ellipsoid.cpp
  src/utils/geometry_utils.cpp
  src/utils/lbfgs.cpp
  src/utils/mvie.cpp
  src/utils/optimization_utils.cpp
  src/utils/piece.cpp
  src/utils/polynomial_interpolation.cpp
  src/utils/polytope.cpp
  src/utils/quickhull.cpp
  src/utils/root_finder.cpp
  src/utils/sdlp.cpp
  src/utils/trajectory.cpp
)

# Function to simplify target creation
function(add_rrt_executable target_name)
  add_executable(${target_name} ${ARGN})
  ament_target_dependencies(${target_name} ${COMMON_DEPS})
  target_link_libraries(${target_name} ${COMMON_LIBS})
  target_compile_features(${target_name} PUBLIC c_std_99 cxx_std_17)
endfunction()

# Main RRT node (uses standard kdtree)
set(SOURCES_RRT_NODE
  ${COMMON_SOURCES}
  src/corridor_finder.cpp
  src/corridor_finder_ellip.cpp
  src/rrt_node.cpp
)
add_rrt_executable(rrt_node ${SOURCES_RRT_NODE} $<TARGET_OBJECTS:kdtree_obj>)

# Trajectory server with yaw
set(SOURCES_TRAJECTORY_SERVER_YAW
  src/trajectory_server_yaw.cpp
  src/non_uniform_bspline.cpp
)
add_rrt_executable(trajectory_server_yaw ${SOURCES_TRAJECTORY_SERVER_YAW})

# PX4 yaw trajectory server
set(SOURCES_TRAJECTORY_SERVER_PX4_YAW
  src/trajectory_server_px4_yaw.cpp
  src/non_uniform_bspline.cpp
)
add_rrt_executable(trajectory_server_px4_yaw ${SOURCES_TRAJECTORY_SERVER_PX4_YAW})
ament_target_dependencies(trajectory_server_px4_yaw px4_msgs)

# MAVROS trajectory server
set(SOURCES_TRAJECTORY_SERVER_MAVROS
  src/trajectory_server_mavros.cpp
  src/non_uniform_bspline.cpp
)
add_rrt_executable(trajectory_server_mavros ${SOURCES_TRAJECTORY_SERVER_MAVROS})
ament_target_dependencies(trajectory_server_mavros 
  mavros_msgs
  geometry_msgs
)

# New replan executable
set(SOURCES_NEW_REPLAN ${SOURCES_RRT_NODE})
list(REMOVE_ITEM SOURCES_NEW_REPLAN src/rrt_node.cpp)
list(APPEND SOURCES_NEW_REPLAN src/rrt_node_new_replan.cpp)
add_rrt_executable(new_replan ${SOURCES_NEW_REPLAN} $<TARGET_OBJECTS:kdtree_obj>)

# Dynamic RRT executable (uses dynamic kdtree)
set(SOURCES_RRT_DYNAMIC
  ${COMMON_SOURCES}
  src/corridor_finder_dynamic.cpp
  src/rrt_dynamic.cpp
)
add_rrt_executable(rrt_dynamic ${SOURCES_RRT_DYNAMIC} $<TARGET_OBJECTS:kdtree_dynamic_obj>)

# PX4 yaw RRT node
set(SOURCES_RRT_NODE_PX4_YAW ${SOURCES_RRT_NODE})
list(REMOVE_ITEM SOURCES_RRT_NODE_PX4_YAW src/rrt_node.cpp)
list(APPEND SOURCES_RRT_NODE_PX4_YAW src/rrt_node_px4_yaw.cpp)
add_rrt_executable(rrt_node_px4_yaw ${SOURCES_RRT_NODE_PX4_YAW} $<TARGET_OBJECTS:kdtree_obj>)
ament_target_dependencies(rrt_node_px4_yaw px4_msgs)

# PCD manipulation tools
add_rrt_executable(pcd_manipulation src/pcd_manipulation.cpp)
add_rrt_executable(pcd_segmentation src/pcd_segmentation.cpp)
ament_target_dependencies(pcd_segmentation cv_bridge 
  tf2
  tf2_ros
  tf2_sensor_msgs
  sensor_msgs
  cv_bridge
  pcl_conversions
  tf2_geometry_msgs
  tf2_eigen
)
target_link_libraries(pcd_segmentation tf2_eigen::tf2_eigen)
# Pointcloud transformer
add_executable(pointcloud_transformer src/pointcloud_transformer.cpp)
ament_target_dependencies(pointcloud_transformer
  rclcpp
  mavros_msgs
  geometry_msgs
  sensor_msgs
  px4_msgs
  nav_msgs
  tf2_ros
  pcl_conversions
  tf2_sensor_msgs
  tf2_geometry_msgs
)
target_link_libraries(pointcloud_transformer ${COMMON_LIBS})
target_compile_features(pointcloud_transformer PUBLIC cxx_std_17)

# Install targets
install(TARGETS 
  rrt_node_px4_yaw 
  trajectory_server_px4_yaw 
  trajectory_server_mavros  # Add MAVROS trajectory server
  pointcloud_transformer 
  trajectory_server_yaw 
  pcd_manipulation 
  new_replan 
  rrt_dynamic 
  pcd_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  nav_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  PCL
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  custom_interface_gym
  px4_msgs
  mavros_msgs  # Export MAVROS dependencies
  geometry_msgs
)

ament_package()