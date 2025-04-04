cmake_minimum_required(VERSION 3.8)
project(rrt_path_finder)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
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
find_package(cv_bridge)

# Manually specify the path to custom_interface_gym
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/share/custom_interface_gym/cmake)
find_package(custom_interface_gym REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/include
  ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/include/custom_interface_gym
  ${CMAKE_SOURCE_DIR}/data_structure
  ${CMAKE_SOURCE_DIR}/data_structure/base
  ${CMAKE_SOURCE_DIR}/utils
  ${CMAKE_SOURCE_DIR}/utils/optimization
  ${CMAKE_SOURCE_DIR}/utils/geometry
  ${CMAKE_SOURCE_DIR}/utils/header
)

# Add source files for rrt_node
set(SOURCES_RRT_NODE
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node.cpp
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

# Declare the executable for rrt_node
add_executable(rrt_node ${SOURCES_RRT_NODE})

# Link dependencies for rrt_node
ament_target_dependencies(rrt_node
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

# Link the PCL libraries for rrt_node
target_link_libraries(rrt_node  ${PCL_LIBRARIES})

# Ensure C++17 and C99 standards for rrt_node
target_compile_features(rrt_node PUBLIC c_std_99 cxx_std_17)

# Add source files for trajectory_server
set(SOURCES_TRAJECTORY_SERVER
  src/trajectory_server.cpp
)

# Declare the executable for trajectory_server
add_executable(trajectory_server ${SOURCES_TRAJECTORY_SERVER})

# Link the PCL libraries for trajectory_server (SAME AS rrt_node)
target_link_libraries(trajectory_server  ${PCL_LIBRARIES})

# Link dependencies for trajectory_server
ament_target_dependencies(trajectory_server
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

# Ensure C++17 and C99 standards for trajectory_server (SAME AS rrt_node)
target_compile_features(trajectory_server PUBLIC c_std_99 cxx_std_17)

# Add source files for trajectory_server
set(SOURCES_TRAJECTORY_SERVER_YAW
  src/trajectory_server_yaw.cpp
  src/non_uniform_bspline.cpp
)

# Declare the executable for trajectory_server
add_executable(trajectory_server_yaw ${SOURCES_TRAJECTORY_SERVER_YAW})

# Link the PCL libraries for trajectory_server (SAME AS rrt_node)
target_link_libraries(trajectory_server_yaw  ${PCL_LIBRARIES})

# Link dependencies for trajectory_server
ament_target_dependencies(trajectory_server_yaw
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

# Ensure C++17 and C99 standards for trajectory_server (SAME AS rrt_node)
target_compile_features(trajectory_server_yaw PUBLIC c_std_99 cxx_std_17)


set(SOURCES_NEW_REPLAN
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node_new_replan.cpp
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

# Declare the executable for rrt_node_new_replan
add_executable(new_replan ${SOURCES_NEW_REPLAN})

# Link dependencies for new_replan
ament_target_dependencies(new_replan
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


set(SOURCES_NEW_REPLAN_NORMAL_YAW
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node_new_replan_normal_yaw.cpp
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

# Declare the executable for rrt_node_new_replan
add_executable(new_replan_normal_yaw ${SOURCES_NEW_REPLAN_NORMAL_YAW})

# Link dependencies for new_replan
ament_target_dependencies(new_replan_normal_yaw
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

set(SOURCES_NEW_REPLAN_FIRI
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node_firi.cpp
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

# Declare the executable for rrt_node_firi
add_executable(new_replan_firi ${SOURCES_NEW_REPLAN_FIRI})

# Link dependencies for rrt_node_firi
ament_target_dependencies(new_replan_firi
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

set(SOURCES_NEW_REPLAN_NORMAL
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node_new_replan_normal.cpp
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

# Declare the executable for rrt_node_firi
add_executable(new_replan_normal ${SOURCES_NEW_REPLAN_NORMAL})

# Link dependencies for rrt_node_firi
ament_target_dependencies(new_replan_normal
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

set(SOURCES_PCD_MANIPULATION
  src/pcd_manipulation.cpp
)

# Declare the executable for rrt_node_new_replan
add_executable(pcd_manipulation ${SOURCES_PCD_MANIPULATION})

# Link dependencies for new_replan
ament_target_dependencies(pcd_manipulation
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

target_compile_features(pcd_manipulation PUBLIC c_std_99 cxx_std_17)


set(SOURCES_PCD_GEN
  src/pcd_gen.cpp
)

# Declare the executable for rrt_node_new_replan
add_executable(pcd_gen ${SOURCES_PCD_GEN})

# Link dependencies for new_replan
ament_target_dependencies(pcd_gen
  rclcpp
  nav_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  PCL
  tf2_ros
  cv_bridge
  tf2_sensor_msgs
  tf2_geometry_msgs
  std_msgs
  custom_interface_gym
)

target_compile_features(pcd_gen PUBLIC c_std_99 cxx_std_17)


############


# Add source files for rrt_node
set(SOURCES_RRT_SAFE_NODE
  src/corridor_finder.cpp
  src/non_uniform_bspline.cpp
  src/corridor_finder_ellip.cpp
  src/kdtree.c
  src/rrt_node_safe_replan.cpp
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

# Declare the executable for rrt_node
add_executable(safe_node ${SOURCES_RRT_SAFE_NODE})

# Link dependencies for rrt_node
ament_target_dependencies(safe_node
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

# Link the PCL libraries for rrt_node
target_link_libraries(safe_node  ${PCL_LIBRARIES})


############
# Install targets
install(TARGETS rrt_node trajectory_server trajectory_server_yaw new_replan safe_node pcd_manipulation new_replan_firi new_replan_normal new_replan_normal_yaw pcd_gen
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
)

# This is necessary for building the package
ament_package()



# --------------------------------------- PX4 server ----------------------------

# cmake_minimum_required(VERSION 3.8)
# project(rrt_path_finder)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Add compiler flags for warnings
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # Find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(nav_msgs REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(PCL REQUIRED)
# find_package(tf2_ros REQUIRED)
# find_package(tf2_sensor_msgs REQUIRED)
# find_package(tf2_geometry_msgs REQUIRED)
# find_package(tf2 REQUIRED) 
# find_package(std_msgs REQUIRED)
# find_package(px4_msgs REQUIRED)

# # Manually specify the path to custom_interface_gym
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/share/custom_interface_gym/cmake)
# find_package(custom_interface_gym REQUIRED)

# # Include directories
# include_directories(
#   include
#   ${PCL_INCLUDE_DIRS}
#   ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/include
#   ${CMAKE_INSTALL_PREFIX}/custom_interface_gym/include/custom_interface_gym
#   ${CMAKE_SOURCE_DIR}/data_structure
#   ${CMAKE_SOURCE_DIR}/data_structure/base  # ADD THIS LINE
#   ${CMAKE_SOURCE_DIR}/utils
#   ${CMAKE_SOURCE_DIR}/utils/optimization
#   ${CMAKE_SOURCE_DIR}/utils/geometry
#   ${CMAKE_SOURCE_DIR}/utils/header
# )

# # Add source files for rrt_node
# set(SOURCES_RRT_NODE
#   src/corridor_finder.cpp
#   src/corridor_finder_ellip.cpp
#   src/kdtree.c
#   src/rrt_node.cpp
#   src/ciri.cpp
#   src/ciri_ellip.cpp
#   src/ciri_sphere.cpp
#   src/utils/banded_system.cpp
#   src/utils/ellipsoid.cpp
#   src/utils/geometry_utils.cpp
#   src/utils/lbfgs.cpp
#   src/utils/mvie.cpp
#   src/utils/optimization_utils.cpp
#   src/utils/piece.cpp
#   src/utils/polynomial_interpolation.cpp
#   src/utils/polytope.cpp
#   src/utils/quickhull.cpp
#   src/utils/root_finder.cpp
#   src/utils/sdlp.cpp
#   src/utils/trajectory.cpp
# )

# # Declare the executable for rrt_node
# add_executable(rrt_node ${SOURCES_RRT_NODE})

# # Link dependencies for rrt_node
# ament_target_dependencies(rrt_node
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )

# # Link the PCL libraries for rrt_node
# target_link_libraries(rrt_node ${PCL_LIBRARIES})

# # Ensure C++17 and C99 standards for rrt_node
# target_compile_features(rrt_node PUBLIC c_std_99 cxx_std_17)


# # Add source files for rrt_node
# set(SOURCES_RRT_NODE_PX4
#   src/corridor_finder.cpp
#   src/corridor_finder_ellip.cpp
#   src/kdtree.c
#   src/rrt_node_px4.cpp
#   src/ciri.cpp
#   src/ciri_ellip.cpp
#   src/ciri_sphere.cpp
#   src/utils/banded_system.cpp
#   src/utils/ellipsoid.cpp
#   src/utils/geometry_utils.cpp
#   src/utils/lbfgs.cpp
#   src/utils/mvie.cpp
#   src/utils/optimization_utils.cpp
#   src/utils/piece.cpp
#   src/utils/polynomial_interpolation.cpp
#   src/utils/polytope.cpp
#   src/utils/quickhull.cpp
#   src/utils/root_finder.cpp
#   src/utils/sdlp.cpp
#   src/utils/trajectory.cpp
# )

# # Declare the executable for rrt_node
# add_executable(rrt_node_px4 ${SOURCES_RRT_NODE_PX4})

# # Link dependencies for rrt_node
# ament_target_dependencies(rrt_node_px4
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )

# # Link the PCL libraries for rrt_node
# target_link_libraries(rrt_node_px4 ${PCL_LIBRARIES})

# # Ensure C++17 and C99 standards for rrt_node
# target_compile_features(rrt_node_px4 PUBLIC c_std_99 cxx_std_17)

# # Add source files for trajectory_server
# set(SOURCES_TRAJECTORY_SERVER
#   src/trajectory_server_px4.cpp
# )

# # Declare the executable for trajectory_server
# add_executable(trajectory_server_px4 ${SOURCES_TRAJECTORY_SERVER})

# # Link the PCL libraries for trajectory_server (SAME AS rrt_node)
# target_link_libraries(trajectory_server_px4 ${PCL_LIBRARIES})

# # Link dependencies for trajectory_server
# ament_target_dependencies(trajectory_server_px4
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )

# # Ensure C++17 and C99 standards for trajectory_server (SAME AS rrt_node)
# target_compile_features(trajectory_server_px4 PUBLIC c_std_99 cxx_std_17)


# set(SOURCES_NEW_REPLAN
#   src/corridor_finder.cpp
#   src/corridor_finder_ellip.cpp
#   src/kdtree.c
#   src/rrt_node_new_replan.cpp
#   src/ciri.cpp
#   src/ciri_ellip.cpp
#   src/ciri_sphere.cpp
#   src/utils/banded_system.cpp
#   src/utils/ellipsoid.cpp
#   src/utils/geometry_utils.cpp
#   src/utils/lbfgs.cpp
#   src/utils/mvie.cpp
#   src/utils/optimization_utils.cpp
#   src/utils/piece.cpp
#   src/utils/polynomial_interpolation.cpp
#   src/utils/polytope.cpp
#   src/utils/quickhull.cpp
#   src/utils/root_finder.cpp
#   src/utils/sdlp.cpp
#   src/utils/trajectory.cpp
# )

# # Declare the executable for rrt_node_new_replan
# add_executable(new_replan ${SOURCES_NEW_REPLAN})

# # Link dependencies for new_replan
# ament_target_dependencies(new_replan
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )


# set(SOURCES_PCD_MANIPULATION
#   src/pcd_manipulation.cpp
# )

# # Declare the executable for rrt_node_new_replan
# add_executable(pcd_manipulation ${SOURCES_PCD_MANIPULATION})

# # Link dependencies for new_replan
# ament_target_dependencies(pcd_manipulation
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )

# target_compile_features(pcd_manipulation PUBLIC c_std_99 cxx_std_17)





# ############


# # Add source files for rrt_node
# set(SOURCES_RRT_SAFE_NODE
#   src/corridor_finder.cpp
#   src/corridor_finder_ellip.cpp
#   src/kdtree.c
#   src/rrt_node_safe_replan.cpp
#   src/ciri.cpp
#   src/ciri_ellip.cpp
#   src/ciri_sphere.cpp
#   src/utils/banded_system.cpp
#   src/utils/ellipsoid.cpp
#   src/utils/geometry_utils.cpp
#   src/utils/lbfgs.cpp
#   src/utils/mvie.cpp
#   src/utils/optimization_utils.cpp
#   src/utils/piece.cpp
#   src/utils/polynomial_interpolation.cpp
#   src/utils/polytope.cpp
#   src/utils/quickhull.cpp
#   src/utils/root_finder.cpp
#   src/utils/sdlp.cpp
#   src/utils/trajectory.cpp
# )

# # Declare the executable for rrt_node
# add_executable(safe_node ${SOURCES_RRT_SAFE_NODE})

# # Link dependencies for rrt_node
# ament_target_dependencies(safe_node
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   std_msgs
#   custom_interface_gym
#   px4_msgs
# )

# # Link the PCL libraries for rrt_node
# target_link_libraries(safe_node ${PCL_LIBRARIES})


# ############
# # Install targets
# install(TARGETS rrt_node trajectory_server_px4 new_replan safe_node pcd_manipulation
#   DESTINATION lib/${PROJECT_NAME}
# )

# # Install headers
# install(DIRECTORY include/
#   DESTINATION include/${PROJECT_NAME}
# )

# # Export dependencies
# ament_export_dependencies(
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   custom_interface_gym
# )


# # ----------------------------
# # Executable: pointcloud_transformer
# # ----------------------------
# # This executable corresponds to the C++ version of your point cloud transformer node.
# set(SOURCES_POINTCLOUD_TRANSFORMER
#   src/pointcloud_transformer.cpp
# )

# add_executable(pointcloud_transformer ${SOURCES_POINTCLOUD_TRANSFORMER})

# ament_target_dependencies(pointcloud_transformer
#   rclcpp
#   sensor_msgs
#   px4_msgs
#   tf2_ros
#   pcl_conversions
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   tf2
# )

# target_link_libraries(pointcloud_transformer $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# target_compile_features(pointcloud_transformer PUBLIC cxx_std_17)

# # ----------------------------
# # Install targets and headers
# # ----------------------------
# install(TARGETS rrt_node rrt_node_px4 trajectory_server_px4 new_replan pointcloud_transformer
#   DESTINATION lib/${PROJECT_NAME}
# )

# install(DIRECTORY include/
#   DESTINATION include/${PROJECT_NAME}
# )

# # Export dependencies
# ament_export_dependencies(
#   rclcpp
#   nav_msgs
#   sensor_msgs
#   visualization_msgs
#   pcl_conversions
#   PCL
#   tf2_ros
#   tf2_sensor_msgs
#   tf2_geometry_msgs
#   custom_interface_gym
#   px4_msgs
# )

# # This is necessary for building the package
# ament_package()