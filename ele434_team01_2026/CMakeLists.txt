cmake_minimum_required(VERSION 3.8)
project(ele434_team01_2026)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}_modules)

# Install Python executables
install(PROGRAMS
  scripts/slam_explorer.py
  scripts/basic_velocity_control.py
  scripts/stop_me.py
  scripts/test.py
  scripts/Vacuum.py
  scripts/newtest.py
  scripts/APF.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY
  launch
  maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
