cmake_minimum_required(VERSION 3.8)
project(go2_ltv_mpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
add_executable(mpc_solver_node
  src/mpc_solver_node.cpp
)

ament_target_dependencies(mpc_solver_node
  rclcpp
  std_msgs
)

install(TARGETS
  mpc_solver_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
