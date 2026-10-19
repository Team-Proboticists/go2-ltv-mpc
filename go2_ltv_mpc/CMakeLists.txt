cmake_minimum_required(VERSION 3.8)

project(go2_ltv_mpc)

# ==========================================
# Compiler settings
# ==========================================

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# Dependencies
# ==========================================

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# ==========================================
# OSQP library
# ==========================================

find_path(OSQP_INCLUDE_DIR
  NAMES osqp/osqp.h
  PATHS /usr/local/include
  REQUIRED
)

find_library(OSQP_LIBRARY
  NAMES osqp
  PATHS /usr/local/lib
  REQUIRED
)

# ==========================================
# MPC executable
# ==========================================

add_executable(mpc_solver_node
  src/mpc_solver_node.cpp
  src/LTV_MPC.cpp
  src/PredictionModel.cpp
  src/MPCCost.cpp
  src/MPCConstraints.cpp
  src/QPSolver.cpp
)

target_include_directories(mpc_solver_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OSQP_INCLUDE_DIR}
)

target_link_libraries(mpc_solver_node
  Eigen3::Eigen
  ${OSQP_LIBRARY}
)

ament_target_dependencies(mpc_solver_node
  rclcpp
  std_msgs
)

# ==========================================
# Installation
# ==========================================

install(
  TARGETS mpc_solver_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ==========================================
# Testing
# ==========================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
