# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(cable_unilateral SHARED)
target_sources(
  cable_unilateral
  PRIVATE register.cc
          native_route.h
          surface_route.cc
          surface_route.h
          unilateral_cable.cc
          unilateral_cable.h
)

target_include_directories(cable_unilateral PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET mujoco::mujoco)
  target_link_libraries(cable_unilateral PRIVATE mujoco::mujoco)
elseif(TARGET mujoco)
  target_link_libraries(cable_unilateral PRIVATE mujoco)
elseif(MUJOCO_EXTERNAL_INCLUDE_DIR)
  target_include_directories(cable_unilateral PRIVATE ${MUJOCO_EXTERNAL_INCLUDE_DIR})
  if(MUJOCO_EXTERNAL_LIBRARY)
    target_link_libraries(cable_unilateral PRIVATE ${MUJOCO_EXTERNAL_LIBRARY})
  elseif(APPLE)
    target_link_options(cable_unilateral PRIVATE -undefined dynamic_lookup)
  endif()
else()
  message(FATAL_ERROR "No MuJoCo target or include directory found.")
endif()

# Detect API capabilities instead of inferring experimental interfaces from
# the version number. Official MuJoCo 3.9 has no mjpPluginImplicit registry.
include(CheckCXXSourceCompiles)
function(cable_check_mujoco_api)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  if(TARGET mujoco::mujoco)
    set(CMAKE_REQUIRED_LIBRARIES mujoco::mujoco)
  elseif(TARGET mujoco)
    get_target_property(CMAKE_REQUIRED_INCLUDES mujoco INTERFACE_INCLUDE_DIRECTORIES)
  else()
    set(CMAKE_REQUIRED_INCLUDES "${MUJOCO_EXTERNAL_INCLUDE_DIR}")
  endif()
  check_cxx_source_compiles("
    #include <mujoco/mujoco.h>
    #include <mujoco/mjplugin.h>
    int main() {
      mjpPluginImplicit plugin;
      mjp_defaultPluginImplicit(&plugin);
      mjp_registerPluginImplicit(&plugin);
    }" CABLE_HAS_PLUGIN_IMPLICIT)
  check_cxx_source_compiles("
    #include <mujoco/mujoco.h>
    int main() {
      mj_rayMesh(nullptr, nullptr, 0, nullptr, nullptr, nullptr);
    }" CABLE_HAS_RAY_MESH_NORMAL)
endfunction()
cable_check_mujoco_api()
target_compile_definitions(cable_unilateral PRIVATE
  CABLE_HAS_PLUGIN_IMPLICIT=$<BOOL:${CABLE_HAS_PLUGIN_IMPLICIT}>
  CABLE_HAS_RAY_MESH_NORMAL=$<BOOL:${CABLE_HAS_RAY_MESH_NORMAL}>)

include(GNUInstallDirs)
install(
  TARGETS cable_unilateral
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
