set(TOOL_NAMES
particle_sim
ray_fire
batch_ray_fire
find_volume
point_in_volume
batch_point_in_volume
overlap_check
walk_elements
tally_segments
)

#===============================================================================
# OpenMP -- use if availabable
#===============================================================================
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP; tools will be built with threaded support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
    message(STATUS "OpenMP not found; building tools without OpenMP support")
endif()

foreach(tool ${TOOL_NAMES})
    string(REPLACE "_" "-" tool_exec ${tool})
    add_executable(${tool_exec} ${tool}.cpp)
    target_link_libraries(${tool_exec} xdg argparse indicators::indicators)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${tool_exec} OpenMP::OpenMP_CXX)
        target_compile_definitions(${tool_exec} PUBLIC XDG_OPENMP)
    endif()
    install(TARGETS ${tool_exec} DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
endforeach()


# Configure GPRT tools
if (XDG_ENABLE_GPRT)
  set(GPRT_TOOLS
    xdg_gprt_render_tool
    ray_benchmark
  )
  find_package(CUDAToolkit REQUIRED)

  foreach(tool ${GPRT_TOOLS})
    string(REPLACE "_" "-" tool_exec ${tool})
    add_executable(${tool_exec} ${tool}.cpp)
    target_link_libraries(${tool_exec} xdg argparse)
    install(TARGETS ${tool_exec} DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
    target_link_libraries(${tool_exec} xdg gprt)

    embed_devicecode(
      OUTPUT_TARGET
      ${tool}_deviceCode
      HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/${tool}_shared.h
      SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/${tool}_deviceCode.slang
    )

    if (OpenMP_CXX_FOUND)
        target_link_libraries(${tool_exec} OpenMP::OpenMP_CXX)
        target_compile_definitions(${tool_exec} PUBLIC XDG_OPENMP)
    endif()

    target_link_libraries(${tool_exec}
      ${tool}_deviceCode
      gprt::gprt
    )
    # Link CUDA runtime only for ray_benchmark
    if (tool_exec STREQUAL "ray-benchmark")
      target_sources(${tool_exec} PRIVATE openmp_memory_interop.cpp)
      target_link_libraries(${tool_exec} CUDA::cudart)
      target_include_directories(${tool_exec} PRIVATE ${CUDA_INCLUDE_DIRS})
      target_compile_definitions(${tool_exec} PUBLIC XDG_ENABLE_CUDA)
    endif()
  endforeach()
endif()