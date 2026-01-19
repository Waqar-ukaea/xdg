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

#===============================================================================
# ray-benchmark (special case - requires linking directly to GPRT)
#===============================================================================
if (XDG_ENABLE_GPRT)
    # Embed and compile the device code
    embed_devicecode(
        OUTPUT_TARGET
        ray_benchmark_deviceCode
        HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/ray_benchmark_shared.h
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ray_benchmark_deviceCode.slang
    )

    # Create the ray-benchmark executable
    add_executable(ray-benchmark ray_benchmark.cpp)
    target_link_libraries(ray-benchmark xdg argparse ray_benchmark_deviceCode)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(ray-benchmark OpenMP::OpenMP_CXX)
        target_compile_definitions(ray-benchmark PUBLIC XDG_OPENMP)
    endif()
    install(TARGETS ray-benchmark DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
endif()