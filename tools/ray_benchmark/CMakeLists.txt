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
    # Keep the runtime output alongside other tools for single- and multi-config generators.
    get_filename_component(TOOLS_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}" DIRECTORY)
    set_target_properties(ray-benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TOOLS_BIN_DIR}"
    )
    foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(ray-benchmark PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config} "${TOOLS_BIN_DIR}"
        )
    endforeach()
    if (OpenMP_CXX_FOUND)
        target_link_libraries(ray-benchmark OpenMP::OpenMP_CXX)
        target_compile_definitions(ray-benchmark PUBLIC XDG_OPENMP)
    endif()
    install(TARGETS ray-benchmark DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
endif()
