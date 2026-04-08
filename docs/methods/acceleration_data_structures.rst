

Acceleration Data Structures
============================

XDG relies primarily on the bounding volume hierarchy (BVH) data structure for
accelerating ray tracing operations. The BVH is a hierarchical data structure that
organizes primitives (mesh elements) in a scene into a tree of bounding volumes.
The BVH is used to accelerate ray intersection tests by allowing the ray to
quickly traverse the tree and only test intersections with primitives that are
likely to be hit.

Based on the XDG design philosophy (see :ref:`design_philosophy`), the BVH is
constructed by leveraging state-of-the-art ray tracing libraries. On CPUs, XDG
relies on the Embree ray tracing kernels for BVH construction and traversal. 

Mixed Precision Ray Tracing
===========================

The paper "Hardware-Accelerated Ray Tracing of CAD-Based Geometry for Monte
Carlo Radiation Transport" discusses the use of mixed-precision algorithms
to efficiently handle complex CAD-based geometries in Monte Carlo radiation
transport simulations. The key contributions of the paper include leveraging
modern ray tracing kernels to significantly speed up the ray tracing process,
which is critical for handling the high computational demands of Monte Carlo
methods [1]_.

By integrating the techniques discussed in the paper, XDG achieves
faster BVH construction and traversal, leading to more efficient simulations.
This is particularly beneficial for applications involving complex geometries
and large-scale simulations, where traditional CPU-based methods may fall short
in terms of performance.

.. [1] P. Shriwise, P. Wilson, A. Davis, P. Romano, "Hardware-Accelerated Ray
       Tracing of CAD-Based Geometry for Monte Carlo Radiation Transport," in
       *IEEE Computing in Science and Engineering*, vol. 24, no. 2, pp. 52-61,
       February 2022, doi: 10.1109/MCSE.2022.3154656.

GPU Accelerated Ray Tracing
==============================

Ray tracing as a technique is highly parallelizable and has been extensively optimized
for GPU architectures in the context of graphics rendering. As a result there exists 
a rich ecosystem of GPU-accelerated software and even hardware support for ray tracing
operations. But historically these capabilities are focused around single precision 
support and as such have not typically been adopted in the scientfic computing community.

For GPUs, a few different libraries are currently being explored with the goal of 
eventually supporting complete feature parity between CPU and GPU backends of XDG, 
with an explicit focus on vendor agnostic GPU support. The current libraries in question include:

- :term:`GPRT` (General Purpose Ray Tracing Toolkit) - A vulkan based GPU only
  ray tracing library that is vendor agnostic being built around the Vulkan API.

- :term:`DPRT` (Double Precision Ray Tracing Toolkit) - A "basics-only" ray
  tracing library built specifically for double precision ray tracing. Historically
  targeting only NVIDIA platforms via CUDA an experimental OpenMP target offload backend
  is also currently in development for vendor agnostic GPU support.
