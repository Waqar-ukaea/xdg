Acceleration Data Structures
============================

Ray tracing against a mesh becomes expensive if every ray is tested against every
primitive. A model with many triangles, tetrahedra, or other mesh elements needs
an acceleration structure so most primitives can be rejected before the more
expensive ray-primitive intersection tests are performed.

XDG relies primarily on :term:`BVH`-based acceleration structures. In keeping
with the XDG design philosophy (:ref:`design_philosophy`), these structures are
built and traversed by the selected ray tracing backend. On CPUs, XDG currently
relies on the :term:`Embree` ray tracing kernels for BVH construction and
traversal.

Axis-Aligned Bounding Boxes
---------------------------

The basic building block is an axis-aligned bounding box (AABB). An AABB is a
conservative box around a primitive or group of primitives, aligned with the
coordinate axes. Ray-box intersection is much cheaper than ray-primitive
intersection, so a ray that misses an AABB can skip everything inside it.

.. figure:: ../assets/AABB.png
   :alt: Axis-aligned bounding box around a primitive
   :align: center
   :width: 45%

   Axis-aligned bounding boxes provide simple bounding volumes for ray
   intersection tests.

Bottom-Level Acceleration Structures
------------------------------------

A :term:`BLAS` is the lower-level acceleration structure built over the
primitives of one piece of geometry. In practice, this is commonly a BVH:
leaf nodes reference primitives, while internal nodes store AABBs that enclose
their child nodes. Traversal starts at the root of the tree and only descends
into child boxes that the ray intersects.

For XDG surface tracking, a BLAS-like structure is a natural representation for
one surface's triangles. For volume tracking, the same idea can be applied to
the volumetric elements inside a volume.

.. figure:: ../assets/BLAS.png
   :alt: Bottom-level acceleration structure diagram
   :align: center
   :width: 100%

   A BLAS partitions geometry into a hierarchy of bounding volumes.

Top-Level Acceleration Structures
---------------------------------

Many ray tracing libraries use a two-level acceleration structure made from a
:term:`TLAS` and one or more BLAS instances. The TLAS is built over higher-level
geometry instances rather than individual mesh primitives. A ray first traverses
the TLAS to reject whole pieces of geometry, then traverses only the relevant
BLASes to test against individual primitives.

.. figure:: ../assets/TLAS-krhonos.png
   :alt: Khronos top-level acceleration structure diagram
   :align: center
   :width: 100%

   Khronos illustration of a TLAS over lower-level BLAS instances.

This AABB-to-BLAS-to-TLAS structure maps onto XDG geometry as follows:

- In :term:`surface tracking`, each surface is represented by a BLAS-like
  structure over its triangles. Each volume is represented by a TLAS-like scene
  over the BLAS instances for its boundary surfaces. This lets the ray tracer
  reject entire boundary surfaces before testing individual triangles.
- In :term:`volume tracking`, each volume is represented by an acceleration
  structure over that volume's volumetric elements. Backends that expose
  two-level scenes can represent this as a TLAS-like scene containing a
  BLAS-like geometry over the elements.

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

GPU-Accelerated Ray Tracing
===========================

Ray tracing as a technique is highly parallelizable and has been extensively
optimized for GPU architectures in the context of graphics rendering. As a
result, there is a rich ecosystem of GPU-accelerated software and even
hardware support (see :term:`RT hardware acceleration`) for ray tracing
operations. Historically, these capabilities have focused on single-precision
support and have not typically been adopted in the scientific computing
community.

XDG is intended to support GPU acceleration and provide an interface for
leveraging GPU-accelerated ray tracing in scientific computing applications.
An explicit focus is being placed on vendor-agnostic GPU support to ensure that
XDG can be used across a wide range of hardware platforms. Currently, initial
scoping of the GPU API is underway with work being done to support :term:`GPRT`
(General Purpose Ray Tracing Toolkit), a Vulkan-based GPU-only ray tracing
library that is vendor-agnostic and built around the Vulkan API. Other GPU ray
tracing libraries will also be explored in the future, with the eventual goal of
providing complete feature parity between CPU and GPU backends of XDG.
