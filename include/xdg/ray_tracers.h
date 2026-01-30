// ray tracing interface concrete implementations
#ifdef XDG_ENABLE_EMBREE
#include "xdg/embree/ray_tracer.h"
#endif

#ifdef XDG_ENABLE_GPRT
#include "xdg/gprt/ray_tracer.h"
#endif

#ifdef XDG_ENABLE_DEEPEE_RT
#include "xdg/DeePeeRT/ray_tracer.h"
#endif