#ifndef _DEGUG_H
#define _DEBUG_H

#ifdef DEBUG
#define IF_DEBUG(x) x
#define IFN_DEBUG(x)
#else
#define IF_DEBUG(x)
#define IFN_DEBUG(x) x
#endif


#ifdef __CUDACC__
#ifdef DEBUG
#include <stdio.h>
#endif

#define CUDA_CHKERR(x) IFN_DEBUG(x) IF_DEBUG({ \
    cudaError_t cuda_error__ = (x); \
    if (cuda_error__) \
        fprintf(stderr, "%s:%d: CUDA ERROR: %d returned \"%s\"\n", \
                __FILE__, __LINE__, cuda_error__, cudaGetErrorString(cuda_error__)); \
})
#endif

#define DO_ONCE(x) {\
    static bool __done__ = false; \
    if (!__done__) { x; __done__ = true; } \
}

#endif // _DEBUG_H defined
