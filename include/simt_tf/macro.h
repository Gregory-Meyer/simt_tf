#ifndef SIMT_TF_MACRO_H
#define SIMT_TF_MACRO_H

#ifdef __CUDACC__
#define SIMT_TF_HOST_DEVICE __host__ __device__
#else
#define SIMT_TF_HOST_DEVICE
#endif

#endif
