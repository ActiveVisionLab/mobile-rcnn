// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)
#include <cstdio>
#include <stdexcept>
#endif

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__) && defined(ANDROID)
#include <android/log.h>
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __device__	// for CUDA device code
#else
#define _CPU_AND_GPU_CODE_ 
#endif

#if defined(__CUDACC__)
#define _CPU_AND_GPU_CODE_TEMPLATE_ __device__ // for CUDA device code
#else
#define _CPU_AND_GPU_CODE_TEMPLATE_
#endif

#if defined(__OPENCL_VERSION__)
#define TEMPLATE1(x) 
#define TEMPLATE2(x,y) 
#define TEMPLATE3(x,y,z) 
#else
#define TEMPLATE1(...) template<__VA_ARGS__>
#define TEMPLATE2(...) template<__VA_ARGS__>
#define TEMPLATE3(...) template<__VA_ARGS__>
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CONSTANT_ __constant__	// for CUDA device code
#else
#define _CPU_AND_GPU_CONSTANT_
#endif

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)
#define THREADPTR(x) x
#define DEVICEPTR(x) x
#define THREADGROUPPTR(x) x
#define CONSTPTR(x) x
#endif

#if defined(__METALC__) // for METAL device code
#define THREADPTR(x) thread x
#define DEVICEPTR(x) device x
#define THREADGRPPTR(x) threadgroup x
#define CONSTPTR(x) constant x
#endif

#if defined(__OPENCL_VERSION__) // for OPENCL device code
#define THREADPTR(x) __private x
#define DEVICEPTR(x) __global x
#define THREADGRPPTR(x) __local x
#define CONSTPTR(x) __global x
#endif

#ifdef ANDROID
#define DIEWITHEXCEPTION(x) { fprintf(stderr, "%s\n", x); exit(-1); }
#else
#if defined(__OPENCL_VERSION__)
#define DIEWITHEXCEPTION(x) {}
#else
#define DIEWITHEXCEPTION(x) throw std::runtime_error(x)
#endif
#endif

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)
#ifdef ANDROID
#define PRINTF(...) __android_log_print(ANDROID_LOG_INFO, "InfiniTAM", __VA_ARGS__);
#else
#define PRINTF(...) printf(__VA_ARGS__);
#endif
#endif
