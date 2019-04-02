// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#ifdef COMPILE_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "PlatformIndependence.h"

#ifndef ORcudaSafeCall
#define ORcudaSafeCall(err) ORUtils::__cudaSafeCall(err, __FILE__, __LINE__)


#if __CUDACC_VER_MAJOR__ >= 9 
#define __or_shfl(x, y, z) __shfl_sync(0xFFFFFFFFU, x, y, z)
#define __or_shfl_xor(x, y, z) __shfl_xor_sync(0xFFFFFFFFU, x, y, z)
#define __or_shfl_up(x, y, z) __shfl_up_sync(0xFFFFFFFFU, x, y, z)
#define __or_shfl_down(x, y, z) __shfl_down_sync(0xFFFFFFFFU, x, y, z)
#else
#define __or_shfl(x, y, z) __shfl(0xFFFFFFFFU, x, y, z)
#define __or_shfl_xor(x, y, z) __shfl_xor(0xFFFFFFFFU, x, y, z)
#define __or_shfl_up(x, y, z) __shfl_up(0xFFFFFFFFU, x, y, z)
#define __or_shfl_down(x, y, z) __shfl_down(0xFFFFFFFFU, x, y, z)
#endif

namespace ORUtils {
	inline void __cudaSafeCall(cudaError err, const char *file, const int line)
	{
		if (cudaSuccess != err) {
			PRINTF("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}
	}
}

#endif

#ifndef ORcudaKernelCheck

// For normal operation
#define ORcudaKernelCheck

// For debugging purposes
//#define ORcudaKernelCheck { cudaDeviceSynchronize(); ORUtils::__cudaSafeCall(cudaPeekAtLastError(), __FILE__, __LINE__); }

#endif

#endif
