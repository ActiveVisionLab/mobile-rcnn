// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include <string>
#include <vector>

#ifdef COMPILE_WITH_CUDNN
#include <cudnn.h>
#include "../../ORUtils/CUDADefines.h"
#endif

namespace LNTLib
{
#ifdef COMPILE_WITH_CUDNN

#ifndef LNTcudnnSafeCall
#define LNTcudnnSafeCall(err) LNTLib::__cudnnSafeCall(err, __FILE__, __LINE__)
#endif

	inline void __cudnnSafeCall(cudnnStatus_t err, const char *file, const int line)
	{
		if (CUDNN_STATUS_SUCCESS != err) {
			printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
				file, line, cudnnGetErrorString(err));
			exit(-1);
		}
	}

#endif

	class Device
	{
	public:
		enum DeviceType
		{
			LNTDEVICE_CUDNN
		};

		DeviceType Type() { return deviceType; }

		Device(std::string deviceType)
		{
			if (deviceType == "CUDNN") this->deviceType = Device::LNTDEVICE_CUDNN;
			else DIEWITHEXCEPTION("Unsupported LNT Device!");

#ifdef COMPILE_WITH_CUDNN
			if (this->deviceType == LNTDEVICE_CUDNN)
			{
				LNTcudnnSafeCall(cudnnCreate(&gpuHandle));
				this->AllocateWorkspace();
			}
#endif
		}

		~Device()
		{
#ifdef COMPILE_WITH_CUDNN
			if (this->deviceType == LNTDEVICE_CUDNN)
			{
				if (totalWorkspaceSize > 0) ORcudaSafeCall(cudaFree(workspace));
				LNTcudnnSafeCall(cudnnDestroy(gpuHandle));
			}
#endif
		}

		void StartCompute()
		{

		}

		void FinishCompute()
		{
#ifdef COMPILE_WITH_MPS
			if (deviceType == LNTDEVICE_MPS) ORUtils::waitForMetalCommandBufferCompletion();
#endif

#ifdef COMPILE_WITH_CUDNN
			if (deviceType == LNTDEVICE_CUDNN) ORcudaSafeCall(cudaDeviceSynchronize());
#endif
#ifdef COMPILE_WITH_OPENCL
			if (deviceType == LNTDEVICE_OPENCL) ORopenclSafeCall(clFinish(OpenCLContext::Instance()->Queue()));
#endif
		}

#ifdef COMPILE_WITH_CUDNN
		cudnnHandle_t CUDAHandle() { return gpuHandle; }
		void *Workspace() { return workspace; }
		size_t TotalWorkspaceSize() { return totalWorkspaceSize; }
#endif

	private:
		DeviceType deviceType;

#ifdef COMPILE_WITH_CUDNN
		cudnnHandle_t gpuHandle;
		void *workspace;
		size_t totalWorkspaceSize;

		void AllocateWorkspace()
		{
			size_t free, total;

			workspace = NULL;
			totalWorkspaceSize = 0;
		}
#endif
	};
}
