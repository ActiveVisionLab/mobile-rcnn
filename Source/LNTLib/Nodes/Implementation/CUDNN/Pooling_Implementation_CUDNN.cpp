// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Pooling_Implementation_CUDNN.h"
#include "../../Pooling.h"

using namespace LNTLib;

void Pooling_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	float alpha = 1.0f, beta = 0.0f;

	LNTcudnnSafeCall(cudnnPoolingForward(device->CUDAHandle(), poolingDesc, &alpha, inputDesc, inputs[0]->Data(MEMORYDEVICE_CUDA), &beta, outputDesc, output->Data(MEMORYDEVICE_CUDA)));
}

void Pooling_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	Pooling::NodeParams params = ((Pooling*)node)->Params();

	cudnnPoolingMode_t poolingMode;

	switch (params.method)
	{
	case Pooling::Pooling_Max:
		poolingMode = CUDNN_POOLING_MAX;
		break;
	case Pooling::Pooling_Avg:
		poolingMode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		break;
	default:
		DIEWITHEXCEPTION("Unsupported pooling mode!\n");
		break;
	}

	// pooling descriptor
	LNTcudnnSafeCall(cudnnCreatePoolingDescriptor(&this->poolingDesc));
	LNTcudnnSafeCall(cudnnSetPooling2dDescriptor(this->poolingDesc, poolingMode, CUDNN_PROPAGATE_NAN, params.poolSize.x, params.poolSize.y, params.padding.z, params.padding.x, params.stride.x, params.stride.y));
}

void Pooling_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	LNTcudnnSafeCall(cudnnDestroyPoolingDescriptor(this->poolingDesc));
}

#endif