// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Relu_Implementation_CUDNN.h"
#include "../../Relu.h"

using namespace LNTLib;

void Relu_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	float alpha = 1.0f, beta = 0.0f;

	LNTcudnnSafeCall(cudnnActivationForward(device->CUDAHandle(), activationDesc, &alpha, this->inputDesc, inputs[0]->Data(MEMORYDEVICE_CUDA), &beta, this->outputDesc, this->output->Data(MEMORYDEVICE_CUDA)));
}

void Relu_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	float relu_max = ((Relu*)node)->Params().relu_max;

	if (relu_max < 1e9f)
	{
		LNTcudnnSafeCall(cudnnCreateActivationDescriptor(&activationDesc));
		LNTcudnnSafeCall(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, relu_max));
	}
	else
	{
		LNTcudnnSafeCall(cudnnCreateActivationDescriptor(&activationDesc));
		LNTcudnnSafeCall(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f));
	}
}

void Relu_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	LNTcudnnSafeCall(cudnnDestroyActivationDescriptor(activationDesc));
}

#endif