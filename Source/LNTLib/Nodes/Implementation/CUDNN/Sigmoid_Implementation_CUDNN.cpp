// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Sigmoid_Implementation_CUDNN.h"
#include "../../Sigmoid.h"

using namespace LNTLib;

void Sigmoid_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	float alpha = 1.0f, beta = 0.0f;

	LNTcudnnSafeCall(cudnnActivationForward(device->CUDAHandle(), activationDesc, &alpha, this->inputDesc, inputs[0]->Data(MEMORYDEVICE_CUDA), &beta, this->outputDesc, this->output->Data(MEMORYDEVICE_CUDA)));
}

void Sigmoid_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	LNTcudnnSafeCall(cudnnCreateActivationDescriptor(&activationDesc));
	LNTcudnnSafeCall(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0f));
}

void Sigmoid_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	LNTcudnnSafeCall(cudnnDestroyActivationDescriptor(activationDesc));
}

#endif