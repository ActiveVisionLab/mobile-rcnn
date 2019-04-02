// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Softmax_Implementation_CUDNN.h"
#include "../../Softmax.h"

using namespace LNTLib;

void Softmax_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	float alpha = 1.0f, beta = 0.0f;

	LNTcudnnSafeCall(cudnnSoftmaxForward(device->CUDAHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, inputDesc,
		inputs[0]->Data(MEMORYDEVICE_CUDA), &beta, outputDesc, this->output->Data(MEMORYDEVICE_CUDA)));
}

#endif