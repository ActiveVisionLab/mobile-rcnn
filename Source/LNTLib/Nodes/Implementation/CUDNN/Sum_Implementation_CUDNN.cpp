// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Sum_Implementation_CUDNN.h"
#include "../../Sum.h"

using namespace LNTLib;

void Sum_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	this->output->SetFrom(inputs[0], MEMCPYDIR_CUDA_TO_CUDA);

	float alpha = 1.0f, beta = 1.0f;
	int noInputs = this->node->NoInputs();

	for (int i = 1; i < noInputs; i++)
		LNTcudnnSafeCall(cudnnAddTensor(device->CUDAHandle(), &alpha, this->inputDesc, inputs[i]->Data(MEMORYDEVICE_CUDA), &beta, this->inputDesc, this->output->Data(MEMORYDEVICE_CUDA)));
}

#endif