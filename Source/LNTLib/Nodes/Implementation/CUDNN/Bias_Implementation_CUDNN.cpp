// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Bias_Implementation_CUDNN.h"
#include "../../Bias.h"

using namespace LNTLib;

void Bias_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	//float alpha = 1.0f, beta = 0.0f;

	//float *outputMem = output->Data(MEMORYDEVICE_CUDA);
	//float *biasesMem = this->biasesData->Data(MEMORYDEVICE_CUDA);

	//output->SetFrom(inputs[0], MEMCPYDIR_CUDA_TO_CUDA);

	//alpha = 1.0f; beta = 1.0f;
	//LNTcudnnSafeCall(cudnnAddTensor(device->CUDAHandle(), &alpha, biasesDesc, biasesMem, &beta, outputDesc, outputMem));
}

void Bias_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	//TensorInfo outputGeometry = node->OutputGeometry();

	//LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->biasesDesc));
	//LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->biasesDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outputGeometry.c, 1, 1));
	//this->biasesData = new ORUtils::MemoryBlock<float>(outputGeometry.c, true, true);
}

void Bias_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->biasesDesc));
	//delete this->biasesData;
}

void Bias_Implementation_CUDNN::ReadWeights(FILE *f)
{
	//fread(biasesData->Data(MEMORYDEVICE_CPU), sizeof(float), biasesData->DataSize(), f);
	//biasesData->UpdateDeviceFromHost();
}

#endif