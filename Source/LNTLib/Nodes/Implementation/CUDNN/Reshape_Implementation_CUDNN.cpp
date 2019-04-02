// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Reshape_Implementation_CUDNN.h"
#include "../../Reshape.h"

using namespace LNTLib;

void Reshape_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	output->SetFrom(inputs[0], MEMCPYDIR_CUDA_TO_CUDA);
	//float alpha = 1.0f, beta = 0.0f;
	//LNTcudnnSafeCall(cudnnTransformTensor(device->CUDAHandle(), &alpha, inputDesc, inputs[0]->Data(MEMORYDEVICE_CUDA), &beta, outputDesc, output->Data(MEMORYDEVICE_CUDA)));
}

void Reshape_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	//this->device = device;

	//TensorInfo inputGeometry = node->InputGeometries()[0];

	//LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&inputDesc));
	//LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
	//	inputGeometry.n, inputGeometry.c, inputGeometry.h, inputGeometry.w));

	//LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&outputDesc));
	//LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
	//	inputGeometry.n, inputGeometry.c, inputGeometry.h, inputGeometry.w));

	//this->output = new ORUtils::MemoryBlock<float>(inputGeometry.n * inputGeometry.c * inputGeometry.h * inputGeometry.w, MEMORYDEVICE_CUDA);
}

//void Reshape_Implementation_CUDNN::ReAllocateOnNewBatchSize()
//{
//	if (output != NULL && !usingExternalOutput)
//	{
//		TensorInfo inputGeometry = node->InputGeometries()[0];
//
//		int newDataSize = inputGeometry.n * inputGeometry.w * inputGeometry.h * inputGeometry.c;
//		if (output->DataSize() != newDataSize) // TODO: this should check individually
//		{
//			LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->outputDesc));
//			LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));
//			delete output;
//
//			LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&inputDesc));
//			LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//				inputGeometry.n, inputGeometry.c, inputGeometry.h, inputGeometry.w));
//
//			LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&outputDesc));
//			LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//				inputGeometry.n, inputGeometry.c, inputGeometry.h, inputGeometry.w));
//
//			this->output = new ORUtils::MemoryBlock<float>(inputGeometry.n * inputGeometry.c * inputGeometry.h * inputGeometry.w, MEMORYDEVICE_CUDA);
//		}
//	}
//}

void Reshape_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();
	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));
	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->outputDesc));

	//delete output;
	//output = NULL;
}

#endif