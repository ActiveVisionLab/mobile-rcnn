// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Crop_Implementation_CUDNN.h"
#include "../../Crop.h"

using namespace LNTLib;

void Crop_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	//float alpha = 1.0f, beta = 0.0f;
	//TensorInfo *inputGeometries = node->InputGeometries();

	//float *startMemory = inputs[0]->Data(MEMORYDEVICE_CUDA) + ((Crop*)node)->DiffLeft() * inputGeometries[0].h + ((Crop*)node)->DiffTop();

	//LNTcudnnSafeCall(cudnnTransformTensor(device->CUDAHandle(), &alpha, inputDesc, startMemory, &beta, outputDesc, output->Data(MEMORYDEVICE_CUDA)));

	//ORcudaSafeCall(cudaThreadSynchronize());
}

void Crop_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	//this->device = device;

	//TensorInfo outputGeometry = node->OutputGeometry();
	//TensorInfo *inputGeometries = node->InputGeometries();

	//this->output = new ORUtils::MemoryBlock<float>(outputGeometry.n * outputGeometry.c * outputGeometry.h * outputGeometry.w, MEMORYDEVICE_CUDA);

	//// output descriptor
	//LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->outputDesc));
	//LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
	//	outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));

	//if (node->NoInputs() > 0)
	//{
	//	// input descriptor
	//	LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->inputDesc));
	//	LNTcudnnSafeCall(cudnnSetTensor4dDescriptorEx(this->inputDesc, CUDNN_DATA_FLOAT, outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w,
	//		outputGeometry.c * outputGeometry.h * inputGeometries[0].w, inputGeometries[0].h * inputGeometries[0].w, inputGeometries[0].h, 1));
	//}
}

void Crop_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));
	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->outputDesc));

	//delete output;
}

#endif