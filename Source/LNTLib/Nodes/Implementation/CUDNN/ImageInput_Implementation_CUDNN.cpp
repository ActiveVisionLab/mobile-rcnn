// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "ImageInput_Implementation_CUDNN.h"
#include "../../ImageInput.h"

using namespace LNTLib;

void ImageInput_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	TensorInfo outputGeometry = node->OutputGeometry();
	int imgSize = outputGeometry.w * outputGeometry.h;

	Vector4f *inputData = ((ImageInput*)node)->Image()->Data(MEMORYDEVICE_CPU);
	Vector3f *buffData = buff->Data(MEMORYDEVICE_CPU);
	for (int i = 0; i < imgSize; i++) buffData[i] = inputData[i].toVector3();

	buff->UpdateDeviceFromHost();

	float alpha = 1.0f, beta = 0.0f;
	LNTcudnnSafeCall(cudnnTransformTensor(device->CUDAHandle(), &alpha, inputDesc, buff->Data(MEMORYDEVICE_CUDA), &beta, outputDesc, output->Data(MEMORYDEVICE_CUDA)));
}

void ImageInput_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	TensorInfo outputGeometry = node->OutputGeometry();

	buff = new ORUtils::MemoryBlock<Vector3f>(outputGeometry.n * outputGeometry.c * outputGeometry.h * outputGeometry.w, true, true);

	//input descriptor
	LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->inputDesc));
	LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
		outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));
}

void ImageInput_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();
	
	LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));

	delete buff;
}

#endif