// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "BatchNorm_Implementation_CUDNN.h"
#include "../../BatchNorm.h"

using namespace LNTLib;

void BatchNorm_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	//float alpha = 1.0f, beta = 0.0f;

	//LNTcudnnSafeCall(cudnnBatchNormalizationForwardInference(device->CUDAHandle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
	//	inputDesc, inputs[0]->Data(MEMORYDEVICE_CUDA), outputDesc, output->Data(MEMORYDEVICE_CUDA),
	//	momentDesc, this->gamma->Data(MEMORYDEVICE_CUDA), this->beta->Data(MEMORYDEVICE_CUDA),
	//	this->mean->Data(MEMORYDEVICE_CUDA), this->variance->Data(MEMORYDEVICE_CUDA), CUDNN_BN_MIN_EPSILON));
}

void BatchNorm_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	//TensorInfo outputGeometry = node->OutputGeometry();

	//// moment descriptor
	//LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&momentDesc));
	//LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(momentDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
	//	1, outputGeometry.c, 1, 1));

	//this->beta = new ORUtils::MemoryBlock<float>(outputGeometry.c, true, true);
	//this->gamma = new ORUtils::MemoryBlock<float>(outputGeometry.c, true, true);
	//this->mean = new ORUtils::MemoryBlock<float>(outputGeometry.c, true, true);
	//this->variance = new ORUtils::MemoryBlock<float>(outputGeometry.c, true, true);
}

void BatchNorm_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	//LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(momentDesc));

	//delete beta;
	//delete gamma;
	//delete mean;
	//delete variance;
}

void BatchNorm_Implementation_CUDNN::ReadWeights(FILE *f)
{
	//TensorInfo outputGeometry = node->OutputGeometry();

	//float *multipliers = new float[outputGeometry.c];
	//float *biases = new float[outputGeometry.c];
	//float *moments = new float[2 * outputGeometry.c];

	//fread(multipliers, sizeof(float), outputGeometry.c, f);
	//fread(biases, sizeof(float), outputGeometry.c, f);
	//fread(moments, sizeof(float), 2 * outputGeometry.c, f);

	//float* _beta = this->beta->Data(MEMORYDEVICE_CPU);
	//float* _gamma = this->gamma->Data(MEMORYDEVICE_CPU);
	//float* _mean = this->mean->Data(MEMORYDEVICE_CPU);
	//float* _variance = this->variance->Data(MEMORYDEVICE_CPU);

	//for (int i = 0; i < outputGeometry.c; i++)
	//{
	//	_beta[i] = biases[i];
	//	_gamma[i] = multipliers[i];
	//	_mean[i] = moments[i];
	//	_variance[i] = pow(moments[outputGeometry.c + i], 2);
	//}

	//this->beta->UpdateDeviceFromHost();
	//this->gamma->UpdateDeviceFromHost();
	//this->mean->UpdateDeviceFromHost();
	//this->variance->UpdateDeviceFromHost();

	//delete[] multipliers;
	//delete[] biases;
	//delete[] moments;
}

#endif