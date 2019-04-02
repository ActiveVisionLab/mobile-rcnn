// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Upsample_Implementation_CUDNN.h"
#include "../../Upsample.h"

using namespace LNTLib;

__global__ void upscale(const float *input, float *output, TensorInfo inputGeometry, TensorInfo outputGeometry, Vector2i scale_factor);

void Upsample_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	Upsample::NodeParams params = ((Upsample*)node)->Params();
	if (params.method != Upsample::Upsample_Nearest)
		DIEWITHEXCEPTION("CUDNN implementation supports only Nearest upsampling!");

	TensorInfo outputGeometry = node->OutputGeometry();
	TensorInfo inputGeometry = node->InputGeometries()[0];

	dim3 blockSize(8, 8, 4);
	dim3 gridSize((unsigned int)ceil((float)outputGeometry.w / (float)blockSize.x), (unsigned int)ceil((float)outputGeometry.h / (float)blockSize.y), outputGeometry.n * outputGeometry.c / blockSize.z);

	upscale << <gridSize, blockSize >> >(inputs[0]->Data(MEMORYDEVICE_CUDA), output->Data(MEMORYDEVICE_CUDA), inputGeometry, outputGeometry, params.upsample);
	ORcudaKernelCheck;
}

void Upsample_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);;
}

void Upsample_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();
}

__global__ void upscale(const float *input, float *output, TensorInfo inputGeometry, TensorInfo outputGeometry, Vector2i scale_factor)
{
	Vector3i gId(
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z);

	if (gId.x >= outputGeometry.w || gId.y >= outputGeometry.h || gId.z >= outputGeometry.n * outputGeometry.c) return;

	const float *srcSlice = &input[gId.z * inputGeometry.w * inputGeometry.h];
	float *dstSlice = &output[gId.z * outputGeometry.w * outputGeometry.h];

	dstSlice[gId.x + gId.y * outputGeometry.w] = srcSlice[(gId.x / scale_factor.x) + (gId.y / scale_factor.y) * inputGeometry.w];
}

#endif