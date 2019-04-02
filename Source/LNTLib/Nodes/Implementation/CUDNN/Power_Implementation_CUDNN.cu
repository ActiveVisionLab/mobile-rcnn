// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Power_Implementation_CUDNN.h"
#include "../../Power.h"

using namespace LNTLib;

__global__ void power_implementation_device(float *input, float *output, Power::NodeParams params, int noEntries);

void Power_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	//Power::NodeParams params = ((Power*)node)->Params();

	//int noEntries = inputs[0]->DataSize();

	//dim3 blockSize(256, 1);
	//dim3 gridSize((int)ceil((float)noEntries / (float)blockSize.x));

	//power_implementation_device << <gridSize, blockSize >> > (output->Data(MEMORYDEVICE_CUDA), inputs[0]->Data(MEMORYDEVICE_CUDA), params, noEntries);
	//ORcudaKernelCheck;
}

__global__ void power_implementation_device(float *output, float *input, Power::NodeParams params, int noEntries)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noEntries - 1) return;

	output[entryId] = pow(input[entryId] * params.scale + params.shift, params.power);
}

#endif