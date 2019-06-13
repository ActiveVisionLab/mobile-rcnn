// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "ConvTransposed_Implementation_CUDNN.h"
#include "../../ConvTransposed.h"

using namespace LNTLib;

void ConvTransposed_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	ConvTransposed::NodeParams params = ((ConvTransposed*)node)->Params();

	float alpha = 1.0f, beta = 0.0f;

	float *inputMem = inputs[0]->Data(MEMORYDEVICE_CUDA), *outputMem = output->Data(MEMORYDEVICE_CUDA);
	float *filtersMem = this->filterData->Data(MEMORYDEVICE_CUDA);

	LNTcudnnSafeCall(cudnnConvolutionBackwardData(device->CUDAHandle(), &alpha, filtersDesc, filtersMem, inputDesc, inputMem, convDesc, algo, device->Workspace(), workspaceSize, &beta, outputDesc, outputMem));

	if (params.hasBiases)
	{
		float *biasesMem = this->biasesData->Data(MEMORYDEVICE_CUDA);

		alpha = 1.0f; beta = 1.0f;
		LNTcudnnSafeCall(cudnnAddTensor(device->CUDAHandle(), &alpha, biasesDesc, biasesMem, &beta, outputDesc, outputMem));
	}
}

void ConvTransposed_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	ConvTransposed::NodeParams params = ((ConvTransposed*)node)->Params();	

	// filter descriptor
	LNTcudnnSafeCall(cudnnCreateFilterDescriptor(&this->filtersDesc));

	if (params.isDepthWise)
		LNTcudnnSafeCall(cudnnSetFilter4dDescriptor(this->filtersDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			params.noOutputs, params.kernelSize.z / params.noOutputs, params.kernelSize.x, params.kernelSize.y));
	else
		LNTcudnnSafeCall(cudnnSetFilter4dDescriptor(this->filtersDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			params.noOutputs, params.kernelSize.z, params.kernelSize.x, params.kernelSize.y));

	// biases
	if (params.hasBiases)
	{
		this->biasesInitialised = true;
		LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->biasesDesc));
		LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->biasesDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, params.noOutputs, 1, 1));
		this->biasesData = new ORUtils::MemoryBlock<float>(1 * params.noOutputs * 1 * 1, true, true);
	}
	else this->biasesInitialised = false;

	// convolution descriptor
	LNTcudnnSafeCall(cudnnCreateConvolutionDescriptor(&this->convDesc));
	LNTcudnnSafeCall(cudnnSetConvolution2dDescriptor(this->convDesc, params.crop.x, params.crop.z, params.upsample.y, params.upsample.x,
		1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	// depthwise convolution noGroups == noOutputs == noInputs
	if (params.isDepthWise)
		LNTcudnnSafeCall(cudnnSetConvolutionGroupCount(this->convDesc, params.noOutputs));

	// algorithm
	if (params.isDepthWise)
		LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataAlgorithm(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc,
			CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, device->TotalWorkspaceSize(), &this->algo));
	else
		LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataAlgorithm(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, device->TotalWorkspaceSize(), &this->algo));

	// workspace size
	LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataWorkspaceSize(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc, this->algo, &this->workspaceSize));

	// kernel size
	if (params.isDepthWise)
		this->filterData = new ORUtils::MemoryBlock<float>(params.kernelSize.x * params.kernelSize.y * params.kernelSize.z, true, true);
	else
		this->filterData = new ORUtils::MemoryBlock<float>(params.noOutputs * params.kernelSize.x * params.kernelSize.y * params.kernelSize.z, true, true);
}

void ConvTransposed_Implementation_CUDNN::ReAllocateOnNewBatchSize(bool descriptorOnly)
{
	Implementation_CUDNN::ReAllocateOnNewBatchSize(descriptorOnly);

	ConvTransposed::NodeParams params = ((ConvTransposed*)node)->Params();

	// algorithm
	if (params.isDepthWise)
		LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataAlgorithm(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc,
			CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, device->TotalWorkspaceSize(), &this->algo));
	else
		LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataAlgorithm(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, device->TotalWorkspaceSize(), &this->algo));

	// workspace size
	LNTcudnnSafeCall(cudnnGetConvolutionBackwardDataWorkspaceSize(device->CUDAHandle(), this->filtersDesc, this->inputDesc, this->convDesc, this->outputDesc, this->algo, &this->workspaceSize));
}

void ConvTransposed_Implementation_CUDNN::Deallocate()
{
	Implementation_CUDNN::Deallocate();

	ConvTransposed::NodeParams params = ((ConvTransposed*)node)->Params();

	LNTcudnnSafeCall(cudnnDestroyFilterDescriptor(this->filtersDesc));
	if (params.hasBiases) LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->biasesDesc));
	LNTcudnnSafeCall(cudnnDestroyConvolutionDescriptor(this->convDesc));

	delete this->filterData;
	if (params.hasBiases) delete this->biasesData;
}

void ConvTransposed_Implementation_CUDNN::ReadWeights(FILE *f)
{
	ConvTransposed::NodeParams params = ((ConvTransposed*)node)->Params();

	fread(filterData->Data(MEMORYDEVICE_CPU), sizeof(float), filterData->DataSize(), f);
	if (params.hasBiases) fread(biasesData->Data(MEMORYDEVICE_CPU), sizeof(float), biasesData->DataSize(), f);

	filterData->UpdateDeviceFromHost();
	if (params.hasBiases) biasesData->UpdateDeviceFromHost();
}

#endif