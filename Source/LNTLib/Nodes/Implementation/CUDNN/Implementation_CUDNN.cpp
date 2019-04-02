// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "Implementation_CUDNN.h"
#include "../../../Core/Node.h"

using namespace LNTLib;

void Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	this->device = device;

	if (!usingExternalOutput)
	{
		TensorInfo outputGeometry = node->OutputGeometry();

		// output descriptor
		LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->outputDesc));
		LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));

		this->output = new ORUtils::MemoryBlock<float>(outputGeometry.n * outputGeometry.c * outputGeometry.h * outputGeometry.w, MEMORYDEVICE_CUDA);
	}

	if (node->NoInputs() > 0)
	{
		TensorInfo *inputGeometries = node->InputGeometries();

		// input descriptor
		LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->inputDesc));
		LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			inputGeometries[0].n, inputGeometries[0].c, inputGeometries[0].h, inputGeometries[0].w));
	}
}

void Implementation_CUDNN::Deallocate()
{
	if (!usingExternalOutput)
	{
		LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->outputDesc));

		delete output;
		output = NULL;
	}

	if (node->NoInputs() > 0)
		LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));
}

void Implementation_CUDNN::ReAllocateOnNewBatchSize()
{
	if (output != NULL && !usingExternalOutput)
	{
		TensorInfo outputGeometry = node->OutputGeometry();

		int newDataSize = outputGeometry.n * outputGeometry.w * outputGeometry.h * outputGeometry.c;
		if (output->DataSize() != newDataSize) // TODO: this should check individually
		{
			LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->outputDesc));
			delete output;

			LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->outputDesc));
			LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
				outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));

			this->output = new ORUtils::MemoryBlock<float>(outputGeometry.n * outputGeometry.c * outputGeometry.h * outputGeometry.w, MEMORYDEVICE_CUDA);

			if (node->NoInputs() > 0)
			{
				LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(this->inputDesc));

				TensorInfo *inputGeometries = node->InputGeometries();

				// input descriptor
				LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&this->inputDesc));
				LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(this->inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					inputGeometries[0].n, inputGeometries[0].c, inputGeometries[0].h, inputGeometries[0].w));
			}
		}
	}
}

void Implementation_CUDNN::Output(ORUtils::MemoryBlock<float> **res, Implementation::OutputOrder outputOrder, int offsetOut)
{
	TensorInfo outputGeometry = node->OutputGeometry();
	int totalDataSize = outputGeometry.n * outputGeometry.c * outputGeometry.h * outputGeometry.w;

	if (*res == NULL) *res = new ORUtils::MemoryBlock<float>(totalDataSize, MEMORYDEVICE_CPU);
	else
	{
		if ((*res)->DataSize() < totalDataSize)
		{
			delete *res;
			*res = new ORUtils::MemoryBlock<float>(totalDataSize, MEMORYDEVICE_CPU);
		}
	}

	switch (outputOrder)
	{
	case OutputOrder::TEST: 
	{
		int noValuesPerImage = (int)(outputGeometry.w * outputGeometry.h * outputGeometry.c);
		int imgId = outputGeometry.n > 1 ? 0 : 0;

		float *dest = (*res)->Data(MEMORYDEVICE_CPU) + offsetOut;
		float *src = this->output->Data(MEMORYDEVICE_CUDA);

		ORcudaSafeCall(cudaMemcpy(dest, src + imgId * noValuesPerImage, noValuesPerImage * sizeof(float), cudaMemcpyDeviceToHost));

		(*res)->ChangeDataSize(noValuesPerImage, true, true);
	}
	break;

	case OutputOrder::HWD:
	{
		float *dest = (*res)->Data(MEMORYDEVICE_CPU) + offsetOut;
		float *src = this->output->Data(MEMORYDEVICE_CUDA);

		ORcudaSafeCall(cudaMemcpy(dest, src, totalDataSize * sizeof(float), cudaMemcpyDeviceToHost));
	}
	break;

	case OutputOrder::DHW:
	{
		cudnnTensorDescriptor_t inputDesc, outputDesc;

		// destination GPU memory
		ORUtils::MemoryBlock<float> *buff = new ORUtils::MemoryBlock<float>(output->DataSize(), MEMORYDEVICE_CUDA);

		// input descriptor of type CUDNN_TENSOR_NCHW
		LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&inputDesc));
		LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));

		// output descriptor of type CUDNN_TENSOR_NHWC
		LNTcudnnSafeCall(cudnnCreateTensorDescriptor(&outputDesc));
		LNTcudnnSafeCall(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
			outputGeometry.n, outputGeometry.c, outputGeometry.h, outputGeometry.w));

		// transform between types
		float alpha = 1.0f, beta = 0.0f;
		LNTcudnnSafeCall(cudnnTransformTensor(device->CUDAHandle(), &alpha, inputDesc, output->Data(MEMORYDEVICE_CUDA), &beta, outputDesc, buff->Data(MEMORYDEVICE_CUDA)));

		float *dest = (*res)->Data(MEMORYDEVICE_CPU) + offsetOut;
		float *src = buff->Data(MEMORYDEVICE_CUDA);

		// copy from transformed source to res at offset location
		ORcudaSafeCall(cudaMemcpy(dest, src, totalDataSize * sizeof(float), cudaMemcpyDeviceToHost));

		// deallocate everything
		LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(inputDesc));
		LNTcudnnSafeCall(cudnnDestroyTensorDescriptor(outputDesc));
		delete buff;
	}
	break;
	}
}

#endif
