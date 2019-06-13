// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class InnerProduct_Implementation_CUDNN : public Implementation_CUDNN
	{
	private:
		size_t workspaceSize;

		cudnnTensorFormat_t tensorFormat;
		cudnnTensorDescriptor_t biasesDesc; bool biasesInitialised;
		cudnnFilterDescriptor_t filtersDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t algo;

		ORUtils::MemoryBlock<float> *biasesData;
		ORUtils::MemoryBlock<float> *filterData;

	public:
		InnerProduct_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Allocate(LNTLib::Device *device);

		void ReAllocateOnNewBatchSize(bool descriptorOnly);

		void Deallocate();

		void ReadWeights(FILE *f);

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif