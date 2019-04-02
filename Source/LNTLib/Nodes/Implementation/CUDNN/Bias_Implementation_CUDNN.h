// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class Bias_Implementation_CUDNN : public Implementation_CUDNN
	{
	private:
		cudnnTensorDescriptor_t biasesDesc;
		ORUtils::MemoryBlock<float> *biasesData;

	public:
		Bias_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Allocate(LNTLib::Device *device);

		void Deallocate();

		void ReadWeights(FILE *f);

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif