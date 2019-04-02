// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class BatchNorm_Implementation_CUDNN : public Implementation_CUDNN
	{
	private:
		cudnnTensorDescriptor_t momentDesc;

		ORUtils::MemoryBlock<float> *beta;
		ORUtils::MemoryBlock<float> *gamma;
		ORUtils::MemoryBlock<float> *mean;
		ORUtils::MemoryBlock<float> *variance;

	public:
		BatchNorm_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Allocate(LNTLib::Device *device);

		void Deallocate();

		void ReadWeights(FILE *f);

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif