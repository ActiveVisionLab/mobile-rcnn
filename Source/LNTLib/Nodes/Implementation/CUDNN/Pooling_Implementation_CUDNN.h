// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class Pooling_Implementation_CUDNN : public Implementation_CUDNN
	{
	private:
		cudnnPoolingDescriptor_t poolingDesc;

	public:
		Pooling_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Allocate(LNTLib::Device *device);

		void Deallocate();

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif