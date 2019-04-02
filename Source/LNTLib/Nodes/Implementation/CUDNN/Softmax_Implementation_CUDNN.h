// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class Softmax_Implementation_CUDNN : public Implementation_CUDNN
	{
	public:
		Softmax_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif