// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "../Implementation.h"
#include "../../../Core/TensorInfo.h"

namespace LNTLib
{
	class Implementation_CUDNN : public Implementation
	{
	protected:
		LNTLib::Device *device;
		cudnnTensorDescriptor_t inputDesc, outputDesc;

	public:
		// Brief:
		//   Calls parent constructor
		Implementation_CUDNN(Node *node) : Implementation(node) { }

		// Brief:
		//   virtual destructor
		virtual ~Implementation_CUDNN() { }

		// Brief:
		//   Allocates the required metal image for output.
		void Allocate(LNTLib::Device *device);

		// Brief:
		//   Deallocates metal image
		void Deallocate();

		// Brief:
		//   Reallocates resources when the batch size is changed
		void ReAllocateOnNewBatchSize();
		
		// Brief:
		//   Copies the output of the network to the memory.
		void Output(ORUtils::MemoryBlock<float> **output, Implementation::OutputOrder outputOrder, int offsetOut) override;
	};
}

#endif
