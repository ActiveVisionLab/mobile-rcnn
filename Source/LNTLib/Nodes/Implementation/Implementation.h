// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Base class for all Implementations. Implementations 
//   are responsible for the underlying platform specific implementation 
//   of CNN computations. 

#pragma once

#include "../../../ORUtils/MemoryBlock.h"
#include "../../../ORUtils/MathTypes.h"

#include "../../Core/Device.h"

namespace LNTLib
{
	// Declarations
	class Node;

	class Implementation
	{
	public:
		// Brief:
		//   Tensor dim order for the Output(MemoryBlock) method
		enum OutputOrder
		{
			HWD, DHW, TEST
		};
		
		// Brief:
		//   Initialises pointer to Node and output to NULL
		Implementation(Node *node) : node(node), output(NULL), usingExternalOutput(false) { }

		// Brief:
		//   Allocates the required CPU/GPU memory, initiating
		//   various handles when possible.
        virtual void Allocate(LNTLib::Device *device) = 0;

		// Brief:
		//   Frees up the resources currently owned by the 
		//   implementation.
		virtual void Deallocate() = 0;

		// Brief:
		//   Reallocates resources when the batch size is changed
		virtual void ReAllocateOnNewBatchSize(bool descriptorOnly) = 0;
		
		// Brief:
		//   Reads weights from input file. Weights are interpreted
		//   in the following heirarchy:
		//     output channel -> input channel -> row -> column.
		//   Not all underlying frameworks require this ordering of 
		//   weights and thus the weights are reordered where approptiate.
		//   In some instances underlying framework handles may be initialised
		//   here also. 
		//   Note: Default implementation is to do nothing. This saves implementing
		//   in classes that don't require it. 
		virtual void ReadWeights(FILE *f) { };

		// Brief:
		//   Performs the node specific (e.g. convolutional) and platform
		//   specific (e.g. CPU) forward pass operation on the given input.
		virtual void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs) = 0;

		// Brief:
		//   Performs the node specific (e.g. convolutional) and platform
		//   specific (e.g. CPU) backward pass operation on the given input.
		//   Note: Currently not supported.
		virtual void Backward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs) { };

		// Brief:
		//   Getter for output in the internal device specific implementation.
		inline ORUtils::MemoryBlock<float>* Output() { return output; }

        // Brief:
        //   Getter for output in standard CPU memory
		virtual void Output(ORUtils::MemoryBlock<float> **output, OutputOrder outputOrder = Implementation::HWD, int offsetOut = 0) = 0;
        
		// Brief:
		//	 Sets the node output to the specified memory block,
		//	 assumed to be allocated on the correct platform.
		virtual void SetOutput(ORUtils::MemoryBlock<float> *output) {
            if (this->output != NULL && !usingExternalOutput) delete this->output;
            
			this->output = output;
            usingExternalOutput = true;
		}

		virtual ~Implementation() { }

	protected:
		Node * node;

		ORUtils::MemoryBlock<float> *output;
        bool usingExternalOutput;
	};
} // namespace LNTLib
