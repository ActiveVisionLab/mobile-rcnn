// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Base class for all LNTNodes. LNTNodes provide  a platform independant 
//   wrapper for an underlying  platform specific Implementation. 
//   LNTNodes represent nodes in a CNN (Convolutional, Pooling, etc...) and 
//   are chained together to build an entire CNN. 

#pragma once

#include <string.h>
#include <vector>

#include "../../ORUtils/ImageTypes.h"
#include "../../ORUtils/MathTypes.h"

#include "TensorInfo.h"

#include "../Core/Device.h"
#include "../Nodes/Implementation/Implementation.h"

namespace LNTLib
{
	class Node
	{
	public:
		// Brief:
		//   Sets the Nodes name and initilaises other properties
		//   to NULL/0 where appropriate.
		Node(std::string name) : implementation(NULL), inputGeometries(NULL), noInputs(0) {
			this->name = name;
		}

		// Brief: 
		//   Configures the input and output geometries off the node.
		//   These are used in the allocation of the nodes Implementation.
		virtual void SetGeometry(TensorInfo *inputGeometries, int noInputs) = 0;

		// Brief:
		//   Sets the number of images in the batch
		virtual void SetMaximumBatchSize(int noTensorsInBatch)
		{
			for (int i = 0; i < noInputs; i++)
				this->inputGeometries[i].n = noTensorsInBatch;
			
			this->outputGeometry.n = noTensorsInBatch;
			
			implementation->ReAllocateOnNewBatchSize(false);
		}

		// Brief:
		//   Sets the number of images in the batch
		virtual void SetCurrentBatchSize(int noTensorsInBatch)
		{
			//TODO assumes currentBatchSize < maximumBatchSize

			for (int i = 0; i < noInputs; i++)
				this->inputGeometries[i].n = noTensorsInBatch;

			this->outputGeometry.n = noTensorsInBatch;

			implementation->ReAllocateOnNewBatchSize(true);
		}
		
		// Brief:
		//   Allocates CPU/GPU memory required as well as initialising handles where
		//   needed. This function call is forwarded on to the underlying Implementation.
		virtual inline void Allocate(LNTLib::Device *device) {
			implementation->Allocate(device);
		}

		// Brief:
		//   Frees up CPU/GPU memory and disposes of handles where needed.
		//   This function call is forwarded on to the underlying Implementation.
		virtual inline void Deallocate() {
			implementation->Deallocate();
		}

		// Brief:
		//   Reads weights from input file. This function call
		//   is forwarded on to the underlying Implementation 
		virtual inline void ReadWeights(FILE *f) {
			implementation->ReadWeights(f);
		}

		// Brief:
		//   Performs a forward pass on given input data, updating the 
		//   nodes output ready for use later in the network. This function
		//   call is forwarded on to the underlying Implementation.
		virtual inline void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs) {
			implementation->Forward(inputs);
		}

		// Brief:
		//   Performs a backwards pass on given input data. 
		//   This function call is forwarded on to the underlying
		//   Implementation.
		//   Note: Not currently supported.
		virtual inline void Backward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs) {
			implementation->Backward(inputs);
		}

		// Brief:
		//   Returns nodes output. This function call is forwarded
		//   on to the underlying LNT_Implementation.
		virtual inline ORUtils::MemoryBlock<float>* Output() {
			return implementation->Output();
		}

		// Brief:
		//   Returns nodes output. This function call is forwarded
		//   on to the underlying LNT_Implementation, but creates a
		//   CPU pointer;
		virtual inline void Output(ORUtils::MemoryBlock<float>** res, Implementation::OutputOrder outputOrder = Implementation::HWD, int offsetOut = 0) {
			return implementation->Output(res, outputOrder, offsetOut);
		}

		// Brief:
		//   Getter for name.
		inline std::string Name() { return name; }

		// Brief:
		//   Getter for outputGeometry.
		virtual inline TensorInfo OutputGeometry() { return outputGeometry; }

		// Brief:
		//   Getter for inputGeometry.
		virtual inline TensorInfo* InputGeometries() { return inputGeometries; }

		// Brief:
		//   Getter for noInputs.
		inline int NoInputs() { return noInputs; }

		// Brief:
		//	 Sets the node output to the specified memory block,
		//	 assumed to be allocated on the correct platform.
		virtual void SetOutput(ORUtils::MemoryBlock<float> *output) {
			implementation->SetOutput(output);
		}

	protected:
		std::string name;

		Implementation *implementation;

		TensorInfo *inputGeometries; int noInputs;
		TensorInfo outputGeometry;

	}; // class Node

} // namespace LNTLib
