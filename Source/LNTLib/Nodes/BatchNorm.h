// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Batch performs a batch normalisation on inputs, given 
//   pre calculated moments, multipliers, and biases.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/BatchNorm_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class BatchNorm : public Node
	{
	public:
		BatchNorm(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new BatchNorm_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: BatchNorm\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is equal to its input geometry.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry = this->inputGeometries[0];
		}
	};
}
