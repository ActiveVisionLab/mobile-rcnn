// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Relu represents a ReLU nonlinear activation.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Relu_Implementation_CUDNN.h"
#endif

namespace LNTLib {
	class Relu : public Node
	{
	public:
		struct NodeParams
		{
			float relu_max = 1e10f;
		};

		// Brief:
		//   Calls parent constructor.
		Relu(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Relu_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Relu\n"); exit(1);
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

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

		void SetReluMax(float relu_max) { params.relu_max = relu_max; }
	private:
		NodeParams params;
	}; // class Relu
} // namespace LNTLib
