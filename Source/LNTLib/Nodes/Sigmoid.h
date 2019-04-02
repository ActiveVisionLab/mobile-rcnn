// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Sigmoid represents a Sigmoid nonlinear activation.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Sigmoid_Implementation_CUDNN.h"
#endif

namespace LNTLib {
	class Sigmoid : public Node
	{
	public:
		struct NodeParams
		{
			float Sigmoid_max = 1e10f;
		};

		// Brief:
		//   Calls parent constructor.
		Sigmoid(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Sigmoid_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Sigmoid\n"); exit(1);
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
	private:
		NodeParams params;
	}; // class Sigmoid
} // namespace LNTLib
