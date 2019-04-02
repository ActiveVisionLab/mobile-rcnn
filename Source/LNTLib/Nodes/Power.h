// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Power represents a bank of convolutional filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Power_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Power : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties: 
		//     shift:
		//     scale: 
		//     power: 
		struct NodeParams
		{
			float shift;
			float scale;
			float power;
		};

		// Brief:
		//   Calls parent constructor and initilises params.
		Power(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Power_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Power\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is the same as the input geometry
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

	}; // class Power
} // namespace LNTLib
