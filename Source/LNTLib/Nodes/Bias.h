// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Bias represents a bank of convolutional filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Bias_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Bias : public Node
	{
	public:
		// Brief:
		//   Enum of supported biasing methods.
		enum Method
		{
			Bias_Features,
			Bias_Weights
		};

		// Brief:
		//   Struct to be passed to constructor.
		struct NodeParams
		{
			Method method;
		};

		// Brief:
		//   Calls parent constructor and initilises params.
		Bias(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			implementation = NULL;

			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Bias_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Bias\n"); exit(1);
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
		//   Getter for params.
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;

	}; // class Bias
} // namespace LNTLib
