// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Reshape reshape a a tensor to a predefined output

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Reshape_Implementation_CUDNN.h"
#endif

namespace LNTLib {
	class Reshape : public Node
	{
	public:
		struct NodeParams
		{
			Vector3i size_out;
		};

		// Brief:
		//   Calls parent constructor.
		Reshape(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Reshape_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Reshape\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is equal to its input geometry.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry.n = this->inputGeometries[0].n;
			this->outputGeometry.c = params.size_out.z;
			this->outputGeometry.w = params.size_out.x;
			this->outputGeometry.h = params.size_out.y;
		}

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }
	private:
		NodeParams params;
	}; // class Reshape
} // namespace LNTLib
