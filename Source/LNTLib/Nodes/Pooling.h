// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Relu represents a pooling filter.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Pooling_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Pooling : public Node
	{
	public:
		// Brief:
		//   Enum of supported pooling methods.
		enum Method
		{
			Pooling_Max,
			Pooling_Avg
		};

		// Brief:
		//   Struct to be passed to constructor.
		//   Properties: 
		//     poolSize: x and y dimensions of the pooling filter
		//     padding: top, bottom, left, and right padding to the input
		//     stride: x and y stride when applying kernels
		//     method: the desired form of pooling to be applied
		struct NodeParams
		{
			Vector2i poolSize;
			Vector4i padding;
			Vector2i stride;
			bool remove_extra;
			Method method;
		};

		// Brief:
		//   Calls parent constructor.
		Pooling(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Pooling_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Pooling\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry given as follows:
		//     output channels = input channels
		//     output width = (input width - pool width + left padding + right padding)/(stride in x) + 1
		//     output height = (input height - pool height + top padding + bottom padding)/(stride in y) + 1
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry.c = this->inputGeometries[0].c;
			this->outputGeometry.n = this->inputGeometries[0].n;

			if (params.remove_extra)
			{
				this->outputGeometry.h = (int)floor((float)(this->inputGeometries[0].h - params.poolSize.y + params.padding.x + params.padding.y) / (float)params.stride.y) + 1;
				this->outputGeometry.w = (int)floor((float)(this->inputGeometries[0].w - params.poolSize.x + params.padding.z + params.padding.w) / (float)params.stride.x) + 1;
			}
			else
			{
				this->outputGeometry.h = (int)ceil((float)(this->inputGeometries[0].h - params.poolSize.y + params.padding.x + params.padding.y) / (float)params.stride.y) + 1;
				this->outputGeometry.w = (int)ceil((float)(this->inputGeometries[0].w - params.poolSize.x + params.padding.z + params.padding.w) / (float)params.stride.x) + 1;
			}

		}

		// Brief:
		//   Getter for params.
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;

	}; // class Pooling
} // namespace LNTLib
