// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Upsample upsamples a tensor.

#pragma once

#include "../Core/Node.h"
#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Upsample_Implementation_CUDNN.h"
#endif

namespace LNTLib {
	class Upsample : public Node
	{
	public:
		// Brief:
		//   Enum of supported upsampling methods.
		enum Method
		{
			Upsample_Bilinear,
			Upsample_Nearest,
			Upsample_SarojBilinear
		};

		// Brief:
		//
		struct NodeParams
		{
			Vector2i upsample;
			int crop;
			Vector2i kernel;
			Vector2i rect;
			Method method;
		};

		// Brief:
		//   Calls parent constructor.
		Upsample(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Upsample_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Upsample\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is equal to its input geometry.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry.n = this->inputGeometries[0].n;
			this->outputGeometry.c = this->inputGeometries[0].c;

			if (params.rect.x > 0 && params.rect.y > 0)
			{
				this->outputGeometry.w = params.rect.x;
				this->outputGeometry.h = params.rect.y;
			}
			else
			{
				this->outputGeometry.w = this->inputGeometries[0].w * params.upsample.x;
				this->outputGeometry.h = this->inputGeometries[0].h * params.upsample.y;
			}
		}

		// Brief:
		//   Getter for params.
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;
	}; // class Upsample
} // namespace LNTLib
