// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Node.h"
#include "../../ORUtils/MathUtils.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Crop_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Crop : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties:
		//     topLeftCorner: row and col offset of precropped image.
		struct NodeParams
		{
			Vector2i topLeft;
		};

		// Brief:
		//   Calls parent constructor, initilises params and selects underlying implementation.
		Crop(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Crop_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Crop\n"); exit(1);

			}
		}

		// Brief:
		//   Nodes output geometry is given as follows:
		//     output channels = input(0) channels
		//     output width = input(1) width
		//     output height = input(1) height
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;

			this->inputGeometries = inputGeometries;

			this->outputGeometry.n = inputGeometries[0].n;
			this->outputGeometry.c = inputGeometries[0].c;
			this->outputGeometry.h = inputGeometries[1].h;
			this->outputGeometry.w = inputGeometries[1].w;

			int diffHeight = inputGeometries[0].h - inputGeometries[1].h;
			int diffWidth = inputGeometries[0].w - inputGeometries[1].w;

			diffBottom = MAX(0, diffHeight - params.topLeft.x);
			diffRight = MAX(0, diffWidth - params.topLeft.y);
			diffTop = diffHeight - diffBottom;
			diffLeft = diffWidth - diffRight;
		}

		// Brief:
		//   Getters for diffs.
		inline int DiffTop() { return diffTop; }
		inline int DiffLeft() { return diffLeft; }

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;

		int diffTop;
		int diffBottom;
		int diffLeft;
		int diffRight;

	};
}
