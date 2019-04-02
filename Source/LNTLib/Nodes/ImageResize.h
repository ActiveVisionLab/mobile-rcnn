// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Node.h"

namespace LNTLib
{
	class ImageResize : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties:
		//     size: input image x, y, and z dimension. 
		struct NodeParams
		{
			Vector4i size;
			Vector4i clipRect;
		};

		// Brief:
		//   Calls parent constructor and initilises params and image.
		ImageResize(std::string name, NodeParams params, Device* device) : Node(name), params(params), image(NULL)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				fprintf(stderr, "Unsupported device type for: ImageResize\n"); exit(1);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: ImageResize\n"); exit(1);

			}
		}

		// Brief:
		//   Sets the image to be forwarded through the network.
		void SetImage(ORFloat4Image *image)
		{
			this->image = image;
		}

		// Brief:
		//   Nodes output geometry simply matches that of the image.
		//   Note: noInputs should be 0, as they are ignored regardless.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			outputGeometry.n = 1;
			outputGeometry.c = params.size.z;
			outputGeometry.h = params.size.y - (params.clipRect.y + params.clipRect.w);
			outputGeometry.w = params.size.x - (params.clipRect.x + params.clipRect.z);
		}

		// Brief:
		//   Getter for image.
		inline ORFloat4Image* Image() { return image; }

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;
		ORFloat4Image *image;

	}; // class ImageResize
} // namespace LNTLib
