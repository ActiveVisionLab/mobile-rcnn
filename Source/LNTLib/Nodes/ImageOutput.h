// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet


#pragma once

#include "../Core/Node.h"

namespace LNTLib
{
	class ImageOutput : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties:
		//     size: input image x, y, and z dimension. 
		struct NodeParams
		{
			Vector4i clipRect;
			Vector2i outputSizeBeforeRescale;
		};

		// Brief:
		//   Calls parent constructor and initilises params and image.
		ImageOutput(std::string name, NodeParams params, Device* device) : Node(name), params(params), image(NULL)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				fprintf(stderr, "Unsupported device type for: ImageOutput\n"); exit(1);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: ImageOutput\n"); exit(1);

			}
		}

		// Brief:
		//   Sets the image to be forwarded through the network.
		void SetImage(ORFloat4Image *image)
		{
			this->image = image;

			outputGeometry.w = image->NoDims().x;
			outputGeometry.h = image->NoDims().y;
		}

		// Brief:
		//   Nodes output geometry simply matches that of the image.
		//   Note: noInputs should be 0, as they are ignored regardless.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			outputGeometry.n = 1;
			outputGeometry.c = inputGeometries[0].c;
			outputGeometry.h = inputGeometries[0].h + (params.clipRect.y + params.clipRect.w);
			outputGeometry.w = inputGeometries[0].w + (params.clipRect.x + params.clipRect.z);

			params.outputSizeBeforeRescale = Vector2i(outputGeometry.w, outputGeometry.h);
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

	}; // class ImageOutput
} // namespace LNTLib
