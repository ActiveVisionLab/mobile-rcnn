// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief:
//   ConvTransposed represents a bank of transposed convolutional
//   filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/ConvTransposed_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class ConvTransposed : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties:
		//     kernelSize: x, y, and z dimension of the kernels
		//     noOutpts: number of output channels
		//     upscale: x and y stride of non transposed convolution
		//     crop: top, bottom, left, and right padding of the non transposed convolution
		//     hasBiases: true if biases should be applied after convolution
		struct NodeParams
		{
			Vector3i kernelSize;
			int noOutputs;
			Vector2i upsample;
			Vector4i crop;
			bool hasBiases;
			bool isDepthWise;
		};

		// Brief:
		//   Calls parent constructor, initilises params and selects underlying implementation.
		ConvTransposed(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new ConvTransposed_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: ConvTransposed\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is given as follows:
		//     output channels = number of kernels
		//     output width = (upsample in x) * (input width - 1) + kernel width - (left crop + right crop)
		//     output height = (upsample in y) * (input height - 1) + kernel height - (top crop + bottom crop)
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry.c = params.noOutputs;
			this->outputGeometry.n = this->inputGeometries[0].n;
			this->outputGeometry.h = (this->inputGeometries[0].h - 1) * params.upsample.y + params.kernelSize.y - (params.crop.x + params.crop.y);
			this->outputGeometry.w = (this->inputGeometries[0].w - 1) * params.upsample.x + params.kernelSize.x - (params.crop.z + params.crop.w);
		}

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

		void SetDepthWise(bool isDepthWise) { params.isDepthWise = isDepthWise; }

	private:
		NodeParams params;

	}; // class ConvTransposed 
} // namespace LNTLib
