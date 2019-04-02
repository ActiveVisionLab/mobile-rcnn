// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Conv represents a bank of convolutional filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Conv_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Conv : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties:
		//     kernelSize: x, y, and z dimension of the kernels
		//     noOutpts: number of output channels
		//     stride: x and y stride when applying kernels
		//     padding: top, bottom, left, and right padding to the input
		//     hasBiases: true if biases should be applied after convolution
		struct NodeParams
		{
			Vector3i kernelSize;
			int noOutputs;
			Vector2i stride;
			Vector4i padding;
			bool hasBiases;
			bool hasRelu;
			bool isDepthWise;
			float relu_max = 1e10f;
		};

		// Brief:
		//   Calls parent constructor and initilises params.
		Conv(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Conv_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Conv\n"); exit(1);
			}
		}


		// Brief:
		//   Nodes output geometry is given as follows:
		//     output channels = number of kernels
		//     output width = (input width - kernel width + left padding + right padding)/(stride in x) + 1
		//     output height = (input height - kernel height + top padding + bottom padding)/(stride in y) + 1
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			int c = params.noOutputs;
			for (int inputId = 1; inputId < this->noInputs; inputId++)
				c += this->inputGeometries[inputId].c;

			this->outputGeometry.c = c;
			this->outputGeometry.n = this->inputGeometries[0].n;
			this->outputGeometry.h = (this->inputGeometries[0].h - params.kernelSize.x + params.padding.x + params.padding.y) / params.stride.x + 1;
			this->outputGeometry.w = (this->inputGeometries[0].w - params.kernelSize.y + params.padding.z + params.padding.w) / params.stride.y + 1;
		}

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

		void SetDepthWise(bool isDepthWise) { params.isDepthWise = isDepthWise; }
		void SetHasRelu(bool hasRelu) { params.hasRelu = hasRelu; }
		void SetReluMax(float relu_max) { params.relu_max = relu_max; }

	private:
		NodeParams params;

	}; // class Conv
} // namespace LNTLib
