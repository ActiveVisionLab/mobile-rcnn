// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   InnerProduct represents a bank of convolutional filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/InnerProduct_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class InnerProduct : public Node
	{
	public:
		// Brief:
		//   Struct to be passed to constructor.
		//   Properties: 
		//     kernelSize: x, y, and z dimension of the kernels
		//     noOutpts: number of output channels
		//     stride: x and y stride when applying kernels
		//     padding: top, bottom, left, and right padding to the input
		//     hasBiases: true if biases should be applied after the inner product
		//     hasReshape: reshapes after the inner product
		struct NodeParams
		{
			Vector3i kernelSize;
			int noOutputs;
			Vector2i stride;
			Vector4i padding;
			TensorInfo newShape;
			TensorInfo innerOutputGeometry;
			bool hasBiases;
			bool hasReshape;
		};

		// Brief:
		//   Calls parent constructor and initilises params.
		InnerProduct(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new InnerProduct_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: InnerProduct\n"); exit(1);
			}
		}

		// Brief:
		//		*
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			params.innerOutputGeometry.c = params.noOutputs;
			params.innerOutputGeometry.n = this->inputGeometries[0].n;
			params.innerOutputGeometry.h = (this->inputGeometries[0].h - params.kernelSize.x + params.padding.x + params.padding.y) / params.stride.x + 1;
			params.innerOutputGeometry.w = (this->inputGeometries[0].w - params.kernelSize.y + params.padding.z + params.padding.w) / params.stride.y + 1;

			TensorInfo newShape = params.newShape;
			params.hasReshape = false;

			if (newShape.n == -1) DIEWITHEXCEPTION("Auto inference not supported for reshape!");
			if (newShape.n == 0) newShape.n = params.innerOutputGeometry.n;
			else params.hasReshape = true;

			if (newShape.c == -1) DIEWITHEXCEPTION("Auto inference not supported for reshape!");
			if (newShape.c == 0) newShape.c = params.innerOutputGeometry.c;
			else params.hasReshape = true;

			if (newShape.h == -1) DIEWITHEXCEPTION("Auto inference not supported for reshape!");
			if (newShape.h == 0) newShape.h = params.innerOutputGeometry.h;
			else params.hasReshape = true;

			if (newShape.w == -1) DIEWITHEXCEPTION("Auto inference not supported for reshape!");
			if (newShape.w == 0) newShape.w = params.innerOutputGeometry.w;
			else params.hasReshape = true;

			this->outputGeometry = newShape;
		}

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

	private:
		NodeParams params;

	}; // class InnerProduct
} // namespace LNTLib
