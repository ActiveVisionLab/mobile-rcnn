// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Concat takes inputs of same height and width and
//   concatenates them into a single output e.g. 5x5x3 and 5x5x5 inputs
//   would produce an output of 5x5x8.

#pragma once

#include "../Core/Node.h"

namespace LNTLib
{
	class Concat : public Node
	{
	public:
		// Brief:
		//   Calls parent constructor.
		Concat(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				fprintf(stderr, "Unsupported device type for: Concat\n"); exit(1);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Concat\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is given as follows:
		//     output channels = SUM(input channels)
		//     output width = input width
		//     output height = input height 
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			int c = 0;
			for (int inputId = 0; inputId < this->noInputs; inputId++)
				c += this->inputGeometries[inputId].c;

			this->outputGeometry.c = c;
			this->outputGeometry.h = this->inputGeometries[0].h;
			this->outputGeometry.w = this->inputGeometries[0].w;
			this->outputGeometry.n = this->inputGeometries[0].n;
		}
	}; // class Concat
} // namespace LNTLib
