// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Norm does a channel normalisation.

#pragma once

#include "../Core/Node.h"

namespace LNTLib {
	class Norm : public Node
	{
	public:
		// Brief:
		//   Calls parent constructor.
		Norm(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				fprintf(stderr, "Unsupported device type for: Upsample\n"); exit(1);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Norm\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is equal to its input geometry.
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry = this->inputGeometries[0];
		}
	}; // class Norm
} // namespace LNTLib
