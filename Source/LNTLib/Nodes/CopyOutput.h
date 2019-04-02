// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/CopyOutput_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class CopyOutput : public Node
	{
	public:
		// Brief:
		//   Calls parent constructor and initilises params and image.
		CopyOutput(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new CopyOutput_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: CopyOutput\n"); exit(1);

			}
		}

		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			outputGeometry = inputGeometries[0];
		}
	}; // class CopyOutput
} // namespace LNTLib
