// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Sum_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Sum : public Node
	{
	public:
		Sum(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Sum_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: Softmax\n"); exit(1);
			}
		}

		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = inputGeometries;

			this->outputGeometry = inputGeometries[0];

			for (int inputId = 1; inputId < noInputs; inputId++)
				this->outputGeometry.c = MIN(this->outputGeometry.c, inputGeometries[inputId].c);
		}
	};
}
