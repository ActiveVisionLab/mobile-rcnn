// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/Softmax_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class Softmax : public Node
	{
	public:
		Softmax(std::string name, Device* device) : Node(name)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new Softmax_Implementation_CUDNN(this);
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

			// does softmax across channels
			this->outputGeometry.w = inputGeometries[0].w;
			this->outputGeometry.h = inputGeometries[0].h;
			this->outputGeometry.c = inputGeometries[0].c;
			this->outputGeometry.n = inputGeometries[0].n;
		}
	};
}
