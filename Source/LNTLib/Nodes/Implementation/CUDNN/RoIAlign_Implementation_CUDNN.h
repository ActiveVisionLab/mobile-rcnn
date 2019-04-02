// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#pragma once

#include "Implementation_CUDNN.h"

namespace LNTLib
{
	class RoIAlign_Implementation_CUDNN : public Implementation_CUDNN
	{
	private:
		ORUtils::MemoryBlock<Vector4f>* boxes;
		ORUtils::MemoryBlock<int>* memberships;
		ORUtils::MemoryBlock<Vector2f> *spatialScales;

		float** fpnLevels_host;
		float** fpnLevels_device;

		Vector2f* spatialScales_host;
		Vector2f* spatialScales_device;

		Vector2i* fpnLevelSizes_device;

	public:
		RoIAlign_Implementation_CUDNN(Node *node) : Implementation_CUDNN(node) { }

		void Allocate(LNTLib::Device *device);

		void Deallocate();

		void Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs);
	};
}

#endif