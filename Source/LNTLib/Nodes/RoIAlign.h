// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   RoiAlign represents a bank of convolutional filters.

#pragma once

#include "../Core/Node.h"

#ifdef COMPILE_WITH_CUDNN
#include "Implementation/CUDNN/RoIAlign_Implementation_CUDNN.h"
#endif

namespace LNTLib
{
	class RoIAlign : public Node
	{
	public:
		struct NodeParams
		{
			Vector2i resolution;
			int noChannels;

			int samplingRatio;

			int noBoxes;
			const ORUtils::MemoryBlock<Vector4f> *boxes;
			const ORUtils::MemoryBlock<int> *memberships;

			Vector2i inputImageSize;
			const Vector2i *fpnLevelSizes;
			int noFPNLevels;

			int currentBatchOffset;
			int actualBatchSize;
		};

		// Brief:
		//   Calls parent constructor and initilises params.
		RoIAlign(std::string name, NodeParams params, Device* device) : Node(name), params(params)
		{
			switch (device->Type())
			{
#ifdef COMPILE_WITH_CUDNN
			case Device::LNTDEVICE_CUDNN:
				implementation = new RoIAlign_Implementation_CUDNN(this);
				break;
#endif
			default:
				fprintf(stderr, "Unsupported device type for: RoiAlign\n"); exit(1);
			}
		}

		// Brief:
		//   Nodes output geometry is the same as the input geometry
		void SetGeometry(TensorInfo *inputGeometries, int noInputs)
		{
			this->noInputs = noInputs;
			this->inputGeometries = NULL;

			this->outputGeometry.n = 1; // this will be changed at runtime based on the number of regions
			this->outputGeometry.c = params.noChannels;

			this->outputGeometry.w = params.resolution.x;
			this->outputGeometry.h = params.resolution.y;
		}

		// Brief:
		//   Getter for params
		inline NodeParams Params() { return params; }

		void SetConfig(const ORUtils::MemoryBlock<Vector4f>* boxes, const ORUtils::MemoryBlock<int> *memberships, int noBoxes,
			const Vector2i *fpnLevelSizes, int noFPNLevels, Vector2i inputImageSize)
		{
			params.boxes = boxes;
			params.memberships = memberships;
			params.noBoxes = noBoxes;

			params.fpnLevelSizes = fpnLevelSizes;
			params.noFPNLevels = noFPNLevels;
			
			params.inputImageSize = inputImageSize;
		}

		void SetBatchConfig(int currentBatchOffset, int currentBatchSize) {
			params.currentBatchOffset = currentBatchOffset;
			params.actualBatchSize = currentBatchSize;
		}

		void SetBaseFeatures(std::vector< ORUtils::GenericMemoryBlock* > baseFeatures) {
			this->baseFeatures.resize(baseFeatures.size());
			for (size_t featureId = 0; featureId < baseFeatures.size(); featureId++)
				this->baseFeatures[featureId] = (ORUtils::MemoryBlock<float>*)(baseFeatures[featureId]);
		}

		std::vector< ORUtils::MemoryBlock<float>* > GetBaseFeatures() { return this->baseFeatures; }

	private:
		std::vector< ORUtils::MemoryBlock<float>* > baseFeatures;
		NodeParams params;

	}; // class RoiAlign
} // namespace LNTLib
