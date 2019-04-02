// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Network.h"

namespace LNTLib
{
	class MobileRCNN : public Network
	{
	public:
		class Detection
		{
		private:
			friend class MobileRCNN;

			float score;
			int classId;
			Vector4f box;
			float *mask;

		public:
			float Score() const { return score; }
			int ClassId() const { return classId; }
			Vector4f Box() const { return box; }
			const float *Mask() const { return mask; }

			Detection() : score(0), classId(-1), box(-1.0f, -1.0f, -1.0f, -1.0f), mask(NULL) {}
		};

		struct MobileRCNN_Score {
			int scoreId;
			float score;

			MobileRCNN_Score() : scoreId(-1), score(0) {}
			MobileRCNN_Score(int in_scoreId, float in_score) : scoreId(in_scoreId), score(in_score) {}

			bool operator > (const MobileRCNN_Score& str) const
			{
				return (score > str.score);
			}
		};

		class Parameters
		{
		public:
			int preNMS_topN, postNMS_topN, rpnPostNMS_topN;
			float minRegionSize, nmsScoreThreshold, goodObjectnessScoreThreshold;
			float lightNMSThreshold, strongNMSThreshold;

			bool computeMask, useConvRelu;

			int verboseLevel;

			Parameters(bool computeMask = true, bool useConvRelu = false, int verboseLevel = 0,
				int preNMS_topN = 512, int postNMS_topN = 128, int rpnPostNMS_topN = 128,
				float minRegionSize = 0.0f, float nmsScoreThreshold = 0.5f, float goodObjectnessScoreThreshold = 0.5f,
				float lightNMSThreshold = 0.5f, float strongNMSThreshold = 0.5f)
			{
				this->computeMask = computeMask;
				this->useConvRelu = useConvRelu;
				this->verboseLevel = verboseLevel;
				
				this->preNMS_topN = preNMS_topN;
				this->postNMS_topN = postNMS_topN;
				this->rpnPostNMS_topN = rpnPostNMS_topN;
				this->minRegionSize = minRegionSize;
				this->nmsScoreThreshold = nmsScoreThreshold;
				this->goodObjectnessScoreThreshold = goodObjectnessScoreThreshold;
				this->lightNMSThreshold = lightNMSThreshold;
				this->strongNMSThreshold = strongNMSThreshold;
			}
		};

	private:
		class BackboneNetwork : public Network
		{
		private:
			std::vector< ORUtils::MemoryBlock<float>* > fpn_features;
			std::vector< ORUtils::MemoryBlock<float>* > cls_scores;
			std::vector< ORUtils::MemoryBlock<float>* > bbox_delta;

			ORFloat4Image *preprocessedImage;

			int noFPNLevels, noRPNLevels;
			std::vector<int> features_ids, cls_logits_ids, bbox_pred_ids;
			std::vector<Vector2i> rpnGridSizes, fpnGridSizes;

			Vector2i inputSize;

			bool useConvRelu;

		private:
			void InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs);
			void NetworkToResult();

		public:
			BackboneNetwork(const char *f_proto, const char *f_weights, bool useConvRelu,
				LNTLib::Device *device, int verboseLevel = 0) : Network(f_proto, f_weights, device, verboseLevel)
			{
				this->useConvRelu = useConvRelu;

				inputSize = Vector2i(this->Node(0)->OutputGeometry().w, this->Node(0)->OutputGeometry().h);

				preprocessedImage = NULL;

				noFPNLevels = 4;
				noRPNLevels = 5;

				for (int levelId = 0; levelId < noRPNLevels; levelId++)
				{
					cls_scores.push_back(nullptr);
					bbox_delta.push_back(nullptr);
				}

				for (int levelId = 0; levelId < noFPNLevels; levelId++)
					fpn_features.push_back(nullptr);

				if (!useConvRelu)
				{
					features_ids = std::vector<int>{ 111, 107, 103, 99 };
					cls_logits_ids = std::vector<int>{ 116, 121, 126, 131, 136 };
					bbox_pred_ids = std::vector<int>{ 117, 122, 127, 132, 137 };
				}
				else
				{
				}

				for (int levelId = 0; levelId < noRPNLevels; levelId++)
				{
					TensorInfo layerInfo = nodes[cls_logits_ids[levelId]]->OutputGeometry();
					rpnGridSizes.push_back(Vector2i(layerInfo.w, layerInfo.h));
				}

				for (int levelId = 0; levelId < noFPNLevels; levelId++)
				{
					TensorInfo layerInfo = nodes[features_ids[levelId]]->OutputGeometry();
					fpnGridSizes.push_back(Vector2i(layerInfo.w, layerInfo.h));
				}
			}

			~BackboneNetwork()
			{
				delete preprocessedImage;

				for (int levelId = 0; levelId < noRPNLevels; levelId++)
				{
					delete cls_scores[levelId];
					delete bbox_delta[levelId];
				}

				// not needs since this is being deallocated by DeallocateNetwork
				//delete lh_conv;
			}

			Vector2i InputSize() { return inputSize; }

			int NoFPNLevels() { return noFPNLevels; }
			int NoRPNLevels() { return noRPNLevels; }

			const Vector2i *RPNGridSizes() { return rpnGridSizes.data(); }
			const Vector2i *FPNGridSizes() { return fpnGridSizes.data(); }

			ORUtils::MemoryBlock<float>* Scores(int levelId) { return cls_scores[levelId]; }
			ORUtils::MemoryBlock<float>* BBoxDeltas(int levelId) { return bbox_delta[levelId]; }
			ORUtils::MemoryBlock<float>* FPNFeature(int levelId) { return fpn_features[levelId]; }

			std::vector< ORUtils::MemoryBlock<float>* > FPNFeatures() { return fpn_features; }
		};

		class DetectionNetwork : public Network
		{
		private:
			std::vector<ORUtils::MemoryBlock<float>*> cls_scores;
			std::vector<ORUtils::MemoryBlock<float>*> bbox_preds;

			ORUtils::MemoryBlock<float> *cls_score;
			ORUtils::MemoryBlock<float> *bbox_delta;

			int maxBatchSize, noBatchesLastRun;

			int noClasses;

		private:
			void InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs);
			void NetworkToResult();

		public:
			DetectionNetwork(const char *f_proto, const char *f_weights, int postNMS_topN,
				LNTLib::Device *device, int verboseLevel = 0) : Network(f_proto, f_weights, device, verboseLevel)
			{
				maxBatchSize = 128;
				noClasses = nodes[nodes.size() - 2]->OutputGeometry().c;

				// this needs to be done in a nicer way
				MemoryDeviceType memoryDeviceType = (device->Type() != Device::LNTDEVICE_CUDNN) ? MEMORYDEVICE_CPU : MEMORYDEVICE_CUDA;

				int total_cls_scores = postNMS_topN * nodes[nodes.size() - 2]->OutputGeometry().c;
				cls_score = new ORUtils::MemoryBlock<float>(total_cls_scores, MEMORYDEVICE_CPU);

				int total_bbox_preds = postNMS_topN * nodes[nodes.size() - 1]->OutputGeometry().c;
				bbox_delta = new ORUtils::MemoryBlock<float>(total_bbox_preds, MEMORYDEVICE_CPU);

				int noMaxBatches = (int)ceil((float)postNMS_topN / (float)maxBatchSize);

				for (int batchId = 0; batchId < noMaxBatches; batchId++)
				{
					int cls_scores_size = maxBatchSize * nodes[nodes.size() - 2]->OutputGeometry().c;
					cls_scores.push_back(new ORUtils::MemoryBlock<float>(cls_scores_size, memoryDeviceType));

					int bbox_preds_size = maxBatchSize * nodes[nodes.size() - 1]->OutputGeometry().c;
					bbox_preds.push_back(new ORUtils::MemoryBlock<float>(bbox_preds_size, memoryDeviceType));
				}

				size_t noNodes = nodes.size();
				for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
					nodes[nodeId]->SetMaximumBatchSize(maxBatchSize);

				noBatchesLastRun = 0;
			}

			~DetectionNetwork()
			{
				for (size_t outputId = 0; outputId < cls_scores.size(); outputId++)
					delete cls_scores[outputId];

				for (size_t outputId = 0; outputId < bbox_preds.size(); outputId++)
					delete bbox_preds[outputId];

				delete cls_score;
				delete bbox_delta;
			}

			void SetConfig(const ORUtils::MemoryBlock<Vector4f> *boxes, const ORUtils::MemoryBlock<int> *memberships,
				int noBoxes, const Vector2i *fpnLevelSizes, int noFPNLevels, Vector2i inputImageSize);

			void Forward(bool waitForComputeToFinish);

			int NoClasses() { return noClasses; }

			ORUtils::MemoryBlock<float> *ClsScore() { return cls_score; }
			ORUtils::MemoryBlock<float> *BBoxDelta() { return bbox_delta; }
		};

		class MaskNetwork : public Network
		{
		private:
			std::vector<ORUtils::MemoryBlock<float>*> batch_masks;
			ORUtils::MemoryBlock<float>* mask_probabilities;

			int maxBatchSize; int noBatchesLastRun;

		private:
			void InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs);
			void NetworkToResult();

		public:
			MaskNetwork(const char *f_proto, const char *f_weights, int postNMS_topN,
				LNTLib::Device *device, int verboseLevel = 0) : Network(f_proto, f_weights, device, verboseLevel)
			{
				maxBatchSize = 128;

				// this needs to be done in a nicer way
				MemoryDeviceType memoryDeviceType = (device->Type() != Device::LNTDEVICE_CUDNN) ? MEMORYDEVICE_CPU : MEMORYDEVICE_CUDA;

				int total_mask_prob = postNMS_topN * nodes[nodes.size() - 1]->OutputGeometry().c * nodes[nodes.size() - 1]->OutputGeometry().w * nodes[nodes.size() - 1]->OutputGeometry().h;
				mask_probabilities = new ORUtils::MemoryBlock<float>(total_mask_prob, MEMORYDEVICE_CPU);

				int noMaxBatches = (int)ceil((float)postNMS_topN / (float)maxBatchSize);

				for (int batchId = 0; batchId < noMaxBatches; batchId++)
				{
					int masks_size = maxBatchSize * nodes[nodes.size() - 1]->OutputGeometry().c * nodes[nodes.size() - 1]->OutputGeometry().w * nodes[nodes.size() - 1]->OutputGeometry().h;
					batch_masks.push_back(new ORUtils::MemoryBlock<float>(masks_size, memoryDeviceType));
				}

				size_t noNodes = nodes.size();
				for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
					nodes[nodeId]->SetMaximumBatchSize(maxBatchSize);

				noBatchesLastRun = 0;
			}

			~MaskNetwork()
			{
				for (size_t outputId = 0; outputId < batch_masks.size(); outputId++)
					delete batch_masks[outputId];

				delete mask_probabilities;
			}

			void SetConfig(const ORUtils::MemoryBlock<Vector4f> *boxes, const ORUtils::MemoryBlock<int> *memberships,
				int noBoxes, const Vector2i *fpnLevelSizes, int noFPNLevels, Vector2i inputImageSize);

			void Forward(bool waitForComputeToFinish);

			Vector2i MaskSize() { return Vector2i(nodes[nodes.size() - 1]->OutputGeometry().w, nodes[nodes.size() - 1]->OutputGeometry().h); }
			ORUtils::MemoryBlock<float> *MaskProb() { return mask_probabilities; }
		};

	private:
		float *sizes;
		float *anchorStrides;

		int noAnchorAspectRatios;
		float *aspectRatios;

		std::vector<ORUtils::MemoryBlock<Vector4f>*> anchors;

		std::vector<int> noBoxes_preNMS;
		std::vector< ORUtils::MemoryBlock<Vector4f>* > boxes_preNMS;
		std::vector< ORUtils::MemoryBlock<float>* > scores_preNMS;

		int noBoxes_postNMS;
		ORUtils::MemoryBlock<Vector4f> *boxes_postNMS;
		ORUtils::MemoryBlock<float> *scores_postNMS;

		int noBoxes_postRPN;
		ORUtils::MemoryBlock<Vector4f> *boxes_postRPN;
		ORUtils::MemoryBlock<float> *scores_postRPN;
		ORUtils::MemoryBlock<int> *memberships_postRPN;

		std::vector<int> noBoxes_postDet;
		std::vector< ORUtils::MemoryBlock<Vector4f>* > boxes_postDet;
		std::vector< ORUtils::MemoryBlock<float>* > scores_postDet;

		int noBoxes_final;
		ORUtils::MemoryBlock<Vector4f>* boxes_final;
		ORUtils::MemoryBlock<float>* scores_final;
		ORUtils::MemoryBlock<int> *classIds_final;
		ORUtils::MemoryBlock<int> *memberships_final;

		std::vector<MobileRCNN_Score> sortArray;
		ORUtils::MemoryBlock<uchar> *nmsArray_1u;
		ORUtils::MemoryBlock<Vector4f> *nmsArray_4f;
		ORUtils::MemoryBlock<float> *nmsArray_1f;
		ORUtils::MemoryBlock<float> *areas;

		BackboneNetwork *baseNetwork;
		DetectionNetwork *detectionNetwork;
		MaskNetwork *maskNetwork;

		bool useConvRelu;

		Parameters parameters;

		std::vector<Detection> detections;

	private:
		void BuildAnchors(const Vector2i *gridSizes);

		bool GetBBoxPrediction(Vector4f &bbox, Vector4f anchor, Vector4f delta, Vector4f weights, float minSize, Vector2i imgSize);

		int SelectBoxes(ORUtils::MemoryBlock<Vector4f> *out_boxes, ORUtils::MemoryBlock<float> *out_scores,
			ORUtils::MemoryBlock<float> *level_bbox_deltas, ORUtils::MemoryBlock<float> *level_scores,
			int preNMS_topN, float minSize, Vector2i imgSize, ORUtils::MemoryBlock<Vector4f> *level_anchors);

		int SelectBoxes(ORUtils::MemoryBlock<Vector4f> *out_boxes, ORUtils::MemoryBlock<float> *out_scores,
			ORUtils::MemoryBlock<Vector4f> *in_boxes, ORUtils::MemoryBlock<float> *in_scores,
			int in_noBoxes, int rpnPostNMS_topN);

		int NMS(ORUtils::MemoryBlock<Vector4f> *boxes_postNMS, ORUtils::MemoryBlock<float> *scores_postNMS, int noBoxes_postNMS,
			const ORUtils::MemoryBlock<Vector4f> *level_boxes_preNMS, const ORUtils::MemoryBlock<float> *level_scores_preNMS, int noBoxes_preNMS,
			int postNMS_topN, float nmsThreshold);

		void MapLevelsToFPN(ORUtils::MemoryBlock<int> *memberships_out, const ORUtils::MemoryBlock<Vector4f> *boxes_in, int noBoxes,
			const Vector2i *fpnGridSizes, int noFPNLevels, Vector2i inputImageSize);

		std::vector<int> SelectBoxes(std::vector< ORUtils::MemoryBlock<Vector4f>* > &boxes_postDet,
			std::vector< ORUtils::MemoryBlock<float>* > &scores_postDet,
			ORUtils::MemoryBlock<Vector4f> *boxes_postRPN, int noBoxes_postRPN,
			ORUtils::MemoryBlock<float> *bbox_deltas, ORUtils::MemoryBlock<float> *scores,
			int noClasses, float scoreThreshold, Vector2i imgSize);

		int NMS(ORUtils::MemoryBlock<Vector4f> *boxes_final, ORUtils::MemoryBlock<float>* scores_final, ORUtils::MemoryBlock<int> *classIds_final,
			const std::vector< ORUtils::MemoryBlock<Vector4f>* > &boxes_postDet, const std::vector< ORUtils::MemoryBlock<float>* > &scores_postDet,
			const std::vector< int > &noBoxes_postDet, int noClasses, int postNMS_topN, float nmsThreshold);

		void ExposeResults(ORUtils::MemoryBlock<Vector4f> *boxes_final, ORUtils::MemoryBlock<float>* scores_final, ORUtils::MemoryBlock<int> *classIds_final,
			int noBoxes_final, ORUtils::MemoryBlock<float> *networkMaskOutput, int noMaskClasses, Vector2i maskSize);

	public:
		const std::vector<Detection> &Detections() { return detections; }
		Vector2i MaskSize() { return maskNetwork->MaskSize(); }

	public:
		MobileRCNN(const char *f_proto_base, const char *f_weights_base, const char *f_proto_det, const char *f_weights_det,
			const char *f_proto_mask, const char *f_weights_mask, LNTLib::Device *device, Parameters parameters)
			: Network(NULL, NULL, device, parameters.verboseLevel)
		{
			this->parameters = parameters;

			baseNetwork = new BackboneNetwork(f_proto_base, f_weights_base, parameters.useConvRelu, device, 0);
			detectionNetwork = new DetectionNetwork(f_proto_det, f_weights_det, parameters.rpnPostNMS_topN, device, 0);
			if (parameters.computeMask) maskNetwork = new MaskNetwork(f_proto_mask, f_weights_mask, parameters.rpnPostNMS_topN, device, 0);
			else maskNetwork = NULL;

			noAnchorAspectRatios = 3;
			aspectRatios = new float[noAnchorAspectRatios];
			aspectRatios[0] = 0.5f; aspectRatios[1] = 1.0f; aspectRatios[2] = 2.0f;

			int noRPNLevels = baseNetwork->NoRPNLevels();
			sizes = new float[noRPNLevels];
			sizes[0] = 32;
			sizes[1] = 64;
			sizes[2] = 128;
			sizes[3] = 256;
			sizes[4] = 512;

			anchorStrides = new float[noRPNLevels];
			anchorStrides[0] = 4;
			anchorStrides[1] = 8;
			anchorStrides[2] = 16;
			anchorStrides[3] = 32;
			anchorStrides[4] = 64;

			const Vector2i *rpnGridSizes = baseNetwork->RPNGridSizes();
			BuildAnchors(rpnGridSizes);

			noBoxes_preNMS.resize(noRPNLevels);

			for (int levelId = 0; levelId < noRPNLevels; levelId++)
			{
				scores_preNMS.push_back(new ORUtils::MemoryBlock<float>(parameters.preNMS_topN, MEMORYDEVICE_CPU));
				boxes_preNMS.push_back(new ORUtils::MemoryBlock<Vector4f>(parameters.preNMS_topN, MEMORYDEVICE_CPU));
			}

			noBoxes_postNMS = 0;
			scores_postNMS = new ORUtils::MemoryBlock<float>(parameters.postNMS_topN * noRPNLevels, MEMORYDEVICE_CPU);
			boxes_postNMS = new ORUtils::MemoryBlock<Vector4f>(parameters.postNMS_topN * noRPNLevels, MEMORYDEVICE_CPU);

			noBoxes_postRPN = 0;
			scores_postRPN = new ORUtils::MemoryBlock<float>(parameters.rpnPostNMS_topN * noRPNLevels, MEMORYDEVICE_CPU);
			boxes_postRPN = new ORUtils::MemoryBlock<Vector4f>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);
			memberships_postRPN = new ORUtils::MemoryBlock<int>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);

			scores_final = new ORUtils::MemoryBlock<float>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);
			boxes_final = new ORUtils::MemoryBlock<Vector4f>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);
			classIds_final = new ORUtils::MemoryBlock<int>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);
			memberships_final = new ORUtils::MemoryBlock<int>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU);

			sortArray.resize(rpnGridSizes[0].x * rpnGridSizes[0].y * noAnchorAspectRatios);
			nmsArray_1u = new ORUtils::MemoryBlock<uchar>(MAX(parameters.rpnPostNMS_topN, parameters.preNMS_topN), MEMORYDEVICE_CPU);
			nmsArray_4f = new ORUtils::MemoryBlock<Vector4f>(MAX(parameters.rpnPostNMS_topN, parameters.preNMS_topN), MEMORYDEVICE_CPU);
			nmsArray_1f = new ORUtils::MemoryBlock<float>(MAX(parameters.rpnPostNMS_topN, parameters.preNMS_topN), MEMORYDEVICE_CPU);
			areas = new ORUtils::MemoryBlock<float>(parameters.preNMS_topN, MEMORYDEVICE_CPU);

			detections = std::vector<Detection>();

			int noClasses = detectionNetwork->NoClasses();
			for (int classId = 0; classId < noClasses; classId++)
			{
				boxes_postDet.push_back(new ORUtils::MemoryBlock<Vector4f>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU));
				scores_postDet.push_back(new ORUtils::MemoryBlock<float>(parameters.rpnPostNMS_topN, MEMORYDEVICE_CPU));
			}

			if (verboseLevel > 1) printf("MobileRCNN initialised!\n");
		}

		void InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs) { }
		void NetworkToResult() { }

		~MobileRCNN()
		{
			int noClasses = detectionNetwork->NoClasses();

			delete baseNetwork;
			delete detectionNetwork;
			delete maskNetwork;

			for (int levelId = 0; levelId < baseNetwork->NoRPNLevels(); levelId++) {
				delete anchors[levelId];
				delete scores_preNMS[levelId];
				delete boxes_preNMS[levelId];
			}

			delete scores_postNMS;
			delete boxes_postNMS;

			delete scores_postRPN;
			delete boxes_postRPN;
			delete memberships_postRPN;

			delete scores_final;
			delete boxes_final;
			delete classIds_final;
			delete memberships_final;

			for (int classId = 0; classId < noClasses; classId++)
			{
				delete boxes_postDet[classId];
				delete scores_postDet[classId];
			}

			delete nmsArray_1u;
			delete nmsArray_4f;
			delete nmsArray_1f;
			delete areas;

			delete aspectRatios;
			delete sizes;
			delete anchorStrides;
		}

		void Process(ORUtils::GenericMemoryBlock *input, bool waitForComputeToFinish = true);

		void DrawResults(ORUChar4Image *output);

		Vector2i InputSize() { return baseNetwork->InputSize(); }
	};
}
