// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#include "MobileRCNN.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>

#include "../../ORUtils/FileUtils.h"
#include "../../ORUtils/PixelPrimitives.h"

using namespace LNTLib;

// ************************
// Backbone Network
// ************************

void MobileRCNN::BackboneNetwork::InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs)
{
	ORUChar4Image *inputImage = (ORUChar4Image*)inputs[0];

	const int hNew = nodes[0]->OutputGeometry().h;
	const int wNew = nodes[0]->OutputGeometry().w;

	if (inputImage->NoDims().x != wNew || inputImage->NoDims().y != hNew)
		DIEWITHEXCEPTION("Input image size different from allocate network input size!");

	if (preprocessedImage == NULL)
		preprocessedImage = new ORFloat4Image(Vector2i(wNew, hNew), true, false, false);

	Vector4u *inputData = inputImage->Data(MEMORYDEVICE_CPU);
	Vector4f *preprocessedData = preprocessedImage->Data(MEMORYDEVICE_CPU);

	int dataSize = preprocessedImage->DataSize();
	for (int i = 0; i < dataSize; i++)
	{
		Vector4f upix = inputData[i].toFloat() / 255.0f;
		Vector4f fpix;

		fpix.x = (upix.x - 0.485f) / 0.229f;
		fpix.y = (upix.y - 0.456f) / 0.224f;
		fpix.z = (upix.z - 0.406f) / 0.225f;
		fpix.w = 0.0f;

		preprocessedData[i] = fpix;
	}

	((ImageInput*)nodes[0])->SetImage(preprocessedImage);
}

void MobileRCNN::BackboneNetwork::NetworkToResult()
{
	//read the feature for the detector head (to be sent to roi align)
	for (int levelId = 0; levelId < noFPNLevels; levelId++)
		fpn_features[levelId] = nodes[features_ids[levelId]]->Output();

	// read the class scores and bbox predictions
	for (int levelId = 0; levelId < noRPNLevels; levelId++)
	{
		// read the outputs of the RPN, to be NMSed
		nodes[cls_logits_ids[levelId]]->Output(&cls_scores[levelId], Implementation::DHW);
		nodes[bbox_pred_ids[levelId]]->Output(&bbox_delta[levelId], Implementation::DHW);
	}
}

// ************************
// Detection Network
// ************************

void MobileRCNN::DetectionNetwork::SetConfig(const ORUtils::MemoryBlock<Vector4f> *boxes, const ORUtils::MemoryBlock<int> *memberships,
	int noBoxes, const Vector2i *fpnLevelSizes, int noFPNLevels, Vector2i inputImageSize)
{
	((RoIAlign*)nodes[0])->SetConfig(boxes, memberships, noBoxes, fpnLevelSizes, noFPNLevels, inputImageSize);
}

void MobileRCNN::DetectionNetwork::InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs)
{
	// sets the base features from which the regions will be extracted
	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	roiAlign->SetBaseFeatures(inputs);
}

void MobileRCNN::DetectionNetwork::NetworkToResult()
{
	//ORUtils::MemoryBlock<float> *test = NULL;
	//bbox_deltas->Output(&test);// , Implementation::TEST);
	//WriteToBIN(test->Data(MEMORYDEVICE_CPU), test->DataSize(), "C:/Temp/cudnn_det_bbox_deltas.bin");

	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	int noBoxes = roiAlign->Params().noBoxes;

	LNTLib::Node *class_prob = nodes[nodes.size() - 2];
	LNTLib::Node *bbox_deltas = nodes[nodes.size() - 1];

	int currentGlobalOffset_scores = 0, currentGlobalOffset_bbox_deltas = 0, currentBoxOffset = 0;
	int batchSizeInEntries_scores = class_prob->OutputGeometry().w * class_prob->OutputGeometry().h * class_prob->OutputGeometry().c;
	int batchSizeInEntries_bbox_deltas = bbox_deltas->OutputGeometry().w * bbox_deltas->OutputGeometry().h * bbox_deltas->OutputGeometry().c;

	for (int batchId = 0; batchId < noBatchesLastRun; batchId++)
	{
		int currentBatchSize = (noBoxes - currentBoxOffset) < maxBatchSize ? noBoxes - currentBoxOffset : maxBatchSize;

		class_prob->SetOutput(cls_scores[batchId]);
		class_prob->Output(&cls_score, Implementation::DHW, currentGlobalOffset_scores);

		bbox_deltas->SetOutput(bbox_preds[batchId]);
		bbox_deltas->Output(&bbox_delta, Implementation::DHW, currentGlobalOffset_bbox_deltas);

		currentGlobalOffset_scores += currentBatchSize * batchSizeInEntries_scores;
		currentGlobalOffset_bbox_deltas += currentBatchSize * batchSizeInEntries_bbox_deltas;
		currentBoxOffset += currentBatchSize;
	}

	class_prob->SetOutput(NULL);
	bbox_deltas->SetOutput(NULL);
}

void MobileRCNN::DetectionNetwork::Forward(bool waitForComputeToFinish)
{
	size_t noNodes = nodes.size();

	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	int noBoxes = roiAlign->Params().noBoxes;
	int noBatches = (int)ceil((float)noBoxes / (float)maxBatchSize);

	noBatchesLastRun = noBatches;

	device->StartCompute();

	std::vector < ORUtils::MemoryBlock<float>* > currentInputs;

	int currentBatchOffset = 0;
	for (int batchNo = 0; batchNo < noBatches; batchNo++)
	{
		// compute current offset in the region array and current batch size
		int currentBatchSize = (noBoxes - currentBatchOffset) < maxBatchSize ? noBoxes - currentBatchOffset : maxBatchSize;

		// update the network with the newly compute batch size
		roiAlign->SetBatchConfig(currentBatchOffset, currentBatchSize);

		// sets the batch size to the current batch
		for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
			nodes[nodeId]->SetCurrentBatchSize(currentBatchSize);

		// set the out matrices for the current batch
		((CopyOutput*)nodes[nodes.size() - 2])->SetOutput(cls_scores[batchNo]);
		((CopyOutput*)nodes[nodes.size() - 1])->SetOutput(bbox_preds[batchNo]);

		// next is the standard forward through the network
		for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
		{
			LNTLib::Node *currentNode = nodes[nodeId];
			LNTLib::Connection *currentConnection = connections[nodeId];

			size_t noInputs = currentConnection->NoInputs();

			currentInputs.resize(noInputs);

			for (size_t inputId = 0; inputId < noInputs; inputId++)
				currentInputs[inputId] = currentConnection->Input(inputId)->Output();

			currentNode->Forward(currentInputs);
		}

		currentBatchOffset += currentBatchSize;
	}

	if (waitForComputeToFinish) device->FinishCompute();
}

// ************************
// Mask Network
// ************************

void MobileRCNN::MaskNetwork::SetConfig(const ORUtils::MemoryBlock<Vector4f> *boxes, const ORUtils::MemoryBlock<int> *memberships,
	int noBoxes, const Vector2i *fpnLevelSizes, int noFPNLevels, Vector2i inputImageSize)
{
	((RoIAlign*)nodes[0])->SetConfig(boxes, memberships, noBoxes, fpnLevelSizes, noFPNLevels, inputImageSize);
}

void MobileRCNN::MaskNetwork::InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs)
{
	// sets the base features from which the regions will be extracted
	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	roiAlign->SetBaseFeatures(inputs);
}

void MobileRCNN::MaskNetwork::NetworkToResult()
{
	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	int noBoxes = roiAlign->Params().noBoxes;

	LNTLib::Node *mask_prob = nodes[nodes.size() - 1];

	int currentGlobalOffset = 0, currentBoxOffset = 0;
	int batchSizeInEntries = mask_prob->OutputGeometry().w * mask_prob->OutputGeometry().h * mask_prob->OutputGeometry().c;

	for (int batchId = 0; batchId < noBatchesLastRun; batchId++)
	{
		int currentBatchSize = (noBoxes - currentBoxOffset) < maxBatchSize ? noBoxes - currentBoxOffset : maxBatchSize;

		mask_prob->SetOutput(batch_masks[batchId]);
		mask_prob->Output(&mask_probabilities, Implementation::HWD, currentGlobalOffset);

		currentGlobalOffset += currentBatchSize * batchSizeInEntries;
		currentBoxOffset += currentBatchSize;
	}

	mask_prob->SetOutput(NULL);
}

void MobileRCNN::MaskNetwork::Forward(bool waitForComputeToFinish)
{
	size_t noNodes = nodes.size();

	RoIAlign *roiAlign = (RoIAlign*)nodes[0];
	int noBoxes = roiAlign->Params().noBoxes;
	int noBatches = (int)ceil((float)noBoxes / (float)maxBatchSize);

	noBatchesLastRun = noBatches;

	device->StartCompute();

	std::vector < ORUtils::MemoryBlock<float>* > currentInputs;

	int currentBatchOffset = 0;
	for (int batchNo = 0; batchNo < noBatches; batchNo++)
	{
		// compute current offset in the region array and current batch size
		int currentBatchSize = (noBoxes - currentBatchOffset) < maxBatchSize ? noBoxes - currentBatchOffset : maxBatchSize;

		// update the network with the newly compute batch size
		roiAlign->SetBatchConfig(currentBatchOffset, currentBatchSize);

		// sets the batch size to the current batch
		for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
			nodes[nodeId]->SetCurrentBatchSize(currentBatchSize);

		// set the out matrices for the current batch
		((CopyOutput*)nodes[nodes.size() - 1])->SetOutput(batch_masks[batchNo]);

		// next is the standard forward through the network
		for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
		{
			LNTLib::Node *currentNode = nodes[nodeId];
			LNTLib::Connection *currentConnection = connections[nodeId];

			size_t noInputs = currentConnection->NoInputs();

			currentInputs.resize(noInputs);

			for (size_t inputId = 0; inputId < noInputs; inputId++)
				currentInputs[inputId] = currentConnection->Input(inputId)->Output();

			currentNode->Forward(currentInputs);
		}

		currentBatchOffset += currentBatchSize;
	}

	if (waitForComputeToFinish) device->FinishCompute();
}

// ************************
// MobileRCNN main code
// ************************

void MobileRCNN::BuildAnchors(const Vector2i *gridSizes)
{
	int noRPNLevels = baseNetwork->NoRPNLevels();

	auto *cell_anchors = new std::vector<Vector4f>[noRPNLevels];

	for (int levelId = 0; levelId < noRPNLevels; levelId++)
	{
		auto *level_anchors = &cell_anchors[levelId];

		float stride = anchorStrides[levelId], size = sizes[levelId];

		for (int ratioId = 0; ratioId < noAnchorAspectRatios; ratioId++)
		{
			float aspectRatio = aspectRatios[ratioId];

			float scale = size / stride;
			float sizeRatio = (stride * stride) / aspectRatio;

			// these below are rounded in the standard mask rcnn implementation
			float w = sqrtf(sizeRatio);
			float h = sqrtf(sizeRatio) * aspectRatio;

			float x_ctr = 0.5f * (stride - 1.0f);
			float y_ctr = 0.5f * (stride - 1.0f);

			Vector4f anchor = Vector4f(x_ctr - 0.5f * (w * scale - 1.0f), y_ctr - 0.5f * (h * scale - 1.0f),
				x_ctr + 0.5f * (w * scale - 1.0f), y_ctr + 0.5f * (h * scale - 1.0f));

			level_anchors->push_back(anchor);
		}
	}

	for (int levelId = 0; levelId < noRPNLevels; levelId++)
	{
		Vector2i gridSize = gridSizes[levelId];

		Vector4f *level_anchors = cell_anchors[levelId].data();
		int noAnchors = (int)cell_anchors[levelId].size();

		anchors.push_back(new ORUtils::MemoryBlock<Vector4f>(noAnchors * gridSize.x * gridSize.y, MEMORYDEVICE_CPU));
		Vector4f *anchors = this->anchors[levelId]->Data(MEMORYDEVICE_CPU);

		int globalAnchorId = 0;
		for (int gy = 0; gy < gridSize.y; gy++) for (int gx = 0; gx < gridSize.x; gx++)
		{
			Vector2i shift = Vector2i(gx, gy) * (int)anchorStrides[levelId];

			for (int anchorId = 0; anchorId < noAnchors; anchorId++)
			{
				Vector4f anchor = level_anchors[anchorId];
				anchors[globalAnchorId] = Vector4f(anchor.x + shift.x, anchor.y + shift.y, anchor.z + shift.x, anchor.w + shift.y);
				globalAnchorId++;
			}
		}
	}

	delete[] cell_anchors;
}

bool MobileRCNN::GetBBoxPrediction(Vector4f &bbox, Vector4f anchor, Vector4f delta, Vector4f weights, float minSize, Vector2i imgSize)
{
	const float BBOX_XFORM_CLIP = logf(1000.0f / 16.0f); // TODO param

	float w = anchor.z - anchor.x + 1.0f;
	float h = anchor.w - anchor.y + 1.0f;
	float ctr_x = anchor.x + 0.5f * w;
	float ctr_y = anchor.y + 0.5f * h;

	float pred_ctr_x = delta.x * weights.x * w + ctr_x;
	float pred_ctr_y = delta.y * weights.y * h + ctr_y;
	float pred_w = expf(fminf(delta.z * weights.z, BBOX_XFORM_CLIP)) * w;
	float pred_h = expf(fminf(delta.w * weights.w, BBOX_XFORM_CLIP)) * h;

	// final prediction
	Vector4f prediction(
		pred_ctr_x - 0.5f * pred_w,
		pred_ctr_y - 0.5f * pred_h,
		pred_ctr_x + 0.5f * pred_w - 1.0f,
		pred_ctr_y + 0.5f * pred_h - 1.0f);

	// clamp to image bounds
	prediction.x = CLAMP(prediction.x, 0, imgSize.x - 1);
	prediction.y = CLAMP(prediction.y, 0, imgSize.y - 1);
	prediction.z = CLAMP(prediction.z, 0, imgSize.x - 1);
	prediction.w = CLAMP(prediction.w, 0, imgSize.y - 1);

	w = prediction.z - prediction.x + 1.0f;
	h = prediction.w - prediction.y + 1.0f;
	ctr_x = prediction.x + 0.5f * w;
	ctr_y = prediction.y + 0.5f * h;

	bbox = prediction;

	return (w > minSize && h > minSize && ctr_x < imgSize.x && ctr_y < imgSize.y);
}

int MobileRCNN::SelectBoxes(ORUtils::MemoryBlock<Vector4f> *out_boxes, ORUtils::MemoryBlock<float> *out_scores,
	ORUtils::MemoryBlock<Vector4f> *in_boxes, ORUtils::MemoryBlock<float> *in_scores,
	int in_noBoxes, int fpnPostNMS_topN)
{
	Vector4f *boxes_in = in_boxes->Data(MEMORYDEVICE_CPU);
	float *scores_in = in_scores->Data(MEMORYDEVICE_CPU);

	// copy scores from network to sort array
	auto *sorted_scores = sortArray.data();
	for (int scoreId = 0; scoreId < in_noBoxes; scoreId++)
		sorted_scores[scoreId] = MobileRCNN_Score(scoreId, scores_in[scoreId]);

	std::vector<MobileRCNN_Score>::iterator it_begin = sortArray.begin();
	std::vector<MobileRCNN_Score>::iterator it_end = sortArray.begin(); std::advance(it_end, in_noBoxes);

	// sort the score+id+membership array
	std::sort(it_begin, it_end, std::greater<MobileRCNN_Score>());

	Vector4f *boxes_out = out_boxes->Data(MEMORYDEVICE_CPU);
	float *scores_out = out_scores->Data(MEMORYDEVICE_CPU);

	// write back actual bounding box prediction
	int noBoxes = 0;
	int noMaxInBoxes = MIN(fpnPostNMS_topN, in_noBoxes);
	for (int scoreId = 0; scoreId < noMaxInBoxes; scoreId++)
	{
		MobileRCNN_Score score = sorted_scores[scoreId];

		boxes_out[noBoxes] = boxes_in[score.scoreId];
		scores_out[noBoxes] = score.score;

		noBoxes++;
	}

	return noBoxes;
}

int MobileRCNN::SelectBoxes(ORUtils::MemoryBlock<Vector4f> *out_boxes, ORUtils::MemoryBlock<float> *out_scores,
	ORUtils::MemoryBlock<float> *level_bbox_deltas, ORUtils::MemoryBlock<float> *level_scores,
	int preNMS_topN, float minSize, Vector2i imgSize, ORUtils::MemoryBlock<Vector4f> *level_anchors)
{
	int noAnchors = level_anchors->DataSize();

	float *scores = level_scores->Data(MEMORYDEVICE_CPU);
	Vector4f *bbox_deltas = (Vector4f*)level_bbox_deltas->Data(MEMORYDEVICE_CPU);

	// copy only good scores from network to sort array -- reduces selection time considerably
	int noGoodScores = 0;
	auto *sorted_scores = sortArray.data();
	for (int scoreId = 0; scoreId < noAnchors; scoreId++)
	{
		float score_in = scores[scoreId];
		if (score_in >= parameters.goodObjectnessScoreThreshold)
		{
			sorted_scores[noGoodScores] = MobileRCNN_Score(scoreId, score_in);
			noGoodScores++;
		}
	}

	std::vector<MobileRCNN_Score>::iterator it_begin = sortArray.begin();
	std::vector<MobileRCNN_Score>::iterator it_end = sortArray.begin(); std::advance(it_end, noGoodScores);

	// sort the score+id array
	std::sort(it_begin, it_end, std::greater<MobileRCNN_Score>());

	Vector4f *anchors = level_anchors->Data(MEMORYDEVICE_CPU);
	Vector4f *res_boxes = out_boxes->Data(MEMORYDEVICE_CPU);
	float *res_scores = out_scores->Data(MEMORYDEVICE_CPU);

	Vector4f weights(1.0f, 1.0f, 1.0f, 1.0f);

	// compute, filter and write back actual bounding box prediction
	int noBoxes = 0;
	int noMaxAnchors = MIN(preNMS_topN, noGoodScores);
	for (int scoreId = 0; scoreId < noMaxAnchors; scoreId++)
	{
		MobileRCNN_Score score = sorted_scores[scoreId];

		Vector4f delta = bbox_deltas[score.scoreId];
		Vector4f anchor = anchors[score.scoreId];

		Vector4f prediction;
		if (GetBBoxPrediction(prediction, anchor, delta, weights, parameters.minRegionSize, imgSize))
		{
			res_boxes[noBoxes] = prediction;
			res_scores[noBoxes] = score.score;
			noBoxes++;
		}
	}
	return noBoxes;
}

int MobileRCNN::NMS(ORUtils::MemoryBlock<Vector4f> *all_boxes_postNMS, ORUtils::MemoryBlock<float> *all_scores_postNMS, int noBoxes_postNMS,
	const ORUtils::MemoryBlock<Vector4f> *level_boxes_preNMS, const ORUtils::MemoryBlock<float> *level_scores_preNMS, int noBoxes_preNMS,
	int postNMS_topN, float nmsThreshold)
{
	nmsArray_1u->Clear();

	const float *scores_preNMS = level_scores_preNMS->Data(MEMORYDEVICE_CPU);
	float *scores_postNMS = all_scores_postNMS->Data(MEMORYDEVICE_CPU);

	const Vector4f *boxes_preNMS = level_boxes_preNMS->Data(MEMORYDEVICE_CPU);
	Vector4f *boxes_postNMS = all_boxes_postNMS->Data(MEMORYDEVICE_CPU);

	uchar *shouldRemove = nmsArray_1u->Data(MEMORYDEVICE_CPU);
	float *areas = this->areas->Data(MEMORYDEVICE_CPU);

	for (int i = 0; i < noBoxes_preNMS; i++)
	{
		Vector4f box = boxes_preNMS[i];
		areas[i] = (box.z - box.x + 1.0f) * (box.w - box.y + 1.0f);;
	}

	int currentNoBoxes_postNMS = noBoxes_postNMS;
	int noBoxesAdd_postNMS = 0;
	for (int i = 0; i < noBoxes_preNMS; i++)
	{
		if (shouldRemove[i]) continue;

		Vector4f pred_i = boxes_preNMS[i];
		float area_i = areas[i];

		boxes_postNMS[currentNoBoxes_postNMS] = pred_i;
		scores_postNMS[currentNoBoxes_postNMS] = scores_preNMS[i];

		currentNoBoxes_postNMS++;
		noBoxesAdd_postNMS++;

		if (noBoxesAdd_postNMS >= postNMS_topN) break;

		for (int j = i + 1; j < noBoxes_preNMS; j++)
		{
			if (shouldRemove[j]) continue;

			Vector4f pred_j = boxes_preNMS[j];

			Vector4f intersection(
				fmaxf(pred_i.x, pred_j.x),	// left
				fmaxf(pred_i.y, pred_j.y),	// top
				fminf(pred_i.z, pred_j.z),	// right
				fminf(pred_i.w, pred_j.w)	// bottom
			);

			float area_intersection = fmaxf(intersection.z - intersection.x + 1.0f, 0.0f) * fmaxf(intersection.w - intersection.y + 1.0f, 0.0f);

			float iou = area_intersection / (area_i + areas[j] - area_intersection);
			if (iou > nmsThreshold) shouldRemove[j] = 1;
		}
	}

	return currentNoBoxes_postNMS;
}

void MobileRCNN::MapLevelsToFPN(ORUtils::MemoryBlock<int> *memberships_out, const ORUtils::MemoryBlock<Vector4f> *boxes_in, int noBoxes,
	const Vector2i *fpnGridSizes, int noFPNLevels, Vector2i inputImageSize)
{
	float scale_min = fpnGridSizes[0].x / (float)inputImageSize.x, scale_max = fpnGridSizes[noFPNLevels - 1].x / (float)inputImageSize.x;

	float k_min = -log2f(scale_min), k_max = -log2f(scale_max);

	float s0 = 224;
	float lvl0 = 4;
	float eps = 1e-6f;

	const Vector4f *boxes = boxes_in->Data(MEMORYDEVICE_CPU);
	int *membership = memberships_out->Data(MEMORYDEVICE_CPU);

	for (int boxId = 0; boxId < noBoxes; boxId++)
	{
		Vector4f box = boxes[boxId];

		float s = sqrtf((box.z - box.x + 1.0f) * (box.w - box.y + 1.0f));

		float target_lvl = floorf(lvl0 + log2f(s / s0 + eps));
		target_lvl = CLAMP(target_lvl, k_min, k_max) - k_min;
		membership[boxId] = (int)target_lvl;
	}
}

std::vector<int> MobileRCNN::SelectBoxes(std::vector< ORUtils::MemoryBlock<Vector4f>* > &boxes_postDet,
	std::vector< ORUtils::MemoryBlock<float>* > &scores_postDet,
	ORUtils::MemoryBlock<Vector4f> *boxes_postRPN, int noBoxes_postRPN,
	ORUtils::MemoryBlock<float> *bbox_deltas, ORUtils::MemoryBlock<float> *cls_scores,
	int noClasses, float scoreThreshold, Vector2i imgSize)
{
	std::vector<int> noBoxes_postDet;
	noBoxes_postDet.resize(noClasses);

	std::vector< std::vector<MobileRCNN_Score> > sorted_scores;
	sorted_scores.resize(noClasses);

	Vector4f *boxes_rpn = boxes_postRPN->Data(MEMORYDEVICE_CPU);
	Vector4f *deltas = (Vector4f*)bbox_deltas->Data(MEMORYDEVICE_CPU);
	float *scores = cls_scores->Data(MEMORYDEVICE_CPU);

	Vector4f weights(1.0f / 10.0f, 1.0f / 10.0f, 1.0f / 5.0f, 1.0f / 5.0f);

	for (int bboxId = 0; bboxId < noBoxes_postRPN; bboxId++) for (int classId = 1; classId < noClasses; classId++)
	{
		float score = scores[classId + bboxId * noClasses];
		if (score > scoreThreshold) sorted_scores[classId].push_back(MobileRCNN_Score(bboxId, score));
	}

	for (int classId = 1; classId < noClasses; classId++)
	{
		if (!sorted_scores[classId].empty())
		{
			Vector4f *boxes_det = this->boxes_postDet[classId]->Data(MEMORYDEVICE_CPU);
			float *scores_det = this->scores_postDet[classId]->Data(MEMORYDEVICE_CPU);
			int noBoxes = (int)sorted_scores[classId].size();

			std::sort(sorted_scores[classId].begin(), sorted_scores[classId].end(), std::greater<MobileRCNN_Score>());

			for (int boxId = 0; boxId < noBoxes; boxId++)
			{
				MobileRCNN_Score pred_score = sorted_scores[classId][boxId];

				int bboxId = pred_score.scoreId;

				Vector4f nms = boxes_rpn[bboxId];
				Vector4f delta = deltas[classId + bboxId * noClasses];
				Vector4f bbox_pred;	GetBBoxPrediction(bbox_pred, nms, delta, weights, 0.0f, imgSize);

				boxes_det[boxId] = bbox_pred;
				scores_det[boxId] = pred_score.score;
			}

			noBoxes_postDet[classId] = noBoxes;
		}
	}

	return noBoxes_postDet;
}

int MobileRCNN::NMS(ORUtils::MemoryBlock<Vector4f> *boxes_final, ORUtils::MemoryBlock<float>* scores_final, ORUtils::MemoryBlock<int> *classIds_final,
	const std::vector< ORUtils::MemoryBlock<Vector4f>* > &boxes_postDet, const std::vector< ORUtils::MemoryBlock<float>* > &scores_postDet,
	const std::vector< int > &noBoxes_postDet, int noClasses, int postNMS_topN, float nmsThreshold)
{
	Vector4f *boxes = boxes_final->Data(MEMORYDEVICE_CPU);
	float *scores = scores_final->Data(MEMORYDEVICE_CPU);
	int *classIds = classIds_final->Data(MEMORYDEVICE_CPU);

	int noBoxes_final = 0;
	for (int classId = 0; classId < noClasses; classId++)
	{
		if (noBoxes_postDet[classId] > 0)
		{
			int noBoxes_perClass = this->NMS(nmsArray_4f, nmsArray_1f, 0,
				boxes_postDet[classId], scores_postDet[classId], noBoxes_postDet[classId], postNMS_topN, nmsThreshold);

			Vector4f *boxesPerClass = nmsArray_4f->Data(MEMORYDEVICE_CPU);
			float *scoresPerClass = nmsArray_1f->Data(MEMORYDEVICE_CPU);

			for (int boxId = 0; boxId < noBoxes_perClass; boxId++)
			{
				Vector4f box = boxesPerClass[boxId];

				//			// round here so the masks are computed on pixel bounds
				//			// this is not in the original Mask RCNN but makes sense to have
				//			detection.x = roundf(detection.x);
				//			detection.y = roundf(detection.y);
				//			detection.z = roundf(detection.z);
				//			detection.w = roundf(detection.w);

				boxes[noBoxes_final] = box;
				scores[noBoxes_final] = scoresPerClass[boxId];
				classIds[noBoxes_final] = classId;

				noBoxes_final++;
			}
		}
	}

	return noBoxes_final;
}

void MobileRCNN::ExposeResults(ORUtils::MemoryBlock<Vector4f> *boxes_final, ORUtils::MemoryBlock<float>* scores_final, ORUtils::MemoryBlock<int> *classIds_final,
	int noBoxes_final, ORUtils::MemoryBlock<float> *networkMaskOutput, int noMaskClasses, Vector2i maskSize)
{
	detections.resize(noBoxes_final);

	Vector4f *boxes = boxes_final->Data(MEMORYDEVICE_CPU);
	float *scores = scores_final->Data(MEMORYDEVICE_CPU);
	int *classIds = classIds_final->Data(MEMORYDEVICE_CPU);
	float *masks = networkMaskOutput != NULL ? networkMaskOutput->Data(MEMORYDEVICE_CPU) : NULL;

	int maskChannelSize = maskSize.x * maskSize.y;
	int fullMaskSize = maskChannelSize * noMaskClasses;

	for (int boxId = 0; boxId < noBoxes_final; boxId++)
	{
		Detection detection;

		detection.box = boxes[boxId];
		detection.score = scores[boxId];
		detection.classId = classIds[boxId];

		if (parameters.computeMask)
		{
			float *fullMask = &masks[fullMaskSize * boxId];

			if (noMaskClasses > 1) detection.mask = &fullMask[maskChannelSize * detection.classId];
			else detection.mask = fullMask;
		}
		else detection.mask = NULL;

		detections[boxId] = detection;
	}
}

void MobileRCNN::Process(ORUtils::GenericMemoryBlock *input, bool waitForComputeToFinish)
{
	ORUChar4Image *inputImage = (ORUChar4Image*)input;

	Vector2i imageSize = inputImage->NoDims(); Vector2i inputSize = baseNetwork->InputSize();
	if (imageSize.x != inputSize.x || imageSize.y != inputSize.y)
		DIEWITHEXCEPTION("Unsupported input image resolution!");

	StopWatchInterface *timerTotal = NULL;
	StopWatchInterface *timerBase = NULL, *timerOthers = NULL, *timerDet = NULL, *timerMask = NULL;

	if (verboseLevel > 0)
	{
		sdkCreateTimer(&timerTotal); sdkResetTimer(&timerTotal);

		if (verboseLevel > 1)
		{
			sdkCreateTimer(&timerBase); sdkResetTimer(&timerBase);
			sdkCreateTimer(&timerOthers); sdkResetTimer(&timerOthers);
			sdkCreateTimer(&timerDet); sdkResetTimer(&timerDet);
			if (parameters.computeMask) { sdkCreateTimer(&timerMask); sdkResetTimer(&timerMask); }
		}

		sdkStartTimer(&timerTotal);
	}

	// base network timer
	if (verboseLevel > 1) sdkStartTimer(&timerBase);

	// base neural net processing (feature network + RPN)
	baseNetwork->Process(inputImage, true);

	// base network timer
	if (verboseLevel > 1) sdkStopTimer(&timerBase);

	// NMS timer
	if (verboseLevel > 1) sdkStartTimer(&timerOthers);

	// box selection for each fpn level
	noBoxes_postNMS = 0;
	for (int levelId = 0; levelId < baseNetwork->NoRPNLevels(); levelId++)
	{
		// compute and filter bounding box predictions
		noBoxes_preNMS[levelId] = this->SelectBoxes(boxes_preNMS[levelId], scores_preNMS[levelId],
			baseNetwork->BBoxDeltas(levelId), baseNetwork->Scores(levelId),
			parameters.preNMS_topN, parameters.minRegionSize, inputImage->NoDims(), anchors[levelId]);

		// initial level-wise NMS
		noBoxes_postNMS = this->NMS(boxes_postNMS, scores_postNMS, noBoxes_postNMS,
			boxes_preNMS[levelId], scores_preNMS[levelId], noBoxes_preNMS[levelId],
			parameters.postNMS_topN, parameters.lightNMSThreshold);
	}

	// NMS timer
	if (verboseLevel > 1) sdkStopTimer(&timerOthers);

	// final selection of the top candidates over all rpn levels
	noBoxes_postRPN = this->SelectBoxes(boxes_postRPN, scores_postRPN, boxes_postNMS,
		scores_postNMS, noBoxes_postNMS, parameters.rpnPostNMS_topN);

	// map RPN detections to the FPN levels
	this->MapLevelsToFPN(memberships_postRPN, boxes_postRPN, noBoxes_postRPN, baseNetwork->FPNGridSizes(),
		baseNetwork->NoFPNLevels(), inputImage->NoDims());

	// give the resulting regions to the detection head
	detectionNetwork->SetConfig(boxes_postRPN, memberships_postRPN, noBoxes_postRPN,
		baseNetwork->FPNGridSizes(), baseNetwork->NoFPNLevels(), inputImage->NoDims());

	// detection head timer
	if (verboseLevel > 1) sdkStartTimer(&timerDet);

	// run the light head network
	std::vector< ORUtils::MemoryBlock<float>* > fpnFeatures = baseNetwork->FPNFeatures();
	detectionNetwork->Process(std::vector < ORUtils::GenericMemoryBlock* >(fpnFeatures.begin(), fpnFeatures.end()), true);

	// detection head timer
	if (verboseLevel > 1) sdkStopTimer(&timerDet);

	// NMS timer
	if (verboseLevel > 1) sdkStartTimer(&timerOthers);

	// now add the newly computed deltas to the bbox predictions we had before,
	// and select bboxed likely to contain stuff based on the scoreThreshold
	noBoxes_postDet = this->SelectBoxes(boxes_postDet, scores_postDet, boxes_postRPN, noBoxes_postRPN,
		detectionNetwork->BBoxDelta(), detectionNetwork->ClsScore(), detectionNetwork->NoClasses(),
		parameters.nmsScoreThreshold, inputImage->NoDims());

	// final NMS for each class
	this->noBoxes_final = this->NMS(boxes_final, scores_final, classIds_final, boxes_postDet, scores_postDet, noBoxes_postDet,
		detectionNetwork->NoClasses(), parameters.rpnPostNMS_topN, parameters.strongNMSThreshold);

	// NMS timer
	if (verboseLevel > 1) sdkStopTimer(&timerOthers);

	if (parameters.computeMask)
	{
		this->MapLevelsToFPN(memberships_final, boxes_final, noBoxes_final, baseNetwork->FPNGridSizes(),
			baseNetwork->NoFPNLevels(), inputImage->NoDims());

		// give the final regions to the mask head
		maskNetwork->SetConfig(boxes_final, memberships_final, noBoxes_final,
			baseNetwork->FPNGridSizes(), baseNetwork->NoFPNLevels(), inputImage->NoDims());

		if (verboseLevel > 1) sdkStartTimer(&timerMask);

		// run the mask head network
		maskNetwork->Process(std::vector < ORUtils::GenericMemoryBlock* >(fpnFeatures.begin(), fpnFeatures.end()), true);

		if (verboseLevel > 1) sdkStopTimer(&timerMask);
	}

	this->ExposeResults(boxes_final, scores_final, classIds_final, noBoxes_final,
		maskNetwork != NULL ? maskNetwork->MaskProb() : NULL, detectionNetwork->NoClasses(),
		maskNetwork != NULL ? maskNetwork->MaskSize() : Vector2i(0, 0));

	if (verboseLevel > 0)
	{
		sdkStopTimer(&timerTotal);

		if (verboseLevel > 1)
			printf("----------------------------\n");

		printf("done, time elapsed: %4.3f\n", sdkGetTimerValue(&timerTotal));
		sdkDeleteTimer(&timerTotal);

		if (verboseLevel > 1)
		{
			printf("base time: %4.3f\n", sdkGetTimerValue(&timerBase));
			printf("det time: %4.3f\n", sdkGetTimerValue(&timerDet));
			if (parameters.computeMask) printf("mask time: %4.3f\n", sdkGetTimerValue(&timerMask));
			printf("others time: %4.3f\n", sdkGetTimerValue(&timerOthers));

			sdkDeleteTimer(&timerBase);
			sdkDeleteTimer(&timerDet);
			if (parameters.computeMask) sdkDeleteTimer(&timerMask);
			sdkDeleteTimer(&timerOthers);

			for (size_t boxId = 0; boxId < detections.size(); boxId++)
			{
				Detection det = detections[boxId];
				printf("%d ", det.classId);
			}
			if (!detections.empty()) printf("\n");

			printf("----------------------------\n");
		}
	}
}

void DrawLine(ORUChar4Image *dst, Vector4u color, Vector2i s, Vector2i e)
{
	bool isSteep = false;
	Vector4u *img = dst->Data(MEMORYDEVICE_CPU);

	if (ABS(s.x - e.x) < ABS(s.y - e.y))
	{
		std::swap(s.x, s.y);
		std::swap(e.x, e.y);

		isSteep = true;
	}

	if (s.x > e.x)
	{
		std::swap(s.x, e.x);
		std::swap(s.y, e.y);
	}

	int dx = e.x - s.x, dy = e.y - s.y;

	float derror = ABS(dy / (float)(dx));
	float error = 0.0f;

	Vector2i dstDims = dst->NoDims();
	int y = s.y;
	for (int x = s.x; x <= e.x; x++)
	{
		if (isSteep)
		{
			if (y >= 0 && y < dstDims.x && x >= 0 && x < dstDims.y)
				img[y + x * dstDims.x] = color;
		}
		else
		{
			if (x >= 0 && x < dstDims.x && y >= 0 && y < dstDims.y)
				img[x + y * dstDims.x] = color;
		}

		error += derror;
		if (error > 0.5f)
		{
			y += (e.y > s.y ? 1 : -1);
			error -= 1.0f;
		}
	}
}
void MobileRCNN::DrawResults(ORUChar4Image *output)
{
	const MobileRCNN::Detection *detections = this->detections.data();
	int noDetections = (int)this->detections.size();

	Vector2i imgSize = output->NoDims();
	Vector4u *pixels = output->Data(MEMORYDEVICE_CPU);

	Vector4u drawingColor_box(0, 0, 255, 255);
	Vector4f drawingColor_mask[128];
	for (int cz = 0; cz < 8; cz++) for (int cy = 0; cy < 4; cy++) for (int cx = 0; cx < 4; cx++)
		drawingColor_mask[cx + cy * 4 + (7 - cz) * 4 * 4] = Vector4f(cx * 255.0f / 4.0f, cy * 255.0f / 4.0f, cz * 255.0f / 8.0f, 255.0f);

	for (int detId = 0; detId < noDetections; detId++)
	{
		const Detection *det = &detections[detId];

		// compute upper/lower left/right
		Vector2i ul = Vector2i((int)det->box.x, (int)det->box.y);
		Vector2i lr = Vector2i((int)det->box.z, (int)det->box.w);
		Vector2i ur = Vector2i(lr.x, ul.y);
		Vector2i ll = Vector2i(ul.x, lr.y);

		// now draw the bounding box
		DrawLine(output, drawingColor_box, ul, ur);
		DrawLine(output, drawingColor_box, ur, lr);
		DrawLine(output, drawingColor_box, lr, ll);
		DrawLine(output, drawingColor_box, ll, ul);

		// draw mask
		if (parameters.computeMask)
		{
			const float *currentMask = det->mask;
			Vector2i maskSize = this->MaskSize();
			Vector2f scale((float)maskSize.x / (float)(ur.x - ul.x), (float)maskSize.y / (float)(ll.y - ul.y));

			for (int y = ul.y; y <= ll.y; y++) for (int x = ul.x; x <= ur.x; x++)
			{
				Vector2f bboxPoint((float)(x - ul.x) * scale.x - 0.5f, (float)(y - ul.y) * scale.y - 0.5f);

				float valMask = interpolateBilinear_1f_checkbounds(currentMask, bboxPoint, maskSize);
				int hasMax = valMask > 0.5f ? 1 : 0;
				if (hasMax)
				{
					Vector4f currentPixel = pixels[x + y * imgSize.x].toFloat();
					pixels[x + y * imgSize.x] = (currentPixel * 0.4f + (drawingColor_mask[det->classId] * (float)hasMax) * 0.6f).toUChar();
				}
			}
		}
	}
}
