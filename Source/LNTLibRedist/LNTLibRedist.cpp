// Copyright 2014-2019 Oxford University Innovation Limited and the authors of LightNet

#include "LNTLibRedist.h"

#include "../LNTLib/LNTLib.h"

static LNTLib::Device *lnt_device = NULL;
static LNTLib::MobileRCNN *lnt_network = NULL;
static ORUChar4Image *lnt_image_in;
static ORUChar4Image *lnt_image_out;

void MobileRCNN_Init(const char *f_proto_base, const char *f_weights_base,
	const char *f_proto_det, const char *f_weights_det,
	const char *f_proto_mask, const char *f_weights_mask, MobileRCNN_Parameters parameters)
{
	lnt_device = new LNTLib::Device("CUDNN");

	LNTLib::MobileRCNN::Parameters params = LNTLib::MobileRCNN::Parameters(parameters.computeMask, parameters.useConvRelu, parameters.verboseLevel,
		parameters.preNMS_topN, parameters.postNMS_topN, parameters.rpnPostNMS_topN, parameters.minRegionSize, parameters.nmsScoreThreshold, parameters.goodObjectnessScoreThreshold,
		parameters.lightNMSThreshold, parameters.strongNMSThreshold);;

	lnt_network = new LNTLib::MobileRCNN(f_proto_base, f_weights_base, f_proto_det, f_weights_det, f_proto_mask, f_weights_mask, lnt_device, params);

	lnt_image_in = new ORUChar4Image(lnt_network->InputSize(), MEMORYDEVICE_CPU);
	lnt_image_out = new ORUChar4Image(lnt_network->InputSize(), MEMORYDEVICE_CPU);
}

i2 MobileRCNN_InputSize()
{
	i2 inputSize_i2; Vector2i inputSize_2i;

	inputSize_2i = lnt_network->InputSize();

	inputSize_i2.x = inputSize_2i.x;
	inputSize_i2.y = inputSize_2i.y;

	return inputSize_i2;
}

int MobileRCNN_Process(MobileRCNN_Detection *detections, int noMaxDetections, unsigned char *bytes)
{
	Vector2i inputSize = lnt_network->InputSize();
	lnt_image_in->SetFrom((Vector4u*)bytes, inputSize.x * inputSize.y, MEMCPYDIR_CPU_TO_CPU);

	lnt_network->Process(lnt_image_in, true);

	const std::vector<LNTLib::MobileRCNN::Detection> &dets = lnt_network->Detections();

	int noDets = MIN(noMaxDetections, (int)dets.size());

	for (int detId = 0; detId < noDets; detId++)
	{
		Vector4f box_in = dets[detId].Box();

		f4 box_out;
		box_out.x = box_in.x; box_out.y = box_in.y; box_out.z = box_in.z; box_out.w = box_in.w;

		detections[detId].box = box_out;
		detections[detId].classId = dets[detId].ClassId();
		detections[detId].score = dets[detId].Score();
		detections[detId].mask = dets[detId].Mask();
	}

	return (int)dets.size();
}

void MobileRCNN_DrawLatestResult(unsigned char *bytes, bool overlay)
{
	if (overlay) lnt_image_out->SetFrom(lnt_image_in, MEMCPYDIR_CPU_TO_CPU);
	else lnt_image_out->Clear();

	lnt_network->DrawResults(lnt_image_out);

	Vector2i inputSize = lnt_network->InputSize();
	memcpy(bytes, lnt_image_out->Data(MEMORYDEVICE_CPU), inputSize.x * inputSize.y * sizeof(Vector4u));
}

void MobileRCNN_Shutdown()
{
	delete lnt_network;
	delete lnt_device;
	delete lnt_image_in;
	delete lnt_image_out;
}
