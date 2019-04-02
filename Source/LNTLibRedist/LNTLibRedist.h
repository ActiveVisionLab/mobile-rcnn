// Copyright 2014-2019 Oxford University Innovation Limited and the authors of LightNet

#ifndef __LNTLIB_REDIST__
#define __LNTLIB_REDIST__

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#else
#    define LIBRARY_API __attribute__ ((visibility ("default")))
#endif

struct _f4 { float x, y, z, w; };
typedef struct _f4 f4;

struct _i2 { int x, y; };
typedef struct _i2 i2;

struct _MobileRCNN_Detection
{
	float score;
	int classId;
	f4 box;
	const float *mask;
};

typedef struct _MobileRCNN_Detection MobileRCNN_Detection;

struct _MobileRCNN_Parameters
{
	bool computeMask = true;
	bool useConvRelu = false;
	int verboseLevel = 0;
	int preNMS_topN = 512;
	int postNMS_topN = 128;
	int rpnPostNMS_topN = 128;
	float minRegionSize = 0.0f;
	float nmsScoreThreshold = 0.5f;
	float goodObjectnessScoreThreshold = 0.5f;
	float lightNMSThreshold = 0.5f;
	float strongNMSThreshold = 0.5f;
};

typedef struct _MobileRCNN_Parameters MobileRCNN_Parameters;

extern "C" LIBRARY_API
void MobileRCNN_Init(const char *f_proto_base, const char *f_weights_base,
	const char *f_proto_det, const char *f_weights_det,
	const char *f_proto_mask, const char *f_weights_mask, MobileRCNN_Parameters parameters);

extern "C" LIBRARY_API
i2 MobileRCNN_InputSize();

extern "C" LIBRARY_API
int MobileRCNN_Process(MobileRCNN_Detection *detections, int noMaxDetections, unsigned char *bytes);

extern "C" LIBRARY_API
void MobileRCNN_DrawLatestResult(unsigned char *bytes, bool overlay);

extern "C" LIBRARY_API
void MobileRCNN_Shutdown();

#endif
