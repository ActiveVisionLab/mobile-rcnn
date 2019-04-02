// Copyright 2014-2019 Oxford University Innovation Limited and the authors of LightNet

#include <string>

#include "FileUtils.h"
#include <LNTLibRedist.h>

int main(int argc, char** argv)
{
	// neural network files, must be specified
	std::string folderName = std::string(argv[1]);
	std::string f_proto_backbone = folderName + "/mobileRCNN_base.txt";
	std::string f_weights_backbone = folderName + "/mobileRCNN_base.bin";
	std::string f_proto_det = folderName + "/mobileRCNN_det.txt";
	std::string f_weights_det = folderName + "/mobileRCNN_det.bin";
	std::string f_proto_mask = folderName + "/mobileRCNN_mask.txt";
	std::string f_weights_mask = folderName + "/mobileRCNN_mask.bin";

	std::string f_input = std::string(argv[2]);

	// initialise the detector
	MobileRCNN_Parameters params;
	params.verboseLevel = 2;
	MobileRCNN_Init(f_proto_backbone.c_str(), f_weights_backbone.c_str(),
		f_proto_det.c_str(), f_weights_det.c_str(), f_proto_mask.c_str(), f_weights_mask.c_str(), params);

	// image dims
	i2 inputSize = MobileRCNN_InputSize();

	// allocate input and out images
	unsigned char *image_in = new unsigned char[inputSize.x * inputSize.y * 4];
	unsigned char *image_out = new unsigned char[inputSize.x * inputSize.y * 4];

	// we'll store detections here
	int noMaxDetections = 256;
	MobileRCNN_Detection *detections = new MobileRCNN_Detection[noMaxDetections];

	// read the image from the file
	// this could be replaced by opencv or such
	ReadImageFromFile(image_in, f_input.c_str());

	// running the same image 10 times
	for (int runId = 0; runId < 10; runId++)
	{
		// run the detector on the images
		MobileRCNN_Process(detections, noMaxDetections, image_in);

		// draw the results on the out image
		MobileRCNN_DrawLatestResult(image_out, true);

		// save the image to disk
		SaveImageToFile(image_out, inputSize.x, inputSize.y, "out.ppm");
	}

	// shutdown the detector
	MobileRCNN_Shutdown();
	
	// cleanup the rest of the memory
	delete image_in;
	delete image_out;
	delete[] detections;

	return 0;
}
