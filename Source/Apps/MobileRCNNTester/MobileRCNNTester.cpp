// Copyright 2015 the authors of LightNet

#include <cstdlib>
#include <exception>
#include <iostream>
#include <fstream>
#include <string>

#include "../../ORUtils/ImageTypes.h"
#include "../../ORUtils/FileUtils.h"
#include "../../ORUtils/NVTimer.h"

#include "../../LNTLib/LNTLib.h"

using namespace LNTLib;

int main(int argc, char** argv)
try
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

	LNTLib::Device *device = new LNTLib::Device("CUDNN");

	MobileRCNN::Parameters params(true, false, 2);
	MobileRCNN *mobileRCNN = new MobileRCNN(f_proto_backbone.c_str(), f_weights_backbone.c_str(),
		f_proto_det.c_str(), f_weights_det.c_str(), f_proto_mask.c_str(), f_weights_mask.c_str(), device, params);

	ORUChar4Image *image = new ORUChar4Image(mobileRCNN->InputSize(), MEMORYDEVICE_CPU);
	ORUChar4Image *output = new ORUChar4Image(mobileRCNN->InputSize(), MEMORYDEVICE_CPU);

	for (int i = 0; i < 10; i++)
	{
		ReadImageFromFile(image, f_input.c_str());

		output->SetFrom(image, MEMCPYDIR_CPU_TO_CPU);

		mobileRCNN->Process(image);
		mobileRCNN->DrawResults(output);

		SaveImageToFile(output, "out.ppm");
	}

	delete mobileRCNN;
	delete device;
	delete image;
}
catch (std::exception& e)
{
	std::cerr << e.what() << '\n';
	return EXIT_FAILURE;
}
