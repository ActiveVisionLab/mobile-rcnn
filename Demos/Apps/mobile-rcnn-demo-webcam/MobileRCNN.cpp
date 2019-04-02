// Copyright 2014-2019 Oxford University Innovation Limited and the authors of LightNet

#include <iostream>
#include <LNTLibRedist.h>

#ifdef COMPILE_WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

int main(int argc, char** argv)
{
#ifdef COMPILE_WITH_OPENCV
	// neural network files, must be specified
	std::string folderName = std::string(argv[1]);
	std::string f_proto_backbone = folderName + "/mobileRCNN_base.txt";
	std::string f_weights_backbone = folderName + "/mobileRCNN_base.bin";
	std::string f_proto_det = folderName + "/mobileRCNN_det.txt";
	std::string f_weights_det = folderName + "/mobileRCNN_det.bin";
	std::string f_proto_mask = folderName + "/mobileRCNN_mask.txt";
	std::string f_weights_mask = folderName + "/mobileRCNN_mask.bin";

	int camera_id = (argc >= 3) ? std::atoi(argv[2]) : 0;

	// opencv capture device
	cv::VideoCapture cap(camera_id);
	
	// change the capture resolution to the requested one
	if (argc >= 5) {
		cap.set(3, std::atoi(argv[3])); cap.set(4, std::atoi(argv[4]));
	}

	// open the camera
	if (!cap.isOpened())
	{
		std::cerr << "unable to open camera!\n";
		return -1;
	}

	// initialise the detector
	MobileRCNN_Parameters params;
	MobileRCNN_Init(f_proto_backbone.c_str(), f_weights_backbone.c_str(), 
		f_proto_det.c_str(), f_weights_det.c_str(), f_proto_mask.c_str(), f_weights_mask.c_str(), params);

	// image dims
	i2 inputSize = MobileRCNN_InputSize();

	// opencv images for the various processing stages
	cv::Mat img_camera; //image from camera
	cv::Mat img_resized = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // image resized to width * ..
	cv::Mat img_padded = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // imaged padded with width * height
	cv::Mat img_4ch = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // image with the 4th channel added
	cv::Mat img_display = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // final displayed result
	
	// we'll store detections here
	int noMaxDetections = 256;
	MobileRCNN_Detection *detections = new MobileRCNN_Detection[noMaxDetections];

	// read images from camera, quit on esc
	int key; 
	while (cap.read(img_camera))
	{
		// add missing 4th channel
		cv::cvtColor(img_camera, img_4ch, CV_BGR2RGBA, 4);

		// resize the camera image to our target width 
		float resize_factor = (float)inputSize.x / (float)img_camera.size().width;
		cv::resize(img_4ch, img_resized, cv::Size(), resize_factor, resize_factor);

		// pad to width * height
		img_resized.copyTo(img_padded(cv::Rect(0, 0, img_resized.cols, img_resized.rows)));

		// run the detector on the images
		MobileRCNN_Process(detections, noMaxDetections, img_padded.data);

		// draw the results on the out image
		MobileRCNN_DrawLatestResult(img_display.data, true);

		// convert back to RGBA to display
		cv::cvtColor(img_display, img_display, CV_RGBA2BGRA, 4);

		// display the result
		cv::imshow("segmentation", img_display(cv::Rect(0, 0, img_resized.cols, img_resized.rows)));

		// check if esc was pressed
		key = cv::waitKey(1);
		if (key == 27) break;
	}

	// shutdown the detector
	MobileRCNN_Shutdown();

	// delete the detections array
	delete[] detections;

	// shutdown opencv
	cv::destroyAllWindows();

#else
	std::cout << "OpenCV needed for the webcam demo!" << std::endl;
#endif

	return 0;
}
