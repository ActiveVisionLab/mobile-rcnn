/* cppsrc/main.cpp */
#include <napi.h>
#include <LNTLibRedist.h>
#include <opencv2/opencv.hpp>

// opencv capture device
cv::VideoCapture cap;

// image dims
i2 inputSize;

// opencv images for the various processing stages
cv::Mat img_camera, img_resized, img_padded, img_4ch, img_display;
	
// we'll store the detections here
MobileRCNN_Detection detections[256]; 
int noMaxDetections = 256;
int noDetections;

// captures images from the camera
void CaptureCameraImage(const Napi::CallbackInfo& info)
{
	// capture the image from the camera
	cap.read(img_camera);
	
	// add missing 4th channel
	cv::cvtColor(img_camera, img_4ch, CV_BGR2RGBA, 4);
	
	// resize the camera image to our target width 
	float resize_factor = (float)inputSize.x / (float)img_camera.size().width;
	cv::resize(img_4ch, img_resized, cv::Size(), resize_factor, resize_factor);

	// pad to width * height
	img_resized.copyTo(img_padded(cv::Rect(0, 0, img_resized.cols, img_resized.rows)));
}

// runs the neural net processing
void RunDetector(const Napi::CallbackInfo& info)
{
	noDetections = MobileRCNN_Process(detections, noMaxDetections, img_padded.data);
}

// returns the latest captured image into a js array buffer
Napi::TypedArrayOf<unsigned char> GetResizedCameraImage(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
    
	// convert back to RGBA to display
	cv::cvtColor(img_resized, img_display, CV_RGBA2BGRA, 4);
	
	std::vector<uchar> buffer;
    cv::imencode(".jpg", img_display, buffer);
	
	Napi::TypedArrayOf<unsigned char> returnValue = Napi::TypedArrayOf<unsigned char>::New(env, buffer.size());
	
	memcpy(returnValue.Data(), buffer.data(), sizeof(unsigned char) * buffer.size());
	
    return returnValue;
}

// returns the latest captured image into a js array buffer
Napi::TypedArrayOf<unsigned char> GetOriginalCameraImage(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
    
	std::vector<uchar> buffer;
    cv::imencode(".jpg", img_camera, buffer);
	
	Napi::TypedArrayOf<unsigned char> returnValue = Napi::TypedArrayOf<unsigned char>::New(env, buffer.size());
	
	memcpy(returnValue.Data(), buffer.data(), sizeof(unsigned char) * buffer.size());
	
    return returnValue;
}

// returns the processed captured image into a js array buffer
Napi::TypedArrayOf<unsigned char> GetResultImage(const Napi::CallbackInfo& info)
{
	// draw the results on the out image
	MobileRCNN_DrawLatestResult(img_display.data, true);
	
	// convert back to RGBA to display
	cv::cvtColor(img_display, img_display, CV_RGBA2BGRA, 4);

	Napi::Env env = info.Env();
    
	std::vector<uchar> buffer;
    cv::imencode(".jpg", img_display(cv::Rect(0, 0, img_resized.cols, img_resized.rows)), buffer);
	
	Napi::TypedArrayOf<unsigned char> returnValue = Napi::TypedArrayOf<unsigned char>::New(env, buffer.size());
	
	memcpy(returnValue.Data(), buffer.data(), sizeof(unsigned char) * buffer.size());
	
    return returnValue;
}

// return number of detections
Napi::Number NoDetections(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
	return Napi::Number::New(env, noDetections);
}
Napi::Number DetectionScore(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
	int detId = info[0].As<Napi::Number>().Int32Value();
	return Napi::Number::New(env, detections[detId].score);
}
Napi::Number DetectionClassId(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
	int detId = info[0].As<Napi::Number>().Int32Value();
	return Napi::Number::New(env, detections[detId].classId);
}
Napi::TypedArrayOf<float> DetectionBox(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
	int detId = info[0].As<Napi::Number>().Int32Value();
	
	Napi::TypedArrayOf<float> buffer = Napi::TypedArrayOf<float>::New(env, 4);
	
	float box[4] = {detections[detId].box.x, detections[detId].box.y, detections[detId].box.z, detections[detId].box.w};
	float *bufferPtr = buffer.Data();

	memcpy(buffer.Data(), box, sizeof(float) * 4);

	return buffer;
}
Napi::TypedArrayOf<float> DetectionMask(const Napi::CallbackInfo& info)
{
	Napi::Env env = info.Env();
	int detId = info[0].As<Napi::Number>().Int32Value();
	
	int maskSize = 28 * 28;
	Napi::TypedArrayOf<float> buffer = Napi::TypedArrayOf<float>::New(env, maskSize);
	
	memcpy(buffer.Data(), detections[detId].mask, sizeof(float) * maskSize);
	
	return buffer;
}

// allocator
void Initialize(const Napi::CallbackInfo& info)
{
	std::string folderName = info[0].As<Napi::String>();
	std::string f_proto_backbone = folderName + "/mobileRCNN_base.txt";
	std::string f_weights_backbone = folderName + "/mobileRCNN_base.bin";
	std::string f_proto_det = folderName + "/mobileRCNN_det.txt";
	std::string f_weights_det = folderName + "/mobileRCNN_det.bin";
	std::string f_proto_mask = folderName + "/mobileRCNN_mask.txt";
	std::string f_weights_mask = folderName + "/mobileRCNN_mask.bin";
		
	Napi::Number camera_id = info[1].As<Napi::Number>();
	Napi::Number camera_width = info[2].As<Napi::Number>();
	Napi::Number camera_height = info[3].As<Napi::Number>();
	
	//read from camera
	cap = cv::VideoCapture(camera_id.Int32Value());
	
	//set camera resolution
	cap.set(3, camera_width.Int32Value()); 
	cap.set(4, camera_height.Int32Value());
	
	// initialise the detector
	MobileRCNN_Parameters params;
	MobileRCNN_Init(f_proto_backbone.c_str(), f_weights_backbone.c_str(), 
		f_proto_det.c_str(), f_weights_det.c_str(), f_proto_mask.c_str(), f_weights_mask.c_str(), params);
	
	// image dims
	inputSize = MobileRCNN_InputSize();
	
	// images for the various processing stages
	img_resized = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // image resized to width * ..
	img_padded = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // imaged padded with width * height
	img_4ch = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // image with the 4th channel added
	img_display = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_8UC4); // final displayed result
}

void Shutdown(const Napi::CallbackInfo& info)
{
	// shutdown the detector
	MobileRCNN_Shutdown();
}

// main constructor method
Napi::Object Init(Napi::Env env, Napi::Object exports)
{
	exports.Set("captureCameraImage", Napi::Function::New(env, CaptureCameraImage));
	exports.Set("runDetector", Napi::Function::New(env, RunDetector));

	exports.Set("initialize", Napi::Function::New(env, Initialize));
	exports.Set("shutdown", Napi::Function::New(env, Shutdown));
	
	exports.Set("resizedCameraImage", Napi::Function::New(env, GetResizedCameraImage));
	exports.Set("originalCameraImage", Napi::Function::New(env, GetOriginalCameraImage));
	exports.Set("resultImage", Napi::Function::New(env, GetResultImage));
	
	exports.Set("noDetections", Napi::Function::New(env, NoDetections));
	exports.Set("score", Napi::Function::New(env, DetectionScore));
	exports.Set("classId", Napi::Function::New(env, DetectionClassId));
	exports.Set("box", Napi::Function::New(env, DetectionBox));
	exports.Set("mask", Napi::Function::New(env, DetectionMask));
	
	return exports;
}

NODE_API_MODULE(mobilercnn, Init)