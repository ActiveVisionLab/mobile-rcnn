# mobile-rcnn - LightNet Version

## 1. Building the System

### 1.1 Requirements

Several 3rd party libraries are needed for compiling MobileRCNN. The given version numbers are checked and working, but different versions might be fine as well. Some of the libraries are optional, and skipping them will reduce functionality.

  - cmake (e.g. version 2.8.10.2 or 3.2.3)
    REQUIRED for Linux, unless you write your own build system
    OPTIONAL for MS Windows, if you use MSVC instead
    available at http://www.cmake.org/
	
   - CUDA (e.g. version 9.0 or higher)
    REQUIRED for all GPU acceleration.

   - CUDNN (version 7.0)
    REQUIRED for all GPU acceleration.
	
   - OpenCV (e.g. version 3.1 or higher)
     REQUIRED for the webcam demo.
	
## 1.2 Build Process

To compile the system, use the standard cmake approach.

# Sample Programm

## Disk demo

`mobile-rcnn-demo-disk` should be run by passing both (i) the various neural network files (proto and weights) and (ii) the location of the target input image as arguments to the program, e.g. from the cmake build folder:

```
mobile-rcnn-demo-disk ../../../../Files/Nets/mobileRCNN_base.txt ../../../../Files/Nets/mobileRCNN_base.bin ../../../../Files/Nets/mobileRCNN_det.txt ../../../../Files/Nets/mobileRCNN_det.bin ../../../../Files/Nets/mobileRCNN_mask.txt ../../../../Files/Nets/mobileRCNN_mask.bin ../../../../Files/Images/mobilercnn.ppm 
```

## Webcam demo

`mobile-rcnn-demo-webcam` should be run by passing (i) the various neural network files (proto and weights), (ii) the id of the webcam to use and (iii) (optionally) the resolution of the input video source, e.g. from the cmake build folder:

```
mobile-rcnn-demo-webcam ../../../../Files/Nets/mobileRCNN_base.txt ../../../../Files/Nets/mobileRCNN_base.bin ../../../../Files/Nets/mobileRCNN_det.txt ../../../../Files/Nets/mobileRCNN_det.bin ../../../../Files/Nets/mobileRCNN_mask.txt ../../../../Files/Nets/mobileRCNN_mask.bin 0 640 480
```

## Nodejs webcam demo

The nodejs version can be built using the standard npm process, as follows:
- make both the OpenCV dll `opencv_world345.dll` and the LNTLibRedist dll `LNTLibRedist.dll` are in the current user or system path;
- navigate to the `Apps/mobile-rcnn-demo-nodejs` folder;
- set the correct library and include paths in the `binding.gyp`, as explained below;
- copy the 6 model files from the `Files/Nets` folder to the `C:/mobile-rcnn` folder; 
- run `npm install node-gyp`;
- run `npm run build`.

The following library and include paths are required in the `binding.gyp` file:
- OpenCV `include` folder -- preset to `C:/SDK/opencv/3.4.5/build/include`;
- OpenCV main library -- preset to `C:/SDK/opencv/3.4.5/build/x64/vc15/lib/opencv_world345.lib`

To run the nodejs demo:
- navigate to the `Apps/mobile-rcnn-demo-nodejs` folder;
- run `node server.js`;
- navigate with your browser to `http://localhost:1337` -- if everthing went well, you should see a webcam photo overlaid with the segmentation.