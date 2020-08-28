#ifndef __APP__
#define __APP__

#include <Windows.h>
#include <comutil.h>
#include <iostream>
#include <wtypes.h>
#include <comdef.h> 
#include <string>
#include <string.h>
#include <tchar.h>
#include <stdio.h>
#include "atlbase.h"
#include "atlwin.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>

#include <wrl/client.h>
using namespace Microsoft::WRL;
using namespace cv;
using namespace std;

class Capture {
private :
	VideoCapture cap;
	//Reference image
	cv::Ptr<cv::aruco::Dictionary> currentDict;
	cv::Ptr<cv::aruco::DetectorParameters> arucoParams;

public:
	// Constructor
	Capture();

	// Destructor
	~Capture();
	void run();

private :
	void initialize();
	void finalize();

	void filterColor(Mat image);
	void detectArucoMarkers(cv::Mat input);
};

#endif // __APP__