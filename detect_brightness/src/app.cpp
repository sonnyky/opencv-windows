#include "app.h"

#include <thread>
#include <chrono>

#include <numeric>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
void detect_brightest_area(Mat image);
// Constructor
Capture::Capture()
{
	// Initialize
	initialize();
}

// Destructor
Capture::~Capture()
{
	// Finalize
	finalize();
}

void Capture::initialize() {
	cap.open(0);
	
	cvNamedWindow("Detected", CV_WINDOW_NORMAL);

}

void Capture::finalize() {

}

void Capture::run()
{
	while (true) {
		Mat frame;
        cap >> frame; // get a new frame from camera
		if (frame.empty()) break; // end of video stream
		detect_brightest_area(frame);
		
        imshow("Detected", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
}

void detect_brightest_area(Mat image) {

	Mat gray;
	Mat copy = image.clone();
	cvtColor(copy, gray, COLOR_BGR2GRAY);

	GaussianBlur(gray, gray, cvSize(5, 5), 0);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(gray, &minVal, &maxVal, &minLoc, &maxLoc);
	circle(image, maxLoc, 10, cvScalar(255, 255, 255), 3, 8, 0);
}