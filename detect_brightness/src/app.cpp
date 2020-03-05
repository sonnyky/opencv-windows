#include "app.h"

#include <thread>
#include <chrono>

#include <numeric>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
void detect_brightest_area(Mat image);
void filter_for_red(Mat image);
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
	cap.open(1);
	cap.set(CV_CAP_PROP_AUTOFOCUS, 0);
	cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0.25);
	cap.set(CAP_PROP_BRIGHTNESS, 10);
	cap.set(CAP_PROP_XI_EXPOSURE, 60);
	cap.set(CAP_PROP_CONTRAST, 40);
	cap.set(CAP_PROP_GAIN, 25);
	cap.set(CAP_PROP_SATURATION, 40);

	
	
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
		filter_for_red(frame);
		
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

void filter_for_red(Mat image) {
	Mat copy = image.clone();
	Mat hsvImg;
	cvtColor(copy, hsvImg, COLOR_BGR2HSV);

	vector<Mat> hsvChannels(3);
	split(hsvImg, hsvChannels);
	
	threshold(hsvChannels[0], hsvChannels[0], 160.0, 0, CV_THRESH_TOZERO_INV);
	threshold(hsvChannels[0], hsvChannels[0], 20.0, 255, THRESH_BINARY);
	bitwise_not(hsvChannels[0], hsvChannels[0]);
	
	threshold(hsvChannels[1], hsvChannels[1], 255.0, 0, CV_THRESH_TOZERO_INV);
	threshold(hsvChannels[1], hsvChannels[1], 100.0, 255, THRESH_BINARY);

	threshold(hsvChannels[2], hsvChannels[2], 255.0, 0, CV_THRESH_TOZERO_INV);
	threshold(hsvChannels[2], hsvChannels[2], 100.0, 255, THRESH_BINARY);

	Mat tmp;
	bitwise_and(hsvChannels[0], hsvChannels[2], tmp);
	bitwise_and(hsvChannels[1], tmp, tmp);

	Mat mergedBack;
	merge(hsvChannels, mergedBack);

	imshow("merged", tmp);

}