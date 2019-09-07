#include "app.h"

#include <thread>
#include <chrono>

#include <numeric>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;

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
        imshow("edges", frame);
	}
}
