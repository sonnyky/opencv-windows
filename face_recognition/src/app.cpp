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
	/* Settings for Buffalo HD camera
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 2048);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1536);
	*/
	cvNamedWindow("Detected", CV_WINDOW_NORMAL);

	cv::Mat markerImage;
	currentDict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
	cv::aruco::drawMarker(currentDict, 846, 500, markerImage, 1);
	//imwrite("../ReferenceImages/arucoMarker849.jpg", markerImage);
	
	////////cv::imshow("Reference", markerImage);

}

void Capture::finalize() {

}

void Capture::run()
{
	while (true) {
		 Mat frame;
        cap >> frame; // get a new frame from camera
		detectArucoMarkers(frame);
        imshow("edges", frame);
        if(waitKey(30) >= 0) break;
	}
}

void Capture::filterColor(Mat image) 
{
	int largest_area = 0;
	int largest_contour_index = 0;

	cv::Scalar   min(90, 190, 80);
	cv::Scalar   max(170, 255, 255);
	cv::Mat mask, mask1, mask2, mask_grey;
	cv::Mat hsv_image;
	cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
	mask1 = image.clone(); mask2 = image.clone();
	//filter the image in BGR color space
	cv::inRange(hsv_image, min, max, mask);
	//cv::inRange(hsv_image, cv::Scalar(55, 100, 50), cv::Scalar(65, 255, 255), mask1);
	//cv::inRange(hsv_image, cv::Scalar(50, 100, 50), cv::Scalar(70, 255, 255), mask2);
	std::vector<std::vector<cv::Point>> contours; // Vector for storing contour
	std::vector<cv::Vec4i> hierarchy;
	//mask = mask1 | mask2;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	vector<Rect> bounding_rect(contours.size());

	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
	/*
	if (a>largest_area) {
	largest_area = a;
	largest_contour_index = i;                //Store the index of largest contour
	bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
	}
	*/

		bounding_rect.push_back(boundingRect(contours[i]));

	}
	cv::Scalar color(255, 255, 255);
	drawContours(mask, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); 
	cv::imshow("filtered", mask);	
}

inline void Capture::detectArucoMarkers(cv::Mat input) {
	vector< int > markerIds;
	vector< vector<Point2f> > markerCorners, rejectedCandidates;
	cv::Mat bgr;
	Mat output;

	//input.convertTo(input, -1, 1.2,-50);
	cv::aruco::detectMarkers(input, currentDict, markerCorners, markerIds, arucoParams, rejectedCandidates);
	//cout <<"Number of markers found : "<< markerIds.size() << endl;

	if (markerIds.size() > 0) {
		//depthString.str(std::string());
		cv::aruco::drawDetectedMarkers(input, markerCorners, markerIds, CvScalar(0, 0, 255));
		for (int i = 0; i < markerIds.size(); i++) {
			Vec3i markerInfo;
			//markerInfo[0] = markerIds[i];
			//markerInfo[1] = (int)((markerCorners[i][0].x + markerCorners[i][1].x + markerCorners[i][2].x + markerCorners[i][3].x) / 4) - rangeTopLeft.x;
			//markerInfo[2] = (int)((markerCorners[i][0].y + markerCorners[i][1].y + markerCorners[i][2].y + markerCorners[i][3].y) / 4) - rangeTopLeft.y;
			//depthString << to_string(markerInfo[0]) << "," << to_string(markerInfo[1]) << "," << to_string(markerInfo[2]) << ",";
		}
	}
	else {
		/*
		markerLostCounter++;
		if (markerLostCounter > 50) {
			depthString.str(std::string());
			markerLostCounter = 0;
		}
		*/
	}
	if (rejectedCandidates.size() > 0) {
		cv::aruco::drawDetectedMarkers(input, rejectedCandidates);
	}
	cv::imshow("Detected", input);
}
