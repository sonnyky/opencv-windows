#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "atlbase.h"
#include "atlwin.h"
#include "wmp.h"
#include <Windows.h>
#include <comutil.h>

using namespace cv;
using namespace std;

#define BUFSIZE 4096

int main(int argc, char** argv)
{

	double min, max;
	Point maxLoc;

	Mat im = imread("truck_gas.jpg");
	Mat gray;
	cvtColor(im, gray, COLOR_BGR2GRAY);
	Mat canny_output;
	int thresh = 100;
	Canny(gray, canny_output, thresh, thresh * 2);
	imshow("canny", canny_output);
	// find connected components. we'll use each component as a mask for distance transformed image,
	// then extract the peak location and its strength. strength corresponds to the radius of the circle
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		
		drawContours(im, contours, idx, Scalar(255, 255, 255), -1);
		// draw the circles
		//circle(im, maxLoc, (int)max, Scalar(0, 0, 255), 2);
	}
	imshow("Detected", im);
	waitKey();
	return EXIT_SUCCESS;
}