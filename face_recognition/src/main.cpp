#define NOMINMAX
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "atlbase.h"
#include "atlwin.h"
#include "wmp.h"
#include <Windows.h>
#include <comutil.h>
#include <algorithm>  

using namespace cv;
using namespace cv::face;
using namespace std;

#define BUFSIZE 4096

/*
main 以外の関数宣言
*/
void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip, Rect roi_area);
void  drawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt);
Rect filterSkinColor(Mat frame);

/*
キャプチャと顔認識の設定
*/
int captureHeight = 1280;
int captureWidth = 720;
string cascadeName = "D:/Libraries/opencv3.1.0/build/etc/haarcascades/haarcascade_frontalface_alt2.xml";
string nestedCascadeName = "D:/Libraries/opencv3.1.0/build/etc/haarcascades/haarcascade_eye.xml";
string fn_csv = "D:/Workspace/opencv-windows/res/training_data/data.csv";
double scale = 1.0;
bool tryflip = true;

string g_listname_t[] =
{
	"Sonny",
	"Jun"
};

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int main(int argc, char** argv)
{
	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;

	cout << "Run?" << endl;
	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}


	int im_width = images[0].cols;
	int im_height = images[0].rows;

	VideoCapture cap(0); // open the default camera
	CascadeClassifier cascade, nestedCascade;


	if (!cap.isOpened())  // check if we succeeded
		return -1;

	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = FisherFaceRecognizer::create();
	model->train(images, labels);

	CascadeClassifier haar_cascade;
	haar_cascade.load(cascadeName);

	
	Mat frame;


	for (;;)
	{		
		cap >> frame; // get a new frame from camera
					  // Clone the current frame:
		Mat original = frame.clone();

		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);



		if (faces.size() == 0) continue;
		cout << " Number of faces : " << faces.size() << endl;
	
		for (int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			Mat face = gray(face_i);
			Mat face_resized;
			resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

			int prediction = model->predict(face_resized);
			// And finally write all we've found out to the original image!
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			// Create the text we will annotate the box with:
			string box_text;
			box_text = format("Prediction = ");
			// Get stringname
			if (prediction >= 0 && prediction <= 1)
			{
				box_text.append(g_listname_t[prediction]);
			}
			else box_text.append("Unknown");
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);

		}


		imshow("face_recognizer", original);
		// And display it:
		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
	return 0;
}