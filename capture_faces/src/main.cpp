#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

#include "wmp.h"
#include <Windows.h>
#include <comutil.h>

#include "pugixml.hpp"

using namespace cv;
using namespace std;

#define BUFSIZE 4096

/*
main 以外の関数宣言
*/
Rect detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip, Rect roi_area, int width, int height);
void  drawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt);
Rect filterSkinColor(Mat frame);

int main(int argc, char** argv)
{

	if (argc == 1) {
		cout << "Path to settings file needed," << endl;
		exit(1);
	}

	const char * path = argv[1];
	
#pragma region read capture settings
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(path);
	cout << "Load result: " << result.description() << endl;

	pugi::xpath_query query_path_to_face_cascade("/settings/capture_settings/path_to_face_cascade");
	string path_to_face_cascade = query_path_to_face_cascade.evaluate_string(doc);

	pugi::xpath_query query_path_to_nested_cascade("/settings/capture_settings/path_to_nested_cascade");
	string path_to_nested_cascade = query_path_to_nested_cascade.evaluate_string(doc);

	pugi::xpath_query query_path_to_training_data("/settings/capture_settings/path_to_training_data");
	string path_to_training_data = query_path_to_training_data.evaluate_string(doc);

	pugi::xpath_query query_width("/settings/capture_params/width");
	int width = stoi(query_width.evaluate_string(doc));

	pugi::xpath_query query_height("/settings/capture_params/height");
	int height = stoi(query_height.evaluate_string(doc));

	pugi::xpath_query query_scale("/settings/capture_params/scale");
	double scale = stod(query_scale.evaluate_string(doc));

	pugi::xpath_query query_flip("/settings/capture_params/flip");
	int flip_face = stoi(query_flip.evaluate_string(doc));

	pugi::xpath_query query_numbering("/settings/capture_params/numbering");
	int numbering = stoi(query_numbering.evaluate_string(doc));

#pragma endregion read capture settings

	VideoCapture cap(0); // open the default camera
	CascadeClassifier cascade, nestedCascade;

			
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	if (!nestedCascade.load(path_to_nested_cascade))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(path_to_face_cascade))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, width);
	cap.set(CAP_PROP_FRAME_HEIGHT, height);
	cout << "Starting capture" << endl;
	for (;;)
	{
		Mat frame;
		int64 start = cv::getTickCount();

		cap >> frame; // get a new frame from camera
		Mat frame1 = frame.clone();
		Rect face_area = filterSkinColor(frame1);
		Rect face_roi = detectAndDraw(frame1, cascade, nestedCascade, scale, flip_face, face_area, width, height);

		Mat onlyFace = frame1(face_roi);
		Size faceSize(230, 280);
		if (onlyFace.rows > 200 && onlyFace.cols > 200) {
			resize(onlyFace, onlyFace, faceSize);
			resizeWindow("Image", faceSize);
			imshow("Image", onlyFace);
			
			int c = waitKey(10);
			if (c == 'c') {
				cout << "Capture" << endl;
				Mat face_gray;
				cvtColor(onlyFace, face_gray, COLOR_BGR2GRAY);
				std::ostringstream name;
				name << path_to_training_data << '/' << numbering << ".jpg";
				imwrite(name.str(), face_gray);
				numbering++;
			}
		}

		int64 end = cv::getTickCount();
		double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
		//std::cout << elapsedMsec << "ms" << std::endl;

		int c = waitKey(10);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
	return 0;
}

Rect detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Rect roi_area, int captureWidth, int captureHeight)
{
	double t = 0;
	double zoomFactor = 1.0;
	vector<Rect> faces, faces2, nestedObjects;
	Rect roiRect;

	roiRect.x = 0; roiRect.y = 0; roiRect.width = 0; roiRect.height = 0;

	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg, smallImgROI;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	
	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE,
		Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 15, 0
			//|CASCADE_FIND_BIGGEST_OBJECT,
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			Rect tempRect = Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height);
			smallImgROI = smallImg(tempRect);
			cascade.detectMultiScale(smallImgROI, nestedObjects,
				1.1, 3, 0
				//|CASCADE_FIND_BIGGEST_OBJECT,
				//|CASCADE_DO_ROUGH_SEARCH
				| CASCADE_SCALE_IMAGE,
				Size(30, 30));
			if (nestedObjects.size() > 0 && roi_area.x >= tempRect.x && roi_area.y >= tempRect.y) {
				faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
			}
		}
	}
	t = (double)getTickCount() - t;
	//printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
	if (faces.size() > 0) {
		for (size_t i = 0; i < 1; i++) // faces.size() ではなく、i < 1 にすれば認識結果を一つだけ利用する
		{
			Rect r = faces[i];
			Point center;
			Scalar color = colors[i % 8];
			int radius;
			int zoomed_x, zoomed_y, zoomed_width, zoomed_height;
			int offset_y_face = r.y - (r.width*0.25);

			vector<cv::Point2f>tgtPt;
			zoomed_height = r.height;
			zoomed_width = (float(captureWidth / captureHeight) * r.height);
			zoomed_x = r.x - ((float(captureWidth / captureHeight) * r.height) - r.width) / 2;
			zoomed_y = r.y + zoomed_height;

			tgtPt.push_back(cv::Point2f(r.x, offset_y_face));
			tgtPt.push_back(cv::Point2f(r.x + r.width, offset_y_face));
			tgtPt.push_back(cv::Point2f(r.x + r.width, offset_y_face + r.height));
			tgtPt.push_back(cv::Point2f(r.x, offset_y_face + r.height));

			double aspect_ratio = (double)r.width / r.height;
			if (0.75 < aspect_ratio && aspect_ratio < 1.3)
			{	
				rectangle(img, r, color, 3, 8, 0);
				roiRect = r;
			}		
		}
	}
	else {
		
	}
	return roiRect;
}

void drawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt)
{
	cv::Mat img_rgb, img_aaa, img_1ma;
	vector<cv::Mat>planes_rgba, planes_rgb, planes_aaa, planes_1ma;
	int maxVal = pow(2, 8 * baseImg.elemSize1()) - 1;

	//透過画像はRGBA, 背景画像はRGBのみ許容。ビット深度が同じ画像のみ許容
	if (transImg.data == NULL || baseImg.data == NULL || transImg.channels()<4 || baseImg.channels()<3 || (transImg.elemSize1() != baseImg.elemSize1()))
	{
		img_dst = cv::Mat(100, 100, CV_8UC3);
		img_dst = cv::Scalar::all(maxVal);
		return;
	}

	//書き出し先座標が指定されていない場合は背景画像の中央に配置する
	if (tgtPt.size()<4)
	{
		//座標指定(背景画像の中心に表示する）
		int ltx = (baseImg.cols - transImg.cols) / 2;
		int lty = (baseImg.rows - transImg.rows) / 2;
		int ww = transImg.cols;
		int hh = transImg.rows;

		tgtPt.push_back(cv::Point2f(ltx, lty));
		tgtPt.push_back(cv::Point2f(ltx + ww, lty));
		tgtPt.push_back(cv::Point2f(ltx + ww, lty + hh));
		tgtPt.push_back(cv::Point2f(ltx, lty + hh));
	}

	//変形行列を作成
	vector<cv::Point2f>srcPt;
	srcPt.push_back(cv::Point2f(0, 0));
	srcPt.push_back(cv::Point2f(transImg.cols - 1, 0));
	srcPt.push_back(cv::Point2f(transImg.cols - 1, transImg.rows - 1));
	srcPt.push_back(cv::Point2f(0, transImg.rows - 1));
	cv::Mat mat = cv::getPerspectiveTransform(srcPt, tgtPt);

	//出力画像と同じ幅・高さのアルファ付き画像を作成
	cv::Mat alpha0(baseImg.rows, baseImg.cols, transImg.type());
	alpha0 = cv::Scalar::all(0);
	cv::warpPerspective(transImg, alpha0, mat, alpha0.size(), cv::INTER_CUBIC, cv::BORDER_TRANSPARENT);

	//チャンネルに分解
	cv::split(alpha0, planes_rgba);

	//RGBA画像をRGBに変換   
	planes_rgb.push_back(planes_rgba[0]);
	planes_rgb.push_back(planes_rgba[1]);
	planes_rgb.push_back(planes_rgba[2]);
	merge(planes_rgb, img_rgb);

	//RGBA画像からアルファチャンネル抽出   
	planes_aaa.push_back(planes_rgba[3]);
	planes_aaa.push_back(planes_rgba[3]);
	planes_aaa.push_back(planes_rgba[3]);
	merge(planes_aaa, img_aaa);

	//背景用アルファチャンネル   
	planes_1ma.push_back(maxVal - planes_rgba[3]);
	planes_1ma.push_back(maxVal - planes_rgba[3]);
	planes_1ma.push_back(maxVal - planes_rgba[3]);
	merge(planes_1ma, img_1ma);

	img_dst = img_rgb.mul(img_aaa, 1.0 / (double)maxVal) + baseImg.mul(img_1ma, 1.0 / (double)maxVal);
}


/*
肌色の領域を返す。この領域内で検出された顔のみ「顔」として扱う
*/
Rect filterSkinColor(Mat input)
{
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;
	//YCrCb threshold
	// You can change the values and see what happens
	int Y_MIN = 0;
	int Y_MAX = 255;
	int Cr_MIN = 133;
	int Cr_MAX = 173;
	int Cb_MIN = 77;
	int Cb_MAX = 127;
	cv::Mat skin, mask, mask_grey;
	//first convert our RGB image to YCrCb
	cvtColor(input, skin, cv::COLOR_BGR2YCrCb);
	mask = skin.clone();
	//filter the image in YCrCb color space
	inRange(skin, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), mask);
	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image

	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a>largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}
	//Scalar color(255, 255, 255);
	//drawContours(skin, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
	//rectangle(skin, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
	//imshow("skin", skin);

	return bounding_rect;
}