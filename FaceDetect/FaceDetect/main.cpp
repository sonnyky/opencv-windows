#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

#include "atlbase.h"
#include "atlwin.h"
#include "wmp.h"

using namespace cv;
using namespace std;

void  DrawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt);

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat& overlayImg);

string cascadeName;
string nestedCascadeName;

int main(int argc, char** argv)
{
	VideoCapture cap(0); // open the default camera
	CascadeClassifier cascade, nestedCascade;
	cascadeName = "C:/Users/Sonny/Desktop/Workspace/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml";
	nestedCascadeName = "C:/Users/Sonny/Desktop/Workspace/opencv/build/etc/haarcascades/haarcascade_eye.xml";
	double scale = 1.0;
	bool tryflip = true;

	BSTR media_url = SysAllocString(L"C:/Users/Sonny/Desktop/knightrider.mp3");
	BSTR another_media_url = SysAllocString(L"C:/Users/Sonny/Desktop/missionimpossible.mp3");

	

	Mat overlay_image = imread("../Stamps/bunny_ears.png", cv::IMREAD_UNCHANGED);
	resize(overlay_image, overlay_image, Size(128, 64));
	imshow("overlay",overlay_image);

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 320);


	/*
	CoInitialize(NULL);
	
	HRESULT hr = S_OK;
	CComBSTR bstrVersionInfo; // Contains the version string.
	CComPtr<IWMPPlayer4> spPlayer;  // Smart pointer to IWMPPlayer interface.
	
	hr = spPlayer.CoCreateInstance(__uuidof(WindowsMediaPlayer), 0, CLSCTX_INPROC_SERVER);

	if (SUCCEEDED(hr))
	{
		hr = spPlayer->get_versionInfo(&bstrVersionInfo);
		spPlayer->openPlayer(media_url);
	}
	*/


	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera

		Mat frame1 = frame.clone();
		detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip, overlay_image);
		
		int c = waitKey(10);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}

	// Clean up.
	//spPlayer->openPlayer(another_media_url);
	//spPlayer.Release();
	//CoUninitialize();
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat& overlayImg)
{
	double t = 0;
	vector<Rect> faces, faces2;
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
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();
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
			1.1, 2, 0
			|CASCADE_FIND_BIGGEST_OBJECT,
			//|CASCADE_DO_ROUGH_SEARCH
			//| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;
	//printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;
		int offset_y_face = r.y - (r.width*0.75);

		vector<cv::Point2f>tgtPt;

		tgtPt.push_back(cv::Point2f(r.x, offset_y_face));
		tgtPt.push_back(cv::Point2f(r.x + r.width, offset_y_face));
		tgtPt.push_back(cv::Point2f(r.x + r.width, offset_y_face + r.height));
		tgtPt.push_back(cv::Point2f(r.x, offset_y_face + r.height));

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		
			DrawTransPinP(img, overlayImg,img, tgtPt);
		}
		
		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);
		
	}
	imshow("result", img);
}

void DrawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt)
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