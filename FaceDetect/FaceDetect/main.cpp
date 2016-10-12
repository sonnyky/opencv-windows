#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void  DrawTransPinP(cv::Mat &img_dst, const cv::Mat transImg, const cv::Mat baseImg, vector<cv::Point2f> tgtPt);
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat& overlayImg);
Mat filterSkinColor(Mat frame);
void zoomPicture(cv::Mat src, cv::Mat dst, cv::Point2i center, double rate);

string cascadeName;
string nestedCascadeName;

bool zoomEffect = false; bool face_detected = false;
int zoomStep = 5;
double maxZoomFactor = 3.0, curZoomFactor = 1.0;

int captureHeight = 320;
int captureWidth = 640;
int main(int argc, char** argv)
{
	VideoCapture cap(0); // open the default camera
	CascadeClassifier cascade, nestedCascade;
	cascadeName = "C:/Users/Sonny/Desktop/Workspace/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml";
	nestedCascadeName = "C:/Users/Sonny/Desktop/Workspace/opencv/build/etc/haarcascades/haarcascade_eye.xml";
	double scale = 1.0;
	bool tryflip = true;

	Mat overlay_image = imread("../Stamps/kabuki.png", cv::IMREAD_UNCHANGED);
	
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, captureWidth);
	cap.set(CAP_PROP_FRAME_HEIGHT, captureHeight);

	for (;;)
	{
		Mat frame;
		int64 start = cv::getTickCount();

		cap >> frame; // get a new frame from camera
		Mat frame1 = frame.clone();
		Mat frame2 = frame1.clone();
			frame2 = filterSkinColor(frame1);
		detectAndDraw(frame2, cascade, nestedCascade, scale, tryflip, overlay_image);
		int64 end = cv::getTickCount();
		double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
		std::cout << elapsedMsec << "ms" << std::endl;

		int c = waitKey(10);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat& overlayImg)
{
	double t = 0;
	double zoomFactor = 1.0;
	vector<Rect> faces, faces2, nestedObjects;
	Mat zoomedImage = img.clone();
	CvRect roiRect;

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
			if (nestedObjects.size() > 0) {
				faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
			}
		}
	}
	t = (double)cvGetTickCount() - t;
	//printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
	if (faces.size() > 0) {
		for (size_t i = 0; i < 1; i++) // i < 1 instead of faces.size() to get only one face
		{
			Rect r = faces[i];
			//Mat smallImgROI;
			//vector<Rect> nestedObjects;
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

				center.x = cvRound((r.x + r.width*0.5)*scale);
				center.y = cvRound((r.y + r.height*0.5)*scale);
				radius = cvRound((r.width + r.height)*0.25*scale);
				//circle(img, center, radius, color, 3, 8, 0);

				DrawTransPinP(img, overlayImg, img, tgtPt);

				if (curZoomFactor > 2.0) curZoomFactor = 2.0;
				zoomPicture(img, zoomedImage, Point2i((r.x + r.width / 2), (r.y + r.height / 3)), curZoomFactor);
				curZoomFactor += maxZoomFactor / zoomStep;
			}		
		}
	}
	else {
		curZoomFactor = 1.0;
	}
	
	imshow("result", img);
	imshow("zoomed", zoomedImage);
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

void zoomPicture(cv::Mat src, cv::Mat dst, cv::Point2i center, double rate)
{
	if (rate < 1.0) {//縮小は未対応なのでそのまま
		src.copyTo(dst);
		return;
	}

	cv::Mat resizeSrc;
	cv::resize(src, resizeSrc, cv::Size2i(0, 0), rate, rate);
	//拡大後の拡大中心
	cv::Point2i resizeCenter(center.x*rate, center.y*rate);

	//拡大中心と拡大率の設定次第で元の画像をはみ出してしまうので余白を入れる
	int blankHeight = src.rows / 2;//元画像の上下にそれぞれ入れる余白の画素数
	int blankWidth = src.cols / 2;//元画像の左右にそれぞれ入れる余白の画素数
	cv::Mat resizeSrcOnBlank = cv::Mat::zeros(resizeSrc.rows + 2 * blankHeight, resizeSrc.cols + 2 * blankWidth, CV_8UC3);
	resizeSrc.copyTo(resizeSrcOnBlank(cv::Rect(blankWidth, blankHeight, resizeSrc.cols, resizeSrc.rows)));
	resizeSrcOnBlank(cv::Rect(resizeCenter.x + blankWidth - src.cols / 2, resizeCenter.y + blankHeight - src.rows / 2, src.cols, src.rows)).copyTo(dst);
	return;

}

//this function will return a skin masked image
Mat filterSkinColor(Mat input)
{
	//YCrCb threshold
	// You can change the values and see what happens
	int Y_MIN = 0;
	int Y_MAX = 255;
	int Cr_MIN = 133;
	int Cr_MAX = 173;
	int Cb_MIN = 77;
	int Cb_MAX = 127;
	cv::Mat skin, mask;
	//first convert our RGB image to YCrCb
	cvtColor(input, skin, cv::COLOR_BGR2YCrCb);
	mask = skin.clone();
	//uncomment the following line to see the image in YCrCb Color Space
	imshow("YCrCb Color Space",skin);

	//filter the image in YCrCb color space
	inRange(skin, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), mask);

	return skin;
}