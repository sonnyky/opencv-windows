#include "util.h"
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
Mat asSamplesVectors(Mat& img);
int main(int argc, char* argv[])
{
	Mat img = imread("../../images/ufo_small.JPG", IMREAD_UNCHANGED);
	namedWindow("image", WINDOW_NORMAL);
	namedWindow("reverted", WINDOW_NORMAL);

	// The variable openCVPointCloud is not used in the process. This is just to show how to build our own vectors
	cv::Mat openCVPointCloud(5, 2, CV_32FC(1));
	{
		cv::Vec<float, 1>  & values1 = openCVPointCloud.at<cv::Vec<float, 1> >(0, 0);
		values1.val[0] = 1.1;

		cv::Vec<float, 1>  & values2 = openCVPointCloud.at<cv::Vec<float, 1> >(0, 1);
		values2.val[0] = 3.2;
	}

	{
		cv::Vec<float, 1>  & values1 = openCVPointCloud.at<cv::Vec<float, 1> >(1, 0);
		values1.val[0] = 1.15;

		cv::Vec<float, 1>  & values2 = openCVPointCloud.at<cv::Vec<float, 1> >(1, 1);
		values2.val[0] = 3.15;
	}

	{
		cv::Vec<float, 1>  & values1 = openCVPointCloud.at<cv::Vec<float, 1> >(2, 0);
		values1.val[0] = 3.1;

		cv::Vec<float, 1>  & values2 = openCVPointCloud.at<cv::Vec<float, 1> >(2, 1);
		values2.val[0] = 4.2;
	}

	{
		cv::Vec<float, 1>  & values1 = openCVPointCloud.at<cv::Vec<float, 1> >(3, 0);
		values1.val[0] = 3.2;

		cv::Vec<float, 1>  & values2 = openCVPointCloud.at<cv::Vec<float, 1> >(3, 1);
		values2.val[0] = 4.3;
	}

	{
		cv::Vec<float, 1>  & values1 = openCVPointCloud.at<cv::Vec<float, 1> >(4, 0);
		values1.val[0] = 5;

		cv::Vec<float, 1>  & values2 = openCVPointCloud.at<cv::Vec<float, 1> >(4, 1);
		values2.val[0] = 5;
	}

	std::cout << openCVPointCloud << std::endl;

	//////////////////////////// we have a 5 x 2 matrice ///////////////////////////////

	cv::Ptr<cv::ml::EM> source_model = cv::ml::EM::create();
	source_model->setClustersNumber(2);
	cv::Mat logs;
	cv::Mat labels;
	cv::Mat probs;

	Mat vector = asSamplesVectors(img);

	if (source_model->trainEM(vector, logs, labels, probs))
	{
		std::cout << "true train em";
		int imgVectorIndex = 0;
		for (cv::MatIterator_<int> it(labels.begin<int>()); it != labels.end<int>(); it++)
		{
			//std::cout << (*it) << std::endl; // int i = *it
			int i = *it;
			if (i == 0) {
				vector.at<float>(imgVectorIndex, 0) = 0;
				vector.at<float>(imgVectorIndex, 1) = 0;
				vector.at<float>(imgVectorIndex, 2) = 0;
			}
			else if (i == 1) {
				vector.at<float>(imgVectorIndex, 0) = 0;
				vector.at<float>(imgVectorIndex, 1) = 255;
				vector.at<float>(imgVectorIndex, 2) = 0;
			}
			else {
				vector.at<float>(imgVectorIndex, 0) = 255;
				vector.at<float>(imgVectorIndex, 1) = 0;
				vector.at<float>(imgVectorIndex, 2) = 0;
			}
			imgVectorIndex++;
		}
	}
	else {
		std::cout << "false train em" << std::endl;
	}
	

	Mat revertImage = Mat(img.size(), CV_32FC3);
	int index = 0;
	for (int i = 0; i < revertImage.rows; i++) {
		for (int j = 0; j < revertImage.cols; j++) {

			float temp0 = vector.at<float>(index, 0);
			float temp1 = vector.at<float>(index, 1);
			float temp2 = vector.at<float>(index, 2);


			revertImage.at<Vec3f>(Point(j, i)).val[0] = temp0 / 255;
			revertImage.at<Vec3f>(Point(j, i)).val[1] = temp1 / 255;
			revertImage.at<Vec3f>(Point(j, i)).val[2] = temp2 / 255;
			index++;
		}
	}

	// Dilate image to remove noise
	// Create a structuring element
	int erosion_size = 2; int dilation_size = 3;
	Mat element = getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	Mat dilate_element = getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * dilation_size + 1),
		cv::Point(erosion_size, erosion_size));

	Mat eroded, dilated, dst;
	// Apply erosion or dilation on the image
	erode(revertImage, eroded, element);
	dilate(eroded, dilated, dilate_element);

	imshow("image", img);
	imshow("reverted", dilated);
	waitKey(0); //wait infinite time for a keypress
	return 0;
}

Mat asSamplesVectors(Mat& img) {
	//convert the input image to float
	cv::Mat floatSource;
	img.convertTo(floatSource, CV_32F);

	//now convert the float image to column vector
	cv::Mat samples(img.rows * img.cols, 3, CV_32FC1);
	int idx = 0;
	for (int y = 0; y < img.rows; y++) {
		cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
		for (int x = 0; x < img.cols; x++) {
			samples.at<cv::Vec3f >(idx++, 0) = row[x];
		}
	}
	return samples;
}