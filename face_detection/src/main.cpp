#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	String xmlFile = "lbpcascade_frontalface.xml";
	CascadeClassifier classifier = CascadeClassifier(xmlFile);

	VideoCapture capture;
	//-- 2. Read the video stream
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		//-- 3. Apply the classifier to the frame
		// Detecting the face in the snap
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_RGBA2GRAY);
		equalizeHist(frame_gray, frame_gray);

		std::vector<Rect> faces;
		classifier.detectMultiScale(frame_gray, faces);
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		}

		if (waitKey(10) == 27)
		{
			break; // escape
		}
		imshow("Capture");
	}
	return 0;
}
