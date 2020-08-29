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
#include "app.h"

using namespace cv;
using namespace std;

#define BUFSIZE 4096

int main(int argc, char** argv)
{

	Capture capt;
	capt.run();
}