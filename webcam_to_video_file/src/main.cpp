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

	cout <<"number of arguments : " << to_string(argc) << endl;

	if (argc == 1) {
		cout << "please specify video save path" << endl;
		exit(1);
	}

	const char * path = argv[1];


	return 0;
}