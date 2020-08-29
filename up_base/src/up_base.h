#include <iostream>
#include <string>
#include <map>
#include <objbase.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#pragma once

using namespace cv;
using namespace std;

class up_base {
public:

	up_base();
	~up_base();

	LPCSTR echo(const char * input);

	void save_black_and_white(unsigned char * bytes, int rows, int cols, int type);
};