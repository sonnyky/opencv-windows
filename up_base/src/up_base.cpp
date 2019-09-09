#include <numeric>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "up_base.h"

using namespace std;

up_base::up_base()
{
}

up_base::~up_base()
{
}

LPCSTR up_base::echo(const char * input)
{
	if (input == NULL) return NULL;
	char something[10];
	strcpy(something, input);
	string new_string(something);

	string test = input;

	ofstream myfile;
	myfile.open("test.txt");
	myfile << test;
	myfile << "\n";
	myfile << new_string;
	myfile << "\n";
	myfile << &input[0];
	myfile << "\n";
	string original = (const char *)input;
	string echo = "Message to echo res: " + original;
	myfile << echo;
	myfile.close();
	return echo.c_str();
}

void up_base::save_black_and_white(unsigned char * bytes, int rows, int cols, int type)
{

	ofstream myfile;
	myfile.open("testImage.txt");
	
	myfile << "rows and cols";
	myfile << "\n";
	myfile << to_string(rows) << ", " << to_string(cols);
	myfile.close();

	Mat img(rows, cols, CV_8UC4);
	memcpy(img.data, bytes, rows * cols * 4);
	imwrite("bw.jpg", img);
	/*
	Mat gray;

	cvtColor(img, gray, CV_BGR2GRAY);

	imwrite("bw.jpg", gray);*/
}
