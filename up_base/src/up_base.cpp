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
