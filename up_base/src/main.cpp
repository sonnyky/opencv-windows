#include <common.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

extern "C" {

	up_base* com_tinker_up_base_create() {
		return new up_base();
	}

	LPCSTR com_tinker_up_base_echo(up_base* instance, const char * input) {
		return instance->echo(input);
	}
}