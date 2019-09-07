#define common_api __declspec(dllexport) 

#include <../up_base.h>

#pragma once

extern "C" {
	common_api up_base* com_tinker_up_base_create();
	common_api LPCSTR com_tinker_up_base_echo(up_base* instance, const char * input);
}