// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#include <stdio.h>

void SaveImageToFile(const unsigned char* image, int width, int height, const char* fileName);
bool ReadImageFromFile(unsigned char* image, const char* fileName);