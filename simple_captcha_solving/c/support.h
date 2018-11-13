#ifndef IMAGEPROCESSING_SUPPORT_H
#define IMAGEPROCESSING_SUPPORT_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;
using namespace std;

typedef struct
{
    Mat image;
    string filename;
} Image;

inline bool exists(const char *fileName);

vector<string> getFiles(const string path);

vector<Image> getImages(string baseDir, vector<string> files);

#endif