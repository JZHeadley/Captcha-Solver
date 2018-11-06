#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;
using namespace std;
#define NUM_STREAMS 100
inline bool exists(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good() && infile.peek() == std::ifstream::traits_type::eof();
}
vector<string> get_files(const string path)
{
	vector<string> files;
	struct dirent *entry;
	DIR *dir = opendir(path.c_str());
	if (dir == NULL)
	{
		printf("Nothing in that directory\n");
		return files;
	}

	string filepath;
	struct stat filestat;

	while ((entry = readdir(dir)) != NULL)
	{
		filepath = path + entry->d_name;
		if (stat(filepath.c_str(), &filestat))
			continue;
		if (S_ISDIR(filestat.st_mode))
			continue;
		files.push_back(entry->d_name);
	}
	return files;
}
vector<Mat> getImages(string baseDir, vector<string> files)
{
	vector<Mat> images;
	int count = 0;
	for (int i = 0; i < files.size(); i++)
	{
//		printf("%s\n", files[i].c_str());
		Mat image = imread(baseDir + files[i], IMREAD_GRAYSCALE);
		if (image.empty())
		{
			printf("Empty image?\n");
			continue;
		}
		image.resize(50);
		images.push_back(image);
		count++;
	}
	return images;
}

int main(int argc, char* argv[])
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
	string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";
	vector<string> files = get_files(baseDir);
	vector<Mat> images = getImages(baseDir, files);
//	Mat* processed = (Mat*) malloc(sizeof(Mat) * images.size());
	vector<Mat> processed;
	Mat thresh, blur, erosion_1, erosion_2, dilation;
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	for (int i = 0; i < images.size(); i++)
	{

		threshold(images[0], thresh, 125, 255, THRESH_BINARY_INV);
		medianBlur(thresh, blur, 3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 3));
		erode(blur, erosion_1, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
		erode(erosion_1, erosion_2, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		dilate(erosion_2, dilation, kernel);
		processed.push_back(dilation);
		imwrite(outDir + files[i], dilation, compression_params);

	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

	printf("The image processing for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int) diff);

	return 0;
}

