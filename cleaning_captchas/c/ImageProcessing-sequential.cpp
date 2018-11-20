#include <stdio.h>

#include "support.h"

#define NUM_ROWS_FROM_BOTTOM 17
#define NUM_ROWS_FROM_TOP 10
#define EXTRACTED_LETTER_DIR "extracted_letters/"

vector<string> files;
vector<Mat> images;
string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";

vector<Mat> processed;
vector<int> compression_params;
int correctSeparations = 0;
int charCounts[256] = {0};

bool compareRects(Rect l, Rect r);
Rect rectUnion(Rect a, Rect b);
void createOutDirsIfNotExists(string processedImageDir, string extracted_letter_dir);
Rect rectIntersection(Rect a, Rect b);
vector<Rect> joinBoundingBoxes(vector<Rect> letterImageRegions, int maxRows, int maxCols);

void extractLetters(string captchaText, Mat image, vector<Rect> letterBoundingBoxes)
{
	Rect letterRegion;
	Mat letter;
	string outFile = EXTRACTED_LETTER_DIR;
	char charDir;
	for (int i = 0; i < (int)letterBoundingBoxes.size(); i++)
	{
		Range rowRange = Range(letterBoundingBoxes[i].x, letterBoundingBoxes[i].x + letterBoundingBoxes[i].width);
		Range colRange = Range(letterBoundingBoxes[i].y, letterBoundingBoxes[i].y + letterBoundingBoxes[i].height);
		letter = Mat(image, colRange, rowRange);
		charDir = captchaText.at(i);
		mkdir((outFile + charDir).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		imwrite(((string)(outFile + charDir + "/" + to_string(charCounts[charDir]) + ".jpg")).c_str(), letter, compression_params);
		charCounts[charDir] += 1;
	}
	return;
}

void findLetters(string fileName, Mat image, vector<int> compression_params)
{
	Mat canny_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
	Rect boundingBox;
	vector<Rect> letterImageRegions;
	vector<vector<Point>> rectContours;
	for (int i = 0; i < (int)contours.size(); i++)
	{
		rectContours.clear();
		boundingBox = boundingRect(contours[i]);
		rectContours = vector<vector<Point>>{
			vector<Point>{Point(boundingBox.x, boundingBox.y)},
			vector<Point>{Point(boundingBox.x + boundingBox.width, boundingBox.y)},
			vector<Point>{Point(boundingBox.x, boundingBox.y + boundingBox.height)},
			vector<Point>{Point(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height)}};
		if (boundingBox.width * boundingBox.height < 200)
		{
			// ignore small amounts of noise.  maybe fill it in later on
			fillPoly(canny_output, rectContours, Scalar(0, 0, 0));
			continue;
		}
		if (boundingBox.width * boundingBox.height > 2000 || boundingBox.width / boundingBox.height > 1.5)
		{
			int halfWidth = boundingBox.width / 2;
			if (halfWidth * boundingBox.height < 200)
			{
				rectContours = vector<vector<Point>>{
					vector<Point>{Point(boundingBox.x, boundingBox.y)},
					vector<Point>{Point(boundingBox.x + halfWidth, boundingBox.y)},
					vector<Point>{Point(boundingBox.x, boundingBox.y + boundingBox.height)},
					vector<Point>{Point(boundingBox.x + halfWidth, boundingBox.y + boundingBox.height)}};
				fillPoly(canny_output, rectContours, Scalar(0, 0, 0));
				continue;
			}
			letterImageRegions.push_back(Rect(boundingBox.x, boundingBox.y, halfWidth, boundingBox.height));
			letterImageRegions.push_back(Rect(boundingBox.x + halfWidth, boundingBox.y, halfWidth, boundingBox.height));
		}
		else
		{
			letterImageRegions.push_back(Rect(boundingBox));
		}
	}
	letterImageRegions = joinBoundingBoxes(letterImageRegions, image.rows - 1, image.cols - 1);
	if (letterImageRegions.size() == 6)
	{
		correctSeparations++;
		replaceString(fileName, "_duplicate", "");
		replaceString(fileName, ".jpg", "");
		sort(letterImageRegions.begin(), letterImageRegions.end(), compareRects);
		extractLetters(fileName.c_str(), image, letterImageRegions);
	}
	return;
}

void processImages(vector<Mat> images)
{
	Mat thresh, blurred, bilateral, erosion_1, erosion_2, dilation, dilation_2, kernel;
	printf("Beginning image processing\n");
	for (int i = 0; i < (int)images.size(); i++)
	{
		threshold(images[i], thresh, 125, 255, THRESH_BINARY_INV);
		medianBlur(thresh, blurred, 3);
		bilateralFilter(blurred, bilateral, 4, 75, 75);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 3));
		erode(blurred, erosion_1, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
		erode(erosion_1, erosion_2, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		kernel.at<Vec3b>(Point(0, 1)) = 0;
		kernel.at<Vec3b>(Point(1, 0)) = 0;
		dilate(erosion_2, dilation, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		dilate(dilation, dilation_2, kernel);
		// processed.push_back(dilation); // damn you c++ and your lack of thread safety

		// write out the intermediate image after all the image processing
		string outFile = outDir + files[i];
		imwrite(outFile, dilation_2, compression_params);
		findLetters(files[i], dilation_2, compression_params);
	}
	return;
}

int main(int argc, char *argv[])
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
	string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";
	if (argc == 3)
	{
		baseDir = argv[1];
		outDir = argv[2];
	}
	printf("reading images from %s and writing to %s\n", baseDir.c_str(), outDir.c_str());
	files = getFiles(baseDir);
	images = getImages(baseDir, files);
	createOutDirsIfNotExists(outDir, EXTRACTED_LETTER_DIR);
	printf("Finished reading in all of the images\n");
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	processImages(images);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
	printf("We had %i correct separattions.\n", correctSeparations);
	printf("The image processing for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

	return 0;
}

void createOutDirsIfNotExists(string processedImageDir, string extracted_letter_dir)
{
	mkdir(processedImageDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	mkdir(extracted_letter_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

bool compareRects(Rect l, Rect r)
{
	if (l.y == r.y)
		return l.x < r.x;
	return (l.y < r.y);
}

Rect rectUnion(Rect a, Rect b)
{
	Rect output;
	output.x = min(a.x, b.x);
	output.y = min(a.y, b.y);
	output.width = max(a.width, b.width);
	output.height = max(a.height, b.height);
	return output;
}

Rect rectIntersection(Rect a, Rect b)
{
	Rect output;
	output.x = max(a.x, b.x);
	output.y = max(a.y, b.y);
	output.width = min(a.x + a.width, b.x + b.width) - output.x;
	output.height = min(a.y + a.height, b.y + b.height) - output.y;
	if (output.width < 0 || output.height < 0)
		return Rect(0, 0, 0, 0);
	return output;
}

vector<Rect> joinBoundingBoxes(vector<Rect> letterImageRegions, int maxRows, int maxCols)
{
	vector<Rect> boundingBoxes;
	Rect temp1, temp2, intersection;
	bool flag = false;
	for (int i = 0; i < (int)letterImageRegions.size(); i++)
	{
		temp1 = Rect(max(0, letterImageRegions[i].x - 2),
					 max(0, letterImageRegions[i].y - 25),
					 min(maxCols, letterImageRegions[i].width + 2),
					 min(maxRows, letterImageRegions[i].height + 25));
		for (int j = 0; j < (int)letterImageRegions.size(); j++)
		{
			if (i == j)
				continue;
			temp2 = Rect(max(0, letterImageRegions[j].x - 2),
						 max(0, letterImageRegions[j].y - 25),
						 min(maxCols, letterImageRegions[j].width + 2),
						 min(maxRows, letterImageRegions[j].height + 25));
			intersection = rectIntersection(temp1, temp2);
			if (intersection.width * intersection.height > 0 && !flag)
			{
				flag = true;
				boundingBoxes.push_back(rectUnion(temp1, temp2));
				break;
			}
		}
		if (!flag)
			boundingBoxes.push_back(letterImageRegions[i]);
	}
	return boundingBoxes;
}