#include <stdio.h>

#include "support.h"
#include <opencv2/highgui/highgui.hpp>

#define NUM_THREADS 256
#define NUM_ROWS_FROM_BOTTOM 17
#define NUM_ROWS_FROM_TOP 10

vector<string> files;
vector<Mat> images;
string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";
vector<Mat> processed;
vector<int> compression_params;
int correctSeparations = 0;

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
	output.width = max(a.x + a.width, b.x + b.width);
	output.height = max(a.y + a.height, b.y + b.height);
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
	for (int i = 0; i < letterImageRegions.size(); i++)
	{
		temp1 = Rect(max(0, letterImageRegions[i].x - 2),
					 max(0, letterImageRegions[i].y - 25),
					 min(maxCols, letterImageRegions[i].width + 2),
					 min(maxRows, letterImageRegions[i].y + letterImageRegions[i].height + 25));
		for (int j = 0; j < letterImageRegions.size(); j++)
		{
			if (i == j)
				continue;
			temp2 = Rect(max(0, letterImageRegions[j].x - 2),
						 max(0, letterImageRegions[j].y - 25),
						 min(maxCols, letterImageRegions[j].width + 2),
						 min(maxRows, letterImageRegions[j].y + letterImageRegions[j].height + 25));
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

void extractLetters(string captchaText, Mat image, vector<Rect> letterBoundingBoxes)
{
	Rect letterRegion;
	Mat letter;
	for (int i = 0; i < letterBoundingBoxes.size(); i++)
	{
		letterRegion = Rect(letterBoundingBoxes[i].x,
							letterBoundingBoxes[i].y,
							letterBoundingBoxes[i].width,
							letterBoundingBoxes[i].height);
		printf("cols: %i rows: %i\n", image.cols, image.rows);
		printf("%i %i %i %i\n", letterBoundingBoxes[i].x,
			   letterBoundingBoxes[i].y,
			   letterBoundingBoxes[i].width,
			   letterBoundingBoxes[i].height);
		letter = image(letterBoundingBoxes[i]);
		// imshow("blah", letter);
		// waitKey(0);
	}
	return;
}

void findLetters(string fileName, Mat image, vector<int> compression_params)
{
	Mat canny_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Canny(image, canny_output, 100, 200, 3);
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
	Rect boundingBox;
	vector<Rect> letterImageRegions;
	vector<vector<Point>> rectContours;
	for (int i = 0; i < contours.size(); i++)
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
			letterImageRegions.push_back(Rect(boundingBox.x + halfWidth, boundingBox.y, boundingBox.x + halfWidth, boundingBox.height));
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
		// printf("Correctly separated %s\n", fileName.c_str());
		sort(letterImageRegions.begin(), letterImageRegions.end(), compareRects);
		extractLetters(fileName.c_str(), image, letterImageRegions);
		// imshow("image", image);
		// waitKey(0);
	}
	return;
}
void *processImages(void *args)
{
	int tid = *((int *)(&args));
	int instancesPerTask = (images.size() + NUM_THREADS) / NUM_THREADS;
	int beginIndex = tid * instancesPerTask;
	int endIndex = MIN((tid + 1) * instancesPerTask, images.size());
	Mat tailed, beheaded, thresh, blurred, erosion_1, erosion_2, dilation, dilation_2, kernel;

	for (int i = beginIndex; i < endIndex; i++)
	{
		// tailed = Mat(images[i].rows - NUM_ROWS_FROM_BOTTOM, images[i].cols, CV_8UC1, images[i].data);
		// beheaded = tailed;
		threshold(images[i], thresh, 125, 255, THRESH_BINARY_INV);
		medianBlur(thresh, blurred, 3);

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
		imwrite(outDir + files[i], dilation_2, compression_params);
		findLetters(files[i], dilation_2, compression_params);
	}
	return NULL;
}

int main(int argc, char *argv[])
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	if (argc == 3)
	{
		baseDir = argv[1];
		outDir = argv[2];
	}
	printf("reading images from %s and writing to %s\n", baseDir.c_str(), outDir.c_str());
	files = getFiles(baseDir);
	images = getImages(baseDir, files);
	printf("Finished reading in all of the images\n");
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	pthread_t *threads = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
	int *threadIds = (int *)malloc(NUM_THREADS * sizeof(int));
	for (int i = 0; i < NUM_THREADS; i++)
		threadIds[i] = i;

	for (int i = 0; i < NUM_THREADS; i++)
	{
		int status = pthread_create(&threads[i], NULL, processImages, (void *)threadIds[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
	free(threads);
	free(threadIds);
	printf("We had %i correct separattions.\n", correctSeparations);
	printf("The image processing for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

	return 0;
}
