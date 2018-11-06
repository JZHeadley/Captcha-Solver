#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;
using namespace std;

inline bool exists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good() && infile.peek() == std::ifstream::traits_type::eof();
}

vector<string> getFiles(const string path)
{
    vector<string> filesOut;
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());
    if (dir == NULL)
    {
        printf("Nothing in that directory\n");
        return filesOut;
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
        filesOut.push_back(entry->d_name);
    }
    return filesOut;
}

vector<Mat> getImages(string baseDir, vector<string> files)
{
    vector<Mat> images_out;
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
        images_out.push_back(image);
        count++;
    }
    return images_out;
}