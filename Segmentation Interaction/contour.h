#pragma once
#include <iostream>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#define PI 3.1415926
using namespace std;
using namespace cv;


class Contour {//提取src图像的轮廓
public:
	Mat image(Mat img);//将输入的img二值化并光滑后返回
	vector<vector<Point2i>> contour_generate(Mat de_noise);//提取src的轮廓，并按照面积从大到小排序，并返回
	double line_point_dist(Point2i mouse, Point2i p1, Point2i p2);
	double get_distance(Point2i p1, Point2i p2);
};

