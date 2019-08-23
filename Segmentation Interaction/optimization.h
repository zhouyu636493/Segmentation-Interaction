#pragma once
#pragma once
#include <iostream>
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
class Optimization
{
public:
	Optimization();
	~Optimization();
	double D_compute(double angle,vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<Point2f>>& segment_pairs);//计算形变量的函数
};