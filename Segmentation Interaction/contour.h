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


class Contour {//��ȡsrcͼ�������
public:
	Mat image(Mat img);//�������img��ֵ�����⻬�󷵻�
	vector<vector<Point2i>> contour_generate(Mat de_noise);//��ȡsrc������������������Ӵ�С���򣬲�����
	double line_point_dist(Point2i mouse, Point2i p1, Point2i p2);
	double get_distance(Point2i p1, Point2i p2);
};

