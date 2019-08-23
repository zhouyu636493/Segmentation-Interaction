#pragma once
#include <iostream>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <random>
#include <math.h>
#include <float.h>
#include "triangle.h"
using namespace std;//����һ��Ҫ�������ռ䣬����vector��ʶ��
using namespace cv;//����һ��Ҫ��cv������Rect�Ȳ�ʶ��
typedef struct Circle {
public:
	Point2f center;
	float rad;
}Circle;
typedef struct Coordinate {
	int x_axis;
	int y_axis;
}Coordinate;
typedef struct Delaunay {
	CvSubdiv2D* div;
	vector <CvPoint2D32f> point;
	vector<Vec2i> constrained_edge;
} Delaunay;

class Triangle {
public:

	Delaunay delaunay_triangle(vector <Point> contour,Mat img);
	CvSubdiv2D* init_delaunay(CvMemStorage* storage,Rect rect);
	Circle circle_generate(vector<Point> contour);
	vector <Point2f> Poisson_disc(vector<Point> contour, double r, int k);//�����̲���
	double distance(double x1, double y1, double x2, double y2);
	void TriSubDiv(vector<Point> contour, Mat dstImage, vector<Vec3i> &tri);
	
};