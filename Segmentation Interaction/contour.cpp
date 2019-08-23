#include <iostream>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "contour.h"
#include <random>
#include <math.h>
#include <float.h>
using namespace std;
using namespace cv;

static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	return (contourArea(contour1) > contourArea(contour2));
}
Mat Contour::image(Mat img) {
	Mat grayImage, binaryImage;
	cvtColor(img, grayImage, COLOR_BGR2GRAY);//转换为灰度图并平滑滤波
	imwrite("gray.jpg", grayImage);

	adaptiveThreshold(grayImage, binaryImage, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 25, 10);
	imwrite("binary.jpg", binaryImage);

	Mat de_noise = binaryImage.clone();
	medianBlur(binaryImage, de_noise, 5);//de_noise是输出
	imwrite("blur.jpg", de_noise);
	return de_noise;
}


vector<vector<Point2i>> Contour::contour_generate(Mat de_noise) {
	Mat  dstImage;
	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(de_noise, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNAL只检测外部轮廓
	std::sort(contours.begin(), contours.end(), ContoursSortFun);
	waitKey(0);
	return contours;
}

double Contour:: line_point_dist(Point2i mouse, Point2i p1, Point2i p2) {//计算mouse点到p1-p1的距离
	double dist;
	double a = p2.y - p1.y;
	double b = p1.x - p2.x;
	double c = p2.x*p1.y - p1.x*p2.y;
	dist = abs(a*mouse.x + b*mouse.y + c) / sqrt(a*a + b*b);
	return dist;
}

double Contour:: get_distance(Point2i p1,Point2i p2) {
	double distance;
	distance = pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
	distance = sqrt(distance);
	return distance;

}