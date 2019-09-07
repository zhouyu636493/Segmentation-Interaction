#ifndef SOLVER_H
#define SOLVER_H
#include <vector>
#include <iostream>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
struct  data
{
	vector<cv::Point> contour;
	double angle;
	Mat Image;
};
class Solver
{
public:
	//double myfunc(int n, const double* var, double *grad, struct  data* mydata);
	//void contour_segmentation(vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<int>>& intersect_index, vector<vector<Point2f>>& segment_pairs);
	void contour_segmentation(vector<vector<Point2f>> point_direction, vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<Point2f>>& segment_pairs);
};

#endif