#pragma once
#include <iostream>
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
class Model
{
public:
	Model();
	~Model();
	void generate(vector<vector<int>>concavity, vector<vector<Point3f>>seg_contours_3,
		vector<vector<Point3f>> segment_pairs_3, int contour_index, double L, double  W, double H,vector<Point3f>& vertex,vector<vector<int>>& face);
	void offset(vector<vector<Point3f>>& seg_contours_3, int base_index, vector<vector<int>>& seg_index, int** adj_matrix,
		vector<vector<Point3f>>& segment_pairs_3, vector<vector<int>>& concavity,double d1);
};