#pragma once
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>//这个头文件一定要有，里面包括平面细分的函数
#include <string>
#include <queue>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_conformer_2.h>
using namespace Eigen;
using namespace std;
using namespace cv;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point CdtPoint;//防止和opencv中的Point混淆
#define N 999 
class Deformation
{
public:
	Deformation();
	~Deformation();
	bool is_in_contour(Point2f mid, vector<Point2f> contour);
	bool is_in_contour(Point3f mid, vector<Point3f> contour);
	bool line_is_in_contour(CdtPoint p1, CdtPoint p2, vector<Point2i> contour);
	vector<Point2f> compute_intersect(vector<Point2f> contour, vector<Point2i> points,vector<vector<int>>& intersect_index);
	bool is_cross(vector<Point2f> contour, vector<Point2i> points);
	vector<Point3f> rotate_axis(vector<Point3f> contour, vector<Point3f> segment);
	Point3f rotating(Point3f point, Point3f axis1, Point3f axis2, double angle1);
	void contour_segmentation(vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<int>>& intersect_index, 
		vector<vector<Point2i>>& points_set, vector<vector<Point2f>>& segment_pairs);
	void rotated_contour_compute(queue<int>& reference, vector<vector<Point3f>>seg_contours_3, vector<vector<Point3f>> segment_pairs_3, double angle1,
		int** adj_matrix, vector<vector<int>> seg_index, int ang);
	vector<vector<int>> concavity(vector<vector<Point3f>> seg_contours_3, vector<vector<Point3f>> segment_pairs_3,
		vector<vector<int>> seg_index, int base_index, int** adj_matrix);

};