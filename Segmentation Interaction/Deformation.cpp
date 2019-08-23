#include "Deformation.h"
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

Deformation::Deformation() {}
Deformation::~Deformation() {}
bool Deformation::is_in_contour(Point2f mid, vector<Point2f> contour)
{
	//判断点mid是否在contour中，在则返回true
	int n = contour.size();
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (mid.y == contour[i].y&&mid.x == contour[i].x)
			return true;
		else if (mid.y == contour[i].y&&mid.y == contour[(i + 1) % n].y)
		{
			if ((mid.x >= contour[i].x&&mid.x <= contour[(i + 1) % n].x) || (mid.x <= contour[i].x&&mid.x >= contour[(i + 1) % n].x))
				return true;
		}
		else if ((mid.y >= contour[i].y&&mid.y < contour[(i + 1) % n].y) || (mid.y >= contour[(i + 1) % n].y&&mid.y < contour[i].y))
		{
			double d = ((mid.y - contour[i].y)*(contour[(i + 1) % n].x - contour[i].x) + (contour[i].x - mid.x)*(contour[(i + 1) % n].y - contour[i].y))
				* (contour[(i + 1) % n].y - contour[i].y);
			if (d > 0)
				count++;
			else if (d == 0)
				return true;
		}
	}
	if (count % 2 == 0)
		return false;
	else
		return true;//在contour中
}
bool Deformation::is_in_contour(Point3f mid, vector<Point3f> contour)
{
	//判断点mid是否在contour中，在则返回true
	int n = contour.size();
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (mid.y == contour[i].y&&mid.x == contour[i].x)
			return true;
		else if (mid.y == contour[i].y&&mid.y == contour[(i + 1) % n].y)
		{
			if ((mid.x >= contour[i].x&&mid.x <= contour[(i + 1) % n].x) || (mid.x <= contour[i].x&&mid.x >= contour[(i + 1) % n].x))
				return true;
		}
		else if ((mid.y >= contour[i].y&&mid.y < contour[(i + 1) % n].y) || (mid.y >= contour[(i + 1) % n].y&&mid.y < contour[i].y))
		{
			double d = ((mid.y - contour[i].y)*(contour[(i + 1) % n].x - contour[i].x) + (contour[i].x - mid.x)*(contour[(i + 1) % n].y - contour[i].y))
				* (contour[(i + 1) % n].y - contour[i].y);
			if (d > 0)
				count++;
			else if (d == 0)
				return true;
		}
	}
	if (count % 2 == 0)
		return false;
	else
		return true;//在contour中
}
bool Deformation::line_is_in_contour(CdtPoint p1, CdtPoint p2, vector<Point2i> contour)
{
	//判断点mid是否在contour中，在则返回true
	Point2f mid;
	mid.x = (p1.x() + p2.x()) / 2;
	mid.y = (p1.y() + p2.y()) / 2;
	int n = contour.size();
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (mid.y == contour[i].y&&mid.x == contour[i].x)
			return true;
		else if (mid.y == contour[i].y&&mid.y == contour[(i + 1) % n].y)
		{
			if ((mid.x >= contour[i].x&&mid.x <= contour[(i + 1) % n].x) || (mid.x <= contour[i].x&&mid.x >= contour[(i + 1) % n].x))
				return true;
		}
		else if ((mid.y >= contour[i].y&&mid.y < contour[(i + 1) % n].y) || (mid.y >= contour[(i + 1) % n].y&&mid.y < contour[i].y))
		{
			double d = ((mid.y - contour[i].y)*(contour[(i + 1) % n].x - contour[i].x) + (contour[i].x - mid.x)*(contour[(i + 1) % n].y - contour[i].y))
				* (contour[(i + 1) % n].y - contour[i].y);
			if (d > 0)
				count++;
			else if (d == 0)
				return true;
		}
	}
	if (count % 2 == 0)
		return false;
	else
		return true;//在contour中
}
vector<Point2f> Deformation::compute_intersect(vector<Point2f> contour, vector<Point2i> points, vector<vector<int>>& intersect_index)
{
	//计算线段points和contour的交点
	vector<Point2f> intersect;
	vector<int> tep;
	//tep中存放的是交点所处的contour中的边的index，因为一共两个交点，每个交点位于两个index之间，所以tep中有四个元素
	//tep中的前两个元素是从小到大排序，后两个也是从小到大排序，它们分别代表两条contour边的index
	int size = contour.size();
	for (int i = 0; i < size; i++)
	{
		if (intersect.size() < 2) {
			int s1 = i;
			int s2 = (i + 1) % size;
			if (s1 > s2)
			{
				int t = s1;
				s1 = s2;
				s2 = t;
			}//确保每对index是从小到大排序
			Point2f d0, d1;
			double t0, t1, d;
			d0.x = contour[s2].x - contour[s1].x;
			d0.y = contour[s2].y - contour[s1].y;
			d1.x = points[1].x - points[0].x;
			d1.y = points[1].y - points[0].y;
			double k;
			k = d0.x*d1.y - d1.x*d0.y;
			Point2f deta;
			deta.x = points[0].x - contour[s1].x;
			deta.y = points[0].y - contour[s1].y;
			if (k == 0)
			{
				if (abs(d0.x) > abs(d0.y))
				{
					t0 = points[0].x - contour[s1].x;
					t1 = points[1].x - contour[s1].x;
					d = d0.x;
				}
				else
				{
					t0 = points[0].y - contour[s1].y;
					t1 = points[1].y - contour[s1].y;
					d = d0.y;
				}
				if (d < 0)
				{
					t0 = -t0;
					t1 = -t1;
					d = -d;
				}
				if (t0 > t1)
				{
					double temp = t0;
					t0 = t1;
					t1 = temp;
				}
				if (t0 > d || t1 < 0)
					continue;//不相交
				else if (t1 == 0)
				{
					Point2f Q;
					Q.x = contour[s1].x;
					Q.y = contour[s1].y;
					vector<Point2f>::iterator iter = find(intersect.begin(), intersect.end(), Q);
					if (iter == intersect.end())
					{
						intersect.push_back(Q);
						tep.push_back(s1);
						tep.push_back(s2);
						continue;
					}
					else
						continue;
				}
				else if (t0 == d)
				{
					Point2f Q;
					Q.x = contour[s2].x;
					Q.y = contour[s2].y;
					vector<Point2f>::iterator iter = find(intersect.begin(), intersect.end(), Q);
					if (iter == intersect.end()) {
						intersect.push_back(Q);
						tep.push_back(s1);
						tep.push_back(s2);
						continue;
					}
					else
						continue;
				}
				else
					cerr << "overlapping wrong" << endl;
			}
			else
			{
				double t;
				t = (deta.x*d0.y - deta.y*d0.x) / k;
				double s;
				s = (deta.x*d1.y - deta.y*d1.x) / k;
				if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
				{
					Point2f Q;
					Q.x = contour[s1].x + s*(contour[s2].x - contour[s1].x);
					Q.y = contour[s1].y + s*(contour[s2].y - contour[s1].y);
					vector<Point2f>::iterator iter = find(intersect.begin(), intersect.end(), Q);
					if (iter == intersect.end()) {
						intersect.push_back(Q);
						tep.push_back(s1);
						tep.push_back(s2);
						continue;
					}
					else
						continue;
				}
				else
					continue;
			}
		}
		else
			break;
	}
	if (!tep.empty())
		intersect_index.push_back(tep);
	return intersect;
}
bool Deformation::is_cross(vector<Point2f> contour, vector<Point2i> points)
{
	//判断线段points和contour的交点是否为两个
	int sum = 0;
	for (int a = 0; a < contour.size(); a++)
	{
		Point2f d0, d1, deta;
		d0.x = contour[(a + 1) % contour.size()].x - contour[a].x;
		d0.y = contour[(a + 1) % contour.size()].y - contour[a].y;
		d1.x = points[1].x - points[0].x;
		d1.y = points[1].y - points[0].y;
		double k;
		k = d0.x*d1.y - d0.y*d1.x;
		deta.x = points[0].x - contour[a].x;
		deta.y = points[0].y - contour[a].y;
		if (k == 0)
		{
			double t0, t1, d;
			if (abs(d0.x) > abs(d0.y))
			{
				t0 = points[0].x - contour[a].x;
				t1 = points[1].x - contour[a].x;
				d = d0.x;
			}
			else
			{
				t0 = points[0].y - contour[a].y;
				t1 = points[1].y - contour[a].y;
				d = d0.y;
			}
			if (d < 0)
			{
				t0 = -t0;
				t1 = -t1;
				d = -d;
			}
			if (t0 > t1)
			{
				double temp = t0;
				t0 = t1;
				t1 = temp;
			}
			if (t1 == 0)
				sum++;
			else if (t0 == d)
				sum++;
		}
		else
		{
			double t = (deta.x*d0.y - deta.y*d0.x) / k;
			double s = (deta.x*d1.y - deta.y*d1.x) / k;
			if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
				sum++;
		}
	}
	if (sum == 2)
		return true;
	else
		return false;
}
//void CGAL_Output_Obj(std::string path, std::vector<Vector3d> &vecs, std::vector<int> &face_id_0, std::vector<int> &face_id_1, std::vector<int> &face_id_2)
//{
//
//	if (vecs.size() < 3 || face_id_0.size() < 1 || face_id_1.size() < 1 || face_id_2.size() < 1)
//	{
//		std::cout << "CGAL_Output_Obj error: vecs.size() < 3 || face_id_0.size() < 1 || face_id_1.size() < 1 || face_id_2.size() < 1" << std::endl;
//		return;
//	}
//
//	for (int i = 0; i < face_id_0.size(); i++)
//	{
//		int index_0 = face_id_0[i];
//		int index_1 = face_id_1[i];
//		int index_2 = face_id_2[i];
//
//		if (index_0 < 0 || index_0 >= vecs.size() || index_1 < 0 || index_1 >= vecs.size() || index_2 < 0 || index_2 >= vecs.size())
//		{
//			std::cout << "CGAL_Output_Obj error: index_0 < 0 || index_0 >= vecs.size() || index_1 < 0 || index_1 >= vecs.size() || index_2 < 0 || index_2 >= vecs.size()" << std::endl;
//			return;
//		}
//	}
//
//	std::ofstream file(path);
//	for (int i = 0; i < vecs.size(); i++)
//	{
//		file << "v " << vecs[i][0] << " " << vecs[i][1] << " " << vecs[i][2] << std::endl;
//	}
//
//	for (int i = 0; i < face_id_0.size(); i++)
//	{
//		int index_0 = face_id_0[i];
//		int index_1 = face_id_1[i];
//		int index_2 = face_id_2[i];
//
//		if (index_0 != index_1&&index_0 != index_2&&index_1 != index_2)
//			file << "f " << index_0 + 1 << " " << index_1 + 1 << " " << index_2 + 1 << std::endl;
//	}
//	file.close();
//}
vector<Point3f> Deformation::rotate_axis(vector<Point3f> contour, vector<Point3f> segment)
{
	Point3f seg1, seg2;
	seg1 = segment[0];
	seg2 = segment[1];
	int a;
	for (a = 0; a < contour.size(); a++)
	{
		if (contour[a] == seg1)
		{
			break;
		}
	}
	if (contour[(a + 1) % contour.size()] != seg2)
	{
		Point3f temp;
		temp = seg1;
		seg1 = seg2;
		seg2 = temp;
	}
	vector<Point3f> result;
	result.push_back(seg2);
	result.push_back(seg1);
	return result;
}

//Point2f is_intersect(Point2f l1_1, Point2f l1_2, Point2f l2_1, Point2f l2_2)
//{
//	//首先判断两条直线是否相交并求出交点
//	
//	if (std::min(l1_1.x, l1_2.x) <= std::max(l2_1.x, l2_2.x) && std::max(l1_1.x, l1_2.x) >= std::min(l2_1.x, l2_2.x)
//		&& std::min(l1_1.y, l1_2.y) <= std::max(l2_1.y, l2_2.y) && std::max(l1_1.y, l1_2.y) >= std::min(l2_1.y, l2_2.y))
//		//通过快速排斥实验
//	{
//		Point2f left21, l22, right21;
//		left21.x = l1_1.x - l2_2.x;
//		left21.y = l1_1.y - l2_2.y;
//		right21.x = l1_2.x - l2_2.x;
//		right21.y = l1_2.y - l2_2.y;
//		l22.x = l2_1.x - l2_2.x;
//		l22.y = l2_1.y - l2_2.y;
//		if (((l1_1.x - l2_2.x)*(l2_1.y - l2_2.y) - (l2_1.x - l2_2.x)*(l1_1.y - l2_2.y))*((l1_2.x - l2_2.x)*(l2_1.y - l2_2.y) - (l2_1.x - l2_2.x)*(l1_2.y - l2_2.y)) < 0)
//		{
//			Point2f left12, l11, right12;
//			left12.x = l2_1.x - l1_1.x;
//			left12.y = l2_1.y - l1_1.y;
//			right12.x = l2_2.x - l1_1.x;
//			right12.y = l2_2.y - l1_1.y;
//			l11.x = l1_2.x - l1_1.x;
//			l11.y = l1_2.y - l1_1.y;
//			if (((l2_2.x - l1_1.x)*(l1_2.y - l1_1.y) - (l1_2.x - l1_1.x)*(l2_2.y - l1_1.y))*((l2_1.x - l1_1.x)*(l1_2.y - l1_1.y) - (l1_2.x - l1_1.x)*(l2_1.y - l1_1.y)) < 0)
//			{
//				//相交，计算交点
//
//			}
//		}
//	}
//}


Point3f Deformation::rotating(Point3f point, Point3f axis1, Point3f axis2, double angle1)
{
	//旋转的时候一定要注意！！！旋转轴不是过原点的就不能单纯的乘以一个矩阵
	//例如：对于点point，旋转轴两端坐标为axis1和axis2，旋转矩阵为W，那么在求result的时候
	//旋转轴方向 axis=axis2-axis1,,   result=(point-axis1)*W+axis1
	Point3f axis;
	axis.x = axis2.x - axis1.x;
	axis.y = axis2.y - axis1.y;
	axis.z = axis2.z - axis1.z;
	Point3f rotate_point;
	rotate_point.x = point.x - axis1.x;
	rotate_point.y = point.y - axis1.y;
	rotate_point.z = point.z - axis1.z;
	Point3f unit_vector;
	unit_vector.x = axis.x / sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	unit_vector.y = axis.y / sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	unit_vector.z = axis.z / sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	Point3f result;
	result.x = rotate_point.x * ((pow(unit_vector.x, 2))*(1 - cos(angle1)) + cos(angle1)) + rotate_point.y * (unit_vector.x * unit_vector.y * (1 - cos(angle1)) + unit_vector.z * sin(angle1)) +
		rotate_point.z * (unit_vector.x*unit_vector.z*(1 - cos(angle1)) - unit_vector.y*sin(angle1));
	result.y = rotate_point.x * (unit_vector.x*unit_vector.y*(1 - cos(angle1)) - unit_vector.z*sin(angle1)) + rotate_point.y * (pow(unit_vector.y, 2)*(1 - cos(angle1)) + cos(angle1)) +
		rotate_point.z * (unit_vector.y* unit_vector.z*(1 - cos(angle1)) + unit_vector.x*sin(angle1));
	result.z = rotate_point.x * (unit_vector.x* unit_vector.z*(1 - cos(angle1)) + unit_vector.y*sin(angle1)) + rotate_point.y * (unit_vector.y* unit_vector.z*(1 - cos(angle1)) - unit_vector.x*sin(angle1)) +
		rotate_point.z * (pow(unit_vector.z, 2)*(1 - cos(angle1)) + cos(angle1));
	result.x = result.x + axis1.x;
	result.y = result.y + axis1.y;
	result.z = result.z + axis1.z;
	return result;
}

void Deformation::contour_segmentation(vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<int>>& intersect_index, 
	vector<vector<Point2i>>&points_set, vector<vector<Point2f>>& segment_pairs)
{
	//提取分割后轮廓的函数
	for (int k = 0; k < points_set.size(); k++)
	{
		vector<Point2f> contour0_temp;
		vector<Point2f> contour_temp;
		vector<Point2f> temp_pair;
		int size = seg_contours.size();
		bool flag = false;
		int e;
		for (e = 0; e < size; e++)
		{
			if (!seg_contours[e].empty()) {
				temp_pair = compute_intersect(seg_contours[e], points_set[k], intersect_index);
				if (temp_pair.size() == 0)
					continue;
				else
				{
					flag = true;
					contour_temp.push_back(temp_pair[0]);
					int min, max;
					int v1 = intersect_index[k][0];//向小的方向转
					if (seg_contours[e][v1] != temp_pair[0])
					{
						min = v1;
						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						if (min > max)
						{
							min = max;
							max = v1;
						}
					}
					else
					{
						v1 = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						min = v1;
						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						if (min > max)
						{
							max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
							min = (max - 1 + seg_contours[e].size()) % seg_contours[e].size();
						}
					}
					while (!(min == intersect_index[k][2] && max == intersect_index[k][3]))
					{
						contour_temp.push_back(seg_contours[e][v1]);
						v1 = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						min = v1;
						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						if (min > max)
						{
							max = v1;
							min = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
						}
					};
					if (seg_contours[e][v1] != temp_pair[1]) {
						contour_temp.push_back(seg_contours[e][v1]);
						contour_temp.push_back(temp_pair[1]);
					}
					else
					{
						contour_temp.push_back(seg_contours[e][v1]);
					}

					for (int g = contour_temp.size() - 1; g >= 0; g--)
					{
						contour0_temp.push_back(contour_temp[g]);//将顺时针变成逆时针
					}




					contour_temp.clear();
					int v2 = intersect_index[k][1];//向大的方向转,原本就是逆时针
					contour_temp.push_back(temp_pair[0]);
					if (seg_contours[e][v2] != temp_pair[0])
					{
						min = v2;
						max = (v2 + 1) % seg_contours[e].size();
						if (min > max)
						{
							min = max;
							max = v1;
						}
					}
					else
					{
						v2 = (v2 + 1) % seg_contours[e].size();
						min = v2;
						max = (v2 + 1) % seg_contours[e].size();
						if (min > max)
						{
							max = (v2 + 1) % seg_contours[e].size();
							min = (max + 1) % seg_contours[e].size();
						}
					}
					while (!(min == intersect_index[k][2] && max == intersect_index[k][3]))
					{
						contour_temp.push_back(seg_contours[e][v2]);
						v2 = (v2 + 1) % seg_contours[e].size();
						min = v2;
						max = (v2 + 1) % seg_contours[e].size();
						if (min > max)
						{
							max = v2;
							min = (v2 + 1) % seg_contours[e].size();
						}
					};
					if (seg_contours[e][v2] != temp_pair[1]) {
						contour_temp.push_back(seg_contours[e][v2]);
						contour_temp.push_back(temp_pair[1]);
					}
					else
					{
						contour_temp.push_back(seg_contours[e][v2]);
					}
					//最终contour0_temp和contour_temp就是分割后的两个轮廓，且都是逆时针放置  

					break;
				}
			}
		}
		if (flag)
		{
			int pos = seg_contours.size();
			seg_contours.push_back(contour_temp);
			seg_contours.push_back(contour0_temp);
			vector<int> index_temp0;
			for (int y = 0; y < seg_index.size(); y++)
			{
				index_temp0.clear();
				bool flag0 = false;
				bool flag1 = false;
				if (seg_index[y][0] == e)
				{
					//查询新生成的轮廓中哪个包含seg_index[y][1]分割线
					vector<Point2f> seg0 = segment_pairs[seg_index[y][1]];
					int x;
					for (x = 0; x < contour_temp.size(); x++)
					{
						if (contour_temp[x] == seg0[0])
							flag0 = true;
						if (contour_temp[x] == seg0[1])
							flag1 = true;
					}
					if (flag0&&flag1)
					{
						index_temp0.push_back(pos);
						index_temp0.push_back(seg_index[y][1]);
						seg_index.push_back(index_temp0);
					}
					index_temp0.clear();
					flag0 = false;
					flag1 = false;
					for (x = 0; x < contour0_temp.size(); x++)
					{
						if (contour0_temp[x] == seg0[0])
							flag0 = true;
						if (contour0_temp[x] == seg0[1])
							flag1 = true;
					}
					if (flag0&&flag1)
					{
						index_temp0.push_back(pos + 1);
						index_temp0.push_back(seg_index[y][1]);
						seg_index.push_back(index_temp0);
					}

				}
			}
			vector<int> index_temp;
			index_temp.push_back(pos);
			index_temp.push_back(k);
			seg_index.push_back(index_temp);
			index_temp.clear();
			index_temp.push_back(pos + 1);
			index_temp.push_back(k);
			seg_index.push_back(index_temp);
			segment_pairs.push_back(temp_pair);
			seg_contours[e].clear();//将被划分的contour清空
		}
	}
}

void Deformation::rotated_contour_compute(queue<int>& reference, vector<vector<Point3f>>seg_contours_3, vector<vector<Point3f>> segment_pairs_3, double angle1,
	int** adj_matrix, vector<vector<int>> seg_index, int ang)
{


	bool* refer_flag = new bool[seg_contours_3.size()];
	for (int i = 0; i < seg_contours_3.size(); i++)
	{
		refer_flag[i] = false;
	}


	while (!reference.empty())
	{
		int refer = reference.front();
		refer_flag[refer] = true;
		reference.pop();
		vector<int> root;
		for (int i = 0; i < seg_contours_3.size(); i++)
		{
			if (adj_matrix[refer][i] == 1 && !refer_flag[i])
			{
				root.push_back(i);
			}
		}
		for (int j = 0; j < root.size(); j++)
		{
			int axis_index = -1;
			for (int a = 0; a < seg_index.size(); a++)
			{
				if (seg_index[a][0] == refer)
				{
					for (int b = 0; b < seg_index.size(); b++)
					{
						if (seg_index[b][0] == root[j])
						{
							if (seg_index[a][1] == seg_index[b][1])
							{
								axis_index = seg_index[a][1];
								break;
							}
						}
					}
					if (axis_index != -1)
						break;
				}
			}

			if (axis_index == -1)
				cerr << "The axis is wrong";
			vector<Point3f> axis = rotate_axis(seg_contours_3[refer], segment_pairs_3[axis_index]);

			bool* tree_flag = new bool[seg_contours_3.size()];
			for (int k = 0; k < seg_contours_3.size(); k++)
				tree_flag[k] = false;
			tree_flag[refer] = true;


			//以root[j]为根深度优先搜索，最后得到的所有块以axis_index为轴进行旋转
			stack <int> area;
			area.push(root[j]);
			tree_flag[root[j]] = true;
			vector<int> result;
			while (!area.empty())
			{
				int sum = 0;
				int cur = area.top();
				for (int r = 0; r < seg_contours_3.size(); r++)
				{
					if (adj_matrix[cur][r] == 1)
					{
						if (!tree_flag[r])
						{
							sum++;
							area.push(r);
							tree_flag[r] = true;
							break;
						}
					}
				}
				if (sum == 0)
				{
					result.push_back(area.top());
					area.pop();
				}
			}










			for (int e = 0; e < result.size(); e++)
			{
				for (int f = 0; f < seg_contours_3[result[e]].size(); f++)
				{
					Point3f new_point = rotating(seg_contours_3[result[e]][f], axis[0], axis[1], angle1);
					seg_contours_3[result[e]][f] = new_point;
				}
			}

			bool* seg_flag = new bool[segment_pairs_3.size()];
			for (int u = 0; u < segment_pairs_3.size(); u++)
				seg_flag[u] = false;

			for (int y = 0; y < seg_index.size(); y++)
			{
				vector<int>::iterator iter = find(result.begin(), result.end(), seg_index[y][0]);
				if (iter == result.end())
					continue;
				else
				{
					if (seg_index[y][1] == axis_index)
						continue;
					else if (!seg_flag[seg_index[y][1]])
					{
						seg_flag[seg_index[y][1]] = true;
						Point3f new_point0 = rotating(segment_pairs_3[seg_index[y][1]][0], axis[0], axis[1], angle1);
						Point3f new_point1 = rotating(segment_pairs_3[seg_index[y][1]][1], axis[0], axis[1], angle1);
						segment_pairs_3[seg_index[y][1]][0] = new_point0;
						segment_pairs_3[seg_index[y][1]][1] = new_point1;
					}
				}
			}
			reference.push(root[j]);
		}
	}



	//ofstream outfile1("data1.obj", ios::out);
	string filename = "data" + to_string(ang) + ".obj";
	ofstream outfile1(filename, ios::out);
	//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	if (!outfile1)
	{
		cerr << "open error";
		exit(1);
	}

	outfile1 << "#List of geometric vertices, with (x,y,z) coordinates" << endl;

	std::srand(time(NULL));
	for (int c = 0; c < seg_contours_3.size(); c++)
	{
		if (seg_contours_3[c].empty())
			continue;
		float c1 = rand() % (N + 1) / (float)(N + 1);
		float c2 = rand() % (N + 1) / (float)(N + 1);
		float c3 = rand() % (N + 1) / (float)(N + 1);
		//注意顶点的颜色c1 c2 c3不能是相同的，所以要用不同随机数生成
		for (int d = 0; d < seg_contours_3[c].size(); d++) {
			outfile1 << "v" << " " << seg_contours_3[c][d].x << " " << seg_contours_3[c][d].y << " " << seg_contours_3[c][d].z <<
				" " << c1 << " " << c2 << " " << c3 << endl;
		}
	}
	outfile1 << "#Polygonal face element" << endl;
	int v_sum1 = 0;
	for (int d = 0; d < seg_contours_3.size(); d++)
	{
		if (seg_contours_3[d].empty())
			continue;
		outfile1 << "f" << " ";//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！
		for (int c = 0; c < seg_contours_3[d].size(); c++)
			outfile1 << v_sum1 + c + 1 << " ";
		outfile1 << endl;
		v_sum1 = v_sum1 + seg_contours_3[d].size();
	}
	outfile1.close();

	/*cv::imwrite("output.bmp", dstImage3);

	cv::imshow("Segmentation", dstImage3);
	cv::waitKey(0);*/
}

vector<vector<int>> Deformation::concavity(vector<vector<Point3f>> seg_contours_3, vector<vector<Point3f>> segment_pairs_3,
	vector<vector<int>> seg_index, int base_index, int** adj_matrix)
{
	bool* visited_contour = new bool[seg_contours_3.size()];
	for (int p = 0; p < seg_contours_3.size(); p++)
		visited_contour[p] = false;
	bool* visited_seg = new bool[segment_pairs_3.size()];
	for (int p = 0; p < segment_pairs_3.size(); p++)
		visited_seg[p] = false;
	queue<int> seg_visited;
	vector<vector<int>>  concavity;
	//该数组每行有三个单元，第一个单元表示轮廓index，第二个单元表示分割线index，第三个单元表示关节凹凸性，关节为凹，则为0，关节为凸，则为1
	seg_visited.push(base_index);
	visited_contour[base_index] = true;
	while (!seg_visited.empty())//这段代码采用广度优先算法，遍历每个contour，标记其各个分割线上关节的凹凸性。
	{
		int cur = seg_visited.front();
		seg_visited.pop();
		for (int s = 0; s < seg_index.size(); s++)
		{
			if (seg_index[s][0] == cur)
			{
				int index = seg_index[s][1];
				if (!visited_seg[index])
				{
					visited_seg[index] = true;
					vector<int> temp;
					temp.push_back(cur);
					temp.push_back(index);
					temp.push_back(1);
					concavity.push_back(temp);
				}
				else
				{
					vector<int> temp;
					temp.push_back(cur);
					temp.push_back(index);
					temp.push_back(0);
					concavity.push_back(temp);
				}
			}
		}
		for (int r = 0; r < seg_contours_3.size(); r++)
		{
			if (adj_matrix[cur][r] == 1)
			{
				if (!visited_contour[r] && !seg_contours_3[r].empty())
				{
					seg_visited.push(r);
					visited_contour[r] = true;
				}
			}
		}
	}
	return concavity;
}