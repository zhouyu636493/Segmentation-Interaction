#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_conformer_2.h>
//#include <CGAL/Polyhedron_3.h>
//#include <CGAL/HalfedgeDS_vector.h>
//#include <CGAL/Polyhedron_items_with_id_3.h>
#include <iostream>
#include <fstream>
#include "joint.h"
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>//这个头文件一定要有，里面包括平面细分的函数
#include "contour.h"
#include "triangle.h"
#include <string>
#include <queue>
#include <cgal_triangle.h>
//#include <Eigen/Dense>
//#include <CGAL/Homogeneous.h>
using namespace Eigen;
using namespace std;
using namespace cv;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point CdtPoint;//防止和opencv中的Point混淆
//typedef CGAL::Polyhedron_3< K,
//	CGAL::Polyhedron_items_3,
//	CGAL::HalfedgeDS_default> Polyhedron_3;
//Polyhedron_3 polyhedron;
//typedef Eigen::Vector3d Vec3d;




vector<cv::Point> points;
Mat dstImage;//必须有一个dstImage用来存储原始的图像，否则最后的直线会有很多中间状态的直线显示
Mat srcImage;//每次从dstImage复制过来图像，然后再此基础上实时显示

vector<vector<Point2i>> points_set;//存储的是用户输入的线段的两端坐标
cv::Point pre;
cv::Point cur;
cv::Point pre_1(-1, -1);

vector<vector<Point2f>> segment_pairs;//存储的是最终分割线在轮廓上的两端坐标

int find_next(vector<vector<int>> intersect_index,vector<vector<Point2f>> segment_pairs,vector<Point2i> contour,Point2f center,int seg_index,int ver_index)
{
//寻找第seg_index条分割线的ver_index端点的下一个contour点是逆时针（contour下标增大的方向）还是顺时针
//如果是逆时针返回1，如果是顺时针返回-1
	Point2f v1, v2, v3, v;
	int f, l;
	if (ver_index == 0)
	{
		f = 0;
		l = 1;
	}
	else if(ver_index==1)
	{
		f = 2;
		l = 3;
	}
	int t = intersect_index[seg_index][f];
	v1.x = contour[t].x - segment_pairs[seg_index][ver_index].x;
	v1.y = contour[t].y - segment_pairs[seg_index][ver_index].y;
	v2.x = center.x - segment_pairs[seg_index][ver_index].x;
	v2.y = center.y - segment_pairs[seg_index][ver_index].y;
	v3.x = contour[intersect_index[seg_index][l]].x - segment_pairs[seg_index][ver_index].x;
	v3.y = contour[intersect_index[seg_index][l]].y - segment_pairs[seg_index][ver_index].y;
	v.x = segment_pairs[seg_index][abs(ver_index-1)].x - segment_pairs[seg_index][ver_index].x;
	v.y = segment_pairs[seg_index][abs(ver_index - 1)].y - segment_pairs[seg_index][ver_index].y;
	if ((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) < 0 && (v3.x*v.y - v3.y*v.x)*(v2.x*v.y - v2.y*v.x) > 0)
	{ 
		return -1;
	}
	else if ((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) > 0 && (v3.x*v.y - v3.y*v.x)*(v2.x*v.y - v2.y*v.x) < 0)
	{
		return 1;
	}
	else if (v1.x == 0 && v1.y == 0)
	{
		if ((v3.x*v.y - v3.y*v.x)*(v2.x*v.y - v2.y*v.x) > 0)
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
	else if (v3.x == 0 && v3.y == 0)
	{
		if ((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) < 0)
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
}
bool is_in_contour(Point2f mid , vector<Point2f> contour)
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
			if ((mid.x >= contour[i].x&&mid.x <= contour[(i + 1) % n].x) || (mid.x <=contour[i].x&&mid.x >= contour[(i + 1) % n].x))
				return true;
		}
		else if ((mid.y >= contour[i].y&&mid.y < contour[(i + 1) % n].y) || (mid.y >= contour[(i + 1) % n].y&&mid.y < contour[i].y))
		{
			double d = ((mid.y- contour[i].y)*(contour[(i + 1) % n].x- contour[i].x)+(contour[i].x-mid.x)*(contour[(i + 1) % n].y- contour[i].y))
				* (contour[(i + 1) % n].y- contour[i].y);
			if (d > 0)
				count++;
			else if(d==0)
				return true;
		}
	}
	if (count % 2 == 0)
		return false;
	else
		return true;//在contour中
}
bool line_is_in_contour(CdtPoint p1 , CdtPoint p2, vector<Point2i> contour)
{
	//判断点mid是否在contour中，在则返回true
	Point2f mid;
	mid.x = (p1.x()+ p2.x()) / 2;
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
vector<vector<int>> intersect_index;//存放的是分割线与contour交点所在contour边的端点的index
vector<Point2f> compute_intersect(vector<Point2f> contour,vector<Point2i> points)
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
	if(!tep.empty())
	   intersect_index.push_back(tep);
	return intersect;
}
bool is_cross(vector<Point2f> contour, vector<Point2i> points)
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
		deta.x = points[0].x- contour[a].x ;
		deta.y =  points[0].y- contour[a].y;
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
vector<Point3f> rotate_axis(vector<Point3f> contour, vector<Point3f> segment)      
{
	Point3f seg1, seg2;
	seg1=segment[0];
	seg2= segment[1];
	int a;
	for (a = 0; a < contour.size(); a++)
	{
		if (contour[a]==seg1)
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

void onMouse(int event, int x, int y, int flags, void* param)
{
	vector<cv::Point> &contour = *(vector<cv::Point>*) param;
	Contour c0;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		dstImage.copyTo(srcImage);
		cv::Point pt;
		pt.x = x;
		pt.y = y;
		pre = pt;
		//circle(srcImage, pt, 2, Scalar(0, 255, 0),CV_FILLED,LINE_AA,0);
		imshow("Segmentation", srcImage);
		srcImage.copyTo(dstImage);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
	{
		dstImage.copyTo(srcImage);
		cur.x = x;
		cur.y = y;
		line(srcImage, pre, cur, Scalar(0, 255, 0), 1, CV_AA, 0);
		imshow("Segmentation", srcImage);

	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		dstImage.copyTo(srcImage);
		cur.x = x;
		cur.y = y;
		line(srcImage, pre, cur, Scalar(0, 255, 0), 2, 8, 0);//这里一定要用8通道，不可以用LINE_AA，因为下面删除的时候只能是最多8通道，不能是LINA_AA的16通道，所以如果是LINA_AA,那么删除的时候会有残渣像素
		//circle(srcImage, cur, 2, Scalar(0, 255, 0), CV_FILLED, LINE_AA, 0);
		points.push_back(pre);
		points.push_back(cur);
		points_set.push_back(points);
		points.clear();
		imshow("Segmentation", srcImage);
		srcImage.copyTo(dstImage);
	}
	else if (event == CV_EVENT_RBUTTONDOWN)//删除用户选定的直线
	{
		Point2i pt1;
		pt1.x = x;
		pt1.y = y;
		for (int i = 0; i < points_set.size(); i++) {
			if (c0.line_point_dist(pt1, points_set[i][0], points_set[i][1]) <= 4) {
				//line(dstImage, points_set[i][0], points_set[i][1], Scalar(0,0,0), 1, LINE_AA, 0);
				//这里应该把这条线段坐标从points_set中去除
				LineIterator lit(dstImage, points_set[i][0], points_set[i][1], 8);
				for (int j = 0; j < lit.count; j++, ++lit)
				{
					//这里需要注意！！！图像中行是坐标的y，列是坐标的x，和惯性想法是相反的！！
					dstImage.at<Vec3b>(lit.pos().y, lit.pos().x) = Vec3b(255,255,255);
				}
				imshow("Segmentation", dstImage);
				points_set.erase(points_set.begin() + i);
				break;
			}
		}

	}
	return;
}

Point3f rotating(Point3f point, Point3f axis1,Point3f axis2,double angle1)
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
	rotate_point.y= point.y - axis1.y;
	rotate_point.z = point.z - axis1.z;
	Point3f unit_vector;
	unit_vector.x= axis.x / sqrt(pow(axis.x,2)+pow(axis.y, 2)+pow(axis.z, 2));
	unit_vector.y= axis.y / sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	unit_vector.z= axis.z / sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	Point3f result;
	result.x = rotate_point.x * ((pow(unit_vector.x, 2))*(1 - cos(angle1)) + cos(angle1)) + rotate_point.y * (unit_vector.x * unit_vector.y * (1 - cos(angle1)) + unit_vector.z * sin(angle1)) +
		rotate_point.z * (unit_vector.x*unit_vector.z*(1-cos(angle1)) - unit_vector.y*sin(angle1));
	result.y = rotate_point.x * (unit_vector.x*unit_vector.y*(1-cos(angle1)) - unit_vector.z*sin(angle1)) + rotate_point.y * (pow(unit_vector.y,2)*(1-cos(angle1))+cos(angle1)) +
		rotate_point.z * (unit_vector.y* unit_vector.z*(1-cos(angle1))+ unit_vector.x*sin(angle1));
	result.z = rotate_point.x * (unit_vector.x* unit_vector.z*(1-cos(angle1))+ unit_vector.y*sin(angle1)) + rotate_point.y * (unit_vector.y* unit_vector.z*(1-cos(angle1)) - unit_vector.x*sin(angle1)) +
		rotate_point.z * (pow(unit_vector.z,2)*(1-cos(angle1))+cos(angle1));
	result.x = result.x + axis1.x;
	result.y= result.y + axis1.y;
	result.z = result.z + axis1.z;
	return result;
}
//static void draw_subdiv_facet(Mat img, CvSubdiv2DEdge edge)//这里画的是voronoi面
//{
//	CvSubdiv2DEdge t = edge;
//	int i, count = 0;
//	CvPoint* buf = 0;
//	Point2d* buf1 = 0;
//	do
//	{
//		count++;
//		t = cvSubdiv2DGetEdge(t, CV_NEXT_AROUND_LEFT);
//	} while (t != edge);
//	buf = (CvPoint*)malloc(count * sizeof(buf[0]));
//	buf1 = (Point2d*)malloc(count * sizeof(buf1[0]));
//	t = edge;
//	for (i = 0; i < count; i++)
//	{
//		CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(t);
//		if (!pt)break;
//		buf[i] = cvPoint(cvRound(pt->pt.x), cvRound(pt->pt.y));
//		t = cvSubdiv2DGetEdge(t, CV_NEXT_AROUND_LEFT);
//	}
//	if (i == count)
//	{
//		CvSubdiv2DPoint* pt = cvSubdiv2DEdgeDst(cvSubdiv2DRotateEdge(edge, 1));
//		for (i = 0; i < count; i++)
//		{
//			buf1[i].x = buf[i].x;
//			buf1[i].y = buf[i].y;
//		}
//		IplImage tmp = IplImage(img);
//		CvArr* arr = (CvArr*)&tmp;
//		//上面是如何将Mat转化为CVARR*
//		cvPolyLine(arr, &buf, &count, 1, 1, CV_RGB(0, 200, 0), 1, CV_AA, 0);//画出线。  
//		circle(img, cvPoint(cvRound(pt->pt.x), cvRound(pt->pt.y)), 5, Scalar(0, 0, 255), CV_FILLED, 8, 0);
//	}
//	free(buf);
//}

//static void paint_voronoi(CvSubdiv2D* subdiv, Mat img)//这个函数画的是voronoi面
//{
//	CvSeqReader reader;
//	int total = subdiv->edges->total;
//	int elem_size = subdiv->edges->elem_size;
//	cvCalcSubdivVoronoi2D(subdiv);
//	cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0);
//	for (int i = 0; i < total; i++)
//	{
//		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);
//		if (CV_IS_SET_ELEM(edge))
//		{
//			CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge;
//			draw_subdiv_facet(img, cvSubdiv2DRotateEdge(e, 1));
//			draw_subdiv_facet(img, cvSubdiv2DRotateEdge(e, 3));
//		}
//		CV_NEXT_SEQ_ELEM(elem_size, reader);
//	}
//
//}


////下面这个函数是删除三角形中和多边形边相交的边
//static void delete_intersection(vector<Point>contour, CvSubdiv2D* subdiv)
//{
//	vector<vector<Point>> triangle_edges;//存的是delaunay三角的边
//	CvSeqReader reader;
//	int  total = subdiv->edges->total;
//	int elem_size = subdiv->edges->elem_size;
//	cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0);
//	vector<Point> temp_edge;
//	for (int i = 0; i < total; i++)
//	{
//		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);
//		if (CV_IS_SET_ELEM(edge))
//		{
//			CvSubdiv2DPoint* org_pt;
//			CvSubdiv2DPoint* dst_pt;
//			CvPoint2D32f org;
//			CvPoint2D32f dst;
//			CvPoint iorg, idst;
//			org_pt = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge)edge);
//			dst_pt = cvSubdiv2DEdgeDst((CvSubdiv2DEdge)edge);
//			if (org_pt&&dst_pt)
//			{
//				org = org_pt->pt;
//				dst = dst_pt->pt;
//				iorg = cvPoint(cvRound(org.x), cvRound(org.y));
//				idst = cvPoint(cvRound(dst.x), cvRound(dst.y));
//				temp_edge.push_back(iorg);
//				temp_edge.push_back(idst);
//				triangle_edges.push_back(temp_edge);
//				temp_edge.clear();
//			}
//		}
//		CV_NEXT_SEQ_ELEM(elem_size, reader);
//	}
//	vector<vector<Point>> edges;//里面存的是约束边的坐标
//	vector<Point> temp;
//	for (int i = 0; i < contour.size()-1; i++)
//	{
//		temp.push_back(contour[i]);
//		temp.push_back(contour[i + 1]);
//		edges.push_back(temp);
//		temp.clear();
//	}
//	temp.push_back(contour[contour.size() - 1]);
//	temp.push_back(contour[0]);
//	edges.push_back(temp);
//	for (int j = 0; 
//		Point vi = edges[j][0];
//		Point vj = edges[j][1];
//		vector<Poinj < edges.size(); j++)
//	{t> edge1, edge2;
//		edge1 = edges[j];
//		edge2.push_back(vi);
//		edge2.push_back(vj);
//		vector<vector<Point>> ::iterator iter1,iter2;
//		iter1 = find(triangle_edges.begin(), triangle_edges.end(),edge1);
//		iter2 = find(triangle_edges.begin(), triangle_edges.end(), edge2);
//		if (iter1 != triangle_edges.end() || iter2 != triangle_edges.end())
//		{
//			continue;
//		}
//		else
//		{
//			vector<vector<Point>> cross_ij;//存放的是三角网格中与vi_vj 边相交的边
//			//遍历所有三角形，为三角形编号，并对每个顶点确定其所在三角形集合
//
//
//
//
//
//		}
//		edge1.clear();
//		edge2.clear();
//
//	}
//}
//
//

//bool is_on_edge(Point2f p1,Point2f p2,Point2f seg)
//{
//	Point2f p, q;
//	p.x = p2.x - p1.x;
//	p.y = p2.y - p1.y;
//	q.x = seg.x - p1.x;
//	q.y = seg.y - p1.y;
//	if (p.x*q.y - p.y*q.x == 0)
//	{
//		if (min(p1.x, p2.x) <= seg.x&&max(p1.x, p2.x) >= seg.x)
//		{
//			if (min(p1.y, p2.y) <= seg.y&&max(p1.y, p2.y) >= seg.y)
//				return true;
//		}
//	}
//	return false;
//}





int main(int argc, char *argv[])
{
	string name;//图片编号
	Point2f center;//轮廓中心



	//下面是用户输入关节的相关参数
	double L, d1, d2, R, H, W;//W代表关节的宽度，在计算旋转角度的时候W不需要考虑
	cout << "请输入关节的参数" << endl;
	cin >> L >> d1 >> d2 >> R >> H >> W;
	while (L <= 0 || d1 <= 0 || d2 <= 0 || R <= 0 || H <= 0)
	{
		cerr << "关节参数都应该大于0,请重新输入关节的参数";
		cin >> L >> d1 >> d2 >> R >> H;
	}
	while (L - d2 - 2 * R - d1 < 0)
	{
		cerr << "L-d2-2*R-d1要大于等于0,请重新输入关节的参数";
		cin >> L >> d1 >> d2 >> R >> H;
	}
	Joint j1(L, d1, d2, R, H);
	double angle1 = j1.angle();
	cout << "当前关节可以旋转的最大角度为：" << angle1 << endl;


	//Input the polygon
	cout << "请输入要处理的图片编号" << endl;
	cin >> name;
	Contour c;
	//下面是提取形状轮廓
	string name1 = "F:/Project/Segmentation Interaction/dataset/" + name + ".png";//这个字符串的读取没有问题
	Mat img = imread(name1, 1);//按三通道方式读入图像，即彩色图像
	Mat de_noise = c.image(img);
	vector<vector<Point2i>>  contours = c.contour_generate(de_noise);
	//绘制轮廓图
	Mat dstImage0, dstImage2;
	dstImage0 = Mat::zeros(de_noise.size(), CV_8UC3);
	//	drawContours(dstImage0, contours, 0, Scalar(255, 0, 0), 1, LINE_AA); //画的是index为0的轮廓，也就是面积最大的
	resize(dstImage0, dstImage2, Size(dstImage0.cols * 1.2, dstImage0.rows * 1.2), 0, 0, INTER_LINEAR);															//这里主要用到的是contours[0]
	//setMouseCallback("Segmentation",onMouse,reinterpret_cast<void*> (&contours[0]));	
	vector<cv::Point> contour;
	approxPolyDP(contours[0], contour, 5, true);//多边形简化轮廓
	dstImage = cv::Mat::zeros(dstImage2.rows, dstImage2.cols, CV_8UC3);
	dstImage.setTo(cv::Scalar(255, 255, 255));//设置背景颜色为白色
	polylines(dstImage, contour, true, (0, 0, 255),2);//画出简化后的多边形，同时也是之后交互的画布
	Mat dstImage1 = dstImage.clone();//之后用来显示三角剖分的画布
	Moments mu = moments(contour, false);
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	center = mc;//这个地方的center可以是用户在多边形上任意pick一个点
	circle(dstImage, center, 4, Scalar(0, 0, 255), CV_FILLED, CV_AA, 0);//画出中心点坐标
	namedWindow("Segmentation", CV_WINDOW_AUTOSIZE);
	imshow("Segmentation",dstImage);
	setMouseCallback("Segmentation", onMouse, (void*)&contour);
	char key = waitKey(0);
	//if (key == 'h' || key == 'H')
	//{//键盘出入回车，则计算图像轮廓内的分割，以及形变量
	//	/*for (int k = 0; k < points_set.size(); k++)
	//	{
	//		vector<Point2i> temp_pair;
	//		double max_dist = DBL_MIN;
	//		cv::Point max_point;
	//		LineIterator iter0(dstImage, points_set[k][0], points_set[k][1]);
	//		for (int m = 0; m < iter0.count; m++, ++iter0) {
	//			if (abs(pointPolygonTest(contour, iter0.pos(), true)) < 1)
	//			{
	//				if (temp_pair.size() == 0) {
	//					circle(dstImage, iter0.pos(), 5, Scalar(0,0, 0),-1);
	//					temp_pair.push_back(iter0.pos());
	//				}
	//				else {
	//					if (c.get_distance(iter0.pos(), temp_pair[0]) > max_dist)
	//					{
	//						max_dist = c.get_distance(iter0.pos(), temp_pair[0]);
	//						max_point = iter0.pos();
	//					}
	//				}
	//			}
	//		}
	//		temp_pair.push_back(max_point);
	//		circle(dstImage, max_point, 5, Scalar(0, 0, 0),-1);
	//		segment_pairs.push_back(temp_pair);
	//		temp_pair.clear();
	//	}
	//	cout << segment_pairs.size() << endl;
	//	imshow("Segmentation", dstImage);*/
	//	//上面代码计算的交点不准确，是利用pointPolygonTest函数得到的，下面是利用http://www.twinklingstar.cn/2016/2916/3-5-linear-intersect/
	//	//线段与线段相交的代码计算出来的
	//	for (int k = 0; k < points_set.size(); k++)
	//	{
	//		vector<Point2f> temp_pair;
	//		temp_pair = compute_intersect(contour, points_set[k]);
	//		circle(dstImage, temp_pair[0], 5, Scalar(0, 0, 0), -1);
	//		circle(dstImage, temp_pair[1], 5, Scalar(0, 0, 0), -1);
	//		segment_pairs.push_back(temp_pair);
	//	}
	//	cout << segment_pairs.size() << endl;
	//	imshow("Segmentation", dstImage); 
	//	
	//}
	//waitKey(0);
	vector<vector<Point2f>> seg_contours;//存放划分得到的轮廓，如果某轮廓被后续分割线分割，则相应位置清空，然后后面继续追加分割后生成的新轮廓
	vector<vector<int>> seg_index;//二维数组，第一个值表示contour的index，第二个表示的是该contour相关的分割线的index
	vector<Point2f> contour0;
	for (int a = 0; a < contour.size(); a++)
	{
		Point2f temp;
		temp.x = contour[a].x;
		temp.y = contour[a].y;
		contour0.push_back(temp);
	}
	seg_contours.push_back(contour0);

	
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
				temp_pair = compute_intersect(seg_contours[e], points_set[k]);
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
						index_temp0.push_back(pos+1);
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
			index_temp.push_back(pos+1);
			index_temp.push_back(k);
			seg_index.push_back(index_temp);
			segment_pairs.push_back(temp_pair);
			seg_contours[e].clear();//将被划分的contour清空
		}
	}
	Mat dstImage3;
	dstImage3 = cv::Mat::zeros(dstImage2.rows, dstImage2.cols, CV_8UC3);
	dstImage3.setTo(cv::Scalar(255, 255, 255));//设置背景颜色为白色
	int scalar = 255;
	for (int t = 0; t < seg_contours.size(); t++)
	{
		Scalar color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);
		for (int p = 0; p < seg_contours[t].size(); p++)
		{
			line(dstImage3, seg_contours[t][p], seg_contours[t][(p + 1)% seg_contours[t].size()],color,2,8,0);
		}
	}
	circle(dstImage3, center, 4, Scalar(0, 0, 255), CV_FILLED, CV_AA, 0);//画出中心点坐标
	imshow("Segmentation", dstImage3);
	cv::waitKey(0);


	int** adj_matrix = new int*[seg_contours.size()];
	//adj_matrix存放的是contour之间的邻接关系，如果两个contour存在共同的分割线，那这两个contour就是相邻的，相应矩阵位置为1，否则为0；
	//是否存在共同分割线，查询seg_index数组就可以
	for (int s = 0; s < seg_contours.size(); s++)
	{
		adj_matrix[s] = new int[seg_contours.size()];
		for (int k = 0; k < seg_contours.size(); k++)
			adj_matrix[s][k] = 0;
	}
	
	for (int s = 0; s < seg_index.size(); s++)
	{
		if (!seg_contours[seg_index[s][0]].empty())//判断轮廓是否被清空，也就是是否被后续分割线分割
		{
			for (int v = s + 1; v < seg_index.size(); v++)
			{
				if (!seg_contours[seg_index[v][0]].empty())
				{
					if (seg_index[s][1] == seg_index[v][1])
					{
						cout << s << endl;
						cout << v << endl;
						cout << seg_contours.size() << endl;
						cout << seg_index[s][0] << endl;
						cout << seg_index[v][0] << endl;
						int a = seg_index[s][0];
						int b = seg_index[v][0];
						adj_matrix[a][b] = 1;
						adj_matrix[b][a] = 1;
					}
				}
			}
		}
	}

	for (int k = 0; k< seg_contours.size(); k++)
	{
		for (int p = 0; p < seg_contours.size(); p++)
		{
			cout << adj_matrix[k][p] << " ";
		}
		cout << endl;
	}


	//用户输入一个点center，判断该点位于哪个contour中，如果找到相应contour则break 
	int base_index;
	//静止contour的编号
	for ( base_index = 0; base_index < seg_contours.size(); base_index++)
	{
		if (!seg_contours[base_index].empty()) {
			if (is_in_contour(center, seg_contours[base_index]))
				break;
		}	
	}
	cout << base_index << endl;


	vector<vector<Point3f>> seg_contours_3;
	for (int w = 0; w < seg_contours.size(); w++)
	{
		vector<Point3f> temp;
		for (int z = 0; z < seg_contours[w].size(); z++)
		{
			Point3f temp1;
			temp1.x = seg_contours[w][z].x;
			temp1.y = seg_contours[w][z].y;
			temp1.z = 0;
			temp.push_back(temp1);
		}
		seg_contours_3.push_back(temp);
	}//将seg_contours原本的二维变成三维
	vector<vector<Point3f>> segment_pairs_3;
	for (int w = 0; w < segment_pairs.size(); w++)
	{
		vector<Point3f> temp;
		for (int z = 0; z < segment_pairs[w].size(); z++)
		{
			Point3f temp1;
			temp1.x = segment_pairs[w][z].x;
			temp1.y = segment_pairs[w][z].y;
			temp1.z = 0;
			temp.push_back(temp1);
		}
		segment_pairs_3.push_back(temp);
	}//将segment_pairs原本的二维变成三维

	//ofstream outfile("data1.obj", ios::out);
	////注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	//if (!outfile)
	//{
	//	cerr << "open error";
	//	exit(1);
	//}

	//outfile << "#List of geometric vertices, with (x,y,z) coordinates" << endl;
	//int vertice_sum0 = 0;
	//vector<vector<int>> f_vector0;
	//vector<Point3f> v_vector0;
	//vector<int> v_vindex;
	//for (int c = 0; c < seg_contours_3.size(); c++)
	//{
	//	if (seg_contours_3[c].empty())
	//		continue;
	//	int repeat_sum0 = 0;
	//	vector<int> f_temp0;
	//	for (int d = 0; d < seg_contours_3[c].size(); d++) {
	//		vector<Point3f>::iterator iter0 = find(v_vector0.begin(), v_vector0.end(), seg_contours_3[c][d]);
	//		if (iter0 == v_vector0.end()) {
	//			v_vector0.push_back(seg_contours_3[c][d]);
	//			outfile << "v" << " " << seg_contours_3[c][d].x << " " << seg_contours_3[c][d].y << " " << seg_contours_3[c][d].z << endl;
	//			f_temp0.push_back(vertice_sum0 + d - repeat_sum0 + 1);
	//		}
	//		else
	//		{
	//			repeat_sum0++;
	//			f_temp0.push_back(distance(v_vector0.begin(), iter0) + 1);
	//		}
	//	}
	//	f_vector0.push_back(f_temp0);
	//	vertice_sum0 = vertice_sum0 + seg_contours_3[c].size() - repeat_sum0;
	//}
	//outfile << "#" << vertice_sum0 << " vertices" << endl;
	//outfile << "#Polygonal face element" << endl;
	//for (int d = 0; d < f_vector0.size(); d++)
	//{
	//	outfile << "f" << " ";
	//	for (int c = 0; c < f_vector0[d].size(); c++)
	//		outfile << f_vector0[d][c] << " ";
	//	outfile << endl;
	//}

	//outfile.close();

	ofstream outfile("data0.obj", ios::out);
	//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	if (!outfile)
	{
		cerr << "open error";
		exit(1);
	}

	outfile << "#List of geometric vertices, with (x,y,z) coordinates" << endl;
	for (int c = 0; c < seg_contours_3.size(); c++)
	{
		if (seg_contours_3[c].empty())
			continue;;
		for (int d = 0; d < seg_contours_3[c].size(); d++) {
			outfile << "v" << " " << seg_contours_3[c][d].x << " " << seg_contours_3[c][d].y << " " << seg_contours_3[c][d].z << endl;
		}
	}
	outfile << "#Polygonal face element" << endl;
	int v_sum = 0;
	for (int d = 0; d < seg_contours_3.size(); d++)
	{
		if (seg_contours_3[d].empty())
			continue;
		outfile<< "f" << " ";//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！
		for (int c = 0; c < seg_contours_3[d].size(); c++)
			outfile<< v_sum + c + 1 << " ";
		outfile << endl;
		v_sum = v_sum + seg_contours_3[d].size();
	}

	outfile.close();
	




	queue<int> reference;
	reference.push(base_index);
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
			if (adj_matrix[refer][i] == 1&&!refer_flag[i])
			{
				root.push_back(i);
			}
		}
		for (int j = 0; j < root.size(); j++)
		{
			int axis_index=-1;
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
					else if(!seg_flag[seg_index[y][1]])
					{
						seg_flag[seg_index[y][1]] =true;
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
	
	ofstream outfile1("data1.obj", ios::out);
	//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	if (!outfile1)
	{
		cerr << "open error";
		exit(1);
	}

	outfile1 << "#List of geometric vertices, with (x,y,z) coordinates" << endl;
	for (int c = 0; c < seg_contours_3.size(); c++)
	{
		if (seg_contours_3[c].empty())
			continue;;
		for (int d = 0; d < seg_contours_3[c].size(); d++) {
			outfile1 << "v" << " " << seg_contours_3[c][d].x << " " << seg_contours_3[c][d].y << " " << seg_contours_3[c][d].z << endl;
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

	cv::imwrite("output.bmp", dstImage3);

	imshow("Segmentation", dstImage3);
	cv::waitKey(0);

	





	









	//遍历
	
	/*CGAL::set_halfedgeds_items_id(tri0);

	std::vector<Vector3d> vecs;
	std::vector<int> face_id_0;
	std::vector<int> face_id_1;
	std::vector<int> face_id_2;

	for (Polyhedron_3::Vertex_iterator iter = polyhedron.vertices_begin();
		iter != polyhedron.vertices_end(); iter++)
	{
		Poly_point_3 p = iter->point();
		vecs.push_back(Vector3d(p[0], p[1], p[2]));
	}

	for (Polyhedron_3::Face_iterator iter = polyhedron.facets_begin(); iter != polyhedron.facets_end(); iter++)
	{
		face_id_0.push_back(iter->halfedge()->next()->next()->vertex()->id());
		face_id_1.push_back(iter->halfedge()->vertex()->id());
		face_id_2.push_back(iter->halfedge()->next()->vertex()->id());
	}
	string out_path = "F:\Project\Segmentation Interaction\Segmentation Interaction";
	CGAL_Output_Obj(out_path, vecs, face_id_0, face_id_1, face_id_2);
*/
	//计算每条分割线在关节驱动下旋转后的位置，具体流程如下：
	//(1)对于每条分割线，其将contour分成两部分，计算不包含多边形重心部分所包含的其他分割线，并加入到数据结构partition中
	//(2)对于每条分割线，寻找partition中所有包含它的分割线，并将这些分割线按照其包含分割线数量的多少从大到小排序，放入partition_sort中
	//(3)根据（2）中的计算结果，首先取包含分割线数量最多的分割线，并检查其他partition中是否有包含它的，如果没有它就是静止的，然后
	
	//vector<vector<int>> partition;
	//vector<vector<int>> boundary;
	//for (int s = 0; s < segment_pairs.size(); s++)
	//{
	//	vector<int> bound;//存放的是当前分割线所形成的新形状的边界所包含的contour中顶点的编号
	//	vector<Point2f> contour1;
	//	Point2f v1, v2,v3,v;
	//	int t = intersect_index[s][0];
	//	v1.x = contour[t].x - segment_pairs[s][0].x;
	//	v1.y= contour[t].y - segment_pairs[s][0].y;
	//	v2.x = center.x - segment_pairs[s][0].x;
	//	v2.y=center.y- segment_pairs[s][0].y;
	//	v3.x =contour[intersect_index[s][1]].x- segment_pairs[s][0].x;
	//	v3.y = contour[intersect_index[s][1]].y- segment_pairs[s][0].y; 
	//	v.x=segment_pairs[s][1].x- segment_pairs[s][0].x;
	//	v.y= segment_pairs[s][1].y - segment_pairs[s][0].y;
	//	if ((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) < 0&&(v3.x*v.y-v3.y*v.x)*(v2.x*v.y - v2.y*v.x)>0)
	//	{
	//		contour1.push_back(segment_pairs[s][0]);
	//		contour1.push_back(contour[t]);
	//		bound.push_back(t);

	//		int j = t;
	//		int max= j;
	//		int min= (j + contour.size() - 1) % contour.size();
	//		if (j < (j + contour.size() - 1) % contour.size())
	//		{
	//			max = (j + contour.size() - 1) % contour.size();
	//			min = j;
	//		}
	//		while(!(min == intersect_index[s][2]&& max== intersect_index[s][3]))
	//		{
	//			contour1.push_back(contour[(j + contour.size() - 1) % contour.size()]);
	//			bound.push_back((j + contour.size() - 1) % contour.size());
	//			j= (j + contour.size() - 1) % contour.size();
	//			max = j;
	//			min= (j + contour.size() - 1) % contour.size();
	//			if (j < (j + contour.size() - 1) % contour.size())
	//			{
	//				max = (j + contour.size() - 1) % contour.size();
	//				min = j;
	//			}
	//		}
	//		contour1.push_back(segment_pairs[s][1]);
	//		
	//	}
	//	else if((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) > 0 && (v3.x*v.y - v3.y*v.x)*(v2.x*v.y - v2.y*v.x)<0)
	//	{
	//		contour1.push_back(segment_pairs[s][0]);
	//		contour1.push_back(contour[intersect_index[s][1]]);
	//		bound.push_back(intersect_index[s][1]);
	//		int j = intersect_index[s][1];
	//		int min = j;
	//		int max = (j + 1) % contour.size();
	//		if (j > (j + 1) % contour.size())
	//		{
	//			min = (j + 1) % contour.size();
	//			max = j;
	//		}
	//		while (!(min == intersect_index[s][2] && max == intersect_index[s][3]))
	//		{
	//			contour1.push_back(contour[(j +1) % contour.size()]);
	//			bound.push_back((j + 1) % contour.size());
	//			j = (j + 1) % contour.size();
	//			max= (j + 1) % contour.size();
	//			min = j;
	//			if (j > (j + 1) % contour.size())
	//			{
	//				min = (j + 1) % contour.size();
	//				max = j;
	//			}
	//		}
	//		contour1.push_back(segment_pairs[s][1]);
	//	}
	//	else if (v1.x==0&&v1.y==0)
	//	{
	//		if ((v3.x*v.y - v3.y*v.x)*(v2.x*v.y - v2.y*v.x) > 0)
	//		{
	//			contour1.push_back(contour[t]);
	//			int j = t;
	//			int max = j;
	//			int min = (j + contour.size() - 1) % contour.size();
	//			if (j < (j + contour.size() - 1) % contour.size())
	//			{
	//				max = (j + contour.size() - 1) % contour.size();
	//				min = j;
	//			}
	//			while (!(min == intersect_index[s][2] && max == intersect_index[s][3]))
	//			{
	//				contour1.push_back(contour[(j + contour.size() - 1) % contour.size()]);
	//				bound.push_back((j + contour.size() - 1) % contour.size());
	//				j = (j + contour.size() - 1) % contour.size();
	//				max = j;
	//				min = (j + contour.size() - 1) % contour.size();
	//				if (j < (j + contour.size() - 1) % contour.size())
	//				{
	//					max = (j + contour.size() - 1) % contour.size();
	//					min = j;
	//				}
	//			}
	//			contour1.push_back(segment_pairs[s][1]);
	//		}
	//		else
	//		{
	//			contour1.push_back(segment_pairs[s][0]);
	//			contour1.push_back(contour[intersect_index[s][1]]);
	//			bound.push_back(intersect_index[s][1]);
	//			int j = intersect_index[s][1];
	//			int min = j;
	//			int max = (j + 1) % contour.size();
	//			if (j > (j + 1) % contour.size())
	//			{
	//				min = (j + 1) % contour.size();
	//				max = j;
	//			}
	//			while (!(min == intersect_index[s][2] && max == intersect_index[s][3]))
	//			{
	//				contour1.push_back(contour[(j + 1) % contour.size()]);
	//				bound.push_back((j + 1) % contour.size());
	//				j = (j + 1) % contour.size();
	//				max = (j + 1) % contour.size();
	//				min = j;
	//				if (j > (j + 1) % contour.size())
	//				{
	//					min = (j + 1) % contour.size();
	//					max = j;
	//				}
	//			}
	//			contour1.push_back(segment_pairs[s][1]);
	//		}
	//	}
	//	else if (v3.x == 0 && v3.y == 0)
	//	{
	//		if ((v1.x*v.y - v1.y*v.x)*(v2.x*v.y - v2.y*v.x) < 0)
	//		{
	//			contour1.push_back(segment_pairs[s][0]);
	//			contour1.push_back(contour[t]);
	//			bound.push_back(t);
	//			int j = t;
	//			int max = j;
	//			int min = (j + contour.size() - 1) % contour.size();
	//			if (j < (j + contour.size() - 1) % contour.size())
	//			{
	//				max = (j + contour.size() - 1) % contour.size();
	//				min = j;
	//			}
	//			while (!(min == intersect_index[s][2] && max == intersect_index[s][3]))
	//			{
	//				contour1.push_back(contour[(j + contour.size() - 1) % contour.size()]);
	//				bound.push_back((j + contour.size() - 1) % contour.size());
	//				j = (j + contour.size() - 1) % contour.size();
	//				max = j;
	//				min = (j + contour.size() - 1) % contour.size();
	//				if (j < (j + contour.size() - 1) % contour.size())
	//				{
	//					max = (j + contour.size() - 1) % contour.size();
	//					min = j;
	//				}
	//			}
	//			contour1.push_back(segment_pairs[s][1]);
	//		}
	//		else
	//		{
	//			contour1.push_back(contour[intersect_index[s][1]]);
	//			int j = intersect_index[s][1];
	//			int min = j;
	//			int max = (j + 1) % contour.size();
	//			if (j > (j + 1) % contour.size())
	//			{
	//				min = (j + 1) % contour.size();
	//				max = j;
	//			}
	//			while (!(min == intersect_index[s][2] && max == intersect_index[s][3]))
	//			{
	//				contour1.push_back(contour[(j + 1) % contour.size()]);
	//				bound.push_back((j + 1) % contour.size());
	//				j = (j + 1) % contour.size();
	//				max = (j + 1) % contour.size();
	//				min = j;
	//				if (j > (j + 1) % contour.size())
	//				{
	//					min = (j + 1) % contour.size();
	//					max = j;
	//				}
	//			}
	//			contour1.push_back(segment_pairs[s][1]);
	//		}
	//	}
	//		
	//	
	//	//分割线将多边形分割为两部分，不包含中心部分的轮廓点在contour1中
	//	//下面是为了检测计算出的contour1是否正确，经过显示可知是正确的
	//	/*line(dstImage, segment_pairs[s][0], segment_pairs[s][1], Scalar(255, 255, 0), 2, CV_AA, 0);
	//	for (int b = 0; b < contour1.size(); b++)
	//		circle(dstImage, contour1[b], 5, Scalar(255, 255, 0), -1);
	//	imshow("Segmentation", dstImage);
	//	waitKey(0);*/
	//	vector<int> temp_index;
	//	for (int g = 0; g <  points_set.size(); g++)//这里不能用segment_pairs来算和contour1是否交叉，而应该用points_set来算！！否则不准确！！
	//	{
	//		if (g == s)
	//		{
	//			continue;
	//		}
	//		else
	//		{
	//			if (is_cross(contour1 , points_set[g]))
	//			{
	//				//line(dstImage, segment_pairs[g][0], segment_pairs[g][1], Scalar(0, 0, 255), 2, CV_AA, 0);
	//				temp_index.push_back(g);//temp_index表示当前分割线所形成的的轮廓中所包含的分割线的index
	//			}
	//		
	//		}
	//	}
	//	/*imshow("Segmentation", dstImage);
	//	waitKey(0);*/
	//	boundary.push_back(bound);
	//	partition.push_back(temp_index);
	//}



	//bool* is_visit = new bool[segment_pairs.size()];
	//for (int j = 0; j < segment_pairs.size(); j++)
	//{
	//	is_visit[j] = false;
	//}




	//vector<Vec3d> contour3D;
	//for (int j = 0; j < contour.size(); j++)//将contour中坐标转换为三维坐标，存储在contour3D中
	//{
	//	Vec3d vec;
	//	vec[0] = contour[j].x;
	//	vec[1] = contour[j].y;
	//	vec[2] = 0;
	//	contour3D.push_back(vec);
	//}





	//vector<vector<Vec3d>> segment_pairs_3D;

	//for (int j = 0; j < segment_pairs.size(); j++)
	//{
	//	vector<Vec3d> temp;
	//	Vec3d val;
	//	val[0] = segment_pairs[j][0].x;
	//	val[1] = segment_pairs[j][0].y;
	//	val[2] = 0;
	//	temp.push_back(val);
	//	val[0] = segment_pairs[j][1].x;
	//	val[1] = segment_pairs[j][1].y;
	//	val[2] = 0;
	//	temp.push_back(val);
	//	segment_pairs_3D.push_back(temp);
	//}

	////主要用到这些数组变量segment_pairs_3D contour3D partition boundary  is_visit
	//for (int s = 0; s < segment_pairs_3D.size(); s++)
	//{
	//	int max = INT_MAX;
	//	int cur = 0;
	//	for (int j = 0; j < partition.size(); j++)
	//	{
	//		if (partition[j].size() > max && !is_visit[j])
	//		{
	//			max = partition[j].size();
	//			cur = j;//cur表示当前拥有分割线数量最多的分割线的index
	//		}
	//	}
	//	is_visit[cur] = true;
	//	Vec3d axis;
	//	axis[0] = segment_pairs_3D[cur][1][0] - segment_pairs_3D[cur][0][0];
	//	axis[1] = segment_pairs_3D[cur][1][1] - segment_pairs_3D[cur][0][1];
	//	axis[2] = segment_pairs_3D[cur][1][2] - segment_pairs_3D[cur][0][2];
	//	for (int a = 0; a < boundary[cur].size(); a++)
	//	{
	//		int index = boundary[cur][a];
	//		Vec3d new_cor = rotating(contour3D[index], axis, angle1);
	//		contour3D[index] = new_cor;
	//	}

	//	for (int b = 0; b < partition[cur].size(); b++)
	//	{
	//		vector<Vec3d> new_segment;
	//		int index = partition[cur][b];
	//		Vec3d f1 = segment_pairs_3D[index][0];
	//		Vec3d f2 = segment_pairs_3D[index][1];
	//		Vec3d new_f1= rotating(f1, axis, angle1);
	//		Vec3d new_f2= rotating(f2, axis, angle1);
	//		segment_pairs_3D[index][0] = new_f1;
	//		segment_pairs_3D[index][1] = new_f2;
	//	}
	//}









	//下面是参考论文自己写的三角剖分的代码，因为有bug，所以不用了
	/*Triangle tr;
	vector<Vec3i> tri;
	tr.TriSubDiv(contour, dstImage1, tri);*/
	//paint_voronoi(del.div, tri_Image);//这里画的是voronoi面
	CTriangle cdt;
	CDT tri0 = cdt.insert_edge(contour);
	/*CGAL::make_conforming_Delaunay_2(tri0);
	CGAL::make_conforming_Gabriel_2(tri0);*/
	//CGAL::refine_Delaunay_mesh_2(tri0, Criteria(0.125, 0.5));
	//CDT tri1 = cdt.insert_point(tri0, dstImage);
	Mesher mesher(tri0);
	mesher.refine_mesh();
	// 0.125 is the default shape bound. It corresponds to abound 20.6 degree.
	// 20 is the upper bound on the length of the longest edge.
	// See reference manual for Delaunay_mesh_size_traits_2<K>.
	mesher.set_criteria(Criteria(0.125, 20));
	mesher.refine_mesh();

	//output

	CDT::Finite_faces_iterator f_iter;
	CDT::Finite_vertices_iterator v_iter;
	vector<CdtPoint> vertice_index;
	vector<vector<int>> face_index;
	int sum = 0;
	for (f_iter = tri0.finite_faces_begin(); f_iter != tri0.finite_faces_end(); f_iter++)
	{
		if(line_is_in_contour(f_iter->vertex(0)->point(), f_iter->vertex(1)->point(),contour)
		&& line_is_in_contour(f_iter->vertex(0)->point(), f_iter->vertex(2)->point(), contour)
		&& line_is_in_contour(f_iter->vertex(1)->point(), f_iter->vertex(2)->point(), contour)){
			vector<int> temp;
			for (int i = 0; i < 3; i++)
			{
				CdtPoint p = f_iter->vertex(i)->point();//顶点本身就是按照逆时针排序的
				vector<CdtPoint>::iterator it;
				it = find(vertice_index.begin(), vertice_index.end(), p);
				if (it == vertice_index.end()) {
					vertice_index.push_back(p);
					temp.push_back(sum);
					sum++;
				}
				else
				{
					int nposition = distance(vertice_index.begin(), it);
					temp.push_back(nposition);
				}
			}
			face_index.push_back(temp);
		}
		else//遍历所有三角形，剔除不在多边形内部的三角形，将每个三角形的顶点逆时针保存
		{
			for (int i = 0; i < 3; i++)
			{
				CdtPoint p = f_iter->vertex(i)->point();
				vector<CdtPoint>::iterator it;
				it = find(vertice_index.begin(), vertice_index.end(), p);
				if (it == vertice_index.end()) {
					vertice_index.push_back(p);
					sum++;
				}
			}
		}	
	}
	/*CDT::Finite_edges_iterator e_iter;
	for (e_iter=tri0.finite_edges_begin();e_iter!=tri0.finite_edges_end();e_iter++)
	{
			Vertex_handle f_v1 = e_iter->first->vertex(tri0.cw(e_iter->second));
			Vertex_handle f_v2 = e_iter->first->vertex(tri0.ccw(e_iter->second));
			CdtPoint p1 = f_v1->point();
			CdtPoint p2 = f_v2->point();
			cout << p1.x() << " " << p1.y() << endl;
			cout << p2.x() << " " << p2.y() << endl;
			line(dstImage, Point2f(p1.x(), p1.y()), Point2f(p2.x(), p2.y()), (0, 0, 255));
	}*/
	
	for (int j = 0; j < face_index.size(); j++)
	{
		Point2f a, b, c;
		a.x = vertice_index[face_index[j][0]].x();
		a.y= vertice_index[face_index[j][0]].y();
		b.x = vertice_index[face_index[j][1]].x();
		b.y = vertice_index[face_index[j][1]].y();
		c.x = vertice_index[face_index[j][2]].x();
		c.y = vertice_index[face_index[j][2]].y();
		circle(dstImage, a, 2, Scalar(0, 0, 255));
		circle(dstImage, b, 2, Scalar(0, 0, 255));
		circle(dstImage, c, 2, Scalar(0, 0, 255));
		line(dstImage, a, b, Scalar(0, 255, 0));
		line(dstImage, a, c, Scalar(0, 255, 0));
		line(dstImage, c, b, Scalar(0, 255, 0));

	}
	imshow("Segmentation", dstImage);
	cv::waitKey(0);
	return 0;
}
