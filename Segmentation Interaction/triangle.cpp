#include <iostream>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <queue>
#include <random>
#include <math.h>
#include <float.h>
#include "triangle.h"
#define TRIANGLE 50
#define POINTSIZE 200
const double eps = 1e-10;

//三角化中插入的点的数量
using namespace std;
using namespace cv;

bool cmp(CvPoint2D32f a, CvPoint2D32f b)
{
	return a.x > b.x;
}
bool unique_finder(CvPoint2D32f a, CvPoint2D32f b)//这里一定要重新定义unique函数中的bool函数，因为CvPoint2D32f属于自定义类型，所以不能重载==操作符
{
	return a.x == b.x&&a.y == b.y;
}
bool isGoodTri(Vec3i &v, vector<Vec3i> & tri)
{
	int a = v[0], b = v[1], c = v[2];
	v[0] = min(a, min(b, c));//v[0]找到点插入的先后顺序（0....N-1，N为点的个数）的最小值
	v[2] = max(a, max(b, c));//v[2]存储最大值.
	v[1] = a + b + c - v[0] - v[2];//v[1]为中间值
	if (v[0] == -1) return false;//说明该顶点的id有为-1的，这种顶点就是虚拟三角形的顶点

	vector<Vec3i>::iterator iter = tri.begin();//开始时为空
	for (; iter != tri.end(); iter++)
	{
		Vec3i &check = *iter;//如果当前待压入的和存储的重复了，则停止返回false。
		if (check[0] == v[0] &&
			check[1] == v[1] &&
			check[2] == v[2])
		{
			break;
		}
	}
	if (iter == tri.end())
	{
		tri.push_back(v);
		return true;
	}
	return false;
}
bool is_cross(Point2f l1_1, Point2f l1_2, Point2f l2_1, Point2f l2_2)
{
	if (std::min(l1_1.x, l1_2.x) <= std::max(l2_1.x, l2_2.x) && std::max(l1_1.x, l1_2.x) >= std::min(l2_1.x, l2_2.x)
		&& std::min(l1_1.y, l1_2.y) <= std::max(l2_1.y, l2_2.y) && std::max(l1_1.y, l1_2.y) >= std::min(l2_1.y, l2_2.y))
		//通过快速排斥实验

	{
		Point2f left21, l22, right21;
		left21.x = l1_1.x - l2_2.x;
		left21.y = l1_1.y - l2_2.y;
		right21.x = l1_2.x - l2_2.x;
		right21.y = l1_2.y - l2_2.y;
		l22.x = l2_1.x - l2_2.x;
		l22.y = l2_1.y - l2_2.y;
		//if((left21.x*l22.y-l22.x*left21.y)*(right21.x*l22.y-l22.x*right21.y)<0)
		if (((l1_1.x - l2_2.x)*(l2_1.y - l2_2.y) - (l2_1.x - l2_2.x)*(l1_1.y - l2_2.y))*((l1_2.x - l2_2.x)*(l2_1.y - l2_2.y)-(l2_1.x - l2_2.x)*(l1_2.y - l2_2.y)) < 0)
		{
			Point2f left12, l11, right12;
			left12.x = l2_1.x - l1_1.x;
			left12.y = l2_1.y - l1_1.y;
			right12.x = l2_2.x - l1_1.x;
			right12.y = l2_2.y - l1_1.y;
			l11.x = l1_2.x - l1_1.x;
			l11.y = l1_2.y - l1_1.y;
			//if ((right12.x*l11.y - l11.x*right12.y)*(left12.x*l11.y - l11.x*left12.y) <0)
			if (((l2_2.x - l1_1.x)*(l1_2.y - l1_1.y) - (l1_2.x - l1_1.x)*(l2_2.y - l1_1.y))*((l2_1.x - l1_1.x)*(l1_2.y - l1_1.y) - (l1_2.x - l1_1.x)*(l2_1.y - l1_1.y)) <0)
				return true;
			else
				return false;
		}
		else
			return false;
	}
	else
	{
		return false;
	}
}
bool is_onedge(Point2f pt,Point2f l1,Point2f l2)//判断点pt是否在线段l1-l2上
{
	Point2f pl1, pl2;
	pl1.x = l1.x - pt.x;
	pl1.y = l1.y - pt.y;
	pl2.x = l2.x - pt.x;
	pl2.y = l2.y - pt.y;
	if (fabs(pl1.x*pl2.y - pl1.y*pl2.x) < eps)
	{
		if (std::min(l1.x, l2.x) - eps <= pt.x&&pt.x - eps <= std::max(l1.x, l2.x))
		{
			if (std::min(l1.y, l2.y) - eps <= pt.y&&pt.y - eps <= std::max(l1.y, l2.y))
				return true;
		}
	}
	return false;
}
bool find(queue<vector<int>> q, vector<int> n)//队列的查找函数
{
	//需要注意的是函数传递参数的时候不能传递引用，否则最后pop的时候会改变原始队列的大小，使之为空
	int size = q.size();
	for (int i = 0; i < size; i++)
	{
		vector<int> temp = q.front();
		if (temp.size() != n.size())
			continue;
		else
		{
			vector<int>:: iterator iter1, iter2;
			iter1 = temp.begin();
			iter2 = n.begin();
			while (iter1 != temp.end() && iter2 != n.end())
			{
				if (*iter1 == *iter2)
				{
					iter1++;
					iter2++;
				}
				else
					break;
			}
			if (iter1 == temp.end() && iter2 == n.end())
				return true;
		}
		q.pop();
	}
	return false;
}
int point_compare(Point A, Point B)
{
	if (A.x == B.x&&A.y == B.y)
		return 0;
	else if (A.x < B.x || (A.x == B.x&&A.y < B.y))
		return 1;
	else
		return -1;
}
//下面是判断线段是否在多边形内部
bool in_Polygon(vector<Vec2i> constrained_edge, vector<CvPoint2D32f> points, int src, int dst)
{
	//基于pointPolygonTest函数的判定方法不够准确,此函数是判断以src和dst为端点的线段是否在以constrained_edge为约束边的多边形内部
	//https://www.zhihu.com/question/62749769按照这个思路来判定
	//由于在生成三角网格时对于和约束边相交的三角网格的边进行了处理，所以这里不需要考虑和约束边相交的情况
	int size = constrained_edge.size();
	int min = src;
	int max = dst;
	if (src > dst)
	{
		min = dst;
		max = src;
	}
	for (int i = 0; i < size; i++)//判断线段是否与多边形的边重合
	{
		Point2f s1, s2;
		s1.x = points[min].x - points[max].x;
		s1.y = points[min].y - points[max].y;
		s2.x = points[constrained_edge[i][0]].x - points[constrained_edge[i][1]].x;
		s2.y = points[constrained_edge[i][0]].y - points[constrained_edge[i][1]].y;
		if (abs(s1.x*s2.y - s2.x*s1.y) == 0)
		{
			Point2f s3;
			s3.x = points[min].x - points[constrained_edge[i][1]].x;
			s3.y = points[min].y - points[constrained_edge[i][1]].y;
			if (abs(s3.x*s2.y - s2.x*s3.y) == 0)
				return true;//说明两条线重合
		}
	}
	//取线段中点，并作射线进行判定,如果该中点在多边形内部，则线段在多边形内部；否则在多边形外部
	Point2f mid;
	mid.x = (points[min].x + points[max].x) / 2;
	mid.y= (points[min].y + points[max].y) / 2;
	//constrained_edge本身就是逆时针排序了
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		if ((points[constrained_edge[i][0]].y <= mid.y&&mid.y < points[constrained_edge[i][1]].y) ||
			(points[constrained_edge[i][1]].y <= mid.y&&mid.y < points[constrained_edge[i][0]].y))
		{
			double d = ((mid.y - points[constrained_edge[i][0]].y)*(points[constrained_edge[i][1]].x - points[constrained_edge[i][0]].x)
				+ (points[constrained_edge[i][0]].x-mid.x)*(points[constrained_edge[i][1]].y - points[constrained_edge[i][0]].y))
				* (points[constrained_edge[i][1]].y - points[constrained_edge[i][0]].y);
			if (d > 0)
				count++;
		}
	}
	if (count % 2 != 0)
		return true;

	return false;
}
//下面是判断点是否在多边形内部
bool point_in_polygon(vector<Vec2i> constrained_edge, vector<CvPoint2D32f>points, int point)
{
	int count = 0;
	for (int i = 0; i < constrained_edge.size(); i++)
	{
		if (points[point].x == points[constrained_edge[i][0]].x&&points[point].y == points[constrained_edge[i][0]].y)
			return true;
		else if (points[point].y == points[constrained_edge[i][0]].y&&points[point].y == points[constrained_edge[i][1]].y)
		{
			if ((points[constrained_edge[i][0]].x <= points[point].x&&points[point].x <= points[constrained_edge[i][1]].x) ||
				(points[constrained_edge[i][1]].x <= points[point].x&&points[point].x <= points[constrained_edge[i][0]].x))
			{
				return true;
			}
		}
		else if((points[constrained_edge[i][0]].y <= points[point].y&&points[point].y < points[constrained_edge[i][1]].y) ||
			(points[constrained_edge[i][1]].y <= points[point].y&&points[point].y < points[constrained_edge[i][0]].y))
		{
			double d = ((points[point].y - points[constrained_edge[i][0]].y)*(points[constrained_edge[i][1]].x - points[constrained_edge[i][0]].x)
				+ (points[constrained_edge[i][0]].x - points[point].x)*(points[constrained_edge[i][1]].y - points[constrained_edge[i][0]].y))
				* (points[constrained_edge[i][1]].y - points[constrained_edge[i][0]].y);
			if (d > 0)
				count++;
			else if (d == 0)
				return true;
		}
	}
	if (count % 2 != 0)
		return true;
	return false;
}

static void draw_subdiv_edge(Mat img, CvSubdiv2DEdge edge, CvScalar color, vector<Point> contour)//去除位于多边形外的三角形
{
	CvSubdiv2DPoint* org_pt;
	CvSubdiv2DPoint* dst_pt;
	CvPoint2D32f org;
	CvPoint2D32f dst;
	CvPoint iorg, idst;
	org_pt = cvSubdiv2DEdgeOrg(edge);
	dst_pt = cvSubdiv2DEdgeDst(edge);
	if (org_pt&&dst_pt)
	{
		org = org_pt->pt;
		dst = dst_pt->pt;
		iorg = cvPoint(cvRound(org.x), cvRound(org.y));
		idst = cvPoint(cvRound(dst.x), cvRound(dst.y));
		IplImage tmp = IplImage(img);
		CvArr* arr = (CvArr*)&tmp;
		/*LineIterator iter(img,iorg,idst,8);*/
		//bool draw = true;
		//for (int i = 0; i < iter.count; i++, iter++)//遍历直线上的点，当直线上的点都在轮廓内，也就是直线在轮廓内部的时候，画出直线，否则不画
		//{
		//	if (pointPolygonTest(contour, iter.pos(), true) >= 0)//
		//		continue;
		//	else
		//	{
		//		draw = false;
		//		break;
		//	}

		//}
		//if(draw==true)
		cvLine(arr, iorg, idst, color, 1, CV_AA, 0);
	}
}
static void draw_subdiv(Mat img, CvSubdiv2D* subdiv, CvScalar delaunay_color, vector<Point> contour)//画出三角剖分
{
	CvSeqReader reader;
	int i, total = subdiv->edges->total;
	int elem_size = subdiv->edges->elem_size;
	cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0);
	for (i = 0; i < total; i++)
	{
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);
		if (CV_IS_SET_ELEM(edge))
		{

			draw_subdiv_edge(img, (CvSubdiv2DEdge)edge, delaunay_color, contour);
		}
		CV_NEXT_SEQ_ELEM(elem_size, reader);
	}
}

Delaunay Triangle::delaunay_triangle(vector<Point> contour,Mat img)
{
	vector<Vec2i> constrained_edge;//存放的是约束边的顶点编号
	Vec2i vec;
	int count = 0;
	//下面是多边形三角化代码
	vector<CvPoint2D32f> points;//插入的所有点
	//Rect rect = boundingRect(contour);
	Rect rect= Rect(0, 0, img.cols, img.rows);
	CvMemStorage* storage;
	CvSubdiv2D* subdiv;
	storage = cvCreateMemStorage(0);
	subdiv = init_delaunay(storage, rect);
	int contour_size = contour.size();
	int j;
	for (j = 0; j < contour_size; j++)
	{
		CvPoint2D32f fp0 = cvPoint2D32f(contour[j].x, contour[j].y);
		CvSubdiv2DPoint * pt = cvSubdivDelaunay2DInsert(subdiv, fp0);
		pt->id = j;//为插入的顶点编号
		points.push_back(fp0);
		if (j != 0)
		{
			vec[0] = j - 1;
			vec[1] = j;
			constrained_edge.push_back(vec);
		}

	}
	if (j == contour_size)
	{
		vec[0] = 0;
		vec[1] = j-1;
		constrained_edge.push_back(vec);
	} 
	/*vector<Point2f>  points0 = Poisson_disc(contour, 20, 30);
	int point_size = points0.size();
	for (int i = 0; i < point_size; i++)
	{
		CvPoint2D32f fp = cvPoint2D32f(points0[i].x, points0[i].y);
		CvSubdiv2DPoint * pt = cvSubdivDelaunay2DInsert(subdiv, fp);
		pt->id = contour_size + i;
		points.push_back(fp);
	}*/
	
	for (int i = 0; i < POINTSIZE; i++)
	{
		CvPoint2D32f fp = cvPoint2D32f((float)(rand() % (img.cols - 10)),//使点约束在距离边框10像素之内。因为之前生成的矩形为Rect(0, 0, img.cols, img.rows)    
			(float)(rand() % (img.rows - 10)));
		CvSubdiv2DPoint * pt = cvSubdivDelaunay2DInsert(subdiv, fp);
		pt->id = contour_size + i;
		points.push_back(fp);
	}



	//sort(points.begin(), points.end(), cmp);
	//points.erase(unique(points.begin(),points.end(), unique_finder),points.end());//去重
	//draw_subdiv(dstImage, subdiv, Scalar(0, 200, 0), contour);
	Delaunay del;
	del.point = points;
	del.div = subdiv;
	del.constrained_edge = constrained_edge;
	return del;//返回细分
}//这个画的是voronoi图，不是三角剖分！！！


CvSubdiv2D* Triangle::init_delaunay(CvMemStorage* storage, Rect rect)
{
	CvSubdiv2D* subdiv;
	subdiv = cvCreateSubdiv2D(CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv), sizeof(CvSubdiv2DPoint), sizeof(CvQuadEdge2D), storage);
	cvInitSubdivDelaunay2D(subdiv, rect);
	return subdiv;
}

double Triangle::distance(double x1, double y1, double x2, double y2) {
	double sum = pow(x2 - x1, 2) + pow(y2 - y1, 2);
	return sqrt(sum);
}



Circle Triangle::circle_generate(vector<Point> contour) {
	//Rect rect = boundingRect(contour);
	//rectangle(dstImage, rect, (0, 0, 255));
	//RotatedRect rect = minAreaRect(contour);//最小外接矩形
	//Point2f P[4];
	//rect.points(P);
	//for (int j = 0; j <= 3; j++) {
	//	line(dstImage, P[j], P[(j + 1) % 4], Scalar(255), 1, LINE_AA);//LINE_AA是为了抗锯齿
	//	cout << P[j] << endl;
	//}
	Point2f center;
	float radius;
	minEnclosingCircle(contour, center, radius);
	/*circle(dstImage, center, radius, Scalar(255),1, LINE_AA);
	namedWindow("circle", WINDOW_NORMAL);
	imshow("circle", dstImage);
	waitKey(0);*/
	Circle c;
	c.center = center;
	c.rad = radius;
	return c;
}


vector<Point2f> Triangle::Poisson_disc(vector<Point> contour, double r, int k) {
	Rect rect = boundingRect(contour);
	//rectangle(dstImage, rect, (0, 0, 255));//绘制外接矩形
	Point2f topl = rect.tl();
	Point2f bottomr = rect.br();
	double dis = r / sqrt(2);//绘制网格时每个单元的边长
	int row = floor((bottomr.y - topl.y) / dis) + 1;
	int column = floor((bottomr.x - topl.x) / dis) + 1;
	int i, j;

	//下面是Poisson——disc算法实现
	Point2f  **grid = new Point2f*[row];
	for (i = 0; i < row; i++)
		grid[i] = new Point2f[column];
	int** flag = new int*[row];
	for (i = 0; i < row; i++)
		flag[i] = new int[column];
	for (i = 0; i < row; i++)
		for (j = 0; j < column; j++)
			flag[i][j] = 0;
	vector<Coordinate> active;
	vector<Point2f> sample;

	Circle o = circle_generate(contour);
	random_device rd;//用来获得随机数引擎的种子
	mt19937 gen(rd());//Standard mersenne_twister_engine seeded with rd()
	uniform_real_distribution<double> theta(0.0, 360.0);
	//uniform_real_distribution<double> rad(0.0,nextafter(c.radius,DBL_MAX));//nextafter(c.radius,DBL_MAX)返回的是第一个参数和第二个参数之间与第一个参数相邻的浮点数
	uniform_real_distribution<double> r0(0, nextafter(1, DBL_MAX));


	Coordinate coor;

	while (1)
	{
		double th = theta(gen);
		double r1 = sqrt(r0(gen))*o.rad;
		double x = o.center.x + r1*sin(th);
		double y = o.center.y + r1*cos(th);
		if (pointPolygonTest(contour, Point2f(x, y), true) > 0)
		{
			//circle(dstImage, Point2f(x, y), 1, Scalar(120, 120, 120), -1);//之后会一起画
			int cy = floor((x - topl.x) / dis);
			int cx = floor((y - topl.y) / dis);
			coor.x_axis = cx;
			coor.y_axis = cy;
			grid[cx][cy] = Point2f(x, y);
			flag[cx][cy] = 1;
			active.push_back(coor);//放的是grid里面的数组index
			sample.push_back(Point2f(x, y));//放的是grid里面的数组值
			break;
		}
		else
			continue;
	}

	while (!active.empty()) {
		int index = r0(gen)*(active.size() - 1);
		bool activeflag = false;
		for (i = 0; i < k; i++) {
			double R = r + r*r0(gen);
			double De = theta(gen);
			Point2f temp = grid[active[index].x_axis][active[index].y_axis];
			double k_1 = temp.x + R*cos(De);
			double k_2 = temp.y + R*sin(De);
			bool A = false;
			bool B = false;
			bool C = false;
			bool D = false;
			bool E = false;
			bool F = false;
			bool G = false;
			bool H = false;

			if (k_1<topl.x || k_2<topl.y || k_1>bottomr.x || k_2>bottomr.y || pointPolygonTest(contour, Point2f(k_1, k_2), true) <= 0) {
				continue;
			}
			else {
				int tempy = floor((k_1 - topl.x) / dis);
				int tempx = floor((k_2 - topl.y) / dis);
				coor.x_axis = tempx;
				coor.y_axis = tempy;
				if (flag[tempx][tempy] == 1) {
					continue;
				}
				else {
					//判断相邻的八个方格里面有没有点存在，若存在距离是否大于r

					if (tempx - 1 >= 0 && tempy - 1 >= 0) {
						if (flag[tempx - 1][tempy - 1] == 0)
							A = true;
						else {
							Point2f temp1 = grid[tempx - 1][tempy - 1];
							if (distance(k_1, k_2, temp1.x, temp1.y) > r)
								A = true;
						}
					}
					else
						A = true;
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx >= 0 && tempy - 1 >= 0) {
						if (flag[tempx][tempy - 1] == 0)
							B = true;
						else {
							Point2f temp2 = grid[tempx][tempy - 1];
							if (distance(k_1, k_2, temp2.x, temp2.y) > r)
								B = true;
						}
					}
					else
						B = true;
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx + 1 < row && tempy - 1 >= 0) {
						if (flag[tempx + 1][tempy - 1] == 0)
							C = true;
						else {
							Point2f temp3 = grid[tempx + 1][tempy - 1];
							if (distance(k_1, k_2, temp3.x, temp3.y) > r)
								C = true;
						}
					}
					else
						C = true;
					///////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx - 1 >= 0 && tempy >= 0) {
						if (flag[tempx - 1][tempy] == 0)
							D = true;
						else {
							Point2f temp4 = grid[tempx - 1][tempy];
							if (distance(k_1, k_2, temp4.x, temp4.y) > r)
								D = true;
						}
					}
					else
						D = true;
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx + 1 < row  && tempy >= 0) {
						if (flag[tempx + 1][tempy] == 0)
							E = true;
						else {
							Point2f temp5 = grid[tempx + 1][tempy];
							if (distance(k_1, k_2, temp5.x, temp5.y) > r)
								E = true;
						}
					}
					else
						E = true;
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx - 1 >= 0 && tempy + 1 < column) {
						if (flag[tempx - 1][tempy + 1] == 0)
							F = true;
						else {
							Point2f temp6 = grid[tempx - 1][tempy + 1];
							if (distance(k_1, k_2, temp6.x, temp6.y) > r)
								F = true;
						}
					}
					else
						F = true;
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx >= 0 && tempy + 1 < column) {
						if (flag[tempx][tempy + 1] == 0)
							G = true;
						else {
							Point2f temp7 = grid[tempx][tempy + 1];
							if (distance(k_1, k_2, temp7.x, temp7.y) > r)
								G = true;
						}
					}
					else
						G = true;
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (tempx + 1 < row && tempy + 1 < column) {
						if (flag[tempx + 1][tempy + 1] == 0)
							H = true;
						else {
							Point2f temp6 = grid[tempx + 1][tempy + 1];
							if (distance(k_1, k_2, temp6.x, temp6.y) > r)
								H = true;
						}
					}
					else
						H = true;

					if (A&&B&&C&&D&&E&&F&&G&&H) {
						grid[tempx][tempy] = Point2f(k_1, k_2);
						flag[tempx][tempy] = 1;
						active.push_back(coor);//放的是grid里面的数组index
						sample.push_back(Point2f(k_1, k_2));//放的是grid里面的数组index
						activeflag = true;
					}

				}
			}
		}
		if (!activeflag)//从active中删除这个点
		{
			vector<Coordinate>::iterator it = active.begin() + index;
			active.erase(it);
		}
	}
	for (i = 0; i < row; i++)
		delete[] grid[i];
	delete[] grid;
	return sample;
}


void Triangle::TriSubDiv(vector<Point> contour, Mat dstImage, vector<Vec3i> &tri)
{
	Mat imgShow=dstImage.clone();
	Delaunay del = delaunay_triangle(contour, imgShow);
	CvSubdiv2D* subdiv = del.div;
	CvSeqReader reader;
	int total = subdiv->edges->total;
	int elem_size = subdiv->edges->elem_size;
	cvStartReadSeq((CvSeq*)(subdiv->edges),&reader,0);
	Point buf[3];
	const Point *pBuf = buf;
	Vec3i verticesIdx;
	//Mat imgShow = dstImage.clone();
	for (int i = 0; i < total; i++)
	{
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

		if (CV_IS_SET_ELEM(edge))
		{
			CvSubdiv2DEdge t = (CvSubdiv2DEdge)edge;
			int iPointNum = 3;
			//Scalar color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);
			Scalar color = CV_RGB(0, 255, 0);
			//Scalar color=CV_RGB(255,0,0);
			//bool isNeg = false;
			int j;
			for (j = 0; j < iPointNum; j++)
			{
				CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(t);//获取t边的源点
				if (!pt) break;
				buf[j] = pt->pt;//将点存储起来
								//if (pt->id == -1) isNeg = true;
				verticesIdx[j] = pt->id;//获取顶点的Id号，将三个点的id存储到verticesIdx中
				t = cvSubdiv2DGetEdge(t, CV_NEXT_AROUND_LEFT);//获取下一条边
			}
			if (j != iPointNum) continue;
			if (isGoodTri(verticesIdx, tri))//tri里面存放的是各个三角面片所包含的顶点的编号
			{
				
				polylines(imgShow, &pBuf, &iPointNum,
					1, true, color,
					1, CV_AA, 0);//画出三条边
								 //printf("(%d, %d)-(%d, %d)-(%d, %d)\n", buf[0].x, buf[0].y, buf[1].x, buf[1].y, buf[2].x, buf[2].y);
								 //printf("%d\t%d\t%d\n", verticesIdx[0], verticesIdx[1], verticesIdx[2]);
								 //imshow("Delaunay", imgShow);
								 //waitKey();
			}

			t = (CvSubdiv2DEdge)edge + 2;//相反边缘 reversed e

			for (j = 0; j < iPointNum; j++)
			{
				CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(t);
				if (!pt) break;
				buf[j] = pt->pt;
				verticesIdx[j] = pt->id;
				t = cvSubdiv2DGetEdge(t, CV_NEXT_AROUND_LEFT);
			}
			if (j != iPointNum) continue;
			if (isGoodTri(verticesIdx, tri))
			{
				
				polylines(imgShow, &pBuf, &iPointNum,
					1, true, color,
					1, CV_AA, 0);
				//printf("(%d, %d)-(%d, %d)-(%d, %d)\n", buf[0].x, buf[0].y, buf[1].x, buf[1].y, buf[2].x, buf[2].y);
				//printf("%d\t%d\t%d\n", verticesIdx[0], verticesIdx[1], verticesIdx[2]);
				//imshow("Delaunay", imgShow);
				//waitKey();
			}

		}

		CV_NEXT_SEQ_ELEM(elem_size, reader);

	}
	vector<vector<int>> vertice_in_triangle;
	vector<int> temp;
	for (int j = 0; j < del.point.size(); j++)
	{
		circle(imgShow, del.point[j], 2.5, Scalar(255, 255, 255),-1);
		vertice_in_triangle.push_back(temp);//数组初始化
	}
    std::cout <<"三角面片的数量为"<<tri.size() << endl;
	cv::imshow("Segmentation", imgShow);
	cv::waitKey();





	

	//下面为<<A FAST ALGORITHM FOR GENERATING CONSTRAINED DELAUNAY TRIANGULATIONS>> 算法实现，主要是去掉contour外部的三角形
	
	for (int j = 0; j < tri.size(); j++)
	{
		//将每个顶点所在的三角形编号放入数组vertice_in_triangle中
		int a = tri[j][0];
		int b = tri[j][1];
		int c = tri[j][2];
		vector<int>::iterator iter0 = find(vertice_in_triangle[a].begin(), vertice_in_triangle[a].end(), j);
		if (iter0 == vertice_in_triangle[a].end())
		{
			vertice_in_triangle[a].push_back(j);
		}
		vector<int>::iterator iter1 = find(vertice_in_triangle[b].begin(), vertice_in_triangle[b].end(), j);
		if (iter1 == vertice_in_triangle[b].end())
		{
			vertice_in_triangle[b].push_back(j);
		}
		vector<int>::iterator iter2 = find(vertice_in_triangle[c].begin(), vertice_in_triangle[c].end(), j);
		if (iter2 == vertice_in_triangle[c].end())
		{
			vertice_in_triangle[c].push_back(j);
		}
	}
	/*====================================下面是测试代码========================================*/
	/*cout << "vertice_in_triangle:" << endl;
	for (int t = 0; t < vertice_in_triangle.size(); t++) {
		for(int t0=0;t0<vertice_in_triangle[t].size();t0++)
		       cout << vertice_in_triangle[t][t0] << " ";
		cout << endl;
	}*/
	/*==============================================================================================*/
	queue<vector<int>> cross_edges;
	vector<vector<int>> new_edges;
	for (int i = 0; i < del.constrained_edge.size(); i++)//constrained_edge中边上点的标号是从小到大排序的
	{
		new_edges.clear();
		queue<vector<int>> empty;
		std::swap(empty, cross_edges);
		//以上是将cross_edges和new_edges队列清空
		int cur_p0 = del.constrained_edge[i][0];
		int cur_p1 = del.constrained_edge[i][1];
		circle(imgShow,del.point[cur_p0],3,Scalar(0,0,200));
		circle(imgShow, del.point[cur_p1], 3, Scalar(0, 0, 200));
		int count = vertice_in_triangle[cur_p0].size();
		bool is_triangle_edge = false;
		for (int j = 0; j < count; j++)
		{
			int index = vertice_in_triangle[cur_p0][j];
			int a = tri[index][0];
			int b = tri[index][1];
			int c = tri[index][2];
			if (cur_p1 == a || cur_p1 == b || cur_p1 == c)
			{
				is_triangle_edge = true;
				break;
			}

		}

		if (is_triangle_edge)
			continue;
		else//说明当前约束边不是三角网格的边
		{
			//寻找以cur_p0为顶点的三角形中，和当前约束边相交的边
			//约束边上除了两个端点，不可能有三角形顶点！！！
			for (int k = 0; k < vertice_in_triangle[cur_p0].size(); k++)
			{
				int tri_index = vertice_in_triangle[cur_p0][k];
				int s1 = tri[tri_index][0];
				int s2 = tri[tri_index][1];
				int s3 = tri[tri_index][2];
				if (is_cross(del.point[s1], del.point[s2], del.point[cur_p0], del.point[cur_p1]))
				{
					vector<int> temp;
					int min = s1;
					int max = s2;
					if (s2 < s1)
					{
						min = s2;
						max = s1;
					}
					temp.push_back(min);
					temp.push_back(max);
					cross_edges.push(temp);
					line(imgShow, del.point[min], del.point[max], Scalar(0, 0, 255), 4);
					break;
				}
				else if (is_cross(del.point[s1], del.point[s3], del.point[cur_p0], del.point[cur_p1]))
				{
					vector<int> temp;
					int min = s1;
					int max = s3;
					if (s3 < s1)
					{
						min = s3;
						max = s1;
					}
					temp.push_back(min);
					temp.push_back(max);
					cross_edges.push(temp);
					line(imgShow, del.point[min], del.point[max], Scalar(0, 0, 255), 4);
					break;
				}
				else if (is_cross(del.point[s2], del.point[s3], del.point[cur_p0], del.point[cur_p1]))
				{
					vector<int> temp;
					int min = s3;
					int max = s2;
					if (s2 < s3)
					{
						min = s2;
						max = s3;
					}
					temp.push_back(min);
					temp.push_back(max);
					cross_edges.push(temp);
					line(imgShow, del.point[min], del.point[max], Scalar(0, 0, 255), 4);
					break;
				}
				
				else
					continue;
			}
			imshow("Segmentation", imgShow);
			waitKey();



			if (cross_edges.empty())
				cerr << "错误，由于该约束边不是三角网格的边，所以一定能找到相交的边或者顶点在约束边上的三角形" << endl;
		

			bool end = false;
			while (!cross_edges.empty())
			{
				if (!pt.empty())
				{
					int count = vertice_in_triangle[pt[0]].size();
					for (int n = 0; n < count; n++)
					{

					}
				}
				vector<int> back = cross_edges.back();//队列中最晚进去的元素
				int count = vertice_in_triangle[back[0]].size();
				for (int m = 0; m < count; m++)
				{
					int tri_index = vertice_in_triangle[back[0]][m];
					int p1 = tri[tri_index][0];
					int p2 = tri[tri_index][1];
					int p3 = tri[tri_index][2];

					if (back[1] == p1 || back[1] == p2 || back[1] == p3)
					{
						if (is_cross(del.point[p1], del.point[p2], del.point[cur_p0], del.point[cur_p1]))
						{
							vector<int>temp;
							int min = p1;
							int max = p2;
							if (p1 > p2)
							{
								min = p2;
								max = p1;
							}
							temp.push_back(min);
							temp.push_back(max);
							if (!find(cross_edges, temp)) {
								cross_edges.push(temp);
							}
						}
						if (is_cross(del.point[p1], del.point[p3], del.point[cur_p0], del.point[cur_p1]))
						{
							vector<int>temp;
							int min = p1;
							int max = p3;
							if (p1 > p3)
							{
								min = p3;
								max = p1;
							}
							temp.push_back(min);
							temp.push_back(max);
							if (!find(cross_edges, temp)) {
								cross_edges.push(temp);
							}
						}
						if (is_cross(del.point[p2], del.point[p3], del.point[cur_p0], del.point[cur_p1]))
						{
							vector<int>temp;
							int min = p2;
							int max = p3;
							if (p2 > p3)
							{
								min = p3;
								max = p2;
							}
							temp.push_back(min);
							temp.push_back(max);
							if (!find(cross_edges, temp)) {
								cross_edges.push(temp);
							}
						}
							if (p1 == cur_p1 || p2 == cur_p1 || p3 == cur_p1)
							{
								end = true;
								break;
							}
					}
				}
					if (end)
						break;
					else
						continue;
			}
		}
			/*====================================下面是测试代码========================================*/
			/*cout << "cross_edges:" << cross_edges.size() << endl;
			int ddd = cross_edges.size();
			for (int t = 0; t < ddd; t++) {
				vector<int> temp = cross_edges.front();
				cross_edges.pop();
				for (int t0 = 0; t0 < temp.size(); t0++)
					cout << temp[t0] <<" ";
				cout << endl;
			}*/
			/*==============================================================================================*/
			/************************************************************************************************\
			*                     以上代码求出了和当前约束边相交的三角网格的边的集合cross_edges，并用红线画出               *
			\************************************************************************************************/

			while (!cross_edges.empty())
			{
				vector<int> cur_cross;
				cur_cross = cross_edges.front();
				cross_edges.pop();
				int first = cur_cross[0];
				int second = cur_cross[1];
				int third = INT_MAX;
				int fourth = INT_MAX;
				int index0, index1;//分别表示first second third所在三角形标号以及first second fourth所在三角形标号
				bool is_convex = true;
				for (int t = 0; t < vertice_in_triangle[first].size(); t++)
				{
					int index = vertice_in_triangle[first][t];
					if (tri[index][0] == second || tri[index][1] == second || tri[index][2] == second)
					{
						if (third == INT_MAX) {
							third = (tri[index][0] + tri[index][1] + tri[index][2] - first - second);
							index0 = index;
						}
						else
						{
							fourth = (tri[index][0] + tri[index][1] + tri[index][2] - first - second);
							index1 = index;
						}
					}

				}
				if (third != INT_MAX&&fourth != INT_MAX)//找到共享当前交叉边的两个三角形的其他两个顶点了
				{
					//以下代码是将两个三角形组成的多边形顶点按照逆时针顺序放进convex中
					vector<int> convex;
					if ((del.point[third].x - del.point[first].x)*(del.point[fourth].y - del.point[first].y) -
						(del.point[third].y - del.point[first].y)*(del.point[fourth].x - del.point[first].x) < 0)
					{
						convex.push_back(first);
						convex.push_back(fourth);
						convex.push_back(second);
						convex.push_back(third);
					}
					else
					{
						convex.push_back(first);
						convex.push_back(third);
						convex.push_back(second);
						convex.push_back(fourth);
					}
					//以下代码是判断多边形是否为严格的凸多边形
					//判定条件1 单调跳变的次数不大于2

					int nums = 0;
					int s = point_compare(del.point[convex[3]], del.point[convex[0]]);
					for (int r = 0; r < 3; r++)
					{
						int val = point_compare(del.point[convex[r]], del.point[convex[r + 1]]);
						if (val == 0)
						{
							//非凸
							is_convex = false;
						}
						else if (val != s)
						{
							nums++;
							s = val;
							if (nums > 2)
							{
								//非凸
								is_convex = false;
							}
						}
					}
					//判定条件2 相邻三个顶点的夹角不大于180度
					if (is_convex) {
						bool isonline = true;
						for (int r = 0; r < 4; r++)
						{
							Point v1 = del.point[convex[r]];
							Point v0, v2;
							if (r == 0)
								v0 = del.point[convex[3]];
							else
								v0 = del.point[convex[r - 1]];
							if (r == 3)
								v2 = del.point[convex[0]];
							else
								v2 = del.point[convex[r + 1]];
							double angle = (v2.x - v1.x)*(v0.y - v1.y) - (v2.y - v1.y)*(v0.x - v1.x);
							if ((v2.x - v1.x)*(v0.y - v1.y) - (v2.y - v1.y)*(v0.x - v1.x) < 0)//说明夹角大于180度
							{
								/*int iPointNum = 3;
								Point pBuf11[3];
								const Point* Buf11 = pBuf11;
								pBuf11[0] = del.point[convex[0]];
								pBuf11[1] = del.point[convex[3]];
								pBuf11[2] = del.point[convex[2]];
								polylines(imgShow, &Buf11, &iPointNum,
									1, true, Scalar(0,255,0),
									1, CV_AA, 0);
								pBuf11[0] = del.point[convex[0]];
								pBuf11[1] = del.point[convex[1]];
								pBuf11[2] = del.point[convex[2]];
								polylines(imgShow, &Buf11, &iPointNum,
									1, true, Scalar(0, 255, 0),
									1, CV_AA, 0);
								circle(imgShow, del.point[convex[0]], 3, Scalar(255, 255, 255), -1);
								circle(imgShow, del.point[convex[2]],3, Scalar(255, 255, 255), -1);
								cv::imshow("Segmentation",imgShow);
								cv::waitKey();*/
								is_convex = false;
							}
							else if (isonline&&angle > 0)
							{
								isonline = false;
							}
						}
						if (isonline)
						{
							is_convex = false;
						}
					}
				}

				if (!is_convex || third == INT_MAX || fourth == INT_MAX) {
					cross_edges.push(cur_cross);
					continue;
				}
				else
				{
					//修改tri表和vertice_in_triangle表
					Vec3i vec;
					int v0 = min(first, min(third, fourth));
					int v2 = max(first, max(third, fourth));
					int v1 = (first + third + fourth) - v0 - v2;
					vec[0] = v0;
					vec[1] = v1;
					vec[2] = v2;
					tri[index0] = vec;
					vector<int> ::iterator iter1 = find(vertice_in_triangle[second].begin(), vertice_in_triangle[second].end(), index0);
					vertice_in_triangle[second].erase(iter1);
					vertice_in_triangle[fourth].push_back(index0);
					v0 = min(second, min(third, fourth));
					v2 = max(second, max(third, fourth));
					v1 = (second + third + fourth) - v0 - v2;
					vec[0] = v0;
					vec[1] = v1;
					vec[2] = v2;
					tri[index1] = vec;
					vector<int> ::iterator iter2 = find(vertice_in_triangle[first].begin(), vertice_in_triangle[first].end(), index1);
					vertice_in_triangle[first].erase(iter2);
					vertice_in_triangle[third].push_back(index1);

					vector<int> new_edge;
					if (third > fourth)//使得new_edges中边上点的编号也是从小到大排序，和constrained_edge中一样
					{
						int temp = third;
						third = fourth;
						fourth = temp;
					}
					new_edge.push_back(third);
					new_edge.push_back(fourth);
					if (is_cross(del.point[third], del.point[fourth], del.point[cur_p0], del.point[cur_p1]))
					{
						cross_edges.push(new_edge);
					}
					else
					{

						new_edges.push_back(new_edge);
					}
				}
			}
			/*====================================下面是测试代码========================================*/
			std::cout << "new_edges:" << endl;
			for (int t = 0; t < new_edges.size(); t++) {
				for (int t0 = 0; t0 < new_edges[t].size(); t0++)
					std::cout << new_edges[t][t0] << " ";
				std::cout << endl;
			}
			/*==============================================================================================*/
			/****************************************************************************************\
			*                     以上代码移除了和当前约束边相交的三角网格的边                    *
			\****************************************************************************************/

			int tri1, tri2;//记录包含当前new-edge在内的三角形的编号
			bool swap = true;
			while (swap) {
				swap = false;
				for (int c = 0; c < new_edges.size(); c++)
				{
					if (new_edges[c][0] == cur_p0&&new_edges[c][1] == cur_p1)
					{
						continue;
					}
					else
					{
						int vk = new_edges[c][0];
						int vl = new_edges[c][1];
						//下面两个是共享vk和vl边的两个三角形的另外两个顶点
						int vo = INT_MAX;
						int vp = INT_MAX;
						for (int v = 0; v < vertice_in_triangle[vk].size(); v++)
						{
							int index = vertice_in_triangle[vk][v];
							if (tri[index][1] == vl || tri[index][2] == vl || tri[index][0] == vl)
							{
								if (vo == INT_MAX) {
									vo = tri[index][0] + tri[index][1] + tri[index][2] - vk - vl;
									tri1 = index;
								}
								else
								{
									vp = tri[index][0] + tri[index][1] + tri[index][2] - vk - vl;
									tri2 = index;
								}

							}
						}
						double A1 = 2 * (del.point[vk].x - del.point[vl].x);
						double B1 = 2 * (del.point[vk].y - del.point[vl].y);
						double C1 = pow(del.point[vk].x, 2) + pow(del.point[vk].y, 2) - pow(del.point[vl].x, 2) - pow(del.point[vl].y, 2);
						double A2 = 2 * (del.point[vo].x - del.point[vk].x);
						double B2 = 2 * (del.point[vo].y - del.point[vk].y);
						double C2 = pow(del.point[vo].x, 2) + pow(del.point[vo].y, 2) - pow(del.point[vk].x, 2) - pow(del.point[vk].y, 2);
						//下面计算的是三角形vk vl vo的外接圆的圆心
						double center_x = ((C1*B2) - (C2*B1)) / ((A1*B2) - (A2*B1));
						double center_y = ((A1*C2) - (A2*C1)) / ((A1*B2) - (A2*B1));
						double redius_2 = pow((del.point[vk].x - center_x), 2) + pow((del.point[vk].y - center_y), 2);
						if (pow(del.point[vp].x - center_x, 2) + pow(del.point[vp].y - center_y, 2) <= redius_2)
							//圆内最多三点共圆
						{
							swap = true;
							Vec3i vec;
							int a0 = min(vo, min(vk, vp));
							int a2 = max(vo, max(vk, vp));
							int a1 = (vo + vk + vp) - a0 - a2;
							vec[0] = a0;
							vec[1] = a1;
							vec[2] = a2;
							tri[tri1] = vec;
							a0 = min(vo, min(vl, vp));
							a2 = max(vo, max(vl, vp));
							a1 = (vo + vl + vp) - a0 - a2;
							vec[0] = a0;
							vec[1] = a1;
							vec[2] = a2;
							tri[tri2] = vec;
							vector<int>::iterator iter = find(vertice_in_triangle[vl].begin(), vertice_in_triangle[vl].end(), tri1);
							vertice_in_triangle[vl].erase(iter);
							vertice_in_triangle[vp].push_back(tri1);
							vector<int>::iterator iter1 = find(vertice_in_triangle[vk].begin(), vertice_in_triangle[vk].end(), tri2);
							vertice_in_triangle[vk].erase(iter1);
							vertice_in_triangle[vo].push_back(tri2);
							int new0 = vo;
							int new1 = vp;
							if (vo > vp)
							{
								new0 = vp;
								new1 = vo;
							}
							new_edges[c][0] = new0;
							new_edges[c][1] = new1;
						}
					}
					if (swap)
						break;
				}
			}
			/****************************************************************************************\
			*                   以上代码对于新生成的边检测了是否满足delaunay三角，并进行了调整             *
			\****************************************************************************************/
		}

	//主要涉及的数组有 tri，vertice_in_triangle，del.constrained_edge, del.point
	//del.point中索引i是顶点del.point[i]的编号
	//tri中索引i表示三角形编号，三角形i是由tri[i][0]  tri[i][1]  tri[i][2]三个编号所表示的顶点组成的
	//del.constrained_edge中索引i表示约束边编号，约束边i是由del.constrained_edge[i][0] del.constrained_edge[i][1] 两个编号所表示的顶点连线所得
	//vertice_in_triangle中索引i表示顶点编号为i的点所在的三角形编号
	int pointnum = 3;
	const Point *pBuf0;
	Point buf0[3];
	pBuf0 = buf0;
//	Mat imShow1 = dstImage.clone();
	std::cout << "三角面片的数量为" << tri.size() << endl;
	vector<Vec3i> ::iterator iter = tri.begin();
	while (iter != tri.end())
	{
		Vec3i temp = *iter;
		buf0[0] = del.point[temp[0]];
		buf0[1] = del.point[temp[1]];
		buf0[2] = del.point[temp[2]];
		if (!in_Polygon(del.constrained_edge, del.point, temp[0], temp[1])
			|| !in_Polygon(del.constrained_edge, del.point, temp[0], temp[2])
			|| !in_Polygon(del.constrained_edge, del.point, temp[2], temp[1]))//三角形的一个边不在多边形内部
		{
			//这里只修改tri表就可以，把不符合要求的三角形三个顶点赋值为INT_MAX.但是vertice_in_triangle表不需要修改，最后可以重新生成
			Vec3i temp0;
			temp0[0] = INT_MAX;
			temp0[1] = INT_MAX;
			temp0[2] = INT_MAX;
			*iter = temp0;
			/*polylines(imShow1, &pBuf0, &pointnum,
				1, true, CV_RGB(0, 0, 255),
				1, CV_AA, 0);*/
		}
		else {
			polylines(imgShow, &pBuf0, &pointnum,
				1, true, CV_RGB(0, 255, 0),
				1, CV_AA, 0);
		}
		iter++;
	}

	for (int j = 0; j < del.point.size(); j++)
	{
		if (point_in_polygon(del.constrained_edge, del.point, j))
		{
			circle(imgShow, del.point[j], 2.5, Scalar(255, 255, 255), -1);
		}
		else
		{
			del.point[j].x = -1;
			del.point[j].y = -1;
		}
	}
	cv::imshow("Segmentation", imgShow);
	cv::waitKey();
	
}



