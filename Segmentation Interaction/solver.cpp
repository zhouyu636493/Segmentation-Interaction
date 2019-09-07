#include "nlopt.h"
#include "optimization.h"
#include "solver.h"
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;


vector<Point2f> compute_intersect(vector<Point2f> contour, vector<Point2f> points, vector<vector<int>>& intersect_index)
{
	//计算线段points和contour的交点
	vector<Point2f> intersect;
	vector<int> tep;
	//tep中存放的是交点所处的contour中的边的index（不是原本整个大的contour）因为一共两个交点，每个交点位于两个index之间，所以tep中有四个元素
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

bool index_compute(vector<Point2f> contour, vector<Point2f> segment_pair, vector<vector<int>>& intersect_index)
{
	//这个函数的目的是计算线段segment_pairs[k]的两个端点分别在seg_contours[e]的哪两条边上，记录相应的边的端点下标
	int size = contour.size();
	bool flag0, flag1;
	flag0 = false;
	flag1 = false;
	for (int i = 0; i < size; i++)
	{
		int e0 = i;
		int e1 = (i + 1) % size;
		for (int j = 0; j < segment_pair.size(); j++)
		{
			Point2f  p1, p2;
			p1.x = segment_pair[j].x - contour[e0].x;
			p1.y = segment_pair[j].y - contour[e0].y;
			p2.x = contour[e1].x - contour[e0].x;
			p2.y = contour[e1].y - contour[e0].y;
			if (p1.x*p2.y - p2.x*p1.y == 0)
			{
				if (min(contour[e0].x, contour[e1].x) <= segment_pair[j].x&&max(contour[e0].x, contour[e1].x) >= segment_pair[j].x
					&&min(contour[e0].y, contour[e1].y) <= segment_pair[j].y&&max(contour[e0].y, contour[e1].y) >= segment_pair[j].x)
				{
					if (e0 < e1)
					{
						vector<int> temp;
						temp.push_back(e0);
						temp.push_back(e1);
						intersect_index.push_back(temp);
						flag0 = true;
					}
					else
					{
						vector<int> temp;
						temp.push_back(e1);
						temp.push_back(e0);
						intersect_index.push_back(temp);
						flag1 = true;
					}
				}
			}
		}
	}
	return (flag0&&flag1);
}
//void  Solver::contour_segmentation(vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<int>>& intersect_index, vector<vector<Point2f>>& segment_pairs)
//{
//	//提取分割后轮廓的函数
//	//这里可以验证一下运行这段代码之前的segment_pairs和运行这段代码之后的segment_pair是否完全相等，是的话就对了。
//	for (int k = 0; k < segment_pairs.size(); k++)
//	{
//		cout << segment_pairs[k][0].x << " " << segment_pairs[k][0].y << endl;
//		cout << segment_pairs[k][1].x << " " << segment_pairs[k][1].y << endl;
//		vector<Point2f> contour0_temp;
//		vector<Point2f> contour_temp;
//		vector<Point2f> temp_pair;
//		int size = seg_contours.size();
//		bool flag = false;
//		int e;
//		for (e = 0; e < size; e++)
//		{
//			if (!seg_contours[e].empty()) {
//				//temp_pair = compute_intersect(seg_contours[e], segment_pairs[k], intersect_index);//这里的intersect_index是seg_contours[e]中顶点下标，而不是相对于整个contour的下标
//				//上面这个函数在这里其实就只是计算分割线两个端点在相应contour哪两个端点之间，并记录在intersect_Index中，不需要就散交点，因为交点已经在segment_pairs中了
//				if (index_compute(seg_contours[e], segment_pairs[k], intersect_index))
//				{
//
//					temp_pair = segment_pairs[k];
//
//					flag = true;
//					contour_temp.push_back(temp_pair[0]);
//					int min, max;
//					int v1 = intersect_index[k][0];//向小的方向转
//					if (seg_contours[e][v1] != temp_pair[0])
//					{
//						min = v1;
//						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						if (min > max)
//						{
//							min = max;
//							max = v1;
//						}
//					}
//					else
//					{
//						v1 = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						min = v1;
//						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						if (min > max)
//						{
//							max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//							min = (max - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						}
//					}
//					while (!(min == intersect_index[k][2] && max == intersect_index[k][3]))
//					{
//						contour_temp.push_back(seg_contours[e][v1]);
//						v1 = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						min = v1;
//						max = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						if (min > max)
//						{
//							max = v1;
//							min = (v1 - 1 + seg_contours[e].size()) % seg_contours[e].size();
//						}
//					};
//					if (seg_contours[e][v1] != temp_pair[1]) {
//						contour_temp.push_back(seg_contours[e][v1]);
//						contour_temp.push_back(temp_pair[1]);
//					}
//					else
//					{
//						contour_temp.push_back(seg_contours[e][v1]);
//					}
//
//					for (int g = contour_temp.size() - 1; g >= 0; g--)
//					{
//						contour0_temp.push_back(contour_temp[g]);//将顺时针变成逆时针
//					}
//
//
//
//
//					contour_temp.clear();
//					int v2 = intersect_index[k][1];//向大的方向转,原本就是逆时针
//					contour_temp.push_back(temp_pair[0]);
//					if (seg_contours[e][v2] != temp_pair[0])
//					{
//						min = v2;
//						max = (v2 + 1) % seg_contours[e].size();
//						if (min > max)
//						{
//							min = max;
//							max = v1;
//						}
//					}
//					else
//					{
//						v2 = (v2 + 1) % seg_contours[e].size();
//						min = v2;
//						max = (v2 + 1) % seg_contours[e].size();
//						if (min > max)
//						{
//							max = (v2 + 1) % seg_contours[e].size();
//							min = (max + 1) % seg_contours[e].size();
//						}
//					}
//					while (!(min == intersect_index[k][2] && max == intersect_index[k][3]))
//					{
//						contour_temp.push_back(seg_contours[e][v2]);
//						v2 = (v2 + 1) % seg_contours[e].size();
//						min = v2;
//						max = (v2 + 1) % seg_contours[e].size();
//						if (min > max)
//						{
//							max = v2;
//							min = (v2 + 1) % seg_contours[e].size();
//						}
//					};
//					if (seg_contours[e][v2] != temp_pair[1]) {
//						contour_temp.push_back(seg_contours[e][v2]);
//						contour_temp.push_back(temp_pair[1]);
//					}
//					else
//					{
//						contour_temp.push_back(seg_contours[e][v2]);
//					}
//					//最终contour0_temp和contour_temp就是分割后的两个轮廓，且都是逆时针放置  
//					break;
//				}
//			}
//		}
//		if (flag)
//		{
//			int pos = seg_contours.size();
//			seg_contours.push_back(contour_temp);
//			seg_contours.push_back(contour0_temp);
//			vector<int> index_temp0;
//			for (int y = 0; y < seg_index.size(); y++)
//			{
//				index_temp0.clear();
//				bool flag0 = false;
//				bool flag1 = false;
//				if (seg_index[y][0] == e)
//				{
//					//查询新生成的轮廓中哪个包含seg_index[y][1]分割线
//					vector<Point2f> seg0 = segment_pairs[seg_index[y][1]];
//					int x;
//					for (x = 0; x < contour_temp.size(); x++)
//					{
//						if (contour_temp[x] == seg0[0])
//							flag0 = true;
//						if (contour_temp[x] == seg0[1])
//							flag1 = true;
//					}
//					if (flag0&&flag1)
//					{
//						index_temp0.push_back(pos);
//						index_temp0.push_back(seg_index[y][1]);
//						seg_index.push_back(index_temp0);
//					}
//					index_temp0.clear();
//					flag0 = false;
//					flag1 = false;
//					for (x = 0; x < contour0_temp.size(); x++)
//					{
//						if (contour0_temp[x] == seg0[0])
//							flag0 = true;
//						if (contour0_temp[x] == seg0[1])
//							flag1 = true;
//					}
//					if (flag0&&flag1)
//					{
//						index_temp0.push_back(pos + 1);
//						index_temp0.push_back(seg_index[y][1]);
//						seg_index.push_back(index_temp0);
//					}
//
//				}
//			}
//			vector<int> index_temp;
//			index_temp.push_back(pos);
//			index_temp.push_back(k);
//			seg_index.push_back(index_temp);
//			index_temp.clear();
//			index_temp.push_back(pos + 1);
//			index_temp.push_back(k);
//			seg_index.push_back(index_temp);
//			seg_contours[e].clear();//将被划分的contour清空
//		}
//	}
//}

//double Solver:: myfunc(int n,const double* var,double *grad, struct  data* mydata)
//{
//	//由于nlopt优化函数中变量必须是浮点类型，所以var应该修改为浮点型数组，前s个表示分割线上的点，后s个表示分割线的方向向量，但是点和方向向量又都是由
//	//两个浮点数表示的，所以最终数组大小为2*2s,也就是：x1,y1,x2,y2,...,xs,ys,xs+1,ys+1,...,x2s,y2s
//	//所以n=4*s 
//	//vartemp是一个二维数组，数组为2列s行，s表示分割线数量，数组第一列表示分割线上某一点，数组第二列表示分割线的方向向量
//	vector<vector<Point2f>> vartemp;
//	int s = n / 4;//s表示分割线的数量
//	int size_i = 0;
//	while(size_i < 2*s)
//	{
//		vector<Point2f> temp;
//		Point2f point, direction;
//		point.x = var[size_i];
//		point.y = var[size_i + 1];
//		direction.x = var[2 * s+ size_i];
//		direction.y = var[2 * s + size_i +1];
//		size_i = size_i + 2;
//		temp.push_back(point);
//		temp.push_back(direction);
//		vartemp.push_back(temp);
//	}
//	//上面的程序将double类型的var装到了vector<vector<Point2f>>类型的vartemp中，从而能够在下面的Seg_compute中调用
//
//	Optimization opt;
//	vector<vector<Point2f>>segment_pairs = opt.Seg_compute(vartemp, mydata->contour);
//	Mat Image;
//	Image = cv::Mat::zeros(mydata->Image.rows, mydata->Image.cols, CV_8UC3);
//	Image.setTo(cv::Scalar(255, 255, 255));//设置背景颜色为白色
//	polylines(Image, mydata->contour, true, (0, 0, 255), 2);
//	for (int i = 0; i < segment_pairs.size(); i++)
//	{
//		line(Image, segment_pairs[i][0], segment_pairs[i][1], Scalar(0, 255, 0), 1, CV_AA, 0);
//	}
//	namedWindow("Segmentation", CV_WINDOW_AUTOSIZE);
//	cv::imshow("Segmentation", Image);
//	char key = cv::waitKey(0);
//	/*
//	优化中的约束：
//	1.如果直线与多变形交点为0，
//	2.如果两个分割线的交点在多边形内部，
//	3.如果生成的新轮廓放不下轮廓，
//	*/
//	vector<vector<int>> intersect_index; //存储的是segment_pairs中相应分割线和轮廓相交的边的顶点index，每个数组分量有四个分量
//	vector<vector<Point2f>> seg_contours;
//	vector<vector<int>> seg_index;
//	vector<Point2f> contour0;
//	for (int a = 0; a < mydata->contour.size(); a++)
//	{
//		Point2f temp;
//		temp.x = mydata->contour[a].x;
//		temp.y = mydata->contour[a].y;
//		contour0.push_back(temp);
//	}
//	seg_contours.push_back(contour0);
//	contour_segmentation(seg_contours, seg_index, intersect_index, segment_pairs);
//
//	double def = opt.D_compute(mydata->angle,  seg_contours,  seg_index, segment_pairs);
//	return def;
//}

//如果按照之前的想法先计算直线对多边形的分割线，然后根据计算出来的分割线再去计算分割后的轮廓，就会导致浮点数之间的比较问题
//所以转换思路：一边计算直线对多边形的分割线，同时计算分割后的轮廓
//主要需要计算出的变量为,vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index,vector<vector<Point2f>>& segment_pairs
bool intersect(vector<Point2f>seg,vector<Point2f> line,Point2f &res)//计算线段和直线的交点
{
	//seg表示线段，line表示直线
	Point2f d0, d1, p0, p1;
	d0 = line[1];
	d1.x = seg[1].x - seg[0].x;
	d1.y= seg[1].y - seg[0].y;
	p0 = line[0];
	p1 = seg[0];
	double k = d0.x*d1.y - d0.y*d1.x;
	Point2f deta;
	deta.x = p1.x - p0.x;
	deta.y = p1.y - p0.y;
	if (k == 0)
	{
		if (deta.x*d0.y - deta.y*d0.x == 0)
		{
			return false;
		}
		else
			return false;
	}
	else
	{
		double t = (deta.x*d0.y - deta.y*d0.x) / k;
		if (t >= 0 && t <= 1)
		{
			res.x = p1.x + t*d1.x;
			res.y = p1.y + t*d1.y;
			return true;
		}
		else
			return false;
	}
	
}
double getSignedDistance(Point2f point,vector<Point2f> point_direction)
{
	//计算点到直线的带符号距离
	Point2f left;
	left.x = -point_direction[1].y;
	left.y= point_direction[1].x;
	Point2f ap;
	ap.x = point.x - point_direction[0].x;
	ap.y = point.y - point_direction[0].y;
	double signedDistance = left.x*ap.x + left.y*ap.y;
	return signedDistance;

}
void Solver::contour_segmentation( vector<vector<Point2f>> point_direction,vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<Point2f>>& segment_pairs)
{
	//该函数points是已知的
	//其他三个变量都是未知的
	//具体算法参考https://www.cnblogs.com/wantnon/p/6384771.html
	for (int s = 0; s < point_direction.size(); s++)
	{
		for (int r = 0; r < seg_contours.size(); r++)
		{
			if (!seg_contours[r].empty())
			{
				vector<Point2f> contour;
				contour = seg_contours[r];
				vector<Point2f> contour_add;//当前contour后面依次插入计算出的交点形成的新数组
				vector<int> new_contour;//插入交点后的新轮廓，每个元素为相应点在contour_add中的下标
				vector<vector<int>> new_contour_edge;//插入交点后新轮廓的边
				int sum = 0;
				for (int i = 0; i < contour.size(); i++)
				{
					contour_add.push_back(contour[i]);
				}
				for (int i = 0; i < contour.size(); i++)
				{

					int e0 = i;
					int e1 = (i + 1) % contour.size();
					vector<Point2f>seg;
					seg.push_back(contour[e0]);
					seg.push_back(contour[e1]);
					Point2f res;
					if (intersect(seg, point_direction[s], res))
					{
						sum++;
						contour_add.push_back(res);
						new_contour.push_back(e0);
						new_contour.push_back(contour.size()+sum-1);
					}
					else
					{
						new_contour.push_back(e0);
					}
				}
				for (int i = 0; i < new_contour.size(); i++)
				{
					vector<int> temp;
					temp.push_back(new_contour[i]);
					temp.push_back(new_contour[(i + 1) % new_contour.size()]);
					new_contour_edge.push_back(temp);
				}
				vector<int> right, left;
				for (int i = 0; i < contour.size(); i++)
				{
					double res = getSignedDistance(contour[i], point_direction[s]);
					if (res >= 0)
					{
						right.push_back(i);
					}
					else
					{
						left.push_back(i);
					}
				}
				vector<vector<int>> right_edge, left_edge;
				for (int i = 0; i < new_contour_edge.size(); i++)
				{
					int index = new_contour_edge[i][0];
					vector<int>::iterator iter = find(right.begin(), right.end(), index);
					if (iter == right.end())
					{
						left_edge.push_back(new_contour_edge[i]);
					}
					else
					{
						right_edge.push_back(new_contour_edge[i]);
					}
				}




			}
		}
	}
	
}