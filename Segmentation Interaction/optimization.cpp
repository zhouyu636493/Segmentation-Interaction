#include "optimization.h"
#include <iostream>
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
Optimization::Optimization() {};
Optimization::~Optimization() {};
//统计每个contour所包含的分割线的数量
int* seg_count(vector<vector<Point2f>>& seg_contours,vector<vector<int>>& seg_index,int* const number0)
{
	if(number0==NULL)
	{
		cerr << "error: null ptr @number0" << endl;
		return NULL;
	}
	//number0中存放每个contour包含的分割线数量，number0的下标对应seg_contours的下标，但是seg_contours中空的contour的number0永远都是0！
	for (int i = 0; i < seg_contours.size(); i++)
	{
		number0[i] = 0;
	}
	for (int j = 0; j < seg_index.size(); j++)
	{
		int index = seg_index[j][0];
		if (!seg_contours[index].empty())
		{
			number0[index]++;
		}
	}
	return number0;
}
Point2f intersect_point(Point2f direction0, Point2f point0, Point2f direction1, Point2f point1)
{
	//计算两条相交直线的交点
	//direction代表方向向量，point代表直线上的任意一点
	Point2f intersect;
	double A1 = direction0.y;
	double B1 = -direction0.x;
	double C1 = point0.y*direction0.x - point0.x*direction0.y;
	double A2 = direction1.y;
	double B2 = -direction1.x;
	double C2 = point1.y*direction1.x - point1.x*direction1.y;
	intersect.x = (B1*C2-B2*C1) / (A1*B2-A2*B1);
	intersect.y = (A2*C1-A1*C2) / (A1*B2-A2*B1);
	return intersect;
}
double Optimization::D_compute(double angle,vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index,
	vector<vector<Point2f>>& segment_pairs)
{
	int* number0 = new int[seg_contours.size()];
	int* number=seg_count(seg_contours, seg_index, number0);
	int size = segment_pairs.size();
	bool flag;
	double Deformation = 0;
	int value = 0;
	for (int i = 0; i < size; i++)
	{
		flag = true;
		vector<int> bi_index;//里面存放的是与当前ruling线i相邻的两条ruling线在segment——pairs中的下标
		for (int j = 0; j < seg_index.size(); j++)
		{
			if (seg_index[j][1] == i)
			{
				if (!seg_contours[seg_index[j][0]].empty()&&number[seg_index[j][0]] <= 1)
				{
					flag = false;
					break;//说明与当前分割线i相邻的contour中，有一个contour是边界contour，所以这条分割线上的形变量忽略；
				}
				else if(!seg_contours[seg_index[j][0]].empty())
				{
					double max_length = DBL_MIN;
					int max_index;
					for (int k = 0; k < seg_index.size(); k++)
					{
						if (seg_index[k][0] == seg_index[j][0] && seg_index[k][1]!=i)
						{
							double length;
							length = powf((segment_pairs[seg_index[k][1]][0].x - segment_pairs[seg_index[k][1]][1].x), 2)
								+ powf((segment_pairs[seg_index[k][1]][0].y - segment_pairs[seg_index[k][1]][1].y), 2);
							length = sqrtf(length);
							if (length > max_length)
							{
								max_length = length;
								max_index = seg_index[k][1];
							}
						}
					}
					bi_index.push_back(max_index);
				}
			}
			else
				continue;
		}
		if (!flag)
		{
			//说明与当前分割线i相邻的contour中，有一个contour是边界contour，所以这条分割线上的形变量忽略；
			continue;
		}
		else
		{
			//计算形变量
			//bi_index中存放的是与当前ruling线相邻的两条ruling线的下标
			double short0, short1, long0, long1;//存储的是阴影部分上下四条小边的长度
			for (int u = 0; u < bi_index.size(); u++)
			{
				int index = bi_index[u];
				Point2f direction0, direction1, point0, point1;
				direction0.x = segment_pairs[i][1].x - segment_pairs[i][0].x;
				direction0.y = segment_pairs[i][1].y - segment_pairs[i][0].y;
				point0.x = (segment_pairs[i][1].x + segment_pairs[i][0].x) / 2;
				point0.y = (segment_pairs[i][1].y + segment_pairs[i][0].y) / 2;
				direction1.x = segment_pairs[index][1].x - segment_pairs[index][0].x;
				direction1.y = segment_pairs[index][1].y - segment_pairs[index][0].y;
				point1.x = (segment_pairs[index][1].x + segment_pairs[index][0].x) / 2;
				point1.y = (segment_pairs[index][1].y + segment_pairs[index][0].y) / 2;
				if ((segment_pairs[i][1].x- segment_pairs[i][0].x == 0 && segment_pairs[index][1].x - segment_pairs[index][0].x == 0)||
					((segment_pairs[i][1].y - segment_pairs[i][0].y)/(segment_pairs[i][1].x - segment_pairs[i][0].x)==
					(segment_pairs[index][1].y - segment_pairs[index][0].y)/(segment_pairs[index][1].x - segment_pairs[index][0].x)))
				{
					//说明这两条ruling线平行
					double A1 = direction0.y;
					double B1 = -direction0.x;
					double C1 = point0.y*direction0.x - point0.x*direction0.y;
					double A2 = direction1.y;
					double B2 = -direction1.x;
					double C2 = point1.y*direction1.x - point1.x*direction1.y;
					double length;
					if (A1 != A2)
					{
						if (A1 > A2)
						{
							C1 = C1 / (A1 / A2);
							A1 = A2;
							B1 = B1 / (A1 / A2);
						}
						else
						{
							C2 = C2 / (A2 / A1);
						}
					}
					length = abs(C1 - C2) / sqrtf(pow(A1,2)+pow(B1,2));
					if (u == 0)
					{
						short0 =length;
						long0 = length;
					}
					else if (u == 1)
					{
						short1 = length;
						long1 = length;
					}
				}
				else
				{
					//说明两条ruling线相交
					Point2f intersect = intersect_point(direction0, point0, direction1, point1);
					double length0 = sqrtf(powf(intersect.x-segment_pairs[i][0].x,2)+powf(intersect.y-segment_pairs[i][0].y,2));
					double length1= sqrtf(powf(intersect.x - segment_pairs[i][1].x, 2) + powf(intersect.y - segment_pairs[i][1].y, 2));
					double length2 = sqrtf(powf(intersect.x - segment_pairs[index][0].x, 2) + powf(intersect.y - segment_pairs[index][0].y, 2));
					double length3= sqrtf(powf(intersect.x - segment_pairs[index][1].x, 2) + powf(intersect.y - segment_pairs[index][1].y, 2));
					Point2f v1, v2;
					//将length0,1,2,3按从小到大排序,同时计算用于计算夹角的向量v1 v2,
					if (length0 > length1)
					{
						double temp = length0;
						length0 = length1;
						length1 = temp;
						v1.x = segment_pairs[i][0].x - segment_pairs[i][1].x;
						v1.y = segment_pairs[i][0].y - segment_pairs[i][1].y;
					}
					else
					{
						v1.x = segment_pairs[i][1].x - segment_pairs[i][0].x;
						v1.y = segment_pairs[i][1].y - segment_pairs[i][0].y;
					}
					if (length2 > length3)
					{
						v2.x = segment_pairs[index][0].x - segment_pairs[index][1].x;
						v2.y = segment_pairs[index][0].y - segment_pairs[index][1].y;
					}
					else
					{
						v2.x = segment_pairs[index][1].x - segment_pairs[index][0].x;
						v2.y = segment_pairs[index][1].y - segment_pairs[index][0].y;
					}
					double cos2b = (v1.x*v2.x+v1.y*v2.y) / (sqrtf(powf(v1.x,2)+powf(v1.y,2))*sqrtf(powf(v2.x,2)+powf(v2.y,2)));
					double cosb = sqrtf((cos2b+1)/2);
					double sinb = sqrtf((1-cos2b)/2);
					double tanb = sinb / cosb;
					if (u == 0)
					{
						short0 = tanb*length0;
						long0 = tanb*length1;
					}
					else if (u == 1)
					{
						short1 = tanb*length0;
						long1 = tanb*length1;
					}
					
				}
			}
			double d =sqrtf(powf(segment_pairs[i][1].x - segment_pairs[i][0].x, 2) + powf(segment_pairs[i][1].y - segment_pairs[i][0].y, 2));
			double Def = d*powf(angle, 2)*((log(long0+long1)-log(short0+short1))/(long0 + long1-(short0 + short1)));
			Deformation = Deformation + Def;
		}
	}
	delete number;
	number = NULL;
	return Deformation;
}

double crosspruduct(Point2f A,Point2f B)
{
	return A.x*B.y - A.y*B.x;
}
bool is_in_contour(Point2f mid, vector<Point2f> contour)
{
	//判断点mid是否在contour中，在则返回true
	int n = contour.size();
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (mid.y == contour[i].y&&mid.x == contour[i].x)
			return false;
		else if (mid.y == contour[i].y&&mid.y == contour[(i + 1) % n].y)
		{
			if ((mid.x >= contour[i].x&&mid.x <= contour[(i + 1) % n].x) || (mid.x <= contour[i].x&&mid.x >= contour[(i + 1) % n].x))
				return false;
		}
		else if ((mid.y >= contour[i].y&&mid.y < contour[(i + 1) % n].y) || (mid.y >= contour[(i + 1) % n].y&&mid.y < contour[i].y))
		{
			double d = ((mid.y - contour[i].y)*(contour[(i + 1) % n].x - contour[i].x) + (contour[i].x - mid.x)*(contour[(i + 1) % n].y - contour[i].y))
				* (contour[(i + 1) % n].y - contour[i].y);
			if (d > 0)
				count++;
			else if (d == 0)
				return false;
		}
	}
	if (count % 2 == 0)
		return false;
	else
		return true;//在contour中
}
vector<vector<Point2f>> Optimization::Seg_compute(vector<vector<Point2f>> segment,vector<Point2f> contour,vector<vector<int>>&  intersecting_index)
{
	//segment是一个二维数组，数组为2列n行，n表示分割线数量，数组第一列表示分割线上某一点，数组第二列表示分割线的方向向量
	//contour表示多边形轮廓
	/*
	对于每条segment，计算其与多边形边的交点
	计算交点的时候，是直线与线段的相交，当他们重合的时候，不需要将线段的两个端点加入进交点集合中，直接忽略即可；
	如果两个交点之间有多边形顶点，则观察多边形方向，添加分割线到segment-pair；
	*/
	vector<vector<Point2f>> segment_pairs;
	vector<int> no_intersect;//存储与多边形无交点的分割线在segment中的下标
	for (int i = 0; i < segment.size(); i++)
	{
		int index = i;
		vector<Point2f> intersecting;//存放分割线与多边形边的交点的集合
		vector<vector<int>> intersect_index;//存放分割线与多边形边交点所处多边形边两端点的index,放置顺序是多边形顶点的顺序
		for (int j = 0; j < contour.size(); j++)
		{
			//e0和e1表示线段的两个端点下标
			int e0 = j;
			int e1 = (j + 1)%contour.size();
			Point2f P1 = contour[e0];
			Point2f d1;//t要求是大于等于0小于等于1
			d1.x = contour[e1].x - contour[e0].x;
			d1.y = contour[e1].y - contour[e0].y;
			Point2f P0 = segment[index][0];
			Point2f d0 = segment[index][1];
			double K = crosspruduct(d0, d1);
			Point2f deta;
			deta.x = P1.x - P0.x;
			deta.y = P1.y - P0.y;
			if (K == 0)
			{
				//说明平行或重合
				continue;
			}
			else
			{
				double t = crosspruduct(deta, d0) / K;
				if (t >= 0 && t <= 1)
				{
					//说明分割线与多边形的边相交
					Point2f intersect;
					intersect = P1 + t*d1;

					//首先应该确定现有的交点中是否已经包含该交点，如果已经包含，则直接忽略；
					vector<Point2f> ::iterator iter;
					iter = find(intersecting.begin(), intersecting.end(), intersect);
					if (iter == intersecting.end())
					{
						intersecting.push_back(intersect);

						vector<int> temp;

						temp.push_back(e0);
						temp.push_back(e1);

						intersect_index.push_back(temp);
					}
					else
						continue;
					
				}
				else
					continue;
			}
		}
		//下面应该抽取segment中下标为i的直线对应多边形形成的分割线集合
		if (intersecting.size() == 0||intersecting.size()==1)
		{
			//分割线i对多边形必定无分割
			no_intersect.push_back(i);
		}
		else
		{
			//将交点按照x轴坐标排序
			int len = intersecting.size();
			if (len > 1 && intersecting[0].x == intersecting[1].x)
			{
				//如果所有交点的横坐标相等，也就是分割线垂直于x轴，那么久按照y坐标排序
				for (int s = 0; s < len - 1; s++)
				{
					for (int w = 0; w < len - 1 - s; w++)
					{
						if (intersecting[w].y > intersecting[w + 1].y)
						{
							Point2f temp = intersecting[w + 1];
							intersecting[w + 1] = intersecting[w];
							intersecting[w] = temp;
							vector<int> temp0;
							temp0.assign(intersect_index[w + 1].begin(), intersect_index[w + 1].end());
							intersect_index[w + 1].clear();
							intersect_index[w + 1].assign(intersect_index[w].begin(), intersect_index[w].end());
							intersect_index[w].clear();
							intersect_index[w].assign(temp0.begin(), temp0.end());
						}
					}
				}
			}
			else
			{
				//否则按照x坐标排序
				for (int s = 0; s < len - 1; s++)
				{
					for (int w = 0; w < len - 1 - s; w++)
					{
						if (intersecting[w].x > intersecting[w + 1].x)
						{
							Point2f temp = intersecting[w + 1];
							intersecting[w + 1] = intersecting[w];
							intersecting[w] = temp;
							vector<int> temp0;
							temp0.assign(intersect_index[w + 1].begin(), intersect_index[w + 1].end());
							intersect_index[w + 1].clear();
							intersect_index[w + 1].assign(intersect_index[w].begin(), intersect_index[w].end());
							intersect_index[w].clear();
							intersect_index[w].assign(temp0.begin(), temp0.end());
						}
					}
				}
			}
			//根据线段中点是否在多边形内部来判断哪些是新生成的分割线
			int number = 0;
			for (int s = 0; s < intersecting.size()-1; s++)
			{
				int w = s + 1;
				Point2f midpoint;
				midpoint.x = (intersecting[s].x + intersecting[w].x) / 2;
				midpoint.y = (intersecting[s].y + intersecting[w].y) / 2;

				if (is_in_contour(midpoint,  contour))
				{
                    //说明分割线的中点在多边形内部，也就是该线段是分割线
					number++;
					vector<Point2f> temp;
					temp.push_back(intersecting[s]);
					temp.push_back(intersecting[w]);
					segment_pairs.push_back(temp);
					vector<int> temp0;
					//这里的每对temp应该从小到大排序
					if (intersect_index[s][0] < intersect_index[s][1])
					{
						temp0.push_back(intersect_index[s][0]);
						temp0.push_back(intersect_index[s][1]);
					}
					else
					{
						temp0.push_back(intersect_index[s][1]);
						temp0.push_back(intersect_index[s][0]);

					}
					if (intersect_index[w][0]<intersect_index[w][1])
					{
						temp0.push_back(intersect_index[w][0]);
						temp0.push_back(intersect_index[w][1]);
					}
					else
					{
						temp0.push_back(intersect_index[w][1]);
						temp0.push_back(intersect_index[w][0]);
					}
					intersecting_index.push_back(temp0);
				}
				else
				{

				}
			}
			if (number == 0)
			{
				//分割线i对多边形必定无分割
				no_intersect.push_back(i);
			}
		}
	}
	return segment_pairs;
}

/*
优化中的约束：
1.如果直线与多变形交点为0，
2.如果两个分割线的交点在多边形内部，
3.如果生成的新轮廓放不下轮廓，

*/
void  myconstraint1()
{

}
