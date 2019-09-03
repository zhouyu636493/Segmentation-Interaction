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
//ͳ��ÿ��contour�������ķָ��ߵ�����
int* seg_count(vector<vector<Point2f>>& seg_contours,vector<vector<int>>& seg_index,int* const number0)
{
	if(number0==NULL)
	{
		cerr << "error: null ptr @number0" << endl;
		return NULL;
	}
	//number0�д��ÿ��contour�����ķָ���������number0���±��Ӧseg_contours���±꣬����seg_contours�пյ�contour��number0��Զ����0��
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
	//���������ֱཻ�ߵĽ���
	//direction������������point����ֱ���ϵ�����һ��
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
		vector<int> bi_index;//�����ŵ����뵱ǰruling��i���ڵ�����ruling����segment����pairs�е��±�
		for (int j = 0; j < seg_index.size(); j++)
		{
			if (seg_index[j][1] == i)
			{
				if (!seg_contours[seg_index[j][0]].empty()&&number[seg_index[j][0]] <= 1)
				{
					flag = false;
					break;//˵���뵱ǰ�ָ���i���ڵ�contour�У���һ��contour�Ǳ߽�contour�����������ָ����ϵ��α������ԣ�
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
			//˵���뵱ǰ�ָ���i���ڵ�contour�У���һ��contour�Ǳ߽�contour�����������ָ����ϵ��α������ԣ�
			continue;
		}
		else
		{
			//�����α���
			//bi_index�д�ŵ����뵱ǰruling�����ڵ�����ruling�ߵ��±�
			double short0, short1, long0, long1;//�洢������Ӱ������������С�ߵĳ���
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
					//˵��������ruling��ƽ��
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
					//˵������ruling���ཻ
					Point2f intersect = intersect_point(direction0, point0, direction1, point1);
					double length0 = sqrtf(powf(intersect.x-segment_pairs[i][0].x,2)+powf(intersect.y-segment_pairs[i][0].y,2));
					double length1= sqrtf(powf(intersect.x - segment_pairs[i][1].x, 2) + powf(intersect.y - segment_pairs[i][1].y, 2));
					double length2 = sqrtf(powf(intersect.x - segment_pairs[index][0].x, 2) + powf(intersect.y - segment_pairs[index][0].y, 2));
					double length3= sqrtf(powf(intersect.x - segment_pairs[index][1].x, 2) + powf(intersect.y - segment_pairs[index][1].y, 2));
					Point2f v1, v2;
					//��length0,1,2,3����С��������,ͬʱ�������ڼ���нǵ�����v1 v2,
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
	//�жϵ�mid�Ƿ���contour�У����򷵻�true
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
		return true;//��contour��
}
vector<vector<Point2f>> Optimization::Seg_compute(vector<vector<Point2f>> segment,vector<Point2f> contour,vector<vector<int>>&  intersecting_index)
{
	//segment��һ����ά���飬����Ϊ2��n�У�n��ʾ�ָ��������������һ�б�ʾ�ָ�����ĳһ�㣬����ڶ��б�ʾ�ָ��ߵķ�������
	//contour��ʾ���������
	/*
	����ÿ��segment�������������αߵĽ���
	���㽻���ʱ����ֱ�����߶ε��ཻ���������غϵ�ʱ�򣬲���Ҫ���߶ε������˵��������㼯���У�ֱ�Ӻ��Լ��ɣ�
	�����������֮���ж���ζ��㣬��۲����η�����ӷָ��ߵ�segment-pair��
	*/
	vector<vector<Point2f>> segment_pairs;
	vector<int> no_intersect;//�洢�������޽���ķָ�����segment�е��±�
	for (int i = 0; i < segment.size(); i++)
	{
		int index = i;
		vector<Point2f> intersecting;//��ŷָ��������αߵĽ���ļ���
		vector<vector<int>> intersect_index;//��ŷָ��������α߽�����������α����˵��index,����˳���Ƕ���ζ����˳��
		for (int j = 0; j < contour.size(); j++)
		{
			//e0��e1��ʾ�߶ε������˵��±�
			int e0 = j;
			int e1 = (j + 1)%contour.size();
			Point2f P1 = contour[e0];
			Point2f d1;//tҪ���Ǵ��ڵ���0С�ڵ���1
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
				//˵��ƽ�л��غ�
				continue;
			}
			else
			{
				double t = crosspruduct(deta, d0) / K;
				if (t >= 0 && t <= 1)
				{
					//˵���ָ��������εı��ཻ
					Point2f intersect;
					intersect = P1 + t*d1;

					//����Ӧ��ȷ�����еĽ������Ƿ��Ѿ������ý��㣬����Ѿ���������ֱ�Ӻ��ԣ�
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
		//����Ӧ�ó�ȡsegment���±�Ϊi��ֱ�߶�Ӧ������γɵķָ��߼���
		if (intersecting.size() == 0||intersecting.size()==1)
		{
			//�ָ���i�Զ���αض��޷ָ�
			no_intersect.push_back(i);
		}
		else
		{
			//�����㰴��x����������
			int len = intersecting.size();
			if (len > 1 && intersecting[0].x == intersecting[1].x)
			{
				//������н���ĺ�������ȣ�Ҳ���Ƿָ��ߴ�ֱ��x�ᣬ��ô�ð���y��������
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
				//������x��������
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
			//�����߶��е��Ƿ��ڶ�����ڲ����ж���Щ�������ɵķָ���
			int number = 0;
			for (int s = 0; s < intersecting.size()-1; s++)
			{
				int w = s + 1;
				Point2f midpoint;
				midpoint.x = (intersecting[s].x + intersecting[w].x) / 2;
				midpoint.y = (intersecting[s].y + intersecting[w].y) / 2;

				if (is_in_contour(midpoint,  contour))
				{
                    //˵���ָ��ߵ��е��ڶ�����ڲ���Ҳ���Ǹ��߶��Ƿָ���
					number++;
					vector<Point2f> temp;
					temp.push_back(intersecting[s]);
					temp.push_back(intersecting[w]);
					segment_pairs.push_back(temp);
					vector<int> temp0;
					//�����ÿ��tempӦ�ô�С��������
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
				//�ָ���i�Զ���αض��޷ָ�
				no_intersect.push_back(i);
			}
		}
	}
	return segment_pairs;
}

/*
�Ż��е�Լ����
1.���ֱ�������ν���Ϊ0��
2.��������ָ��ߵĽ����ڶ�����ڲ���
3.������ɵ��������Ų���������

*/
void  myconstraint1()
{

}
