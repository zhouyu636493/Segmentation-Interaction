#include <nlopt.h>
#include <optimization.h>

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






void  contour_segmentation(vector<vector<Point2f>>& seg_contours, vector<vector<int>>& seg_index, vector<vector<int>>& intersect_index, vector<vector<Point2f>>& segment_pairs)
{
	//提取分割后轮廓的函数
	//这里可以验证一下运行这段代码之前的segment_pairs和运行这段代码之后的segment_pair是否完全相等，是的话就对了。
	for (int k = 0; k < segment_pairs.size(); k++)
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
				temp_pair = compute_intersect(seg_contours[e], segment_pairs[k], intersect_index);
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






double myfunc(int n,const vector<vector<Point2f>> var,double *grad, vector<Point2f> contour)
{
	//var是一个二维数组，数组为2列n行，n表示分割线数量，数组第一列表示分割线上某一点，数组第二列表示分割线的方向向量
	Optimization opt;
	vector<vector<int>>  intersecting_index;

	vector<vector<Point2f>>segment_pairs = opt.Seg_compute(var,  contour, intersecting_index);
	//需要注意的是Seg_compute中计算的intersecting_index里面的下标是针对整个contour而言的
	/*
	优化中的约束：
	1.如果直线与多变形交点为0，
	2.如果两个分割线的交点在多边形内部，
	3.如果生成的新轮廓放不下轮廓，
	*/
	vector<vector<int>> intersect_index; //存储的是segment_pairs中相应分割线和轮廓相交的边的顶点index，每个数组分量有四个分量
	vector<vector<Point2f>> seg_contours;
	vector<vector<int>> seg_index;
	vector<Point2f> contour0;
	for (int a = 0; a < contour.size(); a++)
	{
		Point2f temp;
		temp.x = contour[a].x;
		temp.y = contour[a].y;
		contour0.push_back(temp);
	}
	seg_contours.push_back(contour0);
	contour_segmentation(seg_contours, seg_index, intersect_index, segment_pairs);

	double def = opt.D_compute(double angle,  seg_contours,  seg_index, segment_pairs);
	return def;
}
