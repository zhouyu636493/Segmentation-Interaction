#include <nlopt.h>
#include <optimization.h>

vector<Point2f> compute_intersect(vector<Point2f> contour, vector<Point2f> points, vector<vector<int>>& intersect_index)
{
	//�����߶�points��contour�Ľ���
	vector<Point2f> intersect;
	vector<int> tep;
	//tep�д�ŵ��ǽ���������contour�еıߵ�index������ԭ���������contour����Ϊһ���������㣬ÿ������λ������index֮�䣬����tep�����ĸ�Ԫ��
	//tep�е�ǰ����Ԫ���Ǵ�С�������򣬺�����Ҳ�Ǵ�С�����������Ƿֱ��������contour�ߵ�index
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
			}//ȷ��ÿ��index�Ǵ�С��������
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
					continue;//���ཻ
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
	//��ȡ�ָ�������ĺ���
	//���������֤һ��������δ���֮ǰ��segment_pairs��������δ���֮���segment_pair�Ƿ���ȫ��ȣ��ǵĻ��Ͷ��ˡ�
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
					int v1 = intersect_index[k][0];//��С�ķ���ת
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
						contour0_temp.push_back(contour_temp[g]);//��˳ʱ������ʱ��
					}




					contour_temp.clear();
					int v2 = intersect_index[k][1];//���ķ���ת,ԭ��������ʱ��
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
					//����contour0_temp��contour_temp���Ƿָ��������������Ҷ�����ʱ�����  
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
					//��ѯ�����ɵ��������ĸ�����seg_index[y][1]�ָ���
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
			seg_contours[e].clear();//�������ֵ�contour���
		}
	}
}






double myfunc(int n,const vector<vector<Point2f>> var,double *grad, vector<Point2f> contour)
{
	//var��һ����ά���飬����Ϊ2��n�У�n��ʾ�ָ��������������һ�б�ʾ�ָ�����ĳһ�㣬����ڶ��б�ʾ�ָ��ߵķ�������
	Optimization opt;
	vector<vector<int>>  intersecting_index;

	vector<vector<Point2f>>segment_pairs = opt.Seg_compute(var,  contour, intersecting_index);
	//��Ҫע�����Seg_compute�м����intersecting_index������±����������contour���Ե�
	/*
	�Ż��е�Լ����
	1.���ֱ�������ν���Ϊ0��
	2.��������ָ��ߵĽ����ڶ�����ڲ���
	3.������ɵ��������Ų���������
	*/
	vector<vector<int>> intersect_index; //�洢����segment_pairs����Ӧ�ָ��ߺ������ཻ�ıߵĶ���index��ÿ������������ĸ�����
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
