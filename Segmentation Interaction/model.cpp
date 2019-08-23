#include "model.h"
#include "Deformation.h"
#include <iostream>
#include <stdlib.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
double dis_compute(Point3f a,Point3f b)
{
	double res;
	res = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
	return res;
}
Model::Model() {}
Model::~Model() {}
void Model::generate(vector<vector<int>>concavity,vector<vector<Point3f>>seg_contours_3,
	vector<vector<Point3f>> segment_pairs_3, int contour_index,double L,double W,double H, vector<Point3f>& vertex, vector<vector<int>>& face)
{
	//concavity  ������ÿ����������Ԫ����һ����Ԫ��ʾ����index���ڶ�����Ԫ��ʾ�ָ���index����������Ԫ��ʾ�ؽڰ�͹�ԣ��ؽ�Ϊ������Ϊ0���ؽ�Ϊ͹����Ϊ1
	//seg_contours_3  ��������������
	//segment_pairs_3  ����ָ��ߵ�����
	// contour_index ��ǰ���������������
	for (int i = 0; i < concavity.size(); i++)
	{
		if (concavity[i][0] == contour_index)
		{
			if (concavity[i][2] == 0)//˵��Ϊ��
			{
				//����ǰ�ָ��ߵ�ֱ�߷����е�����ϵ��
				Point3f p1= segment_pairs_3[concavity[i][1]][0];
				Point3f p2 = segment_pairs_3[concavity[i][1]][1];
				double A = p2.y - p1.y;
				double B = p1.x - p2.x;
				double C = p2.x*p1.y - p1.x*p2.y;
				//����ָ������ĵ������
				Point3f mid;
				mid.x = (p1.x + p2.x) / 2;
				mid.y = (p1.y + p2.y) / 2;
				mid.z = (p1.z + p2.z) / 2;
				//r1��r2�ֱ��ʾ�ָ����ϵĹؽ��ϵ���������
				Point3f r1, r2;
				r1.y = abs(p1.y - p2.y)*(W / 2) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) + mid.y;
				r2.y = mid.y- abs(p1.y - p2.y)*(W / 2) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) ;
				if (p1.y == p2.y)
				{
					r1.x = mid.x - W / 2;
					r2.x = mid.x + W / 2;
				}
				else
				{
					r1.x = (-C - B*r1.y) / A;
					r2.x= (-C - B*r2.y) / A;
				}
				r1.z = 0;
				r2.z = 0;
				//������Ϊ��ȷ��p1��r1�Ǿ��������һ�ԣ�p2��r2�������һ��
				double res1 = dis_compute(p1, r1);
				double res2 = dis_compute(p1, r2);
				if (res1 > res2)
				{
					Point3f temp = r1;
					r1 = r2;
					r2 = temp;
				}
				//�����Ǽ��㳤���������������������
				Point3f r3, r4, r5, r6;
				if (r1.x == r2.x)
				{
					r3.x = r1.x + L;
					r4.x = r1.x - L;
					r3.y = r1.y;
					r4.y = r1.y;
					r5.x = r2.x + L;
					r6.x = r2.x - L;
					r5.y = r2.y;
					r6.y = r2.y;
				}
				else if (r1.y == r2.y)
				{
					r3.y = r1.y + L;
					r4.y = r1.y - L;
					r3.x = r1.x;
					r4.x = r1.x;
					r5.y = r2.y + L;
					r6.y = r2.y - L;
					r5.x = r2.x;
					r6.x = r2.x;
				}
				else
				{
					double k = -1 / ((p1.y-p2.y)/(p1.x-p2.x));
					r3.x = abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))*L + r1.x;
					r4.x = r1.x - abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))*L;
					r3.y = -1 / ((p1.y - p2.y) / (p1.x - p2.x))*(r3.x - r1.x) + r1.y;
					r4.y= -1 / ((p1.y - p2.y) / (p1.x - p2.x))*(r4.x - r1.x) + r1.y;
					r5.x= abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))*L + r2.x;
					r6.x= r2.x - abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))*L;
					r5.y= -1 / ((p1.y - p2.y) / (p1.x - p2.x))*(r5.x - r2.x) + r2.y;
					r6.y= -1 / ((p1.y - p2.y) / (p1.x - p2.x))*(r6.x - r2.x) + r2.y;
				}
				//�������ж�r3 r4 r5 r6����������λ�ڵ�ǰcontour�ڲ���������λ���ⲿ���ⲿ����Ҫ����,���ֻʣ��r3,r5
				Deformation def;
				if (def.is_in_contour(r4, seg_contours_3[contour_index]))
				{
					r3 = r4;
				}
				if (def.is_in_contour(r6,seg_contours_3[contour_index]))
				{
					r5 = r6;
				}
				r3.z = 0;
				r5.z = 0;
				//�ؽڵ��ĸ������ζ���Ϊr1,r2,r3,r5
				for (int j = 0; j < seg_contours_3[contour_index].size(); j++)
				{
					if (seg_contours_3[contour_index][j] == p1)
					{
						if (seg_contours_3[contour_index][(j + 1)] == p2)
						{
							vector<Point3f> insert_arr;
							insert_arr.push_back(r1);
							insert_arr.push_back(r3);
							insert_arr.push_back(r5);
							insert_arr.push_back(r2);
							seg_contours_3[contour_index].insert(seg_contours_3[contour_index].begin() + j+1,insert_arr.begin(), insert_arr.end());
						}
						else
						{
							vector<Point3f> insert_arr;
							insert_arr.push_back(r2);
							insert_arr.push_back(r5);
							insert_arr.push_back(r3);
							insert_arr.push_back(r1);
							seg_contours_3[contour_index].insert(seg_contours_3[contour_index].end(),insert_arr.begin(),insert_arr.end());
						}
						break;
					}
					else if(seg_contours_3[contour_index][j] == p2)
					{
						if (seg_contours_3[contour_index][(j + 1)] == p1)
						{
							vector<Point3f> insert_arr;
							insert_arr.push_back(r2);
							insert_arr.push_back(r5);
							insert_arr.push_back(r3);
							insert_arr.push_back(r1);
							seg_contours_3[contour_index].insert(seg_contours_3[contour_index].begin() + j+1, insert_arr.begin(), insert_arr.end());
						}
						else
						{
							vector<Point3f> insert_arr;
							insert_arr.push_back(r1);
							insert_arr.push_back(r3);
							insert_arr.push_back(r5);
							insert_arr.push_back(r2);
							seg_contours_3[contour_index].insert(seg_contours_3[contour_index].end(), insert_arr.begin(), insert_arr.end());
						}
						break;
					}
				}
			}
		}
	}
	//vector<Point3f> vertex;
	for (int i = 0; i < seg_contours_3[contour_index].size(); i++)
	{
		vertex.push_back(seg_contours_3[contour_index][i]);
	}
	for (int i = 0; i < seg_contours_3[contour_index].size(); i++)
	{
		Point3f p;
		p.x = seg_contours_3[contour_index][i].x;
		p.y = seg_contours_3[contour_index][i].y;
		p.z = H;
		vertex.push_back(p);
	}
	string filename = "contour0.obj";
	ofstream outfile1(filename, ios::out);
	//ע��������.obj�ļ���ʱ�������Ϣ���ɶ����±�+1��ɵģ��Ǵ�1��ʼ�ģ������������ж����±���ɣ�Ҳ���ǲ�������0��ʼ������
	if (!outfile1)
	{
		cerr << "open error";
		exit(1);
	}

	outfile1 << "#List of geometric vertices, with (x,y,z) coordinates" << endl;

	for (int c = 0; c < vertex.size(); c++)
	{
		outfile1 << "v" << " " << vertex[c].x << " " << vertex[c].y << " " << vertex[c].z << endl;
		
	}
	outfile1 << "#Polygonal face element" << endl;
	
	//vector<vector<int>> face;
	vector<int> f;
	int c;
	for (c =0; c<seg_contours_3[contour_index].size(); c++)
	{
		f.push_back(c + 1);
	}
	face.push_back(f);
	f.clear();
	for (int d = seg_contours_3[contour_index].size()-1; d>=0; d--)
	{
		f.push_back(c +d+1);
	}
	face.push_back(f);
	f.clear();
	for (c = 0; c < seg_contours_3[contour_index].size(); c++)
	{
		f.push_back(c+1+ seg_contours_3[contour_index].size());
		f.push_back((c + 1) % (seg_contours_3[contour_index].size()) + 1 + seg_contours_3[contour_index].size());
		f.push_back((c + 1) % (seg_contours_3[contour_index].size()) + 1);
		f.push_back(c + 1);
	
		face.push_back(f);
		f.clear();
	}
	for (int d = 0; d < face.size(); d++)
	{
		outfile1 << "f" ;//ע��������.obj�ļ���ʱ�������Ϣ���ɶ����±�+1��ɵģ��Ǵ�1��ʼ�ģ������������ж����±���ɣ�Ҳ���ǲ�������0��ʼ����
		for (int e = 0; e < face[d].size(); e++)
		{
			outfile1 << " " << face[d][e];

		}
		outfile1 << endl;
	}
	outfile1.close();

}

void Model::offset(vector<vector<Point3f>>& seg_contours_3, int base_index, vector<vector<int>>& seg_index, int** adj_matrix,
	vector<vector<Point3f>>& segment_pairs_3, vector<vector<int>>& concavity,double d1)
{

	//��һ��������ƫ��d1֮���ģ�ͣ������µ�seg_contours_3, segment_pairs_3,concavity
	bool* refer_flag = new bool[seg_contours_3.size()];
	for (int i = 0; i < seg_contours_3.size(); i++)
	{
		refer_flag[i] = false;
	}

	queue<int> reference;
	reference.push(base_index);
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

			bool* tree_flag = new bool[seg_contours_3.size()];
			for (int k = 0; k < seg_contours_3.size(); k++)
				tree_flag[k] = false;
			tree_flag[refer] = true;


			//��root[j]Ϊ������������������õ������п���axis_indexΪ�������ת
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
			//����Ӧ�ý�result�е�����ƽ��
			//�������£�
			//���жϷָ�����x��ļн��Ƕ۽ǻ������,Ȼ�������Ӧ���������������������
			double detax1, detay1, detax2, detay2;
			Point3f p1 = segment_pairs_3[axis_index][0];
			Point3f p2 = segment_pairs_3[axis_index][1];
			if (p1.x == p2.x)
			{
				detax1 = d1;
				detay1 = 0;
				detax2 = -d1;
				detay2 = 0;
			}
			else if (p1.y == p2.y)
			{
				detax1 = 0;
				detay1 = d1;
				detax2 = 0;
				detay2 = -d1;
			}
			else if ((p1.y - p2.y) / (p1.x - p2.x) > 0)
			{
				//˵�������
				detax1 = d1*(abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detax2 = -d1*(abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detay1 = -d1*(abs(p1.x - p2.x) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detay2 = d1*(abs(p1.x - p2.x) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
			}
			else if ((p1.y - p2.y) / (p1.x - p2.x) < 0)
			{
				//˵���Ƕ۽�
				detax1 = d1*(abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detax2 = -d1*(abs(p1.y - p2.y) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detay1 = d1*(abs(p1.x - p2.x) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
				detay2 = -d1*(abs(p1.x - p2.x) / sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
			}
			//Ȼ�����������������ҵ����±�Ϊrefer��contour��������ĵ��������
			double detax, detay;
			Point3f mid;
			mid.x = (p1.x + p2.x) / 2;
			mid.y = (p1.y + p2.y) / 2;
			mid.z = (p1.z + p2.z) / 2;
			Point3f mid1, mid2;
			mid1.x = mid.x + detax1;
			mid2.x = mid.x + detax2;
			mid1.y = mid.y + detay1;
			mid2.y = mid.y + detay2;
			Point3f temp;
			for (int q = 0; q < seg_contours_3[refer].size(); q++)
			{
				if (seg_contours_3[refer][q] == p1)
				{
					if (seg_contours_3[refer][(q + 1) % seg_contours_3[refer].size()] == p2)
					{
						temp = seg_contours_3[refer][(q + seg_contours_3[refer].size() - 1) % seg_contours_3[refer].size()];
					}
					else
						temp = seg_contours_3[refer][(q + 1) % seg_contours_3[refer].size()];
					break;
				}
			}

			Point3f v1, v2, v3, v4;
			v1.x = temp.x - p1.x;
			v1.y = temp.y - p1.y;
			v1.z = temp.z - p1.z;
			v2.x = mid1.x - p1.x;
			v2.y = mid1.y - p1.y;
			v2.z = mid1.z - p1.z;
			v3.x = p2.x - p1.x;
			v3.y = p2.y - p1.y;
			v3.z = p2.z - p1.z;
			double res1 = v2.x*v3.y - v2.y*v3.x;
			double res2 = v1.x*v3.y - v1.y*v3.x;
			if (res1*res2 > 0)
			{
				detax = detax2;
				detay = detay2;
			}
			else
			{
				detax = detax1;
				detay = detay1;
			}
			//������result�еĵ�����detax,detay
			for (int g = 0; g < result.size(); g++)
			{
				for (int h = 0; h < seg_contours_3[result[g]].size(); h++)
				{
					Point3f new_point;
					new_point.x = seg_contours_3[result[g]][h].x + detax;
					new_point.y = seg_contours_3[result[g]][h].y + detay;
					new_point.z = seg_contours_3[result[g]][h].z;
					seg_contours_3[result[g]][h] = new_point;
				}
			}

			//�������ҵ�����result�а����ķָ��ߣ�����������detax��detay
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
						//��Ӧ�ָ���ҲҪƽ��
						Point3f new_point1, new_point2;
						new_point1.x = segment_pairs_3[seg_index[y][1]][0].x + detax;
						new_point1.y = segment_pairs_3[seg_index[y][1]][0].y + detay;
						new_point1.z = segment_pairs_3[seg_index[y][1]][0].z;
						new_point2.x = segment_pairs_3[seg_index[y][1]][1].x + detax;
						new_point2.y = segment_pairs_3[seg_index[y][1]][1].y + detay;
						new_point2.z = segment_pairs_3[seg_index[y][1]][1].z;
						segment_pairs_3[seg_index[y][1]][0] = new_point1;
						segment_pairs_3[seg_index[y][1]][1] = new_point2;
					}
				}
			}

			//�������޸�segment_pairs_3�����concavity����
			Point3f new_p1, new_p2;
			new_p1.x = segment_pairs_3[axis_index][0].x + detax;
			new_p1.y = segment_pairs_3[axis_index][0].y + detay;
			new_p1.z = segment_pairs_3[axis_index][0].z;
			new_p2.x = segment_pairs_3[axis_index][1].x + detax;
			new_p2.y = segment_pairs_3[axis_index][1].y + detay;
			new_p2.z = segment_pairs_3[axis_index][1].z;
			vector<Point3f> new_segment_pair;
			new_segment_pair.push_back(new_p1);
			new_segment_pair.push_back(new_p2);
			segment_pairs_3.push_back(new_segment_pair);
			for (int r = 0; r < concavity.size(); r++)
			{
				if (concavity[r][1] == axis_index)
				{
					if (concavity[r][0] != refer)
					{
						concavity[r][1] = segment_pairs_3.size() - 1;
					}
				}
			}
			reference.push(root[j]);
		}
	}




}