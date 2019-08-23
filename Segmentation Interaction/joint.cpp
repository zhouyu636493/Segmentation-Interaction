#include "joint.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>//���ͷ�ļ�һ��Ҫ�У��������ƽ��ϸ�ֵĺ���
using namespace std;
using namespace cv;
Joint::Joint() {}
Joint::~Joint() {}
Joint::Joint(double a, double b, double c, double d, double e) :
	L(a), d1(b), d2(c), R(d), H(e) {}

double Joint::angle()
{
	if (d1 + d2 + R >= sqrt(pow((H / 2), 2) + pow((d2 + R), 2)))
	{
		if (L >= d1 / 2 + pow(H, 2) / (8 * d1) + d2 + R)
		{
			std::cout << "��ת�Ƕȳ���90��" << std::endl;
			return (PI / 2 + atan((d1 + d2 + R) / (H / 2)) - asin((H / 2) / sqrt(pow(H / 2, 2) + pow(d1 + d2 + R, 2))))*(180 / PI);
		}
		else
		{
			std::cout << "�ؽ��ұ߿�ס" << std::endl;
			return(PI / 2 - acos((L - d2 - R) / sqrt(pow(H / 2, 2) + pow((L - d1 - d2 - R), 2))) - atan(2 * (L - d1 - d2 - R) / H))*(180 / PI);
		}
	}
	else
	{
		if (L >= d1 / 2 + pow(H, 2) / (8 * d1) + d2 + R)
		{
			std::cout << "�ؽ���߿�ס" << std::endl;
			return(PI / 2 - acos((R + d2 + d1) / sqrt(pow(H / 2, 2) + pow(R + d2, 2))) - atan(2 * (R + d2) / H))*(180 / PI);
		}
		else
		{
			if ((PI / 2 - acos((L - d2 - R) / sqrt(pow(H / 2, 2) + pow((L - d1 - d2 - R), 2))) - atan(2 * (L - d1 - d2 - R) / H)) <= (PI / 2 - acos((R + d2 + d1) / sqrt(pow(H / 2, 2) + pow(R + d2, 2))) - atan(2 * (R + d2) / H)))
			{
				std::cout << "�ؽ��ұ߿�ס" << std::endl;
				return(PI / 2 - acos((L - d2 - R) / sqrt(pow(H / 2, 2) + pow((L - d1 - d2 - R), 2))) - atan(2 * (L - d1 - d2 - R) / H))*(180 / PI);
			}
			else
			{
				std::cout << "�ؽ���߿�ס" << std::endl;
				return(PI / 2 - acos((R + d2 + d1) / sqrt(pow(H / 2, 2) + pow(R + d2, 2))) - atan(2 * (R + d2) / H))*(180 / PI);
			}
		}
	}
}

//����������ɵĹؽ��У�������������ľ�������ϵ��ԭ��
void Joint::hollow_joint(double  L,double  d,double r,double H,double W, vector<Point3f>& hollow_joint_vertex, vector<vector<int>>& hollow_joint_face)
{
	//����H����ģ�͵ĺ�ȣ�W����ؽڵĿ��
	//��������Χ�ĳ������ϵĶ���
	vector<Point3f> vertex;
	Point3f v1(0,-H/2,-W/2);
	Point3f v2(L,-H/2,-W/2);
	Point3f v3(L,H/2,-W/2);
	Point3f v4(0,H/2,-W/2);
	Point3f v5(0, -H / 2, W / 2);
	Point3f v6(L, -H / 2, W / 2);
	Point3f v7(L, H / 2, W / 2);
	Point3f v8(0, H / 2, W / 2);
	vertex.push_back(v1);
	vertex.push_back(v2);
	vertex.push_back(v3);
	vertex.push_back(v4);
	vertex.push_back(v5);
	vertex.push_back(v6);
	vertex.push_back(v7);
	vertex.push_back(v8);
	//�������������Բ���ϵĶ���
	for (int i = 0; i < 12; i++)
	{
		Point3f p;
		p.x = (d + r) + r*cos(2*PI/12*i);
		p.y = r*sin(2*PI/12*i);
		p.z = -W / 2;
		vertex.push_back(p);
	}
	for (int i = 0; i < 12; i++)
	{
		Point3f p;
		p.x = (d + r) + r*cos(2 * PI / 12 * i);
		p.y = r*sin(2 * PI / 12 * i);
		p.z = W / 2;
		vertex.push_back(p);
	}
	//��������Χ�������ϵ���
	vector<vector<int>> face;
	vector<int> f1{1,2,6,5 };
	vector<int> f2{2,3,7,6};
	vector<int> f3{3,4,8,7};
	vector<int> f4{1,5,8,4};
	face.push_back(f1);
	face.push_back(f2);
	face.push_back(f3);
	face.push_back(f4);
	//�������������Բ���ϵ���
	for (int i = 9; i < 21; i++)
	{
		vector<int> f;
		f.push_back(i + 12);
		f.push_back((i + 1 - 9) % 12 + 9 + 12);
		f.push_back((i + 1 - 9) % 12 + 9);
		f.push_back(i);
		face.push_back(f);
	}
	//��������Χ����������������Ƕ��Բ������
	vector<int> f5{5,6,21,32,31,30,29,28,27};
	vector<int> f6{6,7,21};
	vector<int> f7{21,7,8,27,26,25,24,23,22};
	vector<int> f8{5,27,8};
	face.push_back(f5);
	face.push_back(f6);
	face.push_back(f7);
	face.push_back(f8);
	vector<int> f9{1,15,16,17,18,19,20,9,2};
	vector<int> f10{3,2,9};
	vector<int> f11{3,9,10,11,12,13,14,15,4};
	vector<int> f12{1,4,15};
	face.push_back(f9);
	face.push_back(f10);
	face.push_back(f11);
	face.push_back(f12);

	string filename = "joint.obj";
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
		outfile1 << "v" << " " << vertex[c].x << " " << vertex[c].y<< " " << vertex[c].z<< endl;
	}
	outfile1 << "#Polygonal face element" << endl;
	for (int d = 0; d < face.size(); d++)
	{
		outfile1 << "f" ;
		for (int e = 0; e < face[d].size(); e++)
		{
			outfile1 <<" "<<face[d][e] ;
		}
		outfile1 << endl;
	}
	outfile1.close();
	hollow_joint_vertex = vertex;
	hollow_joint_face = face;
}


//����������ɹؽڱ��ڿղ��ֵ�Բ��
void Joint:: cylinder(double r,double W, vector<Point3f>&cylinder_vertex, vector<vector<int>>& cylinder_face)
{
	vector<Point3f> vertex;
	vector<vector<int>> face;
	//Բ��������λ������ϵ��ԭ��
	for (int i = 0; i < 12; i++)
	{
		Point3f v;
		v.x = r*cos(2 * PI / 12 * i);
		v.y = r*sin(2*PI/12*i);
		v.z = -W / 2;
		vertex.push_back(v);
	}
	for (int i = 0; i < 12; i++)
	{
		Point3f v;
		v.x = r*cos(2 * PI / 12 * i);
		v.y = r*sin(2 * PI / 12 * i);
		v.z = W / 2;
		vertex.push_back(v);
	}
	vector<int> f;
	for (int i = 12; i >=1; i--)
	{
		f.push_back(i);
	}
	face.push_back(f);
	f.clear();
	for (int i = 13; i < 25; i++)
	{
		f.push_back(i);
	}
	face.push_back(f);
	f.clear();
	for (int i = 1; i <= 12; i++)
	{	
		f.push_back(i);
		f.push_back(i % 12 + 1);
		f.push_back(i % 12 + 1 + 12);
		f.push_back(i + 12);
		face.push_back(f);
		f.clear();
	}
	string filename = "cylinder.obj";
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
	for (int d = 0; d < face.size(); d++)
	{
		outfile1 << "f";
		for (int e = 0; e < face[d].size(); e++)
		{
			outfile1 << " " << face[d][e];
		}
		outfile1 << endl;
	}
	outfile1.close();
	cylinder_vertex = vertex;
	cylinder_face = face;
}