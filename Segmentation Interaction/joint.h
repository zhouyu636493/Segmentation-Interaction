#pragma once
#include "joint.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>//���ͷ�ļ�һ��Ҫ�У��������ƽ��ϸ�ֵĺ���
using namespace std;
using namespace cv;
#define PI 3.1415926
class Joint {
private:
	double L, d1, d2, R, H;
public:
	Joint();
	~Joint();
	Joint(double a, double b, double c, double d, double e);
	double angle();//���㵱ǰ�ؽڿ�����ת�����Ƕ�
	void hollow_joint(double  L, double  d, double r, double H, double W, vector<Point3f>& hollow_joint_vertex, vector<vector<int>>& hollow_joint_face);
	void cylinder(double r, double W, vector<Point3f>&cylinder_vertex, vector<vector<int>>& cylinder_face);
};