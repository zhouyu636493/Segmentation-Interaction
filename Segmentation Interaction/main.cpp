#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_conformer_2.h>
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
#include "Deformation.h"
#include "model.h"
#include "optimization.h"
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

/*1. 关节处，外半径和内半径之差为0.4mm。
  2. 内外宽度之差为2mm。
  3. d1为2mm，d2为1mm，半径为2mm，长度L为8mm，宽度W为5mm。
  4. 厚度H 10mm。
*/




vector<cv::Point> points;
Mat dstImage;//必须有一个dstImage用来存储原始的图像，否则最后的直线会有很多中间状态的直线显示
Mat srcImage;//每次从dstImage复制过来图像，然后再此基础上实时显示
cv::Point pre;
cv::Point cur;
cv::Point pre_1(-1, -1);
vector<vector<int>> intersect_index;//存储的是segment_pairs中相应分割线和轮廓相交的边的顶点index，每个数组分量有四个分量
vector<vector<Point2i>> points_set;//存储的是用户输入的线段的两端坐标
vector<vector<Point2f>> segment_pairs;//存储的是最终分割线在轮廓上的两端坐标

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
		cv::imshow("Segmentation", srcImage);
		srcImage.copyTo(dstImage);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
	{
		dstImage.copyTo(srcImage);
		cur.x = x;
		cur.y = y;
		line(srcImage, pre, cur, Scalar(0, 255, 0), 1, CV_AA, 0);
		cv::imshow("Segmentation", srcImage);

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
		cv::imshow("Segmentation", srcImage);
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
					dstImage.at<Vec3b>(lit.pos().y, lit.pos().x) = Vec3b(255, 255, 255);
				}
				cv::imshow("Segmentation", dstImage);
				points_set.erase(points_set.begin() + i);
				break;
			}
		}

	}
	return;
}
Point3f rotate_y(Point3f p, double angle)
{
	Point3f res;
	res.x = p.z*sin(angle / 180 * PI) + p.x*cos(angle / 180 * PI);
	res.y = p.y;
	res.z = p.z*cos(angle / 180 * PI) - p.x*sin(angle / 180 * PI);
	return res;
}
Point3f rotate_z(Point3f p, double angle)
{
	Point3f res;
	res.x = p.x*cos(angle / 180 * PI) - p.y*sin(angle / 180 * PI);
	res.y = p.x*sin(angle / 180 * PI) + p.y*cos(angle / 180 * PI);
	res.z = p.z;
	return res;

}
//下面这段代码是聪哥的，比之前的想法简单，坐标轴旋转就可以！！！！
/*vector<Vec3> Mesh::meshRotate(Vec3 z_dire, vector<Vec3> point)
{
z_dire.Normalized();

Vec3 x_dire, y_dire;

float t = powf(z_dire.m_x, 2) + powf(z_dire.m_y, 2);

if (t == 0)
{
x_dire.m_x = z_dire.m_z;
}
else
{
x_dire.m_x = -z_dire.m_y;
x_dire.m_y = z_dire.m_x;
x_dire.Normalized();
}

y_dire = Cross(z_dire, x_dire);

vector<Vec3> result;
for (int i = 0; i < point.size(); i++)
{
Vec3 temp = point[i].m_x*x_dire + point[i].m_y*y_dire + point[i].m_z*z_dire;
result.push_back(temp);
}

return result;
}

vector<Vec3> Mesh::meshRotate(Vec3 x_dire, Vec3 z_dire, vector<Vec3> point)
{
x_dire.Normalized();
z_dire.Normalized();

Vec3 y_dire;
y_dire = Cross(z_dire, x_dire);

vector<Vec3> result;
for (int i = 0; i < point.size(); i++)
{
Vec3 temp = point[i].m_x*x_dire + point[i].m_y*y_dire + point[i].m_z*z_dire;
result.push_back(temp);
}

return result;
}*/
Point3f normalized(Point3f a)
{
	Point3f b;
	b.x = a.x / sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
	b.y=a.y/ sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
	b.z = a.z / sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
	return b;
}

Point3f rotate(Point3f x_dire, Point3f z_dire, Point3f p)
{
	//单位向量正交，则叉乘结果一定还是单位向量
	Point3f xdire = normalized(x_dire);
	Point3f zdire = normalized(z_dire);
	Point3f ydire;
	//y轴应该是z叉乘x,因为在建模的时候一直用的是右手系，如果这里用了x叉乘z，则结果就不对
	
	ydire.x = zdire.y*xdire.z - zdire.z*xdire.y;
	ydire.y = zdire.z*xdire.x - zdire.x*xdire.z;
	ydire.z = zdire.x*xdire.y - zdire.y*xdire.x;
	Point3f res;
	res.x = p.x*xdire.x + p.y*ydire.x + p.z*zdire.x;
	res.y = p.x*xdire.y + p.y*ydire.y + p.z*zdire.y;
	res.z = p.x*xdire.z + p.y*ydire.z + p.z*zdire.z;
	return res;
}

int main(int argc, char *argv[])
{
	string name;//图片编号
	Point2f center;//轮廓中心
	//下面是用户输入关节的相关参数
	double L, d1, d2, R, H, W;//W代表关节的宽度，在计算旋转角度的时候W不需要考虑
	std::cout << "请输入关节的参数" << endl;
	cin >> L >> d1 >> d2 >> R >> H >> W;
	while (L <= 0 || d1 <= 0 || d2 <= 0 || R <= 0 || H <= 0)
	{
		cerr << "关节参数都应该大于0,请重新输入关节的参数";
		cin >> L >> d1 >> d2 >> R >> H >> W;
	}
	
	while (L - d2 - 2 * R - d1 < -DBL_EPSILON)//因为double类型数据相减的时候存在精度问题，所以要和DBL_EPSILON比较，而不是单单和0比较
	{
		cerr << "L-d2-2*R-d1要大于等于0,请重新输入关节的参数";
		cin >> L >> d1 >> d2 >> R >> H >> W;
	}
	Joint j1(L, d1, d2, R, H);
	double angle1 = j1.angle();
	std::cout << "当前关节可以旋转的最大角度为：" << angle1 << endl;




	/*double angles[2];
	angles[0] = 0.3490659;
	angles[1] = 0.6981317007977318;*/
	//Input the polygon
	std::cout << "请输入要处理的图片编号" << endl;
	cin >> name;
	Contour c;
	//下面是提取形状轮廓
	string name1 = "F:/Project/Segmentation Interaction/dataset/" + name + ".png";//这个字符串的读取没有问题
	Mat img = imread(name1, 1);//按三通道方式读入图像，即彩色图像





	Mat de_noise = c.image(img);
	vector<vector<Point2i>>  contours = c.contour_generate(de_noise);
	//绘制轮廓图
	Mat  dstImage0,dstImage2;
	dstImage0 = Mat::zeros(de_noise.size(), CV_8UC3);





	resize(dstImage0, dstImage2, Size(dstImage0.cols * 1.2, dstImage0.rows * 1.2), 0, 0, INTER_LINEAR);	




		
	vector<cv::Point> contour;
	approxPolyDP(contours[0], contour, 5, true);//多边形简化轮廓


	//下面代码是为了缩小轮廓
	Rect rect = boundingRect(contour);
	std::cout << rect.width << " " << rect.height << endl;
	//rect.width代表x轴，rect.height代表y轴

	//下面代码是为了缩小轮廓
	/*resize(dstImage0, dstImage2, Size(dstImage0.cols * 230/ rect.width, dstImage0.rows * 180/ rect.height), 0, 0, INTER_LINEAR);*/


	dstImage = cv::Mat::zeros(dstImage2.rows, dstImage2.cols, CV_8UC3);
	dstImage.setTo(cv::Scalar(255, 255, 255));//设置背景颜色为白色


 //   //下面代码也是为了缩小轮廓
	//for (int i = 0; i < contour.size(); i++)
	//{
	//	contour[i].x = contour[i].x * 230 / rect.width;
	//	contour[i].y= contour[i].y * 180 / rect.height;
	//}



	polylines(dstImage, contour, true, (0, 0, 255),2);//画出简化后的多边形，同时也是之后交互的画布
	


	Mat dstImage1 = dstImage.clone();//之后用来显示三角剖分的画布
	Moments mu = moments(contour, false);
	Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
	center = mc;//这个地方的center可以是用户在多边形上任意pick一个点
	circle(dstImage, center, 4, Scalar(0, 0, 255), CV_FILLED, CV_AA, 0);//画出中心点坐标
	namedWindow("Segmentation", CV_WINDOW_AUTOSIZE);
	cv::imshow("Segmentation",dstImage);
	setMouseCallback("Segmentation", onMouse, (void*)&contour);
	char key =cv:: waitKey(0);
	
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
	Deformation def;
	def.contour_segmentation(seg_contours, seg_index, intersect_index, points_set,segment_pairs);



 //下面是基于我设计的形变量formulation计算形变量具体数值
	Optimization opt;
	double deformation=opt.D_compute(angle1 / 180 * PI,  seg_contours,  seg_index, segment_pairs);
	cout << "The deformation of this model is" << deformation << endl;

//下面应该是对分割线基于梯度下降不断优化
//1.这里第一步应该是当点和方向改变之后求出相应的直线对多变形进行分割的分割线
//如果有直线没有对多边形分割，则将该直线平移到任意两个分割线之间距离最大的两个分割线中间；
//2. 判断任意两个分割线是否在多边形内相交，如果是则将交点投影到距离该点最近的多边形边上；
//3. 判断任意两个相邻的分割线之间能否放的下一个关节，如果不能










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
	cv::imshow("Segmentation", dstImage3);
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
						/*cout << s << endl;
						cout << v << endl;
						cout << seg_contours.size() << endl;
						cout << seg_index[s][0] << endl;
						cout << seg_index[v][0] << endl;*/
						int a = seg_index[s][0];
						int b = seg_index[v][0];
						adj_matrix[a][b] = 1;
						adj_matrix[b][a] = 1;
					}
				}
			}
		}
	}

	cv::imwrite("output.bmp", dstImage3);
	cv::imshow("Segmentation", dstImage3);
	cv::waitKey(0);

	//用户输入一个点center，判断该点位于哪个contour中，如果找到相应contour则break 
	int base_index;
	//静止contour的编号
	for ( base_index = 0; base_index < seg_contours.size(); base_index++)
	{
		if (!seg_contours[base_index].empty()) {
			if (def.is_in_contour(center, seg_contours[base_index]))
				break;
		}	
	}
	

	//以下seg_contours_3和segment_pairs_3在生成的时候，之所以x轴缩放230 / rect.width，y轴缩放180 / rect.height，是因为弘瑞打印机
	//的x轴和y轴的大小是分别限制在240和190mm的，而rect是计算出的轮廓的外接矩形，因此要乘以这两个缩放因子使得生成的模型可以不发生碰撞错误。
	vector<vector<Point3f>> seg_contours_3;
	for (int w = 0; w < seg_contours.size(); w++)
	{
		vector<Point3f> temp;
		for (int z = 0; z < seg_contours[w].size(); z++)
		{
			Point3f temp1;
			temp1.x = seg_contours[w][z].x* 120 / rect.width;
			temp1.y = seg_contours[w][z].y* 100 / rect.height;
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
			temp1.x = segment_pairs[w][z].x * 120 / rect.width;
			temp1.y = segment_pairs[w][z].y *100 / rect.height;
			temp1.z = 0;
			temp.push_back(temp1);
		}
		segment_pairs_3.push_back(temp);
	}//将segment_pairs原本的二维变成三维
	
	/*for (int ang = 0; ang < 2; ang++)
	{
		queue<int> reference;
		reference.push(base_index);
		angle1 = angles[ang];
		def.rotated_contour_compute(reference, seg_contours_3, segment_pairs_3, angle1,
			adj_matrix, seg_index,ang);
	}*/
	
	queue<int> reference;
	reference.push(base_index);
	def.rotated_contour_compute(reference, seg_contours_3, segment_pairs_3, angle1/180*PI,
		adj_matrix, seg_index, 0);

	



	//静止块的边上的关节是凸出去的
    //关节的位置都是位于segment_pair的中点
	//广度优先搜索，生成标记有凹凸性的数组concavity
	vector<vector<int>> concavity = def.concavity(seg_contours_3,segment_pairs_3,seg_index, base_index, adj_matrix);
    

	
	Model m;
	
	//这一段是生成偏移d1之后的模型，计算新的seg_contours_3, segment_pairs_3,concavity
	m.offset(seg_contours_3, base_index, seg_index, adj_matrix,
		 segment_pairs_3, concavity,d1);

	//下面是采用广度优先搜索，生成每个contour的三维模型

	bool* visited_contour = new bool[seg_contours_3.size()];
	queue<int> seg_visited;
	vector<Point3f> vertex_sum;
	vector<vector<int>> face_sum;
	
	int sum_vertex = 0;
	queue<int> empty;
	swap(empty, seg_visited);
	for (int p = 0; p < seg_contours_3.size(); p++)
		visited_contour[p] = false;
	seg_visited.push(base_index);
	visited_contour[base_index] = true;
	while (!seg_visited.empty())//还是采用广度优先搜索，生成每个contour的三维模型
	{
		int cur = seg_visited.front();
		seg_visited.pop();
		//下面是生成当前下标为cur的contour的三维模型
		vector<Point3f> vertex;
		vector<vector<int>> face;
		m.generate(concavity,seg_contours_3, segment_pairs_3, cur,L,W,H,vertex,face);
		//生成模型后还应该拼接相应关节
		for (int i = 0; i < vertex.size(); i++)
		{
			vertex_sum.push_back(vertex[i]);
		}
		for (int i = 0; i < face.size(); i++)
		{
			vector<int> temp;
			for (int j = 0; j < face[i].size(); j++)
			{
				temp.push_back(face[i][j]+ sum_vertex);
			}
			face_sum.push_back(temp);
		}
		sum_vertex = sum_vertex + vertex.size();
		for (int h = 0; h < seg_contours_3.size(); h++)
		{
			if (adj_matrix[cur][h] == 1&&!seg_contours_3[h].empty())
			{
				if (!visited_contour[h])
				{
					seg_visited.push(h);
					visited_contour[h] = true;
				}
			}
		}
	}


	//下面生成的是关节，点数组和面数据都存储在相应数组中
	vector<Point3f> hollow_joint_vertex, cylinder_vertex;
	vector<vector<int>> hollow_joint_face, cylinder_face;
	j1.hollow_joint(L, d2, R, H, W-2, hollow_joint_vertex, hollow_joint_face);
	j1.cylinder(R - 0.4, W, cylinder_vertex, cylinder_face);
	for (int i = 0; i < cylinder_vertex.size(); i++)
	{
		cylinder_vertex[i].x = cylinder_vertex[i].x + d2 + R;
	}
	vector<Point3f> joint_vertex;
	vector<vector<int>> joint_face;
	for (int i = 0; i < hollow_joint_vertex.size(); i++)
	{
		joint_vertex.push_back(hollow_joint_vertex[i]);
	}
	for (int i = 0; i < hollow_joint_face.size(); i++)
	{
		vector<int> temp;
		for (int j = 0; j < hollow_joint_face[i].size(); j++)
		{
			temp.push_back(hollow_joint_face[i][j]);
		}
		joint_face.push_back(temp);
	}
	for (int i = 0; i < cylinder_vertex.size(); i++)
	{
		joint_vertex.push_back(cylinder_vertex[i]);
	}
	for (int i = 0; i < cylinder_face.size(); i++)
	{
		vector<int> temp;
		for (int j = 0; j < cylinder_face[i].size(); j++)
		{
			temp.push_back(cylinder_face[i][j] + hollow_joint_vertex.size());
		}
		joint_face.push_back(temp);
	}

	string filename2 = "joint.obj";
	ofstream outfile2(filename2, ios::out);
	//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	if (!outfile2)
	{
		cerr << "open error";
		exit(1);
	}
	outfile2 << "#List of geometric vertices, with (x,y,z) coordinates" << endl;
	for (int c = 0; c < joint_vertex.size(); c++)
	{
		outfile2 << "v" << " " << joint_vertex[c].x << " " << joint_vertex[c].y << " " << joint_vertex[c].z << endl;
	}
	outfile2<< "#Polygonal face element" << endl;
	for (int d = 0; d <joint_face.size(); d++)
	{
		outfile2 << "f";
		for (int e = 0; e < joint_face[d].size(); e++)
		{
			outfile2 << " " << joint_face[d][e];
		}
		outfile2 << endl;
	}
	outfile2.close();
	

	
	int sum_v;
	sum_v = vertex_sum.size();
	for (int i = 0; i < seg_contours_3.size(); i++)
	{
		for (int j = 0; j < concavity.size(); j++)
		{
			if (concavity[j][0] == i)
			{
				if (concavity[j][2] == 0)
				{
					
					int index = concavity[j][1];
					Point3f p1 = segment_pairs_3[index][0];
					Point3f p2 = segment_pairs_3[index][1];
					//计算分割线中点的方向
					Point3f mid;
					mid.x = (p1.x + p2.x) / 2;
					mid.y = (p1.y + p2.y) / 2;
					mid.z = H / 2;
					double d3 = L - d1 - d2 -  R;
					Point3f unit1, unit2;
					unit1.x = -(p1.y-p2.y) / sqrt(pow((p1.y - p2.y), 2) + pow((p1.x - p2.x), 2));
					unit1.y = (p1.x - p2.x) / sqrt(pow((p1.y - p2.y), 2) + pow((p1.x - p2.x), 2));
					unit1.z = 0;
					unit2.x = (p1.y - p2.y) / sqrt(pow((p1.y - p2.y), 2) + pow((p1.x - p2.x), 2));
					unit2.y = -(p1.x - p2.x) / sqrt(pow((p1.y - p2.y), 2) + pow((p1.x - p2.x), 2));
					unit2.z = 0;
					
					Point3f nv1, nv2, nv;
					Point3f z_axis, x_axis;//用于计算分割线在轮廓中逆时针的向量方向
					Point3f pivot;//用于判断计算出的两个点哪个位于关节内部的辅助点
					//nv就是最终圆柱向关节内部偏移后的位置
					if (d3 > 0)
					{
						nv1.x = mid.x + d3*unit1.x;
						nv1.y = mid.y + d3*unit1.y;
						nv1.z = mid.z;
						nv2.x = mid.x + d3*unit2.x;
						nv2.y = mid.y + d3*unit2.y;
						nv2.z = mid.z;
					}
					else if (d3 == 0)
					{
						int dt = 1;
						nv1.x = mid.x + dt*unit1.x;
						nv1.y = mid.y + dt*unit1.y;
						nv1.z = mid.z;
						nv2.x = mid.x + dt*unit2.x;
						nv2.y = mid.y + dt*unit2.y;
						nv2.z = mid.z;
					}
					for (int g = 0; g < seg_contours_3[i].size(); g++)
					{
						if (seg_contours_3[i][g] == p1)
						{
							Point3f p3 = seg_contours_3[i][g + 1];
							Point2f v1, v2;
							v1.x = p1.x - p3.x;
							v1.y = p1.y - p3.y;
							v2.x = p2.x - p3.x;
							v2.y = p2.y - p3.y;
							if (v1.x*v2.y - v1.y*v2.x == 0) {
								pivot = seg_contours_3[i][(g + seg_contours_3[i].size() - 1) % seg_contours_3[i].size()];
								z_axis.x = p2.x - p1.x;
								z_axis.y = p2.y - p1.y;
								z_axis.z = p2.z - p1.z;
							}
							else {
								pivot = p3;
								z_axis.x = p1.x - p2.x;
								z_axis.y = p1.y - p2.y;
								z_axis.z = p1.z - p2.z;
							}
							Point3f va, vb, vc;
							va.x = pivot.x - p1.x;
							va.y = pivot.y - p1.y;
							vb.x = nv1.x - p1.x;
							vb.y = nv1.y - p1.y;
							vc.x = p2.x - p1.x;
							vc.y = p2.y - p1.y;

							if ((va.x*vc.y - va.y*vc.x)*(vb.x*vc.y - vb.y*vc.x) > 0)
							{
								if (d3 == 0)
									nv = mid;
								else
									nv = nv1;
								x_axis = unit2;
							}
							else if ((va.x*vc.y - va.y*vc.x)*(vb.x*vc.y - vb.y*vc.x) < 0)
							{
								if (d3 == 0)
									nv = mid;
								else
									nv = nv2;
								x_axis = unit1;
							}
							break;
						}
						else if (seg_contours_3[i][g] == p2)
						{
							Point3f p3 = seg_contours_3[i][g + 1];
							Point2f v1, v2;
							v1.x = p2.x - p3.x;
							v1.y = p2.y - p3.y;
							v2.x = p1.x - p3.x;
							v2.y = p1.y - p3.y;
							if (v1.x*v2.y - v1.y*v2.x == 0)
							{
								pivot = seg_contours_3[i][(g + seg_contours_3[i].size() - 1) % seg_contours_3[i].size()];
								z_axis.x = p1.x - p2.x;
								z_axis.y = p1.y - p2.y;
								z_axis.z = p1.z - p2.z;
							}
							else {
								pivot = p3;
								z_axis.x = p2.x - p1.x;
								z_axis.y = p2.y - p1.y;
								z_axis.z = p2.z - p1.z;
							}
							Point3f va, vb, vc;
							va.x = pivot.x - p2.x;
							va.y = pivot.y - p2.y;
							vb.x = nv1.x - p2.x;
							vb.y = nv1.y - p2.y;
							vc.x = p1.x - p2.x;
							vc.y = p1.y - p2.y;
							if ((va.x*vc.y - va.y*vc.x)*(vb.x*vc.y - vb.y*vc.x) > 0)
							{
								if (d3 == 0)
									nv = mid;
								else
									nv = nv1;
								x_axis = unit2;
							}
							else
							{
								if (d3 == 0)
									nv = mid;
								else
									nv = nv2;
								x_axis = unit1;
							}
							break;
						}
					}

				
					
					Point3f x_dire= x_axis;
					
					Point3f z_dire = z_axis;


					
					for (int z = 0; z <  joint_vertex.size(); z++)
					{
						Point3f new_cy_1;
						Point3f deta= rotate(x_dire, z_dire, Point3f(d2 + R,0,0));
						Point3f new_cy= rotate(x_dire, z_dire, joint_vertex[z]);
						new_cy_1.x = new_cy.x + (nv.x - deta.x);
						new_cy_1.y = new_cy.y + (nv.y - deta.y);
						new_cy_1.z = new_cy.z + (nv.z - deta.z);
						vertex_sum.push_back(new_cy_1);
					}
					for (int z = 0; z < joint_face.size(); z++)
					{
						vector <int> temp;
						for (int m = 0; m < joint_face[z].size(); m++)
						{
							temp.push_back(joint_face[z][m] + sum_v);
						}
						face_sum.push_back(temp);
					}
					sum_v = sum_v + joint_vertex.size();
					
				}
			}
		}
		
	}


	
	string filename = "model.obj";
	ofstream outfile1(filename, ios::out);
	//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！！
	if (!outfile1)
	{
		cerr << "open error";
		exit(1);
	}

	outfile1 << "#List of geometric vertices, with (x,y,z) coordinates" << endl;
	cout << vertex_sum.size() << endl;
	for (int c = 0; c < vertex_sum.size(); c++)//vertex_sum是多边形上的点，也就是各个contour的点
	{
		outfile1 << "v" << " " << vertex_sum[c].x << " " << vertex_sum[c].y << " " << vertex_sum[c].z << endl;

	}
	
	outfile1 << "#Polygonal face element" << endl;
	
	for (int d = 0; d < face_sum.size(); d++)
	{
		outfile1 << "f";//注意在生成.obj文件的时候，面的信息是由顶点下标+1组成的，是从1开始的！！！并不是有顶点下标组成，也就是并不是由0开始！！
		for (int e = 0; e < face_sum[d].size(); e++)
		{
			outfile1 << " " << face_sum[d][e];

		}
		outfile1 << endl;
	}
	
	outfile1.close();






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
		if(def.line_is_in_contour(f_iter->vertex(0)->point(), f_iter->vertex(1)->point(),contour)
		&& def.line_is_in_contour(f_iter->vertex(0)->point(), f_iter->vertex(2)->point(), contour)
		&& def.line_is_in_contour(f_iter->vertex(1)->point(), f_iter->vertex(2)->point(), contour)){
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
	cv::imshow("Segmentation", dstImage);
	cv::waitKey(0);
	return 0;
}
