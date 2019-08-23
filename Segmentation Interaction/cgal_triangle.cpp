#include <cgal_triangle.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#define POINTSIZE 60
using namespace std;
using namespace cv;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point CdtPoint;//防止和opencv中的Point混淆

CDT CTriangle::insert_edge(vector<Point2i> contour)
{
	CDT cdt;
	int n = contour.size();
	Vertex_handle va, vb, vc;
	for (int i = 0; i < n-1; i++)
	{
		 va = cdt.insert(CdtPoint(contour[i].x, contour[i].y));
		 if (i == 0)
			 vc = va;
		 vb = cdt.insert(CdtPoint(contour[i+1].x, contour[i+1].y));
		 cdt.insert_constraint(va,vb);
	}
    cdt.insert_constraint(vb,vc);
	return cdt;
}
CDT CTriangle::insert_point(CDT cdt,Mat img)
{
	for (int i = 0; i < POINTSIZE; i++)
	{

		Vertex_handle v = cdt.insert(CdtPoint((float)(rand() % (img.cols - 10)),//使点约束在距离边框10像素之内。因为之前生成的矩形为Rect(0, 0, img.cols, img.rows)    
			(float)(rand() % (img.rows - 10))));
	}
	return cdt;
}