#include<opencv2/core/core.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<stdio.h>
#include<math.h>

#define PI 3.1415926535


using namespace std;
using namespace cv;
const double center_x = 720;
const double center_y = 720;
const double eps = 1e-11;
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
vector<Rect> faces(10000);
double point_x_src, point_y_src;

void _reverse(double point_x, double point_y)
{
	double len = sqrt((point_x - center_x)*(point_x - center_x) + (point_y - center_y)*(point_y - center_y));
	double len1 = point_x - center_x, len1_abs = fabs(len1);
	double len2 = point_y - center_y, len2_abs = fabs(len2);
	if (len1 > eps)
	{
		if (len2 > eps)
		{
			double angle_tan = atan(len2_abs / len1_abs) * 180 / PI;
			if (angle_tan - 60 > eps)
			{
				angle_tan = 180 - (angle_tan + 30);
				point_x_src = center_x - len * cos(angle_tan*PI / 180);
				point_y_src = center_y + len * sin(angle_tan*PI / 180);
			}
			else if (fabs(angle_tan - 60) < eps)
			{
				point_y_src = center_y + len;
				point_x_src = center_x;
			}
			else
			{
				angle_tan = angle_tan + 30;
				point_x_src = center_x + len * cos(angle_tan*PI / 180);
				point_y_src = center_y + len * sin(angle_tan*PI / 180);
			}
		}
		else if (fabs(len2) < eps)
		{
			point_y_src = center_y + len * 0.5;
			point_x_src = center_x + sqrt(len*len - len * 0.5* len * 0.5);
		}
		else
		{
			double angle_tan = atan(len2_abs / len1_abs) * 180 / PI;
			if (angle_tan - 30 < eps)
			{
				angle_tan = 30 - angle_tan;
				point_x_src = center_x + len * cos(angle_tan*PI / 180);
				point_y_src = center_y + len * sin(angle_tan*PI / 180);
			}
			else if (fabs(angle_tan - 30)<eps)
			{
				point_x_src = center_x + len;
				point_y_src = center_y;
			}
			else
			{
				angle_tan = angle_tan - 30;
				point_x_src = center_x + len * cos(angle_tan*PI / 180);
				point_y_src = center_y - len * sin(angle_tan*PI / 180);
			}
		}
	}
	else if (fabs(len1) < eps)
	{
		if (len2 > eps)
		{
			point_x_src = center_x - 0.5*len;
			point_y_src = center_y + sqrt(len*len - len * 0.5* len * 0.5);
		}
		else if (fabs(len2) < eps)
		{
			point_x_src = center_x;
			point_y_src = center_y;
		}
		else
		{
			point_x_src = center_x + 0.5*len;
			point_y_src = center_y - sqrt(len*len - len * 0.5* len * 0.5);
		}
	}
	else
	{
		if (len2 > eps)
		{
			double angle_tan = atan(len2_abs / len1_abs) * 180 / PI;
			if (angle_tan - 30 > eps)
			{
				angle_tan = angle_tan - 30;
				point_x_src = center_x - len * cos(angle_tan*PI / 180);
				point_y_src = center_y + len * sin(angle_tan*PI / 180);
			}
			else if (fabs(angle_tan - 30) < eps)
			{
				point_x_src = center_x - len;
				point_y_src = center_y;
			}
			else
			{
				angle_tan = 30 - angle_tan;
				point_x_src = center_x - len * cos(angle_tan*PI / 180);
				point_y_src = center_y - len * sin(angle_tan*PI / 180);
			}
		}
		else if (fabs(len2) < eps)
		{
			point_y_src = center_y - len * 0.5;
			point_x_src = center_x - sqrt(len*len - len * 0.5* len * 0.5);
		}
		else
		{
			double angle_tan = atan(len2_abs / len1_abs) * 180 / PI;
			if (angle_tan - 60 >= eps)
			{
				angle_tan = 180 - (angle_tan + 30);
				point_x_src = center_x + len * cos(angle_tan*PI / 180);
				point_y_src = center_y - len * sin(angle_tan*PI / 180);
			}
			else if (fabs(angle_tan - 60) < eps)
			{
				point_y_src = center_y - len;
				point_x_src = center_x;
			}
			else
			{
				angle_tan = angle_tan + 30;
				point_x_src = center_x - len * cos(angle_tan*PI / 180);
				point_y_src = center_y - len * sin(angle_tan*PI / 180);
			}
		}
	}
}
int main()
{
	faces.clear();
	Mat face = imread("facee.jpg");
	imshow("fffa", face);
	if (!face_cascade.load(face_cascade_name))
	{
		printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
		return -1;
	}

	Mat face_gray, rotMat;
	copyMakeBorder(face, face, face.rows*0.5, face.rows*0.5, face.cols*0.5, face.cols*0.5, BORDER_CONSTANT, Scalar(255, 255, 255));
	cvtColor(face, face_gray, CV_BGR2GRAY);
	equalizeHist(face_gray, face_gray);
	//imshow("asff", face);
	for(int i=0;i<12;i++)
	{
		Point center = Point(face.rows / 2, face.cols / 2);
		double angle = 30;
		rotMat = getRotationMatrix2D(center, angle, 1);
		warpAffine(face_gray, face_gray, rotMat, face_gray.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
		//warpAffine(face, face, rotMat, face.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
		face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
		for (int j = 0; j < faces.size(); j++)
		{
			double point_x = faces[j].x + faces[j].width*0.5;
			double point_y = faces[j].y + faces[j].height*0.5;
			point_x_src = point_x;
			point_y_src = point_y;
			for (int k = 0; k <= i; k++)
			{
				_reverse(point_x_src, point_y_src);
			}
			Point center(point_x_src, point_y_src);
			ellipse(face, center, Size(faces[j].width*0.5, faces[j].height*0.5), 0, 0, 360, Scalar(0, 233, 0), 2, 7, 0);
		}
	}
	//Mat ROIimg = face(Rect(360,360,720,720));
	Mat ROIimg = face;
	imshow("人脸检测", ROIimg(Rect(360, 360, 720, 720)));
	waitKey(0);
	system("pause");
}

