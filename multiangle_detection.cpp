#include<opencv2/core/core.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
vector<Rect> faces(10000);
int main()
{
	faces.clear();
	Mat face = imread("facee.jpg");
	if (!face_cascade.load(face_cascade_name))
	{
		printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
		return -1;
	}

	Mat face_gray, rotMat;
	copyMakeBorder(face, face, face.rows*0.1, face.rows*0.1, face.cols*0.1, face.cols*0.1, BORDER_CONSTANT, Scalar(255, 255, 255));
	cvtColor(face, face_gray, CV_BGR2GRAY);
	equalizeHist(face_gray, face_gray);
	imshow("asff", face);
	int times = 12;
	while (times--)
	{
		Point center = Point(face.rows / 2, face.cols / 2);
		double angle = 30;
		rotMat = getRotationMatrix2D(center, angle, 1);
		warpAffine(face, face, rotMat, face.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

		rotMat = getRotationMatrix2D(center, angle, 1);
		warpAffine(face_gray, face_gray, rotMat, face_gray.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

		face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
		for (int i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(0, 233, 0), 2, 7, 0);
		}
	}

	imshow("人脸检测", face);
	waitKey(0);
	system("pause");
}

