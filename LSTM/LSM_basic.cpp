// LSTM.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <cmath>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
// 相关系数图像匹配
float Get_coefficient(cv::Mat matchLeftWindow, cv::Mat imageRight, int x, int y);
// 将vector内的最大数据移至首位
void vectorsort(vector<cv::Point3f> &Temp_sort);
// 合并左右核线影像，同名点连接
void lastview(cv::Mat imageLeftRGB, cv::Mat imageRightRGB, vector <cv::Point3f> featurePointLeft, vector<cv::Point3f> featurePointRight);
// sift初匹配
void alignImages(Mat im1, Mat im2, Mat& img1_wrapped, Mat& h, vector<cv::Point3f> &featurePointLeft, vector<cv::Point3f> &featurePointRight_sift);
// 最小二乘匹配，使用sift结果作为初值进行迭代
int LSM_Matching(cv::Mat ImgL, cv::Mat ImgR, std::vector<cv::Point3f> featurePointLeft, std::vector<cv::Point3f> featurePointRight_sift, cv::Mat H, std::vector<cv::Point3f>& featureLeftPointLST, std::vector<cv::Point3f>& featureRightPointLST);
// 三次样条插值函数
float Rifman(float a, float b);
int main(int argc, char ** argv)
{
	if (argc != 15)
	{
		cerr << "Wrong Input" << endl;
	}
	// 预留接口
	double distance = atof(argv[1]);
	double focal = atof(argv[2]);
	double pixlsize = atof(argv[3]);

	cv::Rect lmask(atof(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[7]));
	cv::Rect rmask(atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]));
	std::string pathL = argv[12];
	std::string pathR = argv[13];
	// Read reference image
	Mat imgL = imread(pathL);
	// Read image to be aligned
	Mat imgR = imread(pathR);

	imgL = imgL(lmask).clone();
	imgR = imgR(rmask).clone();

	Mat img_Wrapped, h;
	vector<cv::Point3f> featurePointLeft;
	vector<cv::Point3f> featurePointRight_sift;
	// Align images
	alignImages(imgL, imgR, img_Wrapped, h, featurePointLeft, featurePointRight_sift);
	cv::imwrite("imL.jpg", imgL);
	cv::imwrite("imR.jpg", imgR);
	cv::imwrite("img_Wrapped.jpg", img_Wrapped);
	std::vector<cv::Point3f> featureLeftPointLST;
	std::vector<cv::Point3f> featureRightPointLST;

	int k = LSM_Matching(imgL, imgR, featurePointLeft, featurePointRight_sift, h, featureLeftPointLST, featureRightPointLST);
	// 提取LSM匹配点
	vector<cv::Point2f> PointsL, PointsR;
	for (size_t i = 0; i < featureLeftPointLST.size(); i++)
	{
		cv::Point2f pointL, pointR;
		pointL.x = featureLeftPointLST[i].x;
		pointL.y = featureLeftPointLST[i].y;
		PointsL.push_back(pointL);
		pointR.x = featureRightPointLST[i].x;
		pointR.y = featureRightPointLST[i].y;
		PointsR.push_back(pointR);
	}
	cv::Mat H;
	double reprojectionThreshold = 0.1;
	H = findHomography(PointsL, PointsR, RANSAC, reprojectionThreshold);
	cv::Mat img_wrapped_LSM;
	warpPerspective(imgL, img_wrapped_LSM, H, imgR.size());
	imwrite("img_wrapped_LSM.jpg", img_wrapped_LSM);
	std::ofstream out(argv[14]);
	
	//out << "H by LSM :" << endl;
	out << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << endl;
	out << H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << endl;
	out << H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2) << endl;
	/*
	out << "h by SIFT :" << endl;
	out << h.at<double>(0, 0) << " " << h.at<double>(0, 1) << " " << h.at<double>(0, 2) << endl;
	out << h.at<double>(1, 0) << " " << h.at<double>(1, 1) << " " << h.at<double>(1, 2) << endl;
	out << h.at<double>(2, 0) << " " << h.at<double>(2, 1) << " " << h.at<double>(2, 2) << endl;
	*/
	/*
	out << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << " " 
		<< H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << " "
		<< H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2) << endl;

	out << h.at<double>(0, 0) << " " << h.at<double>(0, 1) << " " << h.at<double>(0, 2) << " " 
		<< h.at<double>(1, 0) << " " << h.at<double>(1, 1) << " " << h.at<double>(1, 2) << " "
		<< h.at<double>(2, 0) << " " << h.at<double>(2, 1) << " " << h.at<double>(2, 2) << endl;
	*/
	out.close();

    return 0;
}
//相关系数图像匹配
float Get_coefficient(cv::Mat matchLeftWindow, cv::Mat imageRight, int x, int y)
{
	//根据左搜索窗口确定右搜索窗口的大小
	cv::Mat Rmatchwindow;
	Rmatchwindow.create(matchLeftWindow.rows, matchLeftWindow.cols, CV_32FC1);
	float aveRImg = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			aveRImg += imageRight.at<uchar>(y + m, x + n);
			Rmatchwindow.at<float>(m, n) = imageRight.at<uchar>(y + m, x + n);
		}
	}
	aveRImg = aveRImg / (matchLeftWindow.rows*matchLeftWindow.cols);
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			Rmatchwindow.at<float>(m, n) -= aveRImg;
		}
	}
	//开始计算相关系数
	
	float cofficent1 = 0;
	float cofficent2 = 0;
	float cofficent3 = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			cofficent1 += matchLeftWindow.at<float>(m, n)*Rmatchwindow.at<float>(m, n);
			cofficent2 += Rmatchwindow.at<float>(m, n)*Rmatchwindow.at<float>(m, n);
			cofficent3 += matchLeftWindow.at<float>(m, n)*matchLeftWindow.at<float>(m, n);
		}
	}
	float cofficent = cofficent1 / sqrt(cofficent2 * cofficent3);
	/*
	float sumL = 0;
	float sumR = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			sumL += matchLeftWindow.at<float>(m, n) * matchLeftWindow.at<float>(m, n);
			sumR += Rmatchwindow.at<float>(m, n) * Rmatchwindow.at<float>(m, n);
		}
	}
	sumL = sqrt(sumL);
	sumR = sqrt(sumR);

	float sumC = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			sumC += (matchLeftWindow.at<float>(m, n) / sumL -  Rmatchwindow.at<float>(m, n) / sumR) * (matchLeftWindow.at<float>(m, n) / sumL - Rmatchwindow.at<float>(m, n) / sumR);
		}
	}
	float cofficent = 2 * (1 - sumC);
	*/
	return cofficent;
}
void vectorsort(std::vector < cv::Point3f> &Temp_sort)
{
	for (int i = 0; i < Temp_sort.size() - 1; i++) {
		float tem = 0;
		float temx = 0;
		float temy = 0;
		// 内层for循环控制相邻的两个元素进行比较
		for (int j = i + 1; j < Temp_sort.size(); j++) {
			if (Temp_sort.at(i).z < Temp_sort.at(j).z) {
				tem = Temp_sort.at(j).z;
				Temp_sort.at(j).z = Temp_sort.at(i).z;
				Temp_sort.at(i).z = tem;

				temx = Temp_sort.at(j).x;
				Temp_sort.at(j).x = Temp_sort.at(i).x;
				Temp_sort.at(i).x = temx;

				temy = Temp_sort.at(j).y;
				Temp_sort.at(j).y = Temp_sort.at(i).y;
				Temp_sort.at(i).y = temy;
			}
		}
	}
}
void lastview(cv::Mat imageLeftRGB, cv::Mat imageRightRGB, std::vector<cv::Point3f> featurePointLeft, std::vector<cv::Point3f> featurePointRight)
{
	cv::Mat bothview;//输出图像
	bothview.create(imageLeftRGB.rows, imageLeftRGB.cols + imageRightRGB.cols, imageLeftRGB.type());
	for (int i = 0; i <imageLeftRGB.rows; i++)
	{
		for (int j = 0; j < imageLeftRGB.cols; j++)
		{
			bothview.at<cv::Vec3b>(i, j) = imageLeftRGB.at<cv::Vec3b>(i, j);
		}
	}

	for (int i = 0; i <imageRightRGB.rows; i++)
	{
		for (int j = imageLeftRGB.cols; j <imageLeftRGB.cols + imageRightRGB.cols; j++)
		{
			bothview.at<cv::Vec3b>(i, j) = imageRightRGB.at<cv::Vec3b>(i, j - imageLeftRGB.cols);
		}
	}//左右影像合二为一	
	for (int i = 0; i < featurePointRight.size(); i++)
	{
		int a = (rand() % 200);
		int b = (rand() % 200 + 99);
		int c = (rand() % 200) - 50;
		if (a > 100 || a < 0)
		{
			a = 255;
		}
		if (b > 255 || b < 0)
		{
			b = 88;
		}
		if (c > 255 || c < 0)
		{
			c = 188;
		}
		int radius = 5;
		//左片
		int lm = int(featurePointLeft.at(i).x);
		int ln = int(featurePointLeft.at(i).y);

		cv::circle(bothview, cv::Point(lm, ln), radius, cv::Scalar(0, 255, 255), 1, 4, 0);
		cv::line(bothview, cv::Point(lm - radius - 2, ln), cv::Point(lm + radius + 2, ln), cv::Scalar(0, 255, 255), 1, 8, 0);
		cv::line(bothview, cv::Point(lm, ln - radius - 2), cv::Point(lm, ln + radius + 2), cv::Scalar(0, 255, 255), 1, 8, 0);

		//右片
		int rm = int(featurePointRight.at(i).x + imageLeftRGB.cols);
		int rn = int(featurePointRight.at(i).y);

		cv::circle(bothview, cv::Point(rm, rn), radius, cv::Scalar(0, 255, 255), 1, 4, 0);
		cv::line(bothview, cv::Point(rm - radius - 2, rn), cv::Point(rm + radius + 2, rn), cv::Scalar(0, 255, 255), 1, 8, 0);
		cv::line(bothview, cv::Point(rm, rn - radius - 2), cv::Point(rm, rn + radius + 2), cv::Scalar(0, 255, 255), 1, 8, 0);
		//连接
		cv::line(bothview, cv::Point(lm, ln), cv::Point(rm, rn), cv::Scalar(a, b, c), 1, 8, 0);
	}
	cv::namedWindow("左右片影像同名点展示", cv::WINDOW_NORMAL);
	cv::imshow("左右片影像同名点展示", bothview);
	cv::waitKey(0);
}
int LSM_Matching(cv::Mat ImgL, cv::Mat ImgR, std::vector<cv::Point3f> featurePointLeft, std::vector<cv::Point3f> featurePointRight_sift, cv::Mat H, std::vector<cv::Point3f>& featureLeftPointLST, std::vector<cv::Point3f>& featureRightPointLST)
{

	cv::Mat imageLeft, imageLeftRGB = ImgL;
	cv::Mat imageRight, imageRightRGB = ImgR;

	cv::cvtColor(imageLeftRGB, imageLeft, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageRightRGB, imageRight, cv::COLOR_BGR2GRAY);

	int matchsize = 51;//相关系数的正方形窗口的边长，必须是奇数
	int half_matchsize = matchsize / 2;//边长的一半
	int window_outside = 2; //搜索窗口边缘需要的半径
	int halflengthsize = 1; //最佳匹配窗口搜索半径
	std::vector<cv::Point3f> featurePointRight;//右片匹配到的数据

	float lowst_door = 0; //相关系数法匹配的阈值
	// 删除超出范围的featurePointLeft;
	for (size_t i = 0; i < featurePointLeft.size(); i++)
	{
		if (featurePointLeft.at(i).x < half_matchsize + window_outside || featurePointLeft.at(i).x > imageLeft.cols - half_matchsize - window_outside)
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			featurePointRight_sift.erase(featurePointRight_sift.begin() + i);
			i--;
			continue;
		}
		if (featurePointLeft.at(i).y < half_matchsize + window_outside || featurePointLeft.at(i).y > imageLeft.rows - half_matchsize - window_outside)
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			featurePointRight_sift.erase(featurePointRight_sift.begin() + i);
			i--;
			continue;
		}
	}
	for (size_t i = 0; i < featurePointRight_sift.size(); i++)
	{
		if (featurePointRight_sift.at(i).x < half_matchsize + window_outside || featurePointRight_sift.at(i).x > imageRight.cols - half_matchsize - window_outside)
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			featurePointRight_sift.erase(featurePointRight_sift.begin() + i);
			i--;
			continue;
		}
		if (featurePointRight_sift.at(i).y < half_matchsize + window_outside || featurePointRight_sift.at(i).y > imageRight.rows - half_matchsize - window_outside)
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			featurePointRight_sift.erase(featurePointRight_sift.begin() + i);
			i--;
			continue;
		}
	}

	//创建左窗口的小窗口
	cv::Mat matchLeftWindow;
	matchLeftWindow.create(matchsize, matchsize, CV_32FC1);
	for (int i = 0; i < featurePointLeft.size(); i++)
	{
		float aveLImg = 0;
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				aveLImg += imageLeft.at<uchar>(featurePointLeft.at(i).y - half_matchsize + m, featurePointLeft.at(i).x - half_matchsize + n);
				matchLeftWindow.at<float>(m, n) = imageLeft.at<uchar>(featurePointLeft.at(i).y - half_matchsize + m, featurePointLeft.at(i).x - half_matchsize + n);
			}
		}
		aveLImg = aveLImg / (matchsize* matchsize);//左窗口平均值
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				matchLeftWindow.at<float>(m, n) = matchLeftWindow.at<float>(m, n) - aveLImg;  // 左边小窗口同时减去均值
			}
		}

		/***************************对右窗口进行计算******************************/
		
		//对右窗口位置进行调整
		
		std::vector < cv::Point3f> tempfeatureRightPoint;
		for (int ii = -halflengthsize; ii <= halflengthsize; ii++)
		{
			for (int jj = -halflengthsize; jj <= halflengthsize; jj++)
			{
				cv::Point3f temphalflengthsize;
				int x = featurePointRight_sift.at(i).x + ii - half_matchsize;
				int y = featurePointRight_sift.at(i).y + jj - half_matchsize;
				float  coffee = Get_coefficient(matchLeftWindow, imageRight, x, y);
				temphalflengthsize.x = featurePointRight_sift.at(i).x + ii;
				temphalflengthsize.y = featurePointRight_sift.at(i).y + jj;
				temphalflengthsize.z = coffee;
				tempfeatureRightPoint.push_back(temphalflengthsize);
			}
		}
		vectorsort(tempfeatureRightPoint);

		if (tempfeatureRightPoint.at(0).z > lowst_door && tempfeatureRightPoint.at(0).z < 1)
		{
			cv::Point3f tempr;
			tempr.x = tempfeatureRightPoint.at(0).x;
			tempr.y = tempfeatureRightPoint.at(0).y;
			tempr.z = tempfeatureRightPoint.at(0).z;
			featurePointRight.push_back(tempr);
		}
		else
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			featurePointRight_sift.erase(featurePointRight_sift.begin() + i);
			i--;
			continue;
		}
	}
	/*正式开始最小二乘匹配*/
	//  求几何畸变的初始值
	//cv::Mat formerp = cv::Mat::eye(2 * featurePointLeft.size(), 2 * featurePointLeft.size(), CV_32F)/*权矩阵*/,
	//	formerl = cv::Mat::zeros(2 * featurePointLeft.size(), 1, CV_32F)/*常数项*/,
	//	formera = cv::Mat::zeros(2 * featurePointLeft.size(), 6, CV_32F)/*系数矩阵*/;
	//for (int i = 0; i < featurePointLeft.size(); i++)
	//{
	//	float x1 = featurePointLeft.at(i).x;
	//	float y1 = featurePointLeft.at(i).y;
	//	float x2 = featurePointRight.at(i).x;
	//	float y2 = featurePointRight.at(i).y;
	//	float coef = featurePointRight.at(i).z;//初始同名点的相关系数作为权重
	//	formerp.at<float>(2 * i, 2 * i) = coef;
	//	formerp.at<float>(2 * i + 1, 2 * i + 1) = coef;
	//	formerl.at<float>(2 * i, 0) = x2;
	//	formerl.at<float>(2 * i + 1, 0) = y2;
	//	formera.at<float>(2 * i, 0) = 1; formera.at<float>(2 * i, 1) = x1; formera.at<float>(2 * i, 2) = y1;
	//	formera.at<float>(2 * i + 1, 3) = 1; formera.at<float>(2 * i + 1, 4) = x1; formera.at<float>(2 * i + 1, 5) = y1;
	//}
	//cv::Mat nbb = formera.t()*formerp*formera, u = formera.t()*formerp*formerl;
	//cv::Mat formerR = nbb.inv()*u;
	clock_t start_t = clock();
	//开始进行最小二乘匹配
#pragma omp parallel for
	for (int i = 0; i < featurePointLeft.size(); i++)
	{
		//坐标的迭代初始值
		float x1 = featurePointLeft.at(i).x;
		float y1 = featurePointLeft.at(i).y;
		float x2 = featurePointRight.at(i).x;
		float y2 = featurePointRight.at(i).y;
		//几何畸变参数迭代初始值
		/*
		float a0 = formerR.at<float>(0, 0); float a1 = formerR.at<float>(1, 0); float a2 = formerR.at<float>(2, 0);
		float b0 = formerR.at<float>(3, 0); float b1 = formerR.at<float>(4, 0); float b2 = formerR.at<float>(5, 0);
		*/
		float a0 = H.at<double>(0, 2); float a1 = H.at<double>(0, 0); float a2 = H.at<double>(0, 1);
		float b0 = H.at<double>(1, 2); float b1 = H.at<double>(1, 0); float b2 = H.at<double>(1, 1);
		//辐射畸变迭代初始值
		float h0 = 0, h1 = 1;
		//当后一次相关系数小于前一次且满足相关系数最小要求，迭代停止
		float beforeCorrelationCoe = 0/*前一个相关系数*/, CorrelationCoe = 0, dp_norm_max;
		float xs = 0, ys = 0;
		float xt = 0, yt = 0;
		int iteration = 0;
		//while (beforeCorrelationCoe <= CorrelationCoe || CorrelationCoe >= 0.0001f) // znssd
		//while (beforeCorrelationCoe >= CorrelationCoe || CorrelationCoe < 0.999) // 相关系数
		//while (true)
		do
		{
			iteration++;
			beforeCorrelationCoe = CorrelationCoe;
			cv::Mat C = cv::Mat::zeros(matchsize*matchsize, 8, CV_32F);//系数矩阵，matchsize为左片目标窗口大小
			cv::Mat L = cv::Mat::zeros(matchsize*matchsize, 1, CV_32F);//常数项
			cv::Mat P = cv::Mat::eye(matchsize*matchsize, matchsize*matchsize, CV_32F);//权矩阵
			float sumgxSquare = 0, sumgySquare = 0, sumXgxSquare = 0, sumYgySquare = 0;
			int dimension = 0;//用于矩阵赋值
			cv::Mat LWindow, RWindow;
			LWindow.create(matchsize, matchsize, CV_32FC1);
			RWindow.create(matchsize, matchsize, CV_32FC1);
			for (int m = x1 - half_matchsize; m <= x1 + half_matchsize; m++)
			{
				for (int n = y1 - half_matchsize; n <= y1 + half_matchsize; n++)
				{
					float x2 = a0 + a1*m + a2*n;
					float y2 = b0 + b1*m + b2*n;
					int I = std::floor(x2); int J = std::floor(y2);//不大于自变量的最大整数
					if (I <= 1 || I >= imageRight.cols - 1 || J <= 1 || J >= imageRight.rows - 1)
					{
						I = 2; J = 2; P.at<float>(cv::Point(((n - int(y1 - half_matchsize))*(2 * half_matchsize + 1) + m - (x1 - half_matchsize - 1)), ((n - int(y1 - half_matchsize))*(2 * half_matchsize + 1) + m - (x1 - half_matchsize - 1)))) = 0;
					}
					/*
					// 三次样条插值
					float linerGray = Rifman(x2 - (I - 1), y2 - J + 1) * imageRight.at<uchar>(cv::Point(I - 1, J - 1)) + Rifman(x2 - I, y2 - J + 1) * imageRight.at<uchar>(cv::Point(I, J - 1)) + Rifman(I + 1 - x2, y2 - J + 1)  * imageRight.at<uchar>(cv::Point(I + 1, J - 1)) + Rifman(I + 2 - x2, y2 - J + 1)  * imageRight.at<uchar>(cv::Point(I + 2, J - 1))
									+ Rifman(x2 - (I - 1), y2 - J)     * imageRight.at<uchar>(cv::Point(I - 1, J))	   + Rifman(x2 - I, y2 - J)     * imageRight.at<uchar>(cv::Point(I, J))     + Rifman(I + 1 - x2, y2 - J)      * imageRight.at<uchar>(cv::Point(I + 1, J))     + Rifman(I + 2 - x2, y2 - J)      * imageRight.at<uchar>(cv::Point(I + 2, J))
									+ Rifman(x2 - (I - 1), y2 - J - 1) * imageRight.at<uchar>(cv::Point(I - 1, J + 1)) + Rifman(x2 - I, y2 - J - 1) * imageRight.at<uchar>(cv::Point(I, J + 1)) + Rifman(I + 1 - x2, y2 - J - 1)  * imageRight.at<uchar>(cv::Point(I + 1, J + 1)) + Rifman(I + 2 - x2, y2 - J - 1)  * imageRight.at<uchar>(cv::Point(I + 2, J + 1))
									+ Rifman(x2 - (I - 1), y2 - J - 2) * imageRight.at<uchar>(cv::Point(I - 1, J + 2)) + Rifman(x2 - I, y2 - J - 2) * imageRight.at<uchar>(cv::Point(I, J + 2)) + Rifman(I + 1 - x2, y2 - J - 2)  * imageRight.at<uchar>(cv::Point(I + 1, J + 2)) + Rifman(I + 2 - x2, y2 - J - 2)  * imageRight.at<uchar>(cv::Point(I + 2, J + 2));
					*/
					//双线性内插重采样
					float linerGray = (J + 1 - y2)*((I + 1 - x2)*imageRight.at<uchar>(cv::Point(I, J)) + (x2 - I)*imageRight.at<uchar>(cv::Point(I + 1, J)))
						+ (y2 - J)*((I + 1 - x2)*imageRight.at<uchar>(cv::Point(I, J + 1)) + (x2 - I)*imageRight.at<uchar>(cv::Point(I + 1, J + 1)));
					
					//辐射校正
					float radioGray = h0 + h1*linerGray;//得到相应灰度
					// 右窗口，用于计算相关系数
					RWindow.at<float>(n - (y1 - half_matchsize), m - (x1 - half_matchsize)) = radioGray;
					//确定系数矩阵
					
					float gy = 0.5*(imageRight.at<uchar>(cv::Point(I, J + 1)) - imageRight.at<uchar>(cv::Point(I, J - 1)));
					float gx = 0.5*(imageRight.at<uchar>(cv::Point(I + 1, J)) - imageRight.at<uchar>(cv::Point(I - 1, J)));
					/*
					float gy = 0.0f;
					gy -= imageRight.at<uchar>(cv::Point(I, J + 2)) / 12.f;
					gy += imageRight.at<uchar>(cv::Point(I, J + 1)) * (2.f / 3.f);
					gy -= imageRight.at<uchar>(cv::Point(I, J - 1)) * (2.f / 3.f);
					gy += imageRight.at<uchar>(cv::Point(I, J - 2)) / 12.f;

					float gx = 0.0f;
					gx -= imageRight.at<uchar>(cv::Point(I + 2, J)) / 12.f;
					gx += imageRight.at<uchar>(cv::Point(I + 1, J)) * (2.f / 3.f);
					gx -= imageRight.at<uchar>(cv::Point(I - 1, J)) * (2.f / 3.f);
					gx += imageRight.at<uchar>(cv::Point(I - 2, J)) / 12.f;
					*/
					C.at<float>(dimension, 0) = 1;		C.at<float>(dimension, 1) = linerGray;
					C.at<float>(dimension, 2) = gx;		C.at<float>(dimension, 3) = x2*gx;
					C.at<float>(dimension, 4) = y2*gx;  C.at<float>(dimension, 5) = gy;
				    C.at<float>(dimension, 6) = x2*gy;  C.at<float>(dimension, 7) = y2*gy;
					//常数项赋值
					L.at<float>(dimension, 0) = imageLeft.at<uchar>(cv::Point(m, n)) - radioGray;

					dimension = dimension + 1;
					//左窗口加权平均
					float gyLeft = 0.5*(imageLeft.at<uchar>(cv::Point(m, n + 1)) - imageLeft.at<uchar>(cv::Point(m, n - 1)));
					float gxLeft = 0.5*(imageLeft.at<uchar>(cv::Point(m + 1, n)) - imageLeft.at<uchar>(cv::Point(m - 1, n)));
					/*
					float gyLeft = 0.0f;
					gyLeft -= imageRight.at<uchar>(cv::Point(m, n + 2)) / 12.f;
					gyLeft += imageRight.at<uchar>(cv::Point(m, n + 1)) * (2.f / 3.f);
					gyLeft -= imageRight.at<uchar>(cv::Point(m, n - 1)) * (2.f / 3.f);
					gyLeft += imageRight.at<uchar>(cv::Point(m, n - 2)) / 12.f;

					float gxLeft = 0.0f;
					gxLeft -= imageRight.at<uchar>(cv::Point(m + 2, n)) / 12.f;
					gxLeft += imageRight.at<uchar>(cv::Point(m + 1, n)) * (2.f / 3.f);
					gxLeft -= imageRight.at<uchar>(cv::Point(m - 1, n)) * (2.f / 3.f);
					gxLeft += imageRight.at<uchar>(cv::Point(m - 2, n)) / 12.f;
					*/
					sumgxSquare += gxLeft*gxLeft;
					sumgySquare += gyLeft*gyLeft;
					sumXgxSquare += m*gxLeft*gxLeft;
					sumYgySquare += n*gyLeft*gyLeft;
					// 左窗口，用于算相关系数
					LWindow.at<float>(n - (y1 - half_matchsize), m - (x1 - half_matchsize)) = imageLeft.at<uchar>(cv::Point(m, n));
				}
			}
			float aveLImg = 0;
			float aveRImg = 0;
			for (int m = 0; m < LWindow.rows; m++)
			{
				for (int n = 0; n < LWindow.cols; n++)
				{
					aveLImg += LWindow.at<float>(m, n);
					aveRImg += RWindow.at<float>(m, n);
				}
			}
			aveLImg = aveLImg / (LWindow.rows*LWindow.cols);
			aveRImg = aveRImg / (LWindow.rows*LWindow.cols);
			for (int m = 0; m < LWindow.rows; m++)
			{
				for (int n = 0; n < LWindow.cols; n++)
				{
					LWindow.at<float>(m, n) -= aveLImg;
					RWindow.at<float>(m, n) -= aveRImg;
				}
			}
			float cofficent1 = 0;
			float cofficent2 = 0;
			float cofficent3 = 0;
			for (int m = 0; m < LWindow.rows; m++)
			{
				for (int n = 0; n < LWindow.cols; n++)
				{
					
					cofficent1 += LWindow.at<float>(m, n)*RWindow.at<float>(m, n);
					cofficent2 += LWindow.at<float>(m, n)*LWindow.at<float>(m, n);
					cofficent3 += RWindow.at<float>(m, n)*RWindow.at<float>(m, n);
				}
			}
			float cofficent4 = 0;
			for (int m = 0; m < LWindow.rows; m++)
			{
				for (int n = 0; n < LWindow.cols; n++)
				{
					cofficent4 += pow((LWindow.at<float>(m, n) / sqrt(cofficent2)) - (RWindow.at<float>(m, n) / sqrt(cofficent3)), 2);
				}
			}
			// CorrelationCoe = cofficent1 / sqrt(cofficent2 * cofficent3);
			CorrelationCoe = cofficent4;
			cv::Mat Nb = C.t()*P*C;
			cv::Mat Ub = C.t()*P*L;
			cv::Mat parameter = Nb.inv()*Ub;
			float dh0 = parameter.at<float>(0, 0); float dh1 = parameter.at<float>(1, 0);
			float da0 = parameter.at<float>(2, 0); float da1 = parameter.at<float>(3, 0); float da2 = parameter.at<float>(4, 0);
			float db0 = parameter.at<float>(5, 0); float db1 = parameter.at<float>(6, 0); float db2 = parameter.at<float>(7, 0);

			a0 = a0 + da0 + a0*da1 + b0*da2;
			a1 = a1 + a1*da1 + b1*da2;
			a2 = a2 + a2*da1 + b2*da2;
			b0 = b0 + db0 + a0*db1 + b0*db2;
			b1 = b1 + a1*db1 + b1*db2;
			b2 = b2 + a2*db1 + b2*db2;
			h0 = h0 + dh0 + h0*dh1;
			h1 = h1 + h1*dh1;

			xt = sumXgxSquare / sumgxSquare;
			yt = sumYgySquare / sumgySquare;
			xs = a0 + a1*xt + a2*yt;
			ys = b0 + b1*xt + b2*yt;
			if (abs(da0) <= 0.1f && abs(db0) <= 0.1f )
			{
				break;
			}
			/*
			dp_norm_max = 0.f;
			dp_norm_max += da0 * da0;
			dp_norm_max += da1 * da1;
			dp_norm_max += da2 * da2;
			dp_norm_max += db0 * db0;
			dp_norm_max += db1 * db1;
			dp_norm_max += db2 * db2;
			dp_norm_max = sqrtf(dp_norm_max);
			*/
		}while (iteration <= 100 && (beforeCorrelationCoe <= CorrelationCoe || CorrelationCoe >= 0.00001f));
		cv::Point3f tempPoint;
		tempPoint.x = xs;
		tempPoint.y = ys;
		tempPoint.z = CorrelationCoe;
		featureRightPointLST.push_back(tempPoint);
		cv::Point3f tempPointOrign;
		tempPointOrign.x = xt;
		tempPointOrign.y = yt;
		tempPoint.z = CorrelationCoe;
		featureLeftPointLST.push_back(tempPointOrign);
	}
	clock_t finish_t = clock();
	long duration = (finish_t - start_t) / CLOCKS_PER_SEC;
	std::cout << "Use time: " << duration << " seconds." << std::endl;
	lastview(imageLeftRGB, imageRightRGB, featureLeftPointLST, featureRightPointLST);
	lastview(imageLeftRGB, imageRightRGB, featurePointLeft, featurePointRight_sift);
	//输出两种匹配策略的结果，观察坐标是否发生变化
	std::ofstream outputfile;
	outputfile.open("FeturePointOutput.txt");
	outputfile << "左图中的sift点：(X, Y, Z)" << endl;
	outputfile << "右图中的sift匹配点：(X, Y, Z)" << endl;
	outputfile << "左图中的LSM点：(X, Y, Z)" << endl;
	outputfile << "右图中的LSM匹配点：(X, Y, Z)" << endl;
	outputfile << "总耗时：" << duration << "s" << endl;
	outputfile << " " << endl;
	if (outputfile.is_open()) {
		for (size_t i = 0; i < featurePointRight_sift.size(); i++)
		{
			outputfile << featurePointLeft.at(i).x << ", " << featurePointLeft.at(i).y << ", " << featurePointLeft.at(i).z << ", " << std::endl;
			outputfile << featurePointRight_sift.at(i).x << ", " << featurePointRight_sift.at(i).y << ", " << featurePointRight_sift.at(i).z << std::endl;
			outputfile << featureLeftPointLST.at(i).x << ", " << featureLeftPointLST.at(i).y << ", " << featureLeftPointLST.at(i).z << std::endl;
			outputfile << featureRightPointLST.at(i).x << ", " << featureRightPointLST.at(i).y << ", " << featureRightPointLST.at(i).z << std::endl;
			outputfile << " " << endl;
		}
	}
	outputfile.close();
	return 0;
}
const float GOOD_MATCH_PERCENT = 1.0f;
void alignImages(Mat im1, Mat im2, Mat& img1_wrapped, Mat& h, vector<cv::Point3f> &featurePointLeft, vector<cv::Point3f> &featurePointRight_sift)
{
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect SIFT features and compute descriptors.
	Ptr<Feature2D> sift = SIFT::create();
	sift->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	cv::Ptr<DescriptorMatcher> matcher = BFMatcher::create(cv::NORM_L2, true);
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

	// Draw matches
	Mat imMatches;
	drawMatches(im1.clone(), keypoints1, im2.clone(), keypoints2, matches, imMatches);
	imwrite("matches.jpg", imMatches);

	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// Find homography
	vector<uchar> inliersMask(points1.size());
	double reprojectionThreshold = 2;
	h = findHomography(points1, points2, RANSAC, reprojectionThreshold, inliersMask);
	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
		{
			inliers.push_back(matches[i]);
		}
	}
	matches.swap(inliers);
	for (size_t i = 0; i < matches.size(); i++)
	{
		cv::Point3f tempPointLeft;
		cv::Point3f tempPointRight;
		tempPointLeft.x = keypoints1[matches[i].queryIdx].pt.x;
		tempPointLeft.y = keypoints1[matches[i].queryIdx].pt.y;
		tempPointLeft.z = matches[i].distance;

		tempPointRight.x = keypoints2[matches[i].trainIdx].pt.x;
		tempPointRight.y = keypoints2[matches[i].trainIdx].pt.y;
		tempPointRight.z = matches[i].distance;

		featurePointLeft.push_back(tempPointLeft);
		featurePointRight_sift.push_back(tempPointRight);
	}

	// Draw top matches
	Mat imMatches_filtered;
	drawMatches(im1.clone(), keypoints1, im2.clone(), keypoints2, matches, imMatches_filtered);

	imwrite("matches_filtered.jpg", imMatches_filtered);
	imwrite("matches.jpg", imMatches);
	// Use homography to warp image
	warpPerspective(im1, img1_wrapped, h, im2.size());
}
float Rifman(float a, float b) {
	float A, B;
	if (0 <= abs(a) && abs(a) < 1)
	{
		A = 1 - 2 * a * a + abs(a) * abs(a) * abs(a);
	}
	else if( 1 <= abs(a) && abs(a) < 2)
	{
		A = 4 - 8 * abs(a) + 5 * a * a - abs(a) * abs(a) * abs(a);
	}
	else if ( 2 <= abs(a))
	{
		A = 0;
	}
	if (0 <= abs(b) && abs(b) < 1)
	{
		B = 1 - 2 * b * b + abs(b) * abs(b) * abs(b);
	}
	else if (1 <= abs(b) && abs(b) < 2)
	{
		B = 4 - 8 * abs(b) + 5 * b * b - abs(b) * abs(b) * abs(b);
	}
	else if (2 <= abs(b))
	{
		B = 0;
	}
	return A * B;
}

