// LeastSquareMethodLearning.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

int main(int agc, char **argv)
{
	double ar = 1.0, br = 2.0, cr = 1.0;  // ��ʵ����ֵ
	double ae = 2.0, be = -1.0, ce = 5.0;  // ���Ʋ���ֵ
	int N = 100;	// ���ݵ�
	double w_signma = 1.0;	// ����sigmaֵ
	cv::RNG rng;	// OpenCV�����������

	vector<double> x_data, y_data;
	for (int i = 0; i < N; i++)
	{
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_signma));
	}

	// ��ʼGauss-Newton����
	int iterations = 100;				// ��������
	double cost = 0, lastCost = 0;		// ���ε�����cost����һ�ε�����cost

	for (int iter = 0; iter < iterations; iter++)
	{
		Matrix3d H = Matrix3d::Zero();		// Hessian = J^T * J in Gauss-Newton
		Vector3d b = Vector3d::Zero();		// bias
		cost = 0;
		for (int i = 0; i < N; i++)
		{
			double xi = x_data[i], yi = y_data[i];  //��i�����ݵ�
			double error = 0;	// ��i�����ݵ�ļ������
			error = yi - exp(ae * xi * xi + be * xi + ce);
			Vector3d J;
			J[0] = -xi*xi*exp(ae*xi*xi + be*xi + ce);
			J[1] = -xi*exp(ae*xi*xi + be*xi + ce);
			J[2] = -exp(ae*xi*xi + be*xi + ce);
			H += J * J.transpose();  // GN���Ƶ�H
			b += -error * J;  
			cost += error * error;
		}
		// ������Է��� Hx = b��������ldlt����
		Vector3d dx;
		// dx = H.inverse() * b; //ֱ�����淽���������
		dx = H.ldlt().solve(b);  // ldlt����
		if (isnan(dx[0]))
		{
			cout << "result is Nan" << endl;
			break;
		}

		if (iter > 0 && cost > lastCost)
		{
			// ��������ˣ�˵�����ƵĲ�����
			cout << "cost:" << cost << ", last cost: " << lastCost << endl;
			break;
		}

		// ����abs����ֵ
		ae += dx[0];
		be += dx[1];
		ce += dx[2];
		lastCost = cost;
		cout << "Total cost:" << cost << endl;	
	}

	cout << "estimated abs = " << ae << ", " << be << ", " << ce << endl;
	cin.get();
    return 0;
}

