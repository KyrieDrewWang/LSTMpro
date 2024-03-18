// LeastSquareMethodLearning.cpp : 定义控制台应用程序的入口点。
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
	double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
	double ae = 2.0, be = -1.0, ce = 5.0;  // 估计参数值
	int N = 100;	// 数据点
	double w_signma = 1.0;	// 噪声sigma值
	cv::RNG rng;	// OpenCV随机数产生器

	vector<double> x_data, y_data;
	for (int i = 0; i < N; i++)
	{
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_signma));
	}

	// 开始Gauss-Newton迭代
	int iterations = 100;				// 迭代次数
	double cost = 0, lastCost = 0;		// 本次迭代的cost和上一次迭代的cost

	for (int iter = 0; iter < iterations; iter++)
	{
		Matrix3d H = Matrix3d::Zero();		// Hessian = J^T * J in Gauss-Newton
		Vector3d b = Vector3d::Zero();		// bias
		cost = 0;
		for (int i = 0; i < N; i++)
		{
			double xi = x_data[i], yi = y_data[i];  //第i个数据点
			double error = 0;	// 第i个数据点的计算误差
			error = yi - exp(ae * xi * xi + be * xi + ce);
			Vector3d J;
			J[0] = -xi*xi*exp(ae*xi*xi + be*xi + ce);
			J[1] = -xi*exp(ae*xi*xi + be*xi + ce);
			J[2] = -exp(ae*xi*xi + be*xi + ce);
			H += J * J.transpose();  // GN近似的H
			b += -error * J;  
			cost += error * error;
		}
		// 求解线性方程 Hx = b，建议用ldlt方法
		Vector3d dx;
		// dx = H.inverse() * b; //直接求逆方法求解增量
		dx = H.ldlt().solve(b);  // ldlt方法
		if (isnan(dx[0]))
		{
			cout << "result is Nan" << endl;
			break;
		}

		if (iter > 0 && cost > lastCost)
		{
			// 误差增长了，说明近似的不够好
			cout << "cost:" << cost << ", last cost: " << lastCost << endl;
			break;
		}

		// 更新abs估计值
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

