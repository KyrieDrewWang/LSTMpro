// openMP_Learning.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <iostream>
#include <string>;
#include <typeinfo>;
#include <cmath>;
#include <vector>
using namespace std;

int main()
{
	vector<int>  vecInt(100);
		#pragma omp parallel for
	for (int i = 0; i < vecInt.size(); i++)
	{
		vecInt[i] = i*i;
	}
	
	return 0;
}

