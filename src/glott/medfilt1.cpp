#include"medfilt1.h"
#include <QDebug>

void medfilt1(double* DataAFNewBase, int Length,int Win,double* DataAFBaseMed)
{
    int j, k;
    double *DataAFMid = new double[Win]();

    for (j = 0; j < Length ; j++) {

        //构建中值数组
        for (k = 0; k < Win; k++) {
            if (((j - (Win - 1) / 2) + k < 0) || ((j - (Win - 1) / 2) + k) > (Length - 1))
				DataAFMid[k] = 0;
			else
				DataAFMid[k] = DataAFNewBase[(j - (Win - 1) / 2)+k];
		}

        //将中值数组升序排序
        std::sort(DataAFMid,DataAFMid+Win-1);

		//将排序后中值赋给输出
		DataAFBaseMed[j] =DataAFMid[k/2];
    }

    delete [] DataAFMid;
}

void quicksort(double* a, int begin, int end)
{
    if (begin >= end)
        return;
    double pivot = a[begin];
    int left = begin;
    int right = end;
    while (left < right)
    {
        while (left < right && a[right] >= pivot)
            --right;
        if (left < right)
            a[left++] = a[right];
        while (left < right && a[left] <= pivot)
            ++left;
        if (left < right)
            a[right--] = a[left];
    }
    a[right] = pivot;
    quicksort(a, begin, left - 1);
    quicksort(a, left + 1, end);
}

void bubblesort(double *a , int &length)
{
    double buff;
    #pragma omp parallel for num_threads(4)
    for (int m = 0; m < length-1; m++) {
        for (int n= 0; n < length - m - 1; n++) {
            if (a[n] > a[n+1]) {
                buff = a[n];
                a[n] = a[n + 1];
                a[n + 1] = buff;
            }
        }
    }
}
