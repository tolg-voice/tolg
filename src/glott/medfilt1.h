#ifndef MEDFILT1_H
#define MEDFILT1_H

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>

void medfilt1(double* DataAFNewBase, int Length,  int Win, double* DataAFBaseMed);
/*
 * 中值滤波
 * * DataAFNewBase  源数据数组
 * Length           源数据长度
 * Win              取中值数组长度
 * * DataAFBaseMed  滤波后数组
*/

void quicksort(double* a, int begin, int end);//快速排序
void bubblesort(double *a , int &length);//冒泡排序

#endif // MEDFILT1_H
