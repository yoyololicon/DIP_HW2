#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <highgui.h>
#include "cv.h"
#include <valarray>
#include <complex>
#include <algorithm>
using namespace std;
using namespace cv;
#define M_PI 3.14159265359
typedef valarray<valarray<complex<double>>> val2d;

void fft(valarray<complex<double>> &x)
{
	const size_t N = x.size();
	if (N <= 1) return;

	// divide
	valarray<complex<double>> even = x[slice(0, N / 2, 2)];
	valarray<complex<double>>  odd = x[slice(1, N / 2, 2)];

	// conquer
	fft(even);
	fft(odd);

	// combine
	for (size_t k = 0; k < N / 2; ++k)
	{
		complex<double> t = polar(1.0, -2 * M_PI * k / N) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

void resizing(val2d &x, int M, int N)
{
	x.resize(M);
	for (int i = 0; i < M; i++)
		x[i].resize(N);
}

void compute_fft_num(int rows, int cols, int &M, int &N)
{
	int base = 1;
	while (base < cols)
		base *= 2;
	N = base;
	base = 1;
	while (base < rows)
		base *= 2;
	M = base;
}

void fft_2d_and_Log_Scale(valarray<valarray<double>> &out, val2d &in, int M, int N)
{
	// do the rows transform
	for (int i = 0; i < M; ++i)
		fft(in[i]);

	val2d temp(valarray<complex<double>>(M), N);
	//do the cols transform
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j)
			temp[i][j] = in[j][i];
		fft(temp[i]);
	}
	double Min = INFINITY, Max = 0.;

	//log transform 
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			out[i][j] = log(abs(temp[j][i]) + 1);
			Max = max(Max, out[i][j]);
			Min = min(Min, out[i][j]);
		}
	Max -= Min;
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			out[i][j] = (out[i][j] - Min) / Max;	//scale to 0~1
}

int main(){

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg = imread("Q1.tif", CV_LOAD_IMAGE_GRAYSCALE);

	//Start Problem 1
	//compute the value that is base on 2
	int M, N;
	compute_fft_num(SrcImg.rows, SrcImg.cols, M, N);

	//create 2d-arrays for 2d_fft
	val2d out(valarray<complex<double>>(N), M);
	valarray<valarray<double>> finalout(valarray<double>(N), M);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			if (i < SrcImg.rows && j < SrcImg.cols)
				out[i][j] = SrcImg.at<uchar>(i, j)*pow(-1, i + j);	//let F(0, 0) move to center point
			else
				out[i][j] = 0;	//zero-padding
		}

	fft_2d_and_Log_Scale(finalout, out, M, N);

	// Create a grayscale output image matrix
	Mat DstImg = Mat(M, N, CV_8UC1);

	//output image
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			//scale the value to fit in [0,255]
			DstImg.at<uchar>(i, j) = finalout[i][j]*255;
		}

	// Show images
	imshow("Input Image", SrcImg);
	imshow("Fig 4.29(b)", DstImg);

	// Write output images
	imwrite("p1_output.tif", DstImg);

	//Start Problem 2
	Mat SrcImg2 = imread("Q2_new.tif", CV_LOAD_IMAGE_GRAYSCALE);
	//compute the value that is base on 2
	compute_fft_num(SrcImg2.rows, SrcImg2.cols, M, N);

	//resize 2d-array
	resizing(out, M, N);
	finalout.resize(M);
	for (int i = 0; i < M; i++)
		finalout[i].resize(N);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			if (i < SrcImg2.rows && j < SrcImg2.cols)
				out[i][j] = SrcImg2.at<uchar>(i, j)*pow(-1, i + j);	//let F(0, 0) move to center point
			else
				out[i][j] = 0;	//zero-padding
		}

	fft_2d_and_Log_Scale(finalout, out, M, N);

	// Create a grayscale output image matrix
	Mat DstImg2 = Mat(M, N, CV_8UC1);

	//output image
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			//scale the value to fit in [0,255]
			DstImg2.at<uchar>(i, j) = finalout[i][j]*255;
		}

	// Show images
	imshow("Input Image2", SrcImg2);
	imshow("Fig 4.36(d)", DstImg2);
	
	// Write output images
	imwrite("p2_output.tif", DstImg2);

	//Start Problem 3
	Mat SrcImg3 = imread("Q3.tif", CV_LOAD_IMAGE_GRAYSCALE);
	//compute the value that is base on 2
	compute_fft_num(SrcImg3.rows, SrcImg3.cols, M, N);

	//resize 2d-array
	resizing(out, M, N);
	finalout.resize(M);
	for (int i = 0; i < M; i++)
		finalout[i].resize(N);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			if (i < SrcImg3.rows && j < SrcImg3.cols)
				out[i][j] = SrcImg3.at<uchar>(i, j)*pow(-1, i + j);	//let F(0, 0) move to center point
			else
				out[i][j] = 0;	//zero-padding
		}

	fft_2d_and_Log_Scale(finalout, out, M, N);

	// Create a grayscale output image matrix
	Mat DstImg3 = Mat(M, N, CV_8UC1);

	//output image
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			//scale the value to fit in [0,255]
			DstImg3.at<uchar>(i, j) = finalout[i][j] * 255;
		}

	// Show images
	imshow("Input Image3", SrcImg3);
	imshow("Fig 4.38(b)", DstImg3);

	// Write output images
	imwrite("p3_output.tif", DstImg3);

	//Start Problem 4
	Mat SrcImg4 = imread("Q4.tif", CV_LOAD_IMAGE_GRAYSCALE);
	//compute the value that is base on 2
	compute_fft_num(SrcImg4.rows, SrcImg4.cols, M, N);

	//resize 2d-array
	resizing(out, M, N);
	finalout.resize(M);
	for (int i = 0; i < M; i++)
		finalout[i].resize(N);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			if (i < SrcImg4.rows && j < SrcImg4.cols)
				out[i][j] = SrcImg4.at<uchar>(i, j)*pow(-1, i + j);	//let F(0, 0) move to center point
			else
				out[i][j] = 0;	//zero-padding
		}

	fft_2d_and_Log_Scale(finalout, out, M, N);

	// Create a grayscale output image matrix
	Mat DstImg4 = Mat(M, N, CV_8UC1);

	//output image
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			//scale the value to fit in [0,255]
			DstImg4.at<uchar>(i, j) = finalout[i][j] * 255;
		}

	// Show images
	imshow("Input Image4", SrcImg4);
	imshow("Fig 4.41(b)", DstImg4);

	// Write output images
	imwrite("p4_output.tif", DstImg4);

	waitKey(0);
	return 0;
}