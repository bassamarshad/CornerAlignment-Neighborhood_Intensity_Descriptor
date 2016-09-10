#pragma once
#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

class HarrisCorner {

public:
	vector<KeyPoint> getHarrisCorners(Mat img_object, int thresh)
	{

		Mat dst_object, dst_norm_object, dst_norm_scaled_object;
		dst_object = Mat::zeros(img_object.size(), CV_32FC1);


		Mat gray_object, gray_scene;
		cvtColor(img_object, gray_object, COLOR_BGR2GRAY);

		/// Detector parameters
		int blockSize = 2;
		int apertureSize = 5;
		double k = 0.045;

		//KeyPoint::convert(corners, keypoints_object);

		/// Detecting corners
		//cornerHarris(gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
		cornerHarris(gray_object, dst_object, blockSize, apertureSize, k, BORDER_DEFAULT);


		/// Normalizing
		normalize(dst_object, dst_norm_object, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm_object, dst_norm_scaled_object);



		//Declaring Keypoints
		std::vector<KeyPoint> keypoints_object;

		for (int j = 0; j < dst_norm_object.rows; j++)
		{
			for (int i = 0; i < dst_norm_object.cols; i++)
			{
				if ((int)dst_norm_object.at<float>(j, i) >thresh)
				{
					//	cout << (int)dst_norm_object.at<float>(j, i);
					keypoints_object.push_back(KeyPoint(Point2f(i, j), 1));
					circle(dst_norm_scaled_object, Point(i, j), 5, Scalar(0), 2, 8, 0);
					//cout << "\n" << Point(i, j);
				}
			}
		}

		namedWindow("Corners in Object", 0);
		imshow("Corners in Object", dst_norm_scaled_object);
		Mat img_keypoints_object = img_object.clone();

		//drawKeypoints(img_object, keypoints_object, img_keypoints_object, CV_RGB(0,0,255));
		//namedWindow("Keypoints in Object", 0);
		//imshow("Keypoints in Object", img_keypoints_object);

		return keypoints_object;

	}




};
