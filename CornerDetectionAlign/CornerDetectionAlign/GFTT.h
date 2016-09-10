#pragma once
#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

RNG rng(12345);

class GFTT {

public:
	vector<KeyPoint> getGFTTCorners(Mat img_object1, int noOfCorners)
	{


		/// Parameters for Shi-Tomasi algorithm
		vector<Point2f> corners;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSizeGFTT = 3;
		bool useHarrisDetector = true;
		double kGFTT = 0.04;

		Mat gray_object1;
		cvtColor(img_object1, gray_object1, COLOR_BGR2GRAY);


		/// Copy the source image
		Mat copy;
		copy = img_object1.clone();

		/// Apply corner detection
		goodFeaturesToTrack(gray_object1,
			corners,
			noOfCorners,
			qualityLevel,
			minDistance,
			Mat(),
			blockSizeGFTT,
			useHarrisDetector,
			kGFTT);

		//Declaring Keypoints
		std::vector<KeyPoint> keypoints_object;


		/// Draw corners detected
		cout << "** Number of corners detected: " << corners.size() << endl;
		//int r = 4;
		for (int i = 0; i < corners.size(); i++)
		{
			//keypoints_object.push_back(KeyPoint(corners[i], 1));
			keypoints_object.push_back(KeyPoint(Point2f(corners[i].x, corners[i].y), 1));
			//circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
			//	rng.uniform(0, 255)), -1, 8, 0);
		}



		/// Show what you got
		//	namedWindow("goodFeaturesToTrack", 0);
		//	imshow("goodFeaturesToTrack", copy);

		return keypoints_object;


	}


};