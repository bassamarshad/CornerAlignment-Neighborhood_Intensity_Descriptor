#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "HarrisCorner.h"
#include "GFTT.h"

using namespace std;
using namespace cv;

struct FeatureMatch {
	int idf1;
	int idf2;
	double score;
};


class Feature {
public:
	int type;
	int id;
	int x;
	int y;

	vector<double> data;
};

class FeatureSet : public vector<Feature> {


};

FeatureSet getFeatureVector(vector<KeyPoint> keypoints, Mat gray_image, int patch_size);
bool OutOfImage(Point2f pt, Mat img, int patch_size);
void NormalizePatch(float *patch, int width, int height);
void MatchFeaturesByDist(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore);
double distanceEuclidean(const vector<double> &v1, const vector<double> &v2);
void transposeVector(vector<Point> &GFTTCornerPtsP);

int main()
{

	Mat img_object = imread("fighter.jpg");
	Mat img_scene = imread("fighter_rotated1.jpg");

	//resize(img_scene, img_scene, img_object.size());

	//Convert to Gray
	Mat gray_object, gray_scene;
	cvtColor(img_object, gray_object, COLOR_BGR2GRAY);
	imshow("Gray object", gray_object);

	//Rotate My image by an angle

	double angle = 180;

	// get rotation matrix for rotating the image around its center
	cv::Point2f center(img_object.cols / 2.0, img_object.rows / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle
	cv::Rect bbox = cv::RotatedRect(center, img_object.size(), angle).boundingRect();
	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	//cv::Mat dst;
	//cv::warpAffine(img_object, img_scene, rot, bbox.size());

	cv::warpAffine(img_object, img_scene, rot,
		bbox.size(),
		cv::INTER_LINEAR,
		cv::BORDER_CONSTANT,
		cv::Scalar(255, 255, 255));
	//resize(img_scene, img_scene, img_object.size(), 1.0, 1.0);



	cvtColor(img_scene, gray_scene, COLOR_BGR2GRAY);
	imshow("Gray scene", gray_scene);

	// Get my Keypoints from the method of choice
	vector<KeyPoint> keypoints_object, keypoints_scene;

	//Harris Corner
	//HarrisCorner harris;
	//keypoints_object=harris.getHarrisCorners(img_object, 170);
	//keypoints_scene = harris.getHarrisCorners(img_scene, 170);

	//GFTT Corners
	GFTT gfttcorner;
	keypoints_object = gfttcorner.getGFTTCorners(img_object, 30);
	keypoints_scene = gfttcorner.getGFTTCorners(img_scene, 30);



	////////////////////////

	vector<Point2f> GFTTCornerPtsObj, GFTTCornerPtsScene;
	KeyPoint::convert(keypoints_object, GFTTCornerPtsObj);
	KeyPoint::convert(keypoints_scene, GFTTCornerPtsScene);

	//Mat H = findHomography(GFTTCornerPtsObj, GFTTCornerPtsScene, CV_RANSAC);

	//Convert gftt points to <Point>
	vector<Point> GFTTCornerPtsObjP, GFTTCornerPtsSceneP;

	//Converting Point2f to Point
	cv::Mat(GFTTCornerPtsObj).convertTo(GFTTCornerPtsObjP, cv::Mat(GFTTCornerPtsObjP).type());
	cv::Mat(GFTTCornerPtsScene).convertTo(GFTTCornerPtsSceneP, cv::Mat(GFTTCornerPtsSceneP).type());

	//Mat es = estimateRigidTransform(GFTTCornerPtsObjP, GFTTCornerPtsSceneP,true);


	Mat outImgObj = Mat(img_object.rows, img_object.cols, CV_8UC3, CV_RGB(255, 255, 255));
	Mat outImgScene = Mat(img_scene.rows, img_scene.cols, CV_8UC3, CV_RGB(255, 255, 255));

	//draw the corner points
	drawContours(outImgObj, Mat(GFTTCornerPtsObjP), -1, CV_RGB(0, 0, 255), 5);
	drawContours(outImgScene, Mat(GFTTCornerPtsSceneP), -1, CV_RGB(0, 0, 0), 5);

	//Get a Rectangle to bound them
	cv::Rect rObj = cv::boundingRect(GFTTCornerPtsObjP);
	rectangle(outImgObj, rObj, Scalar(138, 43, 226));

	cv::Rect rScene = cv::boundingRect(GFTTCornerPtsSceneP);
	rectangle(outImgScene, rScene, Scalar(138, 226, 43));

	//Mark the center of the rectangle - this is our reference
	// Center of the Rectangles
	Point rObjCenter = Point(rObj.x + rObj.width / 2, rObj.y + rObj.height / 2);
	Point rSceneCenter = Point(rScene.x + rScene.width / 2, rScene.y + rScene.height / 2);

	drawMarker(outImgObj, rObjCenter, Scalar(138, 43, 226), 0, 10, 3, 8);
	drawMarker(outImgScene, rSceneCenter, Scalar(138, 226, 43), 0, 10, 3, 8);

	//drawMarker(polyImg, Point(mc[0].x, mc[0].y), CV_RGB(0, 255, 0), 0, 10, 3, 8);
	//find which image w*h is larger - object or scene
	int newImgRows = (outImgObj.rows > outImgScene.rows ? outImgObj.rows : outImgScene.rows) * 2;
	int newImgCols = (outImgObj.cols > outImgScene.cols ? outImgObj.cols : outImgScene.cols) * 2;


	Mat newImgObj(outImgObj.rows * 2, outImgObj.cols * 2, CV_8UC3, CV_RGB(255, 255, 255));
	copyMakeBorder(outImgObj, newImgObj, outImgObj.rows / 2, outImgObj.rows / 2, outImgObj.cols / 2, outImgObj.cols / 2, 1);

	// Getting the rectangle centroid co-ordinate from this image  -->newImgObj
	//Rectangle center in expanded image
	Point newImgobjCenter = Point(outImgObj.cols / 2 + rObjCenter.x, outImgObj.rows / 2 + rObjCenter.y);
	//drawMarker(newImgObj, newImgobjCenter, Scalar(22, 143, 22), 0, 20, 2, 8);
	cout << "\n Centroid of Rectangle in newImageObj " << newImgobjCenter;
	//Create o/p image for the scene of the same size as newImgObj
	Mat newImgScene(newImgObj.rows, newImgObj.cols, CV_8UC3, CV_RGB(255, 255, 255));
	Mat sceneROI = outImgScene(rScene);
	// sceneROI.copyTo(newImgScene).rowRange(newImgobjCenter.x - rScene.x, 2).colRange(newImgobjCenter.y - rScene.y, 4);
	cout << "\n rows " << sceneROI.rows;
	cout << " \n cols " << sceneROI.cols;

	if (sceneROI.rows % 2 != 0) { resize(sceneROI, sceneROI, Size(sceneROI.cols, sceneROI.rows + 1)); }
	if (sceneROI.cols % 2 != 0) { resize(sceneROI, sceneROI, Size(sceneROI.cols + 1, sceneROI.rows)); }
	cout << "\n rows " << sceneROI.rows;
	cout << " \n cols " << sceneROI.cols;

	//Shifting/Moving the SceneROI by a new location inorder to align the centroid of the SceneROI with the centroid of the ObjImage 
	sceneROI.copyTo(newImgScene.rowRange(newImgobjCenter.y - sceneROI.rows / 2, sceneROI.rows / 2 + newImgobjCenter.y).colRange(newImgobjCenter.x - sceneROI.cols / 2, sceneROI.cols / 2 + newImgobjCenter.x));



	//Center  aligned at --> center of Rect newImgScene
	//Point newImgSceneCenter = Point(outImgObj.cols / 2 + rObjCenter.x, outImgObj.rows / 2 + rObjCenter.y);

	//sceneROI.copyTo(newImgScene.rowRange(45, sceneROI.rows+45).colRange(45, sceneROI.cols+45));

	// sceneROI.copyTo(newImgScene(cv::Rect(1,1, sceneROI.cols, sceneROI.rows)));
	// newImgobjCenter.x - sceneROI.cols / 2, newImgobjCenter.y - sceneROI.rows / 2

	//copyMakeBorder(outImgScene, newImgScene, outImgScene.rows / 2, outImgScene.rows / 2, outImgScene.cols / 2, outImgScene.cols / 2, 1);

	//resize(outImgScene, outImgScene, outImgObj.size());

	//Mat newImg;

	//cv::warpAffine(outImgObj, outImgScene, es,newImg.size());

	// Output image
	//	Mat im_out;
	// Warp source image to destination based on homography
	//	warpPerspective(outImgObj, im_out, H, outImgScene.size());

	//outImgObj.colRange(rObj.y, rObj.height).rowRange(rObj.x, rObj.width).copyTo(newImgScene.colRange(rScene.y + outImgScene.cols / 2, rScene.height).rowRange(rScene.x + outImgScene.rows / 2, rScene.width));

	//outImgObj.rowRange(rObj.x, rObj.width).colRange(rObj.y, rObj.height).copyTo(newImgScene);

	//cv::Mat objROI = outImgObj(rObj);
	//cv::Mat sceneROI = outImgScene(rScene);//.rowRange(rScene.x + outImgScene.rows / 2, rScene.x + outImgScene.rows / 2 + rScene.height).colRange(rScene.y + outImgScene.cols / 2, rScene.y + outImgScene.cols / 2 + rScene.width);
	//destinationROI.copyTo(sourceROI);


	//Mat mergeImage(newImgObj.rows, newImgObj.cols, CV_8UC3, CV_RGB(255, 255, 255));

	Mat mergeImage;
	addWeighted(newImgObj, 0.5, newImgScene, 0.5, 0.0, mergeImage);




	// namedWindow("approx polygon over corners", 0);
	imshow("Bounded Corners Object", newImgObj);
	imshow("Bounded Corners Scene", newImgScene);
	imshow("new ", mergeImage);


	/*
	FeatureSet f1, f2;

	f1 = getFeatureVector(keypoints_object, gray_object, 9);
	f2 = getFeatureVector(keypoints_scene, gray_scene, 9);

	vector<FeatureMatch> matches;
	double totalScore;

	vector<DMatch> good_matches;
	DMatch DMobj;

	MatchFeaturesByDist(f1, f2, matches, totalScore);

	cout << "\n" << totalScore;
	for (int i = 0; i < matches.size(); i++)
	{
	cout << "\n" << matches[i].idf1 << "  " << matches[i].idf2 << "  " << matches[i].score;
	DMobj.distance = (float)matches[i].score;
	DMobj.queryIdx = matches[i].idf1;
	DMobj.trainIdx = matches[i].idf2;
	good_matches.push_back(DMobj);
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < matches.size(); i++)
	{
	//-- Get the keypoints from the good matches
	obj.push_back(keypoints_object[matches[i].idf1].pt);
	scene.push_back(keypoints_scene[matches[i].idf2].pt);
	}


	//Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	//obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	//std::vector<Point2f> scene_corners(4);

	//perspectiveTransform(obj_corners, scene_corners, H);

	//DMatch::


	Mat img_matches;
	transpose(img_object, img_object);
	transpose(img_scene, img_scene);

	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	namedWindow("Good Matches & Object detection", 0);
	imshow("Good Matches & Object detection", img_matches);
	*/

	waitKey();

	return 0;
}

void transposeVector(vector<Point> &GFTTCornerPtsP1)
{

	//Need to transpose all points in GFTTCornerPtsP
	Mat GFTTCornerPtsPMat;
	GFTTCornerPtsPMat = Mat(GFTTCornerPtsP1);
	transpose(GFTTCornerPtsPMat, GFTTCornerPtsPMat);
	GFTTCornerPtsP1 = Mat(GFTTCornerPtsPMat);
}



FeatureSet getFeatureVector(vector<KeyPoint> keypoints, Mat gray_image, int patch_size)
{
	FeatureSet features;
	Feature f;
	//vector<double> fv;
	// vector<vector<double>> fs;


	//cout << "5*5 patch descriptor...";
	int id = 1;
	float *patch = new float[patch_size * patch_size];

	int x, y;
	int index;

	for (int l = 0; l < keypoints.size(); l++)
	{
		index = 0;
		x = keypoints[l].pt.x;
		y = keypoints[l].pt.y;

		if ((OutOfImage(keypoints[l].pt, gray_image, patch_size)))
			continue;

		f.type = 1;
		f.id = id;
		id += 1;
		f.x = x;
		f.y = y;
		f.data.clear();

		//fv.clear();

		for (int i = -(patch_size / 2); i <= patch_size / 2; i++)
			for (int j = -(patch_size / 2); j <= patch_size / 2; j++)
			{
				patch[index++] = gray_image.at<uchar>(x + i, y + j); //swap indexes and check !
			}
		NormalizePatch(patch, patch_size, patch_size);

		//fv.insert(fv.end(), patch, patch + (patch_size*patch_size));
		f.data.insert(f.data.end(), patch, patch + (patch_size*patch_size));
		features.push_back(f);
		//fs.push_back(f);

	}
	delete[] patch;

	return features;
}

bool OutOfImage(Point2f pt, Mat img, int patch_size)
{
	cv::Rect rect(cv::Point2f(), img.size());
	int size1 = patch_size / 2;


	if (rect.contains(Point2f(pt.x + size1, pt.y + size1)) && rect.contains(Point2f(pt.x - size1, pt.y - size1)))
	{
		return false;
	}
	else
		return true;
}


void NormalizePatch(float *patch, int width, int height)
{
	float tsum = 0.0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			tsum += patch[y * width + x];
		}
	}


	float avg = tsum / (width * height);
	float var = 0.0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			var += pow(patch[y * width + x] - avg, 2);
		}
	}

	float sqrtVar = sqrt(var / (width * height));
	sqrtVar = (sqrtVar == 0) ? 1 : sqrtVar;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			patch[y * width + x] = (patch[y * width + x] - avg) / sqrtVar;
		}
	}

}



void MatchFeaturesByDist(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore)
{
	//cout<<"threshold for match score: "<<Match_threshold_for_score<<endl;
	int m = f1.size();
	int n = f2.size();

	matches.clear();
	totalScore = 0;

	double d;
	double dBest;
	int idBestf1;
	int idBestf2;

	FeatureMatch feamatch;
	for (int i = 0; i<m; i++) {
		dBest = 1e100;
		idBestf2 = 0;

		for (int j = 0; j<n; j++) {
			d = distanceEuclidean(f1[i].data, f2[j].data);

			if (d < dBest) {
				dBest = d;
				idBestf1 = f1[i].id;
				idBestf2 = f2[j].id;
			}
		}
		feamatch.score = exp(-dBest);
		if (dBest < 10000000.0f)
		{
			feamatch.idf1 = idBestf1;
			feamatch.idf2 = idBestf2;
			matches.push_back(feamatch);
			totalScore += feamatch.score;
		}
		else
		{
			feamatch.idf2 = -1;
			matches.push_back(feamatch);
		}
	}
}


// Compute Euclidean distance between two vectors.
double distanceEuclidean(const vector<double> &v1, const vector<double> &v2) {
	int m = v1.size();
	int n = v2.size();

	if (m != n) {
		return 1e100;
	}

	double dist = 0;

	for (int i = 0; i<m; i++) {
		dist += pow(v1[i] - v2[i], 2);
	}

	return dist;
}