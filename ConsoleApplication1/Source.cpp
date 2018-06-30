#include <opencv2\opencv.hpp>
#include <highgui\highgui.hpp>
#include <imgproc\imgproc.hpp>
#include <cv.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\legacy\legacy.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat obj, objGray, scene, sceneGray, resultObj, resultScene, descriptorObj, descriptorScene;
	vector<KeyPoint> keyPointsObj, keyPointsScene;

	obj = imread("obj.jpg", CV_LOAD_IMAGE_COLOR); //imshow("obj", obj);
	objGray = imread("obj.jpg", CV_LOAD_IMAGE_GRAYSCALE); //imshow("objGray", objGray);
	scene = imread("scene.jpg", CV_LOAD_IMAGE_COLOR); //imshow("scene", scene);
	sceneGray = imread("scene.jpg", CV_LOAD_IMAGE_GRAYSCALE); //imshow("sceneGray", sceneGray);

	SIFT siftImageObj, siftImageScene(100, 3, 0.04, 10, 1.6f);

	siftImageObj(objGray, noArray(), keyPointsObj, descriptorObj);
	siftImageScene(sceneGray, noArray(), keyPointsScene, descriptorScene);

	drawKeypoints(obj, keyPointsObj, resultObj, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //imshow("resultObj", resultObj);
	drawKeypoints(scene, keyPointsScene, resultScene, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //imshow("resultScene", resultScene);

	BruteForceMatcher< L2<float> > matcher;
	std::vector< DMatch > matches, good_matches;
	matcher.match(descriptorObj, descriptorScene, matches);

	Mat matchImg;
	double minDist = 100, maxDist = 0;

	for (int i = 0; i < descriptorObj.rows; i++){
		double dist = matches[i].distance;
		if (dist < minDist){
			dist = minDist;
		}
		else if (dist> maxDist){
			dist = maxDist;
		}
	}

	for (int i = 0; i < descriptorObj.rows; i++){
		if (matches[i].distance < 3 * minDist){
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches, allMergeResult;
	std::vector<Point2f> src, target;

	drawMatches(obj, keyPointsObj, scene, keyPointsScene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
	for (int i = 0; i < good_matches.size(); i++){
		src.push_back(keyPointsObj[good_matches[i].queryIdx].pt);
		target.push_back(keyPointsScene[good_matches[i].trainIdx].pt);
	}

	//Homography process
	Mat H = findHomography(target, src, CV_RANSAC);
	warpPerspective(scene, allMergeResult, H, Size(scene.cols * 2, scene.rows));
	Mat target_in_big_mat(allMergeResult, Rect(0, 0, obj.cols, obj.rows));
	obj.copyTo(target_in_big_mat);

	imshow("Draw Matches", img_matches);
	imshow("Merging Result", allMergeResult);
	
	imwrite("drawMatches.jpg", img_matches);
	imwrite("mergingResult.jpg", allMergeResult);

	waitKey(0);
	return 0;
}