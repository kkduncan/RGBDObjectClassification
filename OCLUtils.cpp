
#include <boost/math/special_functions/fpclassify.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "OCLUtils.h"
#include "OCLPose3D.h"

namespace ocl {
	
void OCLUtils::pointCloudToIntensityImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat3b& rgbImage) {
	ocl::Pose3D depthCameraPose;
	depthCameraPose.setCameraParameters(533.069, 533.069, 320, 240);
	depthCameraPose.resetCameraTransform();
	depthCameraPose.applyTransformBefore(cv::Vec3f(0.f, 0.f, 0.f), cv::Vec3f(0.f, 0.f, 0.f));

	cv::Mat3b tempMat;
	float minX = 999.0f;
	float maxX = -999.0f;
	float minY = 999.0f;
	float maxY = -999.0f;
	int width = 0, height = 0;
	pcl::PointCloud<pcl::PointXYZRGB> tempCloud;	
	tempCloud.resize(cloud.size());

	int count = 0;

	for (size_t i = 0; i < cloud.size(); ++i) {
		cv::Point3f origPt;
		pcl::PointXYZRGB pclPt;

		origPt.x = cloud.points[i].x;
		origPt.y = cloud.points[i].y;
		origPt.z = cloud.points[i].z;

		count++;

		if (boost::math::isnan(origPt.x) || boost::math::isinf(origPt.x)
			|| boost::math::isnan(origPt.y) || boost::math::isinf(origPt.y)
			|| boost::math::isnan(origPt.z) || boost::math::isinf(origPt.z)) {
				pclPt.x = 0;
				pclPt.y = 0;
				pclPt.z = 0;			

		} else {
			cv::Point3f projPt = depthCameraPose.projectToImage(origPt);

			if (projPt.x < minX) minX = projPt.x;
			if (projPt.x > maxX) maxX = projPt.x;

			if (projPt.y < minY) minY = projPt.y;
			if (projPt.y > maxY) maxY = projPt.y;

			pclPt.x = projPt.x;
			pclPt.y = projPt.y;
			pclPt.z = projPt.z;
		}
		tempCloud.points[i] = pclPt;		
	}

	width = static_cast<int>(maxX - minX);
	height = static_cast<int>(maxY - minY);
	tempMat = cv::Mat3b(height, width);

	for (int i = 0; i < tempMat.rows; i++) {
		for (int j= 0; j < tempMat.cols; j++) {
			tempMat(i, j) = cv::Vec3b(128, 128, 128);			
		}
	}

	for (size_t k = 0; k < tempCloud.points.size(); k++) {
		int imgXLoc = static_cast<int>(std::ceil(tempCloud.points[k].x - minX));
		int imgYLoc = static_cast<int>(std::ceil(tempCloud.points[k].y - minY));

		if (imgXLoc >= width) imgXLoc = width - 1;
		if (imgXLoc < 0) imgXLoc = 0;

		if (imgYLoc >= height) imgYLoc = height - 1;
		if (imgYLoc < 0) imgYLoc = 0;
		
		cv::Vec3b color;
		color(0) = static_cast<unsigned char>(cloud[k].b);
		color(1) = static_cast<unsigned char>(cloud[k].g);
		color(2) = static_cast<unsigned char>(cloud[k].r);
		tempMat(imgYLoc, imgXLoc) = color;		
	}

	tempMat.copyTo(rgbImage);
}



}
