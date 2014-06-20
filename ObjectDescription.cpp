#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>

#include "OCLUtils.h"
#include "ObjectDescription.h"


namespace ocl {


ObjectDescription::ObjectDescription(const std::string& pathToVocabs) {
	setFPFHVocabPath(pathToVocabs + "/FPFHVocab.dat");
	setSIFTVocabPath(pathToVocabs + "/SIFTVocab.dat");
	setHOGVocabPath(pathToVocabs + "/HOGVocab.dat");

	setupBOWAssignment();
}


ObjectDescription::~ObjectDescription()
{
}



void ObjectDescription::extractFPFHDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampledCloudPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB> objectPtCloudSampled;

	*cloudPtr = *cloud;
	
	pcl::VoxelGrid<pcl::PointXYZRGB> grid;
	grid.setInputCloud (cloudPtr);
	grid.setLeafSize (0.002, 0.002, 0.002);
	grid.filter (objectPtCloudSampled);

	*sampledCloudPtr = objectPtCloudSampled;

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud (sampledCloudPtr);

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr normalsTree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	ne.setSearchMethod (normalsTree);

	pcl::PointCloud<pcl::Normal>::Ptr cloudNormals (new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch (0.03);
	ne.compute (*cloudNormals);

	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud (sampledCloudPtr);
	fpfh.setInputNormals (cloudNormals);

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr fpfhTree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	fpfh.setSearchMethod (fpfhTree);

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr ptFeatHistograms (new pcl::PointCloud<pcl::FPFHSignature33> ());
	fpfh.setRadiusSearch (0.05);
	fpfh.compute (*ptFeatHistograms);

	fpfhDescriptors = cv::Mat(ptFeatHistograms->size(), FPFH_LEN, CV_32F);
	for (size_t i = 0; i < ptFeatHistograms->size(); i++) {
		pcl::FPFHSignature33 feat = ptFeatHistograms->points[i];
		float sum = 0.0;
		
		for(int j = 0; j < FPFH_LEN; j++ )	{
			fpfhDescriptors.at<float>(i, j) = feat.histogram[j];
			sum += feat.histogram[j];
		}

		for(int j = 0; j < FPFH_LEN; j++ )	{
			fpfhDescriptors.at<float>(i, j) /= sum;				
		}		
	}	
}


void ObjectDescription::extractSIFTDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
	cv::SiftDescriptorExtractor extractor;
	cv::SiftFeatureDetector detector(0, 10);
				
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat3b srcImg, tempImg;
	cv::Mat srcImgGray;
	int zoomFactor = 1;
		
	OCLUtils::pointCloudToIntensityImage(*cloud, tempImg);
	cv::resize(tempImg, srcImg, cv::Size(tempImg.cols * zoomFactor, tempImg.rows * zoomFactor));
	cv::cvtColor(srcImg, srcImgGray, CV_BGR2GRAY);
	detector.detect(srcImgGray, keypoints);
	extractor.compute(srcImgGray, keypoints, siftDescriptors);
	
	for (size_t i = 0; i < static_cast<size_t>(siftDescriptors.rows); i++) {
		float sum = 0.f;
		for (int j = 0; j < SIFT_LEN; j++) {
			sum += siftDescriptors.at<float>(i, j);
		}

		for (int j = 0; j < SIFT_LEN; j++) {
			siftDescriptors.at<float>(i, j) /= sum;
		}		
	}
}


void ObjectDescription::extractHOGDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
	//cv::Size winSize(64, 64);
	//cv::Size blockSize(16, 16);
	//cv::Size blockStride(48, 48);
	//cv::Size cellSize(8, 8);
	//int numBins = 9;

	//cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, numBins);
	cv::HOGDescriptor hog;

	std::vector<float> descriptors;
	cv::Mat3b tempImg, srcImg;
	cv::Mat srcImgGray;
		
	OCLUtils::pointCloudToIntensityImage(*cloud, tempImg);
	cv::resize(tempImg, srcImg, cv::Size(64, 128));	
	cv::cvtColor(srcImg, srcImgGray, CV_BGR2GRAY);
	hog.compute(srcImgGray, descriptors); // was on srcImgGray
		
	/*
	int numDescriptors = 0;
	size_t descSize = hog.getDescriptorSize();

	for (size_t j = 0; j < descriptors.size(); ++j) {	
		if ((j + 1) % descSize == 0) {
			numDescriptors++;
		}			
	}

	if (numDescriptors > 0) {
		hogDescriptors = cv::Mat(numDescriptors, HOG_LEN, CV_32F);
	}
	
	for (size_t i = 0; i < numDescriptors; ++i) {
		for (size_t j = 0; j < HOG_LEN; ++j) {	
			int ij = (i * HOG_LEN) + j;	
			hogDescriptors.at<float>(i, j) = descriptors[ij];			
		}		
	}
	*/
	
	/* 
	 * Separate the descriptor into the block histograms 
	 *(4 cells x 9-bin histogram) 
	 *
	 * NOTE:
	 *
	 * For each of the 105 blocks used by the default HOG implementation, we define
	 * our descriptor as the concatenation of the 4 9-binned histograms found in the
	 * block. Therefore our descriptor has a dimension of 36 and there 105 of them
	 */
	hogDescriptors = cv::Mat(NUM_HOG_BLOCKS, HOG_LEN, CV_32F);
	for (size_t i = 0; i < static_cast<size_t>(NUM_HOG_BLOCKS); ++i) {
		for (size_t j = 0; j < static_cast<size_t>(HOG_LEN); ++j) {
			int ij = (i * HOG_LEN) + j;	
			hogDescriptors.at<float>(i, j) = descriptors[ij];			
		}		
	}
}


void ObjectDescription::extractFeatureDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
	boost::thread_group workerThreads;
	boost::thread *siftWorker, *hogWorker, *fpfhWorker;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudForSift (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudForHog (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudForFPFH (new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZRGB>(cloud, *cloudForSift);
	siftWorker = new boost::thread(boost::bind(&ObjectDescription::extractSIFTDescriptors, this, cloudForSift));
	workerThreads.add_thread(siftWorker);
	
	pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZRGB>(cloud, *cloudForHog);
	hogWorker = new boost::thread(boost::bind(&ObjectDescription::extractHOGDescriptors, this, cloudForHog));
	workerThreads.add_thread(hogWorker);
	
	pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZRGB>(cloud, *cloudForFPFH);
	fpfhWorker = new boost::thread(boost::bind(&ObjectDescription::extractFPFHDescriptors, this, cloudForFPFH));
	workerThreads.add_thread(fpfhWorker);

	workerThreads.join_all();
}


void ObjectDescription::setupBOWAssignment() {
	// TODO: ensure that the files are valid here
	cv::FileStorage fpfhFS(fpfhVocabPath.c_str(), cv::FileStorage::READ);
	cv::Mat fpfhVocab;
	fpfhFS["vocab"] >> fpfhVocab;
	fpfhAssigner.setVocabulary(fpfhVocab);

	cv::FileStorage siftFS(siftVocabPath.c_str(), cv::FileStorage::READ);
	cv::Mat siftVocab;
	siftFS["vocab"] >> siftVocab;
	siftAssigner.setVocabulary(siftVocab);

	cv::FileStorage hogFS(hogVocabPath.c_str(), cv::FileStorage::READ);
	cv::Mat hogVocab;
	hogFS["vocab"] >> hogVocab;
	hogAssigner.setVocabulary(hogVocab);

}


void ObjectDescription::assignFPFHBOW() {
	// TODO: Ensure that the FPFH BoW assigner is set up
	fpfhAssigner.compute(fpfhDescriptors, fpfhBOW);
}


void ObjectDescription::assignSIFTBOW() {
	// TODO: Ensure that the SIFT BoW assigner is set up
	if (!siftDescriptors.empty()) {
		siftAssigner.compute(siftDescriptors, siftBOW);
	} else {
		fprintf(stderr, "Error: No SIFT descriptors!\n");
	}
}


void ObjectDescription::assignHOGBOW() {
	// TODO: Ensure that the HOG BoW assigner is set up
	std::cout << "HOG Descriptors columns: " << hogDescriptors.cols << std::endl;
	std::cout << "HOG Vocabulary columns: " << hogAssigner.getVocabulary().cols << std::endl;

	hogAssigner.compute(hogDescriptors, hogBOW);

}


void ObjectDescription::assignBOWs() {
	// TODO: apply multi-threading here
	assignSIFTBOW();
	assignHOGBOW();
	assignFPFHBOW();

}


} /* ocl */
