/*
 * Software License Agreement (BSD License)
 *
 *  Object Classification
 *  Copyright (c) 2013, Kester Duncan
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *	\file	ObjectDescription.h
 *	\brief	Describes an object using 2D and 3D features
 *	\author	Kester Duncan
 */
#pragma once
#ifndef OBJECT_DESCRIPTION_H_
#define OBJECT_DESCRIPTION_H_


#include <opencv2/core/core.hpp>
#include "BOWAssigner.h"



/// Forward declarations to avoid unnecessary includes
namespace ntk {
	class Pose3D;
}

namespace pcl {
	template <typename PointType> 
	class PointCloud;
	struct PointXYZRGB;
}


/**
 * \brief Object Classfication namespace
 */
namespace ocl {


/// Size of an FPFH descriptor
const int FPFH_LEN = 33;

/// Size of a SIFT descriptor
const int SIFT_LEN = 128;

/// Size of custom HoG descriptor
const int HOG_LEN = 36;

/// Number of blocks which make up a HoG descriptor
const int NUM_HOG_BLOCKS = 105;
	

/**
 * \brief Describes an object using FPFH, SIFT, HOG, and BoW descriptors
 */
class ObjectDescription
{
private:
	cv::Mat fpfhDescriptors;
	cv::Mat siftDescriptors;
	cv::Mat hogDescriptors;
	cv::Mat fpfhBOW;
	cv::Mat siftBOW;
	cv::Mat hogBOW;
	std::string fpfhVocabPath;
	std::string siftVocabPath;
	std::string hogVocabPath;
	BOWAssigner fpfhAssigner;
	BOWAssigner siftAssigner;
	BOWAssigner hogAssigner;

	/// Sets up the respective feature type vocabularies
	void setupBOWAssignment();

	/// Assign the Bag of Words representation for a set of FPFH descriptors
	void assignFPFHBOW();

	/// Assign the Bag of Words representation for a set of SIFT descriptors
	void assignSIFTBOW();

	/// Assign the Bag of Words representation for a set of HOG descriptors
	void assignHOGBOW();

	/// Get the Fast Point Feature Histogram representation for the given object point cloud
	void extractFPFHDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

	/// Get the Scale Invariant Feature Transform representation for the given object point cloud
	void extractSIFTDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

	/// Get the Histogram of Oriented Gradients representation for the given object point cloud
	void extractHOGDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);


public:
	ObjectDescription(const std::string& pathToVocabs = ".");
	virtual ~ObjectDescription();

	/// Sets the path of the FPFH visual vocabulary
	inline void setFPFHVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->fpfhVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}
	
	/// Sets the path of the SIFT visual vocabulary
	inline void setSIFTVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->siftVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}

	/// Sets the path of the HOG visual vocabulary
	inline void setHOGVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->hogVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}

	/// Get this object's FPFH BoW representation
	inline cv::Mat getFPFHBow() const {
		return fpfhBOW;
	}

	/// Get this object's SIFT BoW representation
	inline cv::Mat getSIFTBow() const {
		return siftBOW;
	}

	/// Get this object's HOG BoW representation
	inline cv::Mat getHOGBow() const {
		return hogBOW;
	}

	/// Get a vector of all BoWs
	inline std::vector<cv::Mat> getAllBows() {
		std::vector<cv::Mat> allBows;

		allBows.push_back(getFPFHBow());
		allBows.push_back(getSIFTBow());
		allBows.push_back(getHOGBow());

		return allBows;
	}

	/// Get the multi-modal feature descriptions for this object
	void extractFeatureDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>& cloud);

	/// Assigns the BoWs representations to describe this object
	void assignBOWs();


	


};

} /* ocl */

#endif /* OBJECT_DESCRIPTION_H_ */

