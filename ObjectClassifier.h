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
 *	\file ObjectClassifier.h
 *	\brief Determines the class to which an object belongs
 *	\author	Kester Duncan
 */
#pragma once
#ifndef OBJECT_CLASSIFIER_H_
#define OBJECT_CLASSIFIER_H_

#include <cstdlib>
#include <string>
#include "SVMUtils.hpp"

// Forward declaration
namespace cv {
	class Mat;
}

namespace ocl {


/// Stores all information relevant to an object category
struct ObjectCategory {
	double categoryLabel;
	std::string categoryName;
	std::vector<double> classConfidence;

	ObjectCategory() : categoryLabel(-1.0), categoryName("Unknown") {}
};


class ObjectClassifier
{
private:
	std::string fpfhModelPath;
	std::string siftModelPath;
	std::string hogModelPath;
	std::string combinedModelPath;
	SVMModelPtr fpfhModel;
	SVMModelPtr siftModel;
	SVMModelPtr hogModel;
	SVMModelPtr combModel;

	/// Converts a BoW vector into a format for use by LibSVM
	void convertBowIntoSVMData(const cv::Mat& bowVector, SVMData& svmData);

public:
	/// Constructor
	ObjectClassifier(const std::string& modelsDirPath);

	/// Destructor
	virtual ~ObjectClassifier();

	/**
	 * \brief Sets the file path for FPFH SVM classifier model.
	 * This function assumes that FPFH.model is present in the working directory
	 */
	void setFPFHModelPath(const std::string& modelPath = "./FPFH.model") {
		if (!modelPath.empty())
			fpfhModelPath = modelPath;
		else
			fpfhModelPath = "./FPFH.model";
	}

	/**
	 * \brief Sets the file path for SIFT SVM classifier model.
	 * This function assumes that SIFT.model is present in the working directory
	 */
	void setSIFTModelPath(const std::string& modelPath = "./SIFT.model") {
		if (!modelPath.empty())
			siftModelPath = modelPath;
		else
			siftModelPath = "./SIFT.model";
	}

	/**
	 * \brief Sets the file path for HOG SVM classifier model.
	 * This function assumes that HOG.model is present in the working directory
	 */
	void setHOGModelPath(const std::string& modelPath = "./HOG.model") {
		if (!modelPath.empty())
			hogModelPath = modelPath;
		else
			hogModelPath = "./HOG.model";
	}

	/**
	 * \brief Sets the file path for Combined Ensemble SVM classifier model.
	 * This function assumes that Combined.model is present in the working directory
	 */
	void setCombinedModelPath(const std::string& modelPath = "./Combined.model") {
		if (!modelPath.empty())
			combinedModelPath = modelPath;
		else
			combinedModelPath = "./Combined.model";
	}

	/// Determines the category to which an object belongs using its feature representation
	void classify(const cv::Mat& fpfhBow, const cv::Mat& siftBow, const cv::Mat& hogBow);

	/// Measures the confidence of the classification result using the Renyi entropy
	double measureConfidence() const;


public:
	ObjectCategory objectCategory;

};

} // ocl

#endif /* OBJECT_CLASSIFIER_H_ */

