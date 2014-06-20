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
 *	\file BOWAssigner.h
 *	\brief Assigns the Bag-of-Words (Bags-of-Features) representation to feature descriptor set
 *	\author	Kester Duncan
 */
#pragma once
#ifndef BOW_ASSIGNER_H_
#define BOW_ASSIGNER_H_

#include <opencv2/features2d/features2d.hpp>

namespace ocl {

class BOWAssigner
{
public:
	BOWAssigner();
	virtual ~BOWAssigner();

	/// Sets the visual vocabulary that would be used to assign the BoW
	void setVocabulary(const cv::Mat& vocab);

	/// Gets a copy of the visual vocabulary
	const cv::Mat& getVocabulary() const;

	/// Gets the size of the visual vocabulary
	int getVocabularySize() const;

	/// Computes the BoW for the given set of feature descriptors
	void compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor);
	

private:
	cv::Mat vocabulary;
	cv::Ptr<cv::DescriptorMatcher> matcher;
};


} /* ocl */

#endif /* BOW_ASSIGNER_H_ */
