
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "BOWAssigner.h"

namespace ocl {

BOWAssigner::BOWAssigner() {
	matcher  = cv::DescriptorMatcher::create("BruteForce");
}


BOWAssigner::~BOWAssigner()
{
}


void BOWAssigner::setVocabulary(const cv::Mat& vocab) {
	matcher->clear();
	vocabulary = vocab;
	matcher->add( std::vector<cv::Mat>(1, vocab) );
}


const cv::Mat& BOWAssigner::getVocabulary() const {
	return vocabulary;
}


int BOWAssigner::getVocabularySize() const {
	return vocabulary.empty() ? 0 : vocabulary.rows;
}


void BOWAssigner::compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor) {
	bowDescriptor.release();

	std::vector<cv::DMatch> matches;
	matcher->match(queryDesc, matches);

	bowDescriptor = cv::Mat(1, getVocabularySize(), CV_32F, cv::Scalar::all(0.0));
	float *descPtr = (float*) bowDescriptor.data;

	for (size_t i = 0; i < matches.size(); i++) {
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx; // cluster index
		CV_Assert( queryIdx == static_cast<int>(i));

		descPtr[trainIdx] = descPtr[trainIdx] + 1.0f;
	}

	bowDescriptor /= queryDesc.rows;

}


} /* ocl */
