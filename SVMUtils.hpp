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
 *	\file SVMUtils.hpp
 *	\brief Utility functions and types for use with LibSVM
 *	\author	Kester Duncan
 */
#pragma once
#ifndef SVM_UTILS_H_
#define SVM_UTILS_H_

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <vector>

#include "svm.h"

namespace ocl {

/* Redefinitions to fit our style */
typedef svm_model SVMModel;
typedef svm_model* SVMModelPtr;
typedef svm_node SVMNode;
typedef svm_node* SVMNodePtr;


/**
 * Represents one row of LibSVM's input file
 */
struct SVMData {
	int classNum;
	std::vector<SVMNode> nodes;
};


/**
 * SVMData comparator to be used for sorting
 */
//static bool svmDataCompare(const SVMData& left, const SVMData& right){
//	if (left.classNum < right.classNum) {
//		return true;
//	} else {
//		return false;
//	}
//}


/**
 * Wrapper for common LibSVM functions
 */
class SVMUtils {
public:
	/**
	 * Load an SVM model
	 */
	static SVMModelPtr loadModel(const char* modelFileName) {
		return svm_load_model(modelFileName);
	}


	/**
	 * Read an SVM input data file and store the values in a vector
	 */
	static std::vector<SVMData> readSVMInputFile(const char* dataFileName) {
		std::vector<SVMData> data;
		
		if (dataFileName != NULL) {
			FILE *dataFile = fopen(dataFileName, "rb");
			if (dataFile == NULL) {
				fprintf(stderr, "Error: Cannot open %s for reading.", dataFileName);
				std::cin.get();
				exit(EXIT_FAILURE);
			}

			int total = 0;
			int max_nr_attr = 1200;
			int maxLineLen = 1024;
			char *line = (char *) malloc(maxLineLen * sizeof(char));

			// Read file
			while ((line = readLine(dataFile, line)) != NULL) {
				SVMData svmData;
				int i = 0;
				double target_label;
				char *idx, *val, *label, *endptr;
				int inst_max_index = -1; 

				label = strtok(line," \t\n");
				if(label == NULL) {
					fprintf(stderr, "Error at line %d of file: %s\n", (total + 1), dataFileName);
					exit(EXIT_FAILURE);
				}

				target_label = strtod(label, &endptr);
				if(endptr == label || *endptr != '\0') {
					fprintf(stderr, "Error at line %d of file: %s\n", (total + 1), dataFileName);
					exit(EXIT_FAILURE);
				}

				svmData.classNum = (int) target_label;

				while(1) {
					SVMNode attributes;
					if(i >= max_nr_attr - 1) {
						max_nr_attr *= 2;						
					}

					idx = strtok(NULL, ":");
					val = strtok(NULL, " \t");

					if(val == NULL) break;
					
					errno = 0;
					attributes.index = (int) strtol(idx, &endptr, 10);
					
					if(endptr == idx || errno != 0 || *endptr != '\0' || attributes.index <= inst_max_index) {
						fprintf(stderr, "Error at line %d of file: %s\n", (total + 1), dataFileName);
						exit(EXIT_FAILURE);
					} else {
						inst_max_index = attributes.index;
					}

					errno = 0;
					attributes.value = strtod(val, &endptr);
					if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
						fprintf(stderr, "Error at line %d of file: %s\n", (total + 1), dataFileName);
						exit(EXIT_FAILURE);
					}
					++i;
					svmData.nodes.push_back(attributes);

				} // while (1) 

				++total;
				data.push_back(svmData);
			
			} // while (readLine)

		} else {
			fprintf(stderr, "Error: No filename provided for opening");
			exit(EXIT_FAILURE);
		}

		return data;

	}


	static SVMNodePtr convertNodeVectorToArray(const std::vector<SVMNode>& svmNode) {
		// +1 for the -1 to signify end of row for LibSVM
		SVMNodePtr svmNodeArr = (SVMNodePtr) malloc((svmNode.size() + 1) * sizeof(SVMNode));
		
		size_t i = 0;
		for (; i < svmNode.size(); ++i) {
			svmNodeArr[i].index = svmNode[i].index;
			svmNodeArr[i].value = svmNode[i].value;
		}
		svmNodeArr[i].index = -1;

		return svmNodeArr;
	}

private:
	static char* readLine(FILE *input, char* line) {
		int len;
		int maxLineLen = 1024;

		if (line == NULL) {
			fprintf(stderr, "Error: Null pointer provided for reading file line");
			exit(EXIT_FAILURE);
		}

		if(fgets(line, maxLineLen, input) == NULL) return NULL;

		while(strrchr(line, '\n') == NULL) {
			maxLineLen *= 2;
			line = (char *) realloc(line, maxLineLen);
			len = (int) strlen(line);
			if(fgets(line + len, maxLineLen - len, input) == NULL) break;
		}
		return line;
	}
	
};


} /* ocl */

#endif
