#include <cstdlib>
#include <cstdio>
#include <boost/timer.hpp>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include "ObjectDescription.h"
#include "ObjectClassifier.h"


using namespace std;

static string getClassName(const size_t & idx) {
	string categories[11] = {"Bottle", "Bowl", "Box", "Can", "Carton", "Cup", "Mug", "Spray-Can", "Tin", "Tube", "Tub"};
	
	if (idx >= 0 && idx < 11) {
		return categories[idx];
	} else {
		return "Unknown";
	}
}


/**
 * Main entry point for testing Object Classification Module
 */
int main(int argc, char *argv[]) {
	pcl::PointCloud<pcl::PointXYZRGB> loadedCloud;
	pcl::io::loadPLYFile(std::string("./MinuteMaid.ply"), loadedCloud);

	// Change these paths to reflect the layout of your machine
	ocl::ObjectDescription objDesc (".");
	ocl::ObjectClassifier classifier(".");
	{
		boost::timer t;
		objDesc.extractFeatureDescriptors(loadedCloud);
		objDesc.assignBOWs();
		classifier.classify(objDesc.getFPFHBow(), objDesc.getSIFTBow(), objDesc.getHOGBow());
		cout << "After classification\n";
		printf("Elapsed computation time: %g\n", t.elapsed());
	}
	
	printf("Object class: %s\n", getClassName(classifier.objectCategory.categoryLabel - 1).c_str());
	printf("Overall confidence value: %g\n", classifier.measureConfidence());
	printf("\t Confidence scores:\n");
	for (size_t j = 0; j < classifier.objectCategory.classConfidence.size(); j++) {
		printf("\t    %02ld -> %g\n", j + 1, classifier.objectCategory.classConfidence[j]);
	}
		
	cin.get();

	return EXIT_SUCCESS;
}
