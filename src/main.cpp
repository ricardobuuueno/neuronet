#include <iostream>
#include <cassert>

#include "Net.h"
#include "TrainingData.h"

void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << '\n';
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << '\n';
}

int main()
{
	TrainingData trainData {"/tmp/trainingData.txt"};

	std::vector<unsigned> topology {};
	trainData.getTopology(topology);

	Net myNet {topology};

	std::vector<double> inputVals {};
	std::vector<double> targetVals {};
	std::vector<double> resultVals {};

	int trainingPass = 0;

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << '\n' << "Pass " << trainingPass;

		if (trainData.getNextInputs(inputVals) != topology[0])
		{
			break;
		}

		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		//report how well the training is working, average over recent samples
		std::cout << "Net recent average error: " << myNet.getRecentAverageError() << '\n';
	}

	std::cout << '\n' << "Done\n";

}
