#include <string>
#include <fstream>
#include <string>
#include <vector>

class TrainingData
{
public:
    TrainingData(const std::string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};
