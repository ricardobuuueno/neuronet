#include <vector>

#include "Connection.h"


class Neuron;

typedef std::vector<Neuron> Layer;


class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal() const { return m_outputVal; }

private:
	static constexpr double eta {0.15};
	static constexpr double alpha {0.5};

	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight() { return (rand() / double(RAND_MAX)); }
	
	double sumDOW(const Layer& nextLayer) const;

	unsigned m_myIndex {};
	double m_outputVal {};
	double m_gradient {};
	std::vector<Connection> m_outputWeights {};

};

