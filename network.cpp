#include "network.h"
using namespace std;

NeuralNet::NeuralNet(int ni, int nh, int no) : ni(ni), nh(nh), no(no) {
	hiddenWeights = new double*[nh];
	for(int i=0; i<nh; i++) {
		hiddenWeights[i] = new double[ni+1];
	}
	outputWeights = new double*[no];
	for(int i=0; i<no; i++) {
		outputWeights[i] = new double[nh+1];
	}
}

//Get weight for edge between hidden node j and input node i
void NeuralNet::set_hidden_weight(int i, int j, double weight) {
	hiddenWeights[j][i] = weight;
}

void NeuralNet::set_output_weight(int i, int j, double weight) {
	outputWeights[j][i] = weight;
}

double NeuralNet::get_hidden_weight(int i, int j) {
	return hiddenWeights[j][i];
}

double NeuralNet::get_output_weight(int i, int j) {
	return outputWeights[j][i];
}

double* NeuralNet::compute(double* input, double* output) {
	double hiddenAct[nh];
	for(int j=0; j<nh; j++) {
		double sum = -1*hiddenWeights[j][0];
		for(int i=0; i<ni; i++) {
			sum+=input[i]*hiddenWeights[j][i+1];
		}
		hiddenAct[j] = sigmoid(sum);
	}
	for(int j=0; j<no; j++) {
		double sum = -1*outputWeights[j][0];
		for(int i=0; i<nh; i++) {
			sum+=hiddenAct[i]*outputWeights[j][i+1];
		}
		output[j] = sigmoid(sum);
	}
	return output;
}

double NeuralNet::sigmoid(double x) {
	return 1/(1+exp(-x));
}

int main() {
	return 0;
}