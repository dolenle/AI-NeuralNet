#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class NeuralNet {
private:
	int ni; //number of input nodes
	int nh; //number of hidden nodes
	int no; //number of outputs

	double** hiddenWeights;
	double** outputWeights;

public:
	NeuralNet(int ni, int nh, int no);
	NeuralNet(std::string filename);
	void set_hidden_weight(int i, int j, double weight);
	void set_output_weight(int i, int j, double weight);
	double get_hidden_weight(int i, int j);
	double get_output_weight(int i, int j);
	double* compute(double* input, double* output);

	int get_ni();
	int get_nh();
	int get_no();

	static double sigmoid(double x);
};