#include "network.h"

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

NeuralNet::NeuralNet(std::string file) {
	using namespace std;
	string line;
	ifstream in(file.c_str());
	if(!in.is_open()) {
		cerr << "Cannot open file" << endl;
		exit(-1);
	}
	getline(in, line);
	stringstream(line) >> ni >> nh >> no;
	cout << "Ni=" << ni << endl;
	cout << "Nh=" << nh << endl;
	cout << "No=" << no << endl;
	
	hiddenWeights = new double*[nh];
	for(int i=0; i<nh; i++) {
		hiddenWeights[i] = new double[ni+1];
	}
	outputWeights = new double*[no];
	for(int i=0; i<no; i++) {
		outputWeights[i] = new double[nh+1];
	}

	for(int j=0; j<nh; j++) {
		if(!getline(in, line)) {
			cerr << "Parse error" << endl;
			exit(-1);
		}
		stringstream linestream(line);
		for(int i=0; i<=ni; i++) {
			linestream >> hiddenWeights[j][i];
			// cout << "hW=" << hiddenWeights[j][i] << endl;
		}
	}
	for(int j=0; j<no; j++) {
		if(!getline(in, line)) {
			cerr << "Parse error" << endl;
			exit(-1);
		}
		stringstream linestream(line);
		for(int i=0; i<=nh; i++) {
			linestream >> outputWeights[j][i];
			// cout << "hO=" << outputWeights[j][i] << endl;
		}
	}
	in.close();
}

//Assign weight for edge between hidden node j and input node i
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

int NeuralNet::get_ni() {
	return ni;
}

int NeuralNet::get_nh() {
	return nh;
}

int NeuralNet::get_no() {
	return no;
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
	using namespace std;
	NeuralNet n("wdbc/sample.NNWDBC.1.100.trained");
	double result[n.get_no()];
	double sample[n.get_ni()];

	int nsamp;
	string line, sampfile = "wdbc/wdbc.test";
	ifstream samples(sampfile.c_str());
	getline(samples, line);
	stringstream(line) >> nsamp;
	cout << "numSamples=" << nsamp << endl;

	int actualClass[n.get_no()][nsamp];
	int estClass[n.get_no()][nsamp];

	for(int x=0; x<nsamp; x++) {
		getline(samples, line);
		stringstream linestream(line);
		for(int i=0; i<n.get_ni(); i++) {
			linestream >> sample[i];
		}
		n.compute(sample, result);
		for(int i=0; i<n.get_no(); i++) {
			linestream >> actualClass[i][x];
			estClass[i][x] = round(result[i]);
		}
	}

	for(int j=0; j<n.get_no(); j++) {
		int A=0,B=0,C=0,D=0;
		for(int i=0; i<nsamp; i++) {
			// cout << "ac=" << actualClass[j][i] << " eC=" << estClass[j][i] << endl;
			A+=actualClass[j][i] & estClass[j][i];
			B+=!actualClass[j][i] & estClass[j][i];
			C+=actualClass[j][i] & !estClass[j][i];
			D+=!actualClass[j][i] & !estClass[j][i];
		}
		cout << "A=" << A << endl;
		cout << "B=" << B << endl;
		cout << "C=" << C << endl;
		cout << "D=" << D << endl;
	}
	samples.close();
	return 0;
}