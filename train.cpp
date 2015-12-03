#include "network.h"
#include <iostream>
#include <iomanip>

int main() {
	using namespace std;
	string filename;
	int epochs;
	double rate;

	// cout << "Initial Neural Network: ";
	// cin >> filename;
	// NeuralNet n(filename);
	NeuralNet n("wdbc/nn.init");

	// cout << "Training Dataset: ";
	// cin >> filename;
	filename = "wdbc/wdbc.train";

	// cout << "Number of Ehpocks: ";
	// cin >> epochs;
	epochs = 1;

	// cout << "Learning Rate: ";
	// cin >> rate;
	rate = 0.1;

	int nsamp, samp_ni, samp_no;
	string line;
	ifstream samples(filename.c_str());
	getline(samples, line);
	stringstream(line) >> nsamp >> samp_ni >> samp_no;
	
	cout << "numSamples=" << nsamp << endl;
	if(samp_ni != n.get_ni() || samp_no != n.get_no()) {
		cerr << "Sample set does not match network" << endl;
		exit(-1);
	}
	
	double sample[n.get_ni()];
	double target[n.get_no()];

	for(int e=0; e<epochs; e++) {
		for(int x=0; x<nsamp; x++) {
			getline(samples, line);
			stringstream linestream(line);
			for(int i=0; i<n.get_ni(); i++) {
				linestream >> sample[i];
			}
			for(int i=0; i<n.get_no(); i++) {
				linestream >> target[i];
			}
			n.train(target, sample, rate);
		}
	}

	cout.setf(ios::fixed,ios::floatfield);

	for(int j=0; j<n.get_nh(); j++) {
		for(int i=0; i<=n.get_ni(); i++) {
			cout << std::setprecision(3) << n.get_hidden_weight(i, j) << " ";
		}
		cout << endl;
	}
	for(int j=0; j<n.get_no(); j++) {
		for(int i=0; i<=n.get_nh(); i++) {
			cout << std::setprecision(3) << n.get_output_weight(i, j) << " ";
		}
		cout << endl;
	}

}