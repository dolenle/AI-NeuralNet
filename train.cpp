#include "network.h"
#include <iostream>
#include <iomanip>

int main() {
	using namespace std;
	string filename;
	int epochs;
	double rate;

	cout << "Initial Neural Network: ";
	cin >> filename;
	NeuralNet n(filename);

	cout << "Training Dataset: ";
	cin >> filename;

	cout << "Number of Ehpocks: ";
	cin >> epochs;

	cout << "Learning Rate: ";
	cin >> rate;

	int nsamp, samp_ni, samp_no;
	string line;
	ifstream samp_file(filename.c_str());
	getline(samp_file, line);
	stringstream(line) >> nsamp >> samp_ni >> samp_no;
	
	cout << "numSamples=" << nsamp << endl;
	if(samp_ni != n.get_ni() || samp_no != n.get_no()) {
		cerr << "Sample set does not match network" << endl;
		exit(-1);
	}
	
	double sample[nsamp][n.get_ni()];
	double target[nsamp][n.get_no()];

	//read file into array
	for(int x=0; x<nsamp; x++) {
		if(!getline(samp_file, line)) {
			cerr << "Insufficient lines in file" << endl;
			exit(-1);
		}
		stringstream linestream(line);
		for(int i=0; i<n.get_ni(); i++) {
			linestream >> sample[x][i];
		}
		for(int i=0; i<n.get_no(); i++) {
			linestream >> target[x][i];
		}
	}

	samp_file.close();

	for(int e=0; e<epochs; e++) {
		for(int x=0; x<nsamp; x++) {
			n.train(target[x], sample[x], rate);
		}
	}

	cout << "Done training. Output file: ";
	cin >> filename;

	ofstream out(filename.c_str());
	if(!out.is_open()) {
		cerr << "Cannot open file" << endl;
		exit(-1);
	}

	out << samp_ni << " " << n.get_nh() << " " << samp_no << endl;
	out.setf(ios::fixed, ios::floatfield);
	for(int j=0; j<n.get_nh(); j++) {
		for(int i=0; i<=n.get_ni(); i++) {
			out << setprecision(3) << n.get_hidden_weight(i, j) << " ";
		}
		out.seekp(-1, ios::cur);
		out << endl;
	}
	for(int j=0; j<n.get_no(); j++) {
		for(int i=0; i<=n.get_nh(); i++) {
			out << setprecision(3) << n.get_output_weight(i, j) << " ";
		}
		out.seekp(-1, ios::cur);
		out << endl;
	}
	out.close();
}