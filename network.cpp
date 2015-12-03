#include "network.h"

NeuralNet::NeuralNet(int ni, int nh, int no) : ni(ni), nh(nh), no(no) {
	hidden_weights = new double*[nh];
	for(int i=0; i<nh; i++) {
		hidden_weights[i] = new double[ni+1];
	}
	output_weights = new double*[no];
	for(int i=0; i<no; i++) {
		output_weights[i] = new double[nh+1];
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
	
	hidden_weights = new double*[nh];
	for(int i=0; i<nh; i++) {
		hidden_weights[i] = new double[ni+1];
	}
	output_weights = new double*[no];
	for(int i=0; i<no; i++) {
		output_weights[i] = new double[nh+1];
	}

	for(int j=0; j<nh; j++) {
		if(!getline(in, line)) {
			cerr << "Parse error" << endl;
			exit(-1);
		}
		stringstream linestream(line);
		for(int i=0; i<=ni; i++) {
			linestream >> hidden_weights[j][i];
			// cout << "hW=" << hidden_weights[j][i] << endl;
		}
	}
	for(int j=0; j<no; j++) {
		if(!getline(in, line)) {
			cerr << "Parse error" << endl;
			exit(-1);
		}
		stringstream linestream(line);
		for(int i=0; i<=nh; i++) {
			linestream >> output_weights[j][i];
			// cout << "hO=" << output_weights[j][i] << endl;
		}
	}
	in.close();
}

//Assign weight for edge between hidden node j and input node i
void NeuralNet::set_hidden_weight(int i, int j, double weight) {
	hidden_weights[j][i] = weight;
}

void NeuralNet::set_output_weight(int i, int j, double weight) {
	output_weights[j][i] = weight;
}

double NeuralNet::get_hidden_weight(int i, int j) {
	return hidden_weights[j][i];
}

double NeuralNet::get_output_weight(int i, int j) {
	return output_weights[j][i];
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
	double hidden_act[nh];
	for(int j=0; j<nh; j++) {
		double sum = -1*hidden_weights[j][0];
		for(int i=0; i<ni; i++) {
			sum+=input[i]*hidden_weights[j][i+1];
		}
		hidden_act[j] = sigmoid(sum);
	}
	for(int j=0; j<no; j++) {
		double sum = -1*output_weights[j][0];
		for(int i=0; i<nh; i++) {
			sum+=hidden_act[i]*output_weights[j][i+1];
		}
		output[j] = sigmoid(sum);
	}
	return output;
}

void NeuralNet::train(double* target, double* input, double rate) {
	using namespace std;
	double hidden_input[nh], hidden_act[nh];
	double hidden_error[nh], output_error[no];
	for(int j=0; j<nh; j++) {
		double sum = -1*hidden_weights[j][0];
		for(int i=0; i<ni; i++) {
			sum+=input[i]*hidden_weights[j][i+1];
		}
		hidden_input[j] = sum;
		hidden_act[j] = sigmoid(sum);
	}
	
	for(int j=0; j<no; j++) {
		double sum = -1*output_weights[j][0];
		for(int i=0; i<nh; i++) {
			sum+=hidden_act[i]*output_weights[j][i+1];
		}
		output_error[j] = sigmoidPrime(sum)*(target[j]-sigmoid(sum));
	}

	//backpropagate to hidden layer and update
	for(int i=0; i<nh; i++) {
		double sum = 0;
		for(int j=0; j<no; j++) {
			sum+=output_weights[j][i+1]*output_error[j];
		}
		hidden_error[i] = sigmoidPrime(hidden_input[i])*sum;
		
		hidden_weights[i][0] += rate*-1*hidden_error[i];;
		for(int j=0; j<ni; j++) {
			hidden_weights[i][j+1] += rate*input[j]*hidden_error[i];
		}
	}


	//update weights to output layer
	for(int j=0; j<no; j++) {
		output_weights[j][0] += -rate*output_error[j];
		for(int i=0; i<nh; i++) {
			output_weights[j][i+1] += rate*hidden_act[i]*output_error[j];
		}
	}

}

double NeuralNet::sigmoid(double x) {
	return 1/(1+exp(-x));
}

double NeuralNet::sigmoidPrime(double x) {
	return sigmoid(x)*(1-sigmoid(x));
}