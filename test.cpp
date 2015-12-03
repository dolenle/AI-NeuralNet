#include "network.h"
#include <iostream>
#include <iomanip>

int main() {
	using namespace std;
	string line, filename = "wdbc/wdbc.test";
	
	cout << "Neural Network File: ";
	cin >> filename;
	NeuralNet n(filename);

	cout << "Testing Dataset: ";
	cin >> filename;

	double result[n.get_no()];
	double sample[n.get_ni()];
	int nsamp;
	
	ifstream samples(filename.c_str());
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

	cout << "Output file: ";
	cin >> filename;
	ofstream out(filename.c_str());
	out.setf(ios::fixed, ios::floatfield);

	int A_tot=0, B_tot=0, C_tot=0, D_tot=0;
	double m_accuracy=0, m_precision=0, m_recall=0;

	for(int j=0; j<n.get_no(); j++) {
		int A=0,B=0,C=0,D=0;
		for(int i=0; i<nsamp; i++) {
			A += actualClass[j][i] & estClass[j][i];
			B += !actualClass[j][i] & estClass[j][i];
			C += actualClass[j][i] & !estClass[j][i];
			D += !actualClass[j][i] & !estClass[j][i];
		}
		A_tot += A;
		B_tot += B;
		C_tot += C;
		D_tot += D;
		double accuracy = ((double) A+D)/(A+B+C+D);
		double precision = A/((double) A+B);
		double recall = A/((double) A+C);
		double f1 = (2*precision*recall)/(precision+recall);
		m_accuracy += accuracy;
		m_precision += precision;
		m_recall += recall;

		out << setprecision(3) << A <<" "<< B <<" "<< C <<" "<< D <<" "<< accuracy <<" "<< precision <<" "<< recall <<" "<< f1 << endl;
	}
	double u_accuracy = ((double) A_tot+D_tot)/(A_tot+B_tot+C_tot+D_tot);
	double u_precision = A_tot/((double) A_tot+B_tot);
	double u_recall = A_tot/((double) A_tot+C_tot);
	double u_f1 = (2*u_precision*u_recall)/(u_precision+u_recall);

	out << setprecision(3) << u_accuracy <<" "<< u_precision <<" "<< u_recall <<" "<< u_f1 << endl;

	m_accuracy /= n.get_no();
	m_precision /= n.get_no();
	m_recall /= n.get_no();
	double m_f1 = (2*m_precision*m_recall)/(m_precision+m_recall);

	out << setprecision(3) << m_accuracy <<" "<< m_precision <<" "<< m_recall <<" "<< m_f1 << endl;

	samples.close();
	out.close();
	return 0;
}