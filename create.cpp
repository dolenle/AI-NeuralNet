#include "network.h"
#include <iostream>
#include <iomanip>

int main() {
	using namespace std;
	string filename;
	int ni, nh, no;

	cout << "Output file: ";
	cin >> filename;

	cout << "Input nodes: ";
	cin >> ni;

	cout << "Hidden nodes: ";
	cin >> nh;

	cout << "Output nodes: ";
	cin >> no;

	ofstream out(filename.c_str());

	NeuralNet n(ni, nh, no);

	out << ni << " " << n.get_nh() << " " << no << endl;
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