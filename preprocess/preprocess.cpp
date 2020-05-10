#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
	ifstream infile("com-lj/com-lj.ungraph.txt");
	assert(infile.is_open());
	
	string line;
	vector<int> nodes;
	int cnt = 0;
	cout << "start..." << endl; 
	while (getline(infile, line)) {
		if (line[0] == '#') continue;
		int a, b;
		sscanf(line.c_str(), "%d\t%d", &a, &b);
		nodes.push_back(a), nodes.push_back(b);
		//cout << a << ' ' << endl;
		cnt ++;
	}
	infile.close();
	
	sort(nodes.begin(),nodes.end());
	nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
	cout << "outfile..." << endl;
	// for (auto x: nodes) cout << x << ' ';
	// cout << endl;
	ofstream outfile("com-lj/id-map.txt");
	// 1. get id-map
	for (int i = 0; i < nodes.size(); i ++)
		outfile << i << '\t' << nodes[i] << '\n';
	
	// 2. get graph.txt
	outfile.close();
	cout << "nodes: " << nodes.size() << "  edges:" << cnt << endl;
	return 0;
}
