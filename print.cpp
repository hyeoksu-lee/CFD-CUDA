#include <fstream>
#include "var.h"
using namespace std;

// Print flow variables
void print_data(string filename, q_prim* qp, double* x_cb, double t){
    ofstream outfile(filename);
    outfile << t << "\n";
    for (int i = 0; i < mp; i++) {
        outfile << x_cb[i] << " " << qp[i].rho << " " << qp[i].u << " " << qp[i].p << "\n";
    }
    outfile.close();
}