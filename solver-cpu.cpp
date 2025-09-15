#include <string>
#include "var.h"
#include "print.hpp"

void integrate_cpu(q_prim* qp, q_cons* qc, q_cons* qc_new, q_cons* flux, double dt) {
    // Compute flux
    for (int i = 0; i < mp; i++) {
        flux[i].rho = qp[i].rho * qp[i].u;
        flux[i].mom = qp[i].rho * qp[i].u * qp[i].u + qp[i].p;
        flux[i].E = qp[i].u * (qc[i].E + qp[i].p);
    }

    // Update conservative variables
    for (int i = 1; i < m; i++) {
        qc_new[i].rho = 0.5*(qc[i - 1].rho + qc[i + 1].rho) + dt/(2*dx) * (flux[i - 1].rho - flux[i + 1].rho);
        qc_new[i].mom = 0.5*(qc[i - 1].mom + qc[i + 1].mom) + dt/(2*dx) * (flux[i - 1].mom - flux[i + 1].mom);
        qc_new[i].E   = 0.5*(qc[i - 1].E + qc[i + 1].E) + dt/(2*dx) * (flux[i - 1].E - flux[i + 1].E);
    }

    // Update conservative & primitive variables
    for (int i = 1; i < m; i++){
        qc[i] = qc_new[i];
        qp[i].rho = qc[i].rho;
        qp[i].u = qc[i].mom / qc[i].rho;
        qp[i].p = (gam - 1) * (qc[i].E - 0.5 * qp[i].rho * qp[i].u * qp[i].u);
    }
}

void cpuSolver(q_prim* qp, q_cons* qc, double* x_cb, double dt) {
    double t = 0.0;
    int i = 1;
    q_cons* flux = new q_cons[mp];
    q_cons* qc_new = new q_cons[mp];

    // Time integration loop
    while (t < t_max) {
        if (t + dt > t_max) {
            dt = t_max - t;
        }

        integrate_cpu(qp, qc, qc_new, flux, dt);

        t += dt;

        if (t >= i*tsave || t == t_max) {
            print_data("results/cpu_t_"+to_string(i)+".dat", qp, x_cb, t);
            i++;
        }
    }

    // Free memory
    delete[] flux;
    delete[] qc_new;
}
