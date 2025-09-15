#include <cuda_runtime.h>
#include <string>
#include "var.h"
#include "print.hpp"

__global__ void integrate_gpu(q_prim* d_qp, q_cons* d_qc, q_cons* d_qc_new, q_cons* d_flux, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute flux
    if (i < mp) {
        d_flux[i].rho = d_qp[i].rho * d_qp[i].u;
        d_flux[i].mom = d_qp[i].rho * d_qp[i].u * d_qp[i].u + d_qp[i].p;
        d_flux[i].E = d_qp[i].u * (d_qc[i].E + d_qp[i].p);
    }

    __syncthreads();

    // Update conservative variables
    if (i > 0 && i < m) {
        d_qc_new[i].rho = 0.5*(d_qc[i - 1].rho + d_qc[i + 1].rho) + dt/(2*dx) * (d_flux[i - 1].rho - d_flux[i + 1].rho);
        d_qc_new[i].mom = 0.5*(d_qc[i - 1].mom + d_qc[i + 1].mom) + dt/(2*dx) * (d_flux[i - 1].mom - d_flux[i + 1].mom);
        d_qc_new[i].E = 0.5*(d_qc[i - 1].E + d_qc[i + 1].E) + dt/(2*dx) * (d_flux[i - 1].E - d_flux[i + 1].E);
    }

    __syncthreads();

    // Update conservative & primitive variables
    if (i > 0 && i < m) {
        d_qc[i] = d_qc_new[i];
        d_qp[i].rho = d_qc[i].rho;
        d_qp[i].u = d_qc[i].mom / d_qc[i].rho;
        d_qp[i].p = (gam - 1) * (d_qc[i].E - 0.5 * d_qp[i].rho * d_qp[i].u * d_qp[i].u);
    }
}

void gpuSolver(q_prim* d_qp, q_cons* d_qc,  double* x_cb, double dt){
    double t = 0.0;
    int i = 1;
    int blockSize = 128;
    int gridSize = (m + blockSize) / blockSize;

    q_prim* qp_tmp = new q_prim[mp];
    q_cons *d_qc_new, *d_flux;
    cudaMalloc((void**)&d_qc_new, mp * sizeof(q_cons));
    cudaMalloc((void**)&d_flux, mp * sizeof(q_cons));
    
    while (t < t_max) {
        if (t + dt > t_max) {
            dt = t_max - t;
        }

        integrate_gpu<<<gridSize, blockSize>>>(d_qp, d_qc, d_qc_new, d_flux, dt);

        t += dt;

        if (t >= i*tsave || t == t_max) {
            cudaMemcpy(qp_tmp, d_qp, mp * sizeof(q_prim), cudaMemcpyDeviceToHost);
            print_data("results/gpu_t_"+to_string(i)+".dat", qp_tmp, x_cb, t);
            i++;
        }
    }

    // Free memory
    delete[] qp_tmp;
    cudaFree(d_qc_new);
    cudaFree(d_flux);
}
