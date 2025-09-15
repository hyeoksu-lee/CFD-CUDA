#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "var.h"
#include "print.hpp"
#include "solver-cpu.hpp"
#include "solver-gpu.cuh"

using namespace std;

/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *         what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Generate grid points (cell boundary value)
double *generate_grid_cb(double start, double end) {
    double* x_cb = new double[mp];
    for(int i = 0; i < mp; i++) {
        x_cb[i] = start + dx * i;
    }
    return x_cb;
}

// Initialize primitive variables
q_prim *initialize_prim_var(double* x_cb, q_prim left, q_prim right){
    q_prim* qp = new q_prim[mp];
    for (int i = 0; i < mp; i++) {
        if (x_cb[i] < 0.5) {
            qp[i] = left;
        } 
        else {
            qp[i] = right;
        }
    }
    return qp;
}

// Initialize conservative variables
q_cons *initialize_cons_var(q_prim* qp){
    q_cons* qc = new q_cons[mp];
    for (int i = 0; i < mp; i++) {
        qc[i].rho = qp[i].rho;
        qc[i].mom = qp[i].rho * qp[i].u;
        qc[i].E = qp[i].p / (gam - 1) + 0.5 * qp[i].rho * qp[i].u * qp[i].u;
    }
    return qc;
}

// Compute dt based on maximum signal speed
double compute_dt(q_prim* qp){
    double s_max = 0.0;
    for (int i = 0; i < mp; i++) {
        double s = abs(qp[i].u) + sqrt(gam * qp[i].p / qp[i].rho);
        if (s > s_max) {
            s_max = s;
        }
    }
    return CFL * dx / s_max;
}

int main(int argc, char *argv[]) {

    // Timer setup ---------------------------------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;

#define START_TIMER() {                                                    \
        gpuErrChk(cudaEventCreate(&start));                                \
        gpuErrChk(cudaEventCreate(&stop));                                 \
        gpuErrChk(cudaEventRecord(start));                                 \
    }

#define STOP_RECORD_TIMER(name) {                                          \
        gpuErrChk(cudaEventRecord(stop));                                  \
        gpuErrChk(cudaEventSynchronize(stop));                             \
        gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
        gpuErrChk(cudaEventDestroy(start));                                \
        gpuErrChk(cudaEventDestroy(stop));                                 \
    }

    // Initialize timers
    float cpu_ms = -1;
    float gpu_ms = -1;

    // Initialize grid -----------------------------------------------------
    double* x_cb = generate_grid_cb(0.0, L);

    // Initialize flow variables -------------------------------------------
    q_prim left = {0.44, 0.698, 3.528}; // Left q_prim
    q_prim right = {0.5, 0.0, 0.571};   // Right q_prim

    q_prim* qp = initialize_prim_var(x_cb, left, right);
    q_cons* qc = initialize_cons_var(qp);
    print_data("results/cpu_t_0.dat", qp, x_cb, 0.0);
    print_data("results/gpu_t_0.dat", qp, x_cb, 0.0);

    // Initialize time integration -----------------------------------------
    double dt = compute_dt(qp);

    // GPU setup -----------------------------------------------------------
    // Allocate memory
    q_prim *d_qp;
    q_cons *d_qc;
    cudaMalloc(&d_qp, mp * sizeof(q_prim));
    cudaMalloc(&d_qc, mp * sizeof(q_cons));

    // Copy input to GPU
    cudaMemcpy(d_qp, qp, mp * sizeof(q_prim), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qc, qc, mp * sizeof(q_cons), cudaMemcpyHostToDevice);

    // Run solver ----------------------------------------------------------
    // CPU implementation
    START_TIMER();
    cpuSolver(qp, qc, x_cb, dt);
    STOP_RECORD_TIMER(cpu_ms);
    printf("CPU: %f ms\n", cpu_ms);

    // GPU implementation
    START_TIMER();
    gpuSolver(d_qp, d_qc, x_cb, dt);
    STOP_RECORD_TIMER(gpu_ms);
    printf("GPU: %f ms\n", gpu_ms);

    // Comparison
    printf("GPU is %f times faster than CPU.", cpu_ms/gpu_ms);
    
    // Free memory ---------------------------------------------------------
    delete[] x_cb;
    delete[] qp;
    delete[] qc;
    cudaFree(d_qp);
    cudaFree(d_qc);

    printf("\n");
}
