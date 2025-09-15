#pragma once

// Input variables
const double gam = 1.4;     // Specific heat ratio of air
const int m = 16384;        // Number of grid cells (0, ..., m)
const int mp = m + 1;       // Number of grid points (0, ..., mp = m + 1)
const double L = 1.0;       // Length of the domain
const double CFL = 0.5;     // CFL number
const double t_max = 0.14;  // Maximum time
const double dx = L / m;    // Grid spacing
const double tsave = 0.01;  // Saving time interval

// Primitive variables
struct q_prim {
    double rho;  // Density
    double u;    // Velocity
    double p;    // Pressure
};

// Conservative variables
struct q_cons {
    double rho;  // Density
    double mom;  // Momentum (rho * u)
    double E;    // Total energy
};
