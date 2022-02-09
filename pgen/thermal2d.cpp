//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <vector>
#include <hdf5.h>
#include <random>
#include <boost/math/tools/roots.hpp>

#ifdef MPI_PARALLEL
#include <mpi.h>   // MPI_COMM_WORLD, MPI_INFO_NULL
#endif

// Athena++ headers
#include "../globals.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../utils/utils.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/townsend_cooling.hpp"

using namespace boost::math::tools;

//======================================================================================
/*! \file beam.cpp
 *  \brief Beam test for the radiative transfer module
 *
 *====================================================================================*/


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================

static Real gg = 5./3.;
static Real gg1 = gg/(gg - 1.);

static Real rho0 = 1.0;
static Real T0 = 1.0;
static Real g0 = 1.0;

static Real H = T0/g0;
static Real a = 0.1*H;
static Real rhoH = rho0*exp(-(a/H)*(sqrt(1. + SQR(H/a)) - 1.));

static Real beta = 10.;
static Real pgH = rhoH*T0;
static Real B = sqrt(2.*pgH/beta);

static Real delta = 0.1;
static Real L0 = (T0/((gg - 1.)*rhoH*delta))*sqrt(0.5*g0/H);
static Real T_floor = 0.01;
static Real T_ceil = 5.0;
static Real epsil = 0.5;

static Real cool_buffer_left = 3.0*a;
static Real cool_buffer_right = 1.0*a;

static Real x_min = -6.0;
static Real x_max = 6.0;
static Real y_min = -6.0;
static Real y_max = 6.0;

static Real x_min_cool_buffer_left = 0.0 + cool_buffer_left;
static Real x_max_cool_buffer_right = x_max - cool_buffer_right;

static Real rms_amp = 0.01;
static Real Lx = x_max - x_min;
static Real Ly = y_max - y_min;

static Real x_start_sum = 0.68;
static Real x_end_sum = 0.88;

void HydroConstTempInnerBoundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void HydroConstTempOuterBoundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons);

Real MsDensity(MeshBlock *pmb, int iout);

void Mesh::UserWorkInLoop() {
  // Part 1: Compute heating to be added to each grid
  MeshBlock *pmb = pblock; // Start with the first block of each processor
  int num_layer = pblock->phydro->num_layer;
  int *local_grid{new int[num_layer]()};
  int *global_grid{new int[num_layer]()};
  Real *local_heating{new Real[num_layer]()};
  Real *global_heating{new Real[num_layer]()};

  int *layer_grid;
  Real *layer_heat;
  while (pmb != nullptr) {
    layer_grid = &pmb->phydro->layer_grid(0);
    layer_heat = &pmb->phydro->layer_heat(0);
    for (int l=0; l<num_layer; ++l) {
      local_grid[l] += layer_grid[l];
      local_heating[l] += layer_heat[l];
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(local_grid, global_grid, num_layer, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_heating, global_heating, num_layer, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int l=0; l<num_layer; ++l) {
    global_heating[l] /= global_grid[l];
  }

  pmb = pblock;
  while (pmb != nullptr) {
    Hydro *ph = pmb->phydro;
    Field *pf = nullptr;
    if (MAGNETIC_FIELDS_ENABLED) {
      pf = pmb->pfield;
    }

    Real il = pmb->is, iu = pmb->ie;
    Real jl = pmb->js, ju = pmb->je;
    Real kl = pmb->ks, ku = pmb->ke;

    for (int i=il; i<=iu; ++i) {
      Real x1 = pmb->pcoord->x1v(i);
      Real *xf = &ph->xf(0);
      int ind = 0;
      while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}
      for (int j=jl; j<=ju; ++j) {
        for (int k=kl; k<=ku; ++k) {

          if (time <= 5.0) {
            ph->u(IEN,k,j,i) += global_heating[ind];
          }

        }
      }
    }
    // Update the primitive variables
    pmb->peos->ConservedToPrimitive(ph->u, ph->w1, pf->b, ph->w, pf->bcc, pmb->pcoord,
      il, iu, jl, ju, kl, ku);
    pmb = pmb->next;
  }

  delete [] local_grid;
  delete [] global_grid;
  delete [] local_heating;
  delete [] global_heating;



  // Part 2: Compute mean density of each layer and number of grids
  // within the volumetric averaging region
  pmb = pblock; // Start with the first block of each processor
  int local_grid_sum = 0;
  int global_grid_sum = 0;
  int *local_num_grid{new int[num_layer]()};
  int *global_num_grid{new int[num_layer]()};
  Real *local_mean_rho{new Real[num_layer]()};
  Real *global_mean_rho{new Real[num_layer]()};
  while (pmb != nullptr) {
    Hydro *ph = pmb->phydro;

    Real il = pmb->is, iu = pmb->ie;
    Real jl = pmb->js, ju = pmb->je;
    Real kl = pmb->ks, ku = pmb->ke;

    for (int i=il; i<=iu; ++i) {
      Real x1 = pmb->pcoord->x1v(i);
      Real *xf = &ph->xf(0);
      int ind = 0;
      while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}
      for (int j=jl; j<=ju; ++j) {
        for (int k=kl; k<=ku; ++k) {

          if ((x1 > x_start_sum) && (x1 < x_end_sum)) {
            local_grid_sum += 1;
          }

          local_num_grid[ind] += 1;
          local_mean_rho[ind] += ph->u(IDN,k,j,i);

        }
      }
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(&local_grid_sum, &global_grid_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_num_grid, global_num_grid, num_layer, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_mean_rho, global_mean_rho, num_layer, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int l=0; l<num_layer; ++l) {
    global_mean_rho[l] /= global_num_grid[l];
  }

  pmb = pblock;
  while (pmb != nullptr) {
    Hydro *ph = pmb->phydro;
    for (int l=0; l<num_layer; ++l) {
      ph->mean_rho(l) = global_mean_rho[l];
    }
    ph->num_grid_sum = global_grid_sum;
    pmb = pmb->next;
  }

  delete [] local_num_grid;
  delete [] global_num_grid;
  delete [] local_mean_rho;
  delete [] global_mean_rho;

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {    
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Enroll user boundary functions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroConstTempInnerBoundary);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] ==  GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroConstTempOuterBoundary);
  }
  // Enroll source terms
  EnrollUserExplicitSourceFunction(SourceTerm);
  // Allocate and enroll history output
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, MsDensity, "drho_ms", UserHistoryOperation::sum);
  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  std::default_random_engine amp_gen;
  std::default_random_engine phase_xgen;
  std::default_random_engine phase_ygen;
  std::normal_distribution<double> amp_dist(0.0, rms_amp);
  std::uniform_real_distribution<double> phase_xdist(0.0, 2*PI);
  std::uniform_real_distribution<double> phase_ydist(0.0, 2*PI);
  int n_min = 1;
  int n_max = 20;
  int total_mode = (n_max - n_min + 1)*(n_max - n_min + 1);

  std::vector<std::vector<Real>> amp((n_max - n_min + 1), std::vector<Real>(n_max - n_min + 1));
  std::vector<std::vector<Real>> phase_x((n_max - n_min + 1), std::vector<Real>(n_max - n_min + 1));
  std::vector<std::vector<Real>> phase_y((n_max - n_min + 1), std::vector<Real>(n_max - n_min + 1));
  for (int n=0; n<=(n_max - n_min); ++n) {
    for (int m=0; m<=(n_max - n_min); ++m) {
      Real amp_rand = amp_dist(amp_gen);
      Real phase_xrand = phase_xdist(phase_xgen);
      Real phase_yrand = phase_ydist(phase_ygen);
      amp[n][m] = amp_rand;
      phase_x[n][m] = phase_xrand;
      phase_y[n][m] = phase_yrand;
    }
  }

  // Initialize hydro variable
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        Real x1 = pcoord->x1v(i);
        Real y1 = pcoord->x2v(j);
        Real rho = rho0*exp(-(a/H)*(sqrt(1. + SQR(x1/a)) - 1.));
        Real pg = rho*T0;

        Real dd = 0.0; Real dd2 = 0.0;
        for (int n=0; n<=(n_max - n_min); ++n) {
          for (int m=0; m<=(n_max - n_min); ++m) {
            int n_now = n_min + n;
            int m_now = n_min + m;
            dd += (amp[n][m]/sqrt(total_mode))*sin(2.*PI*n_now*x1/Lx + phase_x[n][m])*sin(2.*PI*m_now*y1/Ly + phase_y[n][m]);
          }
        }
        
        // Initialize hydro variables
        
        phydro->u(IDN,k,j,i) = rho*(1. + dd);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = pg/(gg - 1.);
        }

      } //end i
    } //end j
  } //end k

  // Add horizontal magnetic field lines, to show streaming and diffusion 
  // along magnetic field ines
  if(MAGNETIC_FIELDS_ENABLED){

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          Real x1 = pcoord->x1v(i);
          pfield->b.x1f(k,j,i) = B;
        }
      }
    }

    if(block_size.nx2 > 1){

      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }

    }

    if(block_size.nx3 > 1){

      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
    }// end nx3

    // set cell centerd magnetic field
    // Add magnetic energy density to the total energy
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        for(int i=is; i<=ie; ++i){
          phydro->u(IEN,k,j,i) +=
            0.5*(SQR((pfield->bcc(IB1,k,j,i)))
               + SQR((pfield->bcc(IB2,k,j,i)))
               + SQR((pfield->bcc(IB3,k,j,i))));
      
        }
      }
    }

  }// end MHD

  // Calculate rms density fluctuations
  // To save computation only do at last meshblock of each process
  if (next == nullptr) {
    MeshBlock *pb = pmy_mesh->pblock;
    int local_grid_sum = 0;
    int global_grid_sum = 0;
    int num_layer = pb->phydro->num_layer;
    int *local_num_grid{new int[num_layer]()};
    int *global_num_grid{new int[num_layer]()};
    Real *local_mean_rho{new Real[num_layer]()};
    Real *global_mean_rho{new Real[num_layer]()};
    while (pb != nullptr) {
      Hydro *ph = pb->phydro;
      Real il = pb->is, iu = pb->ie;
      Real jl = pb->js, ju = pb->je;
      Real kl = pb->ks, ku = pb->ke;
      for (int i=is; i<=ie; ++i) {
        Real x1 = pb->pcoord->x1v(i);
        Real *xf = &ph->xf(0);
        int ind = 0;
        while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}
        for (int j=jl; j<=ju; ++j) {
          for (int k=kl; k<=ku; ++k) {

            if ((x1 > x_start_sum) && (x1 < x_end_sum)) {
              local_grid_sum += 1;
            }

            local_num_grid[ind] += 1;
            local_mean_rho[ind] += ph->u(IDN,k,j,i);

          }
        }
      }
      pb = pb->next;
    }

#ifdef MPI_PARALLEL
      MPI_Allreduce(&local_grid_sum, &global_grid_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(local_num_grid, global_num_grid, num_layer, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(local_mean_rho, global_mean_rho, num_layer, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

    for (int l=0; l<num_layer; ++l) {
      global_mean_rho[l] /= global_num_grid[l];
    }

    pb = pmy_mesh->pblock;
    while (pb != nullptr) {
      Hydro *ph = pb->phydro;
      for (int l=0; l<num_layer; ++l) {
        ph->mean_rho(l) = global_mean_rho[l];
      }
      ph->num_grid_sum = global_grid_sum;
      pb = pb->next;
    }

    delete [] local_num_grid;
    delete [] global_num_grid;
    delete [] local_mean_rho;
    delete [] global_mean_rho;
  }

  return;
}// end ProblemGenerator

void HydroConstTempInnerBoundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        
        Real x1 = pco->x1v(is-i);
        Real x2 = pco->x1v(is-i+1);
        Real x2f = pco->x1f(is-i+1);
        Real dx = x2 - x1;
        
        Real rho2 = prim(IDN,k,j,is-i+1);
        Real pg2 = prim(IPR,k,j,is-i+1);

        Real v_is = prim(IVX,k,j,is);

        auto g = [=](Real x) {
          return g0*(x/a)/sqrt(1. + SQR(x/a));
        };

        Real pg1 = (pg2 + 0.5*g(x2f)*dx*rho2)/(1. - 0.5*g(x2f)*dx*rho2/pg2);
        Real rho1 = rho2*pg1/pg2;
        Real v1 = v_is > 0.0 ? 0.0 : v_is;

        prim(IDN,k,j,is-i) = rho1;
        prim(IVX,k,j,is-i) = v1;
        prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
        prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
        prim(IPR,k,j,is-i) = pg1;
        
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) { 
      for (int j=js; j<=je; ++j) { 
        for (int i=1; i<=(NGHOST); ++i) { 
          b.x1f(k,j,is-i) = b.x1f(k,j,is); 
        } 
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
        }
      }
    }      
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
        }
      }
    }
  }
  return;
}// end HydroConstTempInnerBoundary

void HydroConstTempOuterBoundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        
        Real x1 = pco->x1v(ie+i);
        Real x1f = pco->x1f(ie+i);
        Real x2 = pco->x1v(ie+i-1);
        Real dx = x1 - x2;
        
        Real rho2 = prim(IDN,k,j,ie+i-1);
        Real pg2 = prim(IPR,k,j,ie+i-1);

        Real v_ie = prim(IVX,k,j,ie);

        auto g = [=](Real x) {
          return g0*(x/a)/sqrt(1. + SQR(x/a));
        };

        Real pg1 = (pg2 - 0.5*g(x1f)*dx*rho2)/(1. + 0.5*g(x1f)*dx*rho2/pg2);
        Real rho1 = rho2*pg1/pg2;
        Real v1 = v_ie > 0.0 ? v_ie : 0.0;

        prim(IDN,k,j,ie+i) = rho1;
        prim(IVX,k,j,ie+i) = v1;
        prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
        prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
        prim(IPR,k,j,ie+i) = pg1;
        
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) { 
      for (int j=js; j<=je; ++j) { 
        for (int i=1; i<=NGHOST; ++i) { 
          b.x1f(k,j,ie+i+1) = b.x1f(k,j,ie+1); 
        } 
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=1; i<=NGHOST; ++i) {
          b.x2f(k,j,(ie+i)) =  b.x2f(k,j,ie);
        }
      }
    }      
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=NGHOST; ++i) {
          b.x3f(k,j,(ie+i)) =  b.x3f(k,j,ie);
        }
      }
    }
  }
  return;
}// end HydroConstTempOuterBoundary

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons)
{
  // Cooling
  Cooling cooler(T_floor, T_ceil, T0, L0, epsil);

  int num_layer = pmb->phydro->num_layer;
  pmb->phydro->layer_grid.ZeroClear();
  pmb->phydro->layer_heat.ZeroClear();
  Real *xf = &pmb->phydro->xf(0);
  int *layer_grid = &pmb->phydro->layer_grid(0);
  Real *layer_heat = &pmb->phydro->layer_heat(0);

  int kl = pmb->ks, ku = pmb->ke;
  int jl = pmb->js, ju = pmb->je;
  int il = pmb->is, iu = pmb->ie;

  for (int i=il; i<=iu; ++i) {
    Real x1 = pmb->pcoord->x1v(i);
    int grid = 0;
    Real heat = 0.0;
    int ind = 0;
    while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}

    for (int j=jl; j<=ju; ++j) {
      for (int k=kl; k<=ku; ++k) {

        Real rho = cons(IDN,k,j,i);
        Real eng = cons(IEN,k,j,i) 
          - 0.5*(cons(IM1,k,j,i)*cons(IM1,k,j,i)
          + cons(IM2,k,j,i)*cons(IM2,k,j,i)
          + cons(IM3,k,j,i)*cons(IM3,k,j,i))/rho;
        if (MAGNETIC_FIELDS_ENABLED) {
          eng -= 0.5*(bcc(IB1,k,j,i) * bcc(IB1,k,j,i)
            + bcc(IB2,k,j,i) * bcc(IB2,k,j,i)
            + bcc(IB3,k,j,i) * bcc(IB3,k,j,i));
        }

        // T = pg/rho
        Real temp = eng*(gg - 1.)/rho;

        Real temp_new = temp;
        if ((fabs(x1) >= x_min_cool_buffer_left) && (fabs(x1) <= x_max_cool_buffer_right)) {
          temp_new = cooler.townsend(temp, rho, dt);
        } 

        temp_new = std::max(temp_new, cooler.T_floor);
        temp_new = std::min(temp_new, cooler.T_ceil);

        grid += 1;
        heat += -(temp_new - temp)*rho/(gg - 1.);
        
        Real mom_src = -rho*g0*(x1/a)/sqrt(1. + SQR(x1/a));
        Real eng_src = mom_src*prim(IVX,k,j,i);

        cons(IM1,k,j,i) += mom_src*dt;
        cons(IEN,k,j,i) += eng_src*dt + (temp_new - temp)*rho/(gg - 1.);

      }
    }

    layer_grid[ind] += grid;
    layer_heat[ind] += heat;

  }
  
  return;
} // End SourceTerm

// Mean Square density fluctuations
Real MsDensity(MeshBlock *pmb, int iout)
{
  Real drho_ms = 0;

  Hydro *ph = pmb->phydro;
  int num_layer = ph->num_layer;
  int num_grid_sum = ph->num_grid_sum;
  Real *xf = &ph->xf(0);

  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  
  for (int i=is; i<=ie; ++i) {
    Real x1 = pmb->pcoord->x1v(i);
    if ((x1 > x_start_sum) && (x1 < x_end_sum)) {
      int ind = 0;
      while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}

      for (int j=js; j<=je; ++j) {
        for (int k=ks; k<=ke; ++k) {
          drho_ms += SQR((ph->u(IDN,k,j,i) - ph->mean_rho(ind))/ph->mean_rho(ind));
        }
      }

    }
  }
  drho_ms /= num_grid_sum;
  return drho_ms;
} // End MsDensity


