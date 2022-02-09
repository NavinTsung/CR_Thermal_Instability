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
#include <complex>
#include <vector>
#include <hdf5.h>
#include <random>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/factorials.hpp>

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
#include "../cr/cr.hpp"
#include "../cr/integrators/cr_integrators.hpp"
#include "../utils/utils.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"
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
static Real gc = 4./3.;
static Real gc1 = gc/(gc - 1.);

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void HydroConstTempInnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void HydroConstTempOuterBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void CRStaticInnerBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh);

void CRStaticOuterBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh);

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons);

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &u_cr);

Real MsDensity(MeshBlock *pmb, int iout);

void Mesh::UserWorkInLoop() {
  // Part 1: Compute heating to be added to each grid
  MeshBlock *pmb = pblock; // Start with the first block of each processor
  int num_layer = pblock->phydro->num_layer;
  int *local_grid{new int[num_layer]()};
  int *global_grid{new int[num_layer]()};
  int *local_grid_cr{new int[num_layer]()};
  int *global_grid_cr{new int[num_layer]()};
  Real *local_heating{new Real[num_layer]()};
  Real *global_heating{new Real[num_layer]()};
  Real *local_heating_cr{new Real[num_layer]()};
  Real *global_heating_cr{new Real[num_layer]()};

  int *layer_grid, *layer_grid_cr;
  Real *layer_heat, *layer_heat_cr;
  while (pmb != nullptr) {
    layer_grid = &pmb->phydro->layer_grid(0);
    layer_heat = &pmb->phydro->layer_heat(0);
    layer_grid_cr = &pmb->pcr->layer_grid_cr(0);
    layer_heat_cr = &pmb->pcr->layer_heat_cr(0);
    for (int l=0; l<num_layer; ++l) {
      local_grid[l] += layer_grid[l];
      local_heating[l] += layer_heat[l];
      local_grid_cr[l] += layer_grid_cr[l];
      local_heating_cr[l] += layer_heat_cr[l];
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(local_grid, global_grid, num_layer, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_heating, global_heating, num_layer, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_grid_cr, global_grid_cr, num_layer, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_heating_cr, global_heating_cr, num_layer, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int l=0; l<num_layer; ++l) {
    global_grid[l] = local_grid[l];
    global_heating[l] = local_heating[l];
    global_grid_cr[l] = local_grid_cr[l];
    global_heating_cr[l] = local_heating_cr[l];
  }
#endif

  for (int l=0; l<num_layer; ++l) {
    global_heating[l] /= global_grid[l];
    global_heating_cr[l] /= global_grid_cr[l];
  }

  pmb = pblock;
  while (pmb != nullptr) {
    Hydro *ph = pmb->phydro;
    Field *pf = nullptr;
    if (MAGNETIC_FIELDS_ENABLED) {
      pf = pmb->pfield;
    }
    CosmicRay *pcr= pmb->pcr;

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

          if (ph->magic_heating > 0) {
            ph->u(IEN,k,j,i) += global_heating[ind] + global_heating_cr[ind];
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
  delete [] local_grid_cr;
  delete [] global_grid_cr;
  delete [] local_heating_cr;
  delete [] global_heating_cr;



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

          if ((x1 > ph->x_start_sum) && (x1 < ph->x_end_sum)) {
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
#else
  global_grid_sum = local_grid_sum;
  for (int l=0; l<num_layer; ++l) {
    global_num_grid[l] = local_num_grid[l];
    global_mean_rho[l] = local_mean_rho[l];
  }
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
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroConstTempInnerBoundaryMHD);
    if(CR_ENABLED){
      EnrollUserCRBoundaryFunction(BoundaryFace::inner_x1, CRStaticInnerBoundaryCR);
    }
  }
  if (mesh_bcs[BoundaryFace::outer_x1] ==  GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroConstTempOuterBoundaryMHD);
    if(CR_ENABLED){
      EnrollUserCRBoundaryFunction(BoundaryFace::outer_x1, CRStaticOuterBoundaryCR);
    }
  }
  EnrollUserExplicitSourceFunction(SourceTerm);
  // Allocate and enroll history output
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, MsDensity, "drho_ms", UserHistoryOperation::sum);
}

void MeshBlock::UserWorkInLoop() {

  for (int i=is; i<=ie; ++i) {
    for (int j=js; j<=je; ++j) {
      for (int k=ks; k<=ke; ++k) {
        for (int n=0; n<NHYDRO; ++n) {
          if (phydro->u(n,k,j,i) != phydro->u(n,k,j,i)) {
            std::stringstream msg;
            msg << "Variable " << n << " NaN detected at" 
                << " x = " << pcoord->x1v(i) 
                << " y = " << pcoord->x2v(j)
                << ", exiting." << std::endl;
            ATHENA_ERROR(msg);
          }
        }

        if (CR_ENABLED) {
          for (int n=0; n<NCR; ++n) {
            if (pcr->u_cr(n,k,j,i) != pcr->u_cr(n,k,j,i)) {
              std::stringstream msg;
              msg << "CR variable " << n << " NaN detected at"
                  << " x = " << pcoord->x1v(i)
                  << " y = " << pcoord->x2v(j) 
                  << ", exiting." << std::endl;
              ATHENA_ERROR(msg);
            }
          }
        }

      }
    }
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED){
    pcr->EnrollOpacityFunction(Diffusion);
    pcr->EnrollUserCRSource(CRSource);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int num_grid = pmy_mesh->mesh_size.nx1;
  Real *xf = &phydro->xf(0);
  Real *yf = &phydro->yf(0);
  Real *zf = &phydro->zf(0);

  // Initialize hydro variable
  for (int i=is; i<=ie; ++i) {
    Real x1 = pcoord->x1v(i);
    int ind = 0;
    while ((ind < num_grid) && (xf[ind+1] < x1)) {ind += 1;}
    for (int j=js; j<=je; ++j) {
      Real y1 = pcoord->x2v(j);
      int indy = 0;
      while ((indy < pmy_mesh->mesh_size.nx2) && (yf[indy+1] < y1)) {indy += 1;}
      for (int k=ks; k<=ke; ++k) {
        Real z1 = pcoord->x3v(k);
        int indz = 0;
        while ((indz < pmy_mesh->mesh_size.nx3) && (zf[indz+1] < z1)) {indz += 1;}

        if ((x1 > pcr->x_min_add_perturb) && (x1 < pcr->x_max_add_perturb)) {
          phydro->u(IDN,k,j,i) = pcr->rho(ind)*(1. + pcr->dd(indz,indy,ind));
        } else {
          phydro->u(IDN,k,j,i) = pcr->rho(ind);
        }
        // phydro->u(IDN,k,j,i) += 0.1*exp(-SQR((x1 - pcr->nu*pcr->Hc)/0.1));
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = pcr->pg(ind)/(gg - 1.);
        }

        // Initialize CR variables
        if (CR_ENABLED) {
          pcr->u_cr(CRE,k,j,i) = pcr->pc(ind)/(gc - 1.);
          pcr->u_cr(CRF1,k,j,i) = pcr->fc(ind)/pcr->vmax;
          pcr->u_cr(CRF2,k,j,i) = 0.0;
          pcr->u_cr(CRF3,k,j,i) = 0.0;
        }

      } // end k
    } // end j
  } // end i

  //Need to set opactiy sigma and pc values in the ghost zones
  if(CR_ENABLED){

    // Default values are 1/3
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if(nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if(nz3 > 1) nz3 += 2*(NGHOST);
    for (int k=0; k<nz3; ++k) {
      for (int j=0; j<nz2; ++j) {
        for (int i=0; i<nz1; ++i) {
          pcr->sigma_diff(0,k,j,i) = (gc - 1.)*pcr->vmax/pcr->kap;
          pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
        }
      }
    }// end k,j,i

    // Set pc ghost zone values only along x1 direction 
    for (int k=0; k<=ke; ++k) {
      for (int j=0; j<=je+2*NGHOST; ++j) {
        for (int i=1; i<=NGHOST; ++i) {
          pcr->u_cr(CRE,k,j,is-i) = pcr->pc_xinner_bval(i-1)/(gc - 1.);
          pcr->u_cr(CRE,k,j,ie+i) = pcr->pc_xouter_bval(i-1)/(gc - 1.);
        }
      }
    }

  }// End CR

    // Add horizontal magnetic field lines, to show streaming and diffusion 
  // along magnetic field ines
  if(MAGNETIC_FIELDS_ENABLED){

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = pcr->B;
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

            if ((x1 > ph->x_start_sum) && (x1 < ph->x_end_sum)) {
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

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
              AthenaArray<Real> &prim, AthenaArray<Real> &bcc)
{
  // set the default opacity to be a large value in the default hydro case
  CosmicRay *pcr=pmb->pcr;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if(pmb->block_size.nx2 > 1){
    jl -= 1;
    ju += 1;
  }
  if(pmb->block_size.nx3 > 1){
    kl -= 1;
    ku += 1;
  }

  Real invlim = 1.0/pcr->vmax;

  for(int k=kl; k<=ku; ++k){
    for(int j=jl; j<=ju; ++j){
#pragma omp simd
      for(int i=il; i<=iu; ++i){
        pcr->sigma_diff(0,k,j,i) = (gc - 1.)*pcr->vmax/pcr->kap;
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;  

      }
    }
  }

  // Need to calculate the rotation matrix 
  // We need this to determine the direction of rotation velocity


  // The information stored in the array
  // b_angle is
  // b_angle[0]=sin_theta_b
  // b_angle[1]=cos_theta_b
  // b_angle[2]=sin_phi_b
  // b_angle[3]=cos_phi_b
 

  if(MAGNETIC_FIELDS_ENABLED && (pcr->stream_flag > 0)){
    //First, calculate B_dot_grad_Pc
    for(int k=kl; k<=ku; ++k){
      for(int j=jl; j<=ju; ++j){
      // diffusion coefficient is calculated with respect to B direction
      // Use a simple estimate of Grad Pc

    // x component
        pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
        for(int i=il; i<=iu; ++i){
          Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                         + pcr->cwidth(i);
          Real dprdx=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
          dprdx /= distance;
          pcr->b_grad_pc(k,j,i) = bcc(IB1,k,j,i) * dprdx;
        }
    // y component
        pmb->pcoord->CenterWidth2(k,j-1,il,iu,pcr->cwidth1);       
        pmb->pcoord->CenterWidth2(k,j,il,iu,pcr->cwidth);
        pmb->pcoord->CenterWidth2(k,j+1,il,iu,pcr->cwidth2);

        for(int i=il; i<=iu; ++i){
          Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                         + pcr->cwidth(i);
          Real dprdy=(u_cr(CRE,k,j+1,i) - u_cr(CRE,k,j-1,i))/3.0;
          dprdy /= distance;
          pcr->b_grad_pc(k,j,i) += bcc(IB2,k,j,i) * dprdy;
        } 
    // z component
        pmb->pcoord->CenterWidth3(k-1,j,il,iu,pcr->cwidth1);       
        pmb->pcoord->CenterWidth3(k,j,il,iu,pcr->cwidth);
        pmb->pcoord->CenterWidth3(k+1,j,il,iu,pcr->cwidth2);

        for(int i=il; i<=iu; ++i){
          Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                          + pcr->cwidth(i);
          Real dprdz=(u_cr(CRE,k+1,j,i) -  u_cr(CRE,k-1,j,i))/3.0;
          dprdz /= distance;
          pcr->b_grad_pc(k,j,i) += bcc(IB3,k,j,i) * dprdz;
        }       

        for(int i=il; i<=iu; ++i){
          // Now calculate the angles of B
          Real bxby = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
          Real btot = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i) + 
                           bcc(IB3,k,j,i)*bcc(IB3,k,j,i));
          
          if(btot > TINY_NUMBER){
            pcr->b_angle(0,k,j,i) = bxby/btot;
            pcr->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
          }else{
            pcr->b_angle(0,k,j,i) = 1.0;
            pcr->b_angle(1,k,j,i) = 0.0;
          }
          if(bxby > TINY_NUMBER){
            pcr->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
            pcr->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
          }else{
            pcr->b_angle(2,k,j,i) = 0.0;
            pcr->b_angle(3,k,j,i) = 1.0;            
          }

          Real inv_sqrt_rho = 1.0/sqrt(prim(IDN,k,j,i));
          Real va1 = bcc(IB1,k,j,i)*inv_sqrt_rho;
          Real va2 = bcc(IB2,k,j,i)*inv_sqrt_rho;
          Real va3 = bcc(IB3,k,j,i)*inv_sqrt_rho;

          Real va = sqrt(btot*btot/prim(IDN,k,j,i));

          Real dpc_sign = 0.0;
          if(pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = 1.0;
          else if(-pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = -1.0;
          
          pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
          pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
          pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;

          if(va < TINY_NUMBER){
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          }else{
            pcr->sigma_adv(0,k,j,i) = fabs(pcr->b_grad_pc(k,j,i))/(btot * va * (1.0 + 1.0/3.0) 
                                               * invlim * u_cr(CRE,k,j,i));
          }
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

        }//end i        

      }// end j
    }// end k

  }// end MHD and streaming
  else{

    for(int k=kl; k<=ku; ++k){
      for(int j=jl; j<=ju; ++j){
#pragma omp simd
        for(int i=il; i<=iu; ++i){

          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;  

          pcr->v_adv(0,k,j,i) = 0.0;   
          pcr->v_adv(1,k,j,i) = 0.0;
          pcr->v_adv(2,k,j,i) = 0.0;
        }
      }
    }

  }// end MHD and stream flag

}// end diffusion

void HydroConstTempInnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  CosmicRay *pcr = pmb->pcr;

  auto G = [=](Real x) {
    Real x1 = x > 0.0 ? x : -x;
    Real ssum = 0.0;
    for (int i=0; i<=pcr->nu; ++i) {
      ssum += pow(x1/pcr->Hc, i)/boost::math::factorial<double>(i);
    }
    Real term = 1. + 0.5*(3. - gc)*boost::math::factorial<double>(pcr->nu)*(1. - exp(-x1/pcr->Hc)*ssum);
    Real g = pcr->g0*pow(x1/pcr->Hc, pcr->nu)*exp(-x1/pcr->Hc)*(pow(term, -1.) + (0.5*gc*pcr->alpha0)*pow(term, 1./(gc - 3.)));
    g = x > 0.0 ? g : -g;
    return g;
  };

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {

        Real x1 = pco->x1v(is-i);
        Real x2 = pco->x1v(is-i+1);
        Real x2f = pco->x1f(is-i+1);
        Real dx = x2 - x1;
        
        Real rho2 = prim(IDN,k,j,is-i+1);
        Real pg2 = prim(IPR,k,j,is-i+1);
        Real pc2 = pcr->u_cr(CRE,k,j,is-i+1)*(gc - 1.);

        Real v_is = prim(IVX,k,j,is);

        Real alpha = pc2/pg2;
        Real zeta = 0.5*G(x2f)*dx*(rho2/pg2);

        auto f = [=](Real r) {
          return alpha*pow(r, 0.5*gc) + (1. - zeta)*r - (1. + alpha + zeta);
        };

        typedef std::pair<Real, Real> Result;
        boost::uintmax_t max_iter=500;
        eps_tolerance<Real> tol(30); 
        Result r1;
        Real r;
        try {
          r1 = toms748_solve(f, 0., 10., tol, max_iter);
          r = r1.first;
        } 
        catch (std::exception &e) {
          r = 1.;
        }

        Real rho1 = r*rho2;
        Real pg1 = r*pg2;
        Real pc1 = pow(r, 0.5*gc)*pc2;
        Real v1 = v_is > 0.0 ? 0.0 : v_is;

        Real rho_face = 0.5*(rho1 + rho2);
        Real pc_face = 0.5*(pc1 + pc2);
        Real dpcdx_face = (pc2 - pc1)/dx;
        Real Lc_face = fabs(pc_face/dpcdx_face);
        Real va_face = b.x1f(k,j,is)/sqrt(rho_face);
        Real thres = (dx/Lc_face)*(pcr->vmax/va_face);

        if (thres > 0.01) {
          prim(IDN,k,j,is-i) = rho1;
          prim(IVX,k,j,is-i) = v1;
          prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
          prim(IVZ,k,j,is-i) = prim(IVY,k,j,is);
          prim(IPR,k,j,is-i) = pg1;

          pcr->u_cr(CRE,k,j,is-i) = pc1/(gc - 1.);
        } else {
          Real r_uncouple = (1. + zeta)/(1. - zeta);
          r_uncouple = r_uncouple < 0. ? 1. : r_uncouple;

          rho1 = r_uncouple*rho2;
          pg1 = r_uncouple*pg2;
          pc1 = 0.99*pc2;

          prim(IDN,k,j,is-i) = rho1;
          prim(IVX,k,j,is-i) = v1;
          prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
          prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
          prim(IPR,k,j,is-i) = pg1;

          pcr->u_cr(CRE,k,j,is-i) = pc1/(gc - 1.); 
        }
        
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
}// end HydroConstTempInnerBoundaryMHD

void HydroConstTempOuterBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  CosmicRay *pcr = pmb->pcr;

  auto G = [=](Real x) {
    Real x1 = x > 0.0 ? x : -x;
    Real ssum = 0.0;
    for (int i=0; i<=pcr->nu; ++i) {
      ssum += pow(x1/pcr->Hc, i)/boost::math::factorial<double>(i);
    }
    Real term = 1. + 0.5*(3. - gc)*boost::math::factorial<double>(pcr->nu)*(1. - exp(-x1/pcr->Hc)*ssum);
    Real g = pcr->g0*pow(x1/pcr->Hc, pcr->nu)*exp(-x1/pcr->Hc)*(pow(term, -1.) + (0.5*gc*pcr->alpha0)*pow(term, 1./(gc - 3.)));
    g = x > 0.0 ? g : -g;
    return g;
  };

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {

        Real x1 = pco->x1v(ie+i);
        Real x1f = pco->x1f(ie+i);
        Real x2 = pco->x1v(ie+i-1);
        Real dx = x1 - x2;
        
        Real rho2 = prim(IDN,k,j,ie+i-1);
        Real pg2 = prim(IPR,k,j,ie+i-1);
        Real pc2 = pcr->u_cr(CRE,k,j,ie+i-1)*(gc - 1.);

        Real v_ie = prim(IVX,k,j,ie);

        Real alpha = pc2/pg2;
        Real zeta = 0.5*G(x1f)*dx*(rho2/pg2);

        auto f = [=](Real r) {
          return alpha*pow(r, 0.5*gc) + (1. + zeta)*r - (1. + alpha - zeta);
        };

        typedef std::pair<Real, Real> Result;
        boost::uintmax_t max_iter = 500;
        eps_tolerance<Real> tol(30); 
        Result r1;
        Real r;
        try {
          r1 = toms748_solve(f, 0., 10., tol, max_iter);
          r = r1.first;
        } 
        catch (std::exception &e) {
          r = 1.;
        }

        Real rho1 = r*rho2;
        Real pg1 = r*pg2;
        Real pc1 = pow(r, 0.5*gc)*pc2;
        Real v1 = v_ie > 0.0 ? v_ie : 0.0;

        Real rho_face = 0.5*(rho1 + rho2);
        Real pc_face = 0.5*(pc1 + pc2);
        Real dpcdx_face = (pc1 - pc2)/dx;
        Real Lc_face = fabs(pc_face/dpcdx_face);
        Real va_face = b.x1f(k,j,ie+1)/sqrt(rho_face);
        Real thres = (dx/Lc_face)*(pcr->vmax/va_face);

        if (thres > 0.01) {
          prim(IDN,k,j,ie+i) = rho1;
          prim(IVX,k,j,ie+i) = v1;
          prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
          prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
          prim(IPR,k,j,ie+i) = pg1;

          pcr->u_cr(CRE,k,j,ie+i) = pc1/(gc - 1.);
        } else {
          Real r_uncouple = (1. - zeta)/(1. + zeta);
          r_uncouple = r_uncouple < 0. ? 1. : r_uncouple;

          rho1 = r_uncouple*rho2;
          pg1 = r_uncouple*pg2;
          pc1 = 0.99*pc2;

          prim(IDN,k,j,ie+i) = rho1;
          prim(IVX,k,j,ie+i) = v1;
          prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
          prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
          prim(IPR,k,j,ie+i) = pg1;

          pcr->u_cr(CRE,k,j,ie+i) = pc1/(gc - 1.);
        }
        
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
}// end HydroConstTempOuterBoundaryMHD

void CRStaticInnerBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh)
{
  Field *pf = pmb->pfield;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {

        Real x1 = pco->x1v(is-i);
        Real x2 = pco->x1v(is-i+1);
        Real x2f = pco->x1f(is-i+1);
        Real dx = x2 - x1;
        
        Real rho1 = w(IDN,k,j,is-i);
        Real rho2 = w(IDN,k,j,is-i+1);
        Real pc1 = u_cr(CRE,k,j,is-i)*(gc - 1.);
        Real pc2 = u_cr(CRE,k,j,is-i+1)*(gc - 1.);
        Real fc2 = u_cr(CRF1,k,j,is-i+1)*pcr->vmax;
        
        Real r = rho1/rho2;
        Real dpcdx = (pc2 - pc1)/dx;

        Real rho_face = 0.5*(rho1 + rho2);
        Real b_grad_pc = pf->b.x1f(k,j,is-i+1)*dpcdx;
        Real dpc_sign = 0.0;
        if(b_grad_pc > TINY_NUMBER) dpc_sign = 1.;
        else if(-b_grad_pc > TINY_NUMBER) dpc_sign = -1.;
        Real va = pf->b.x1f(k,j,is-i+1)/sqrt(rho_face);
        Real vs = -va*dpc_sign;
        Real fc1 = fc2 - vs*dpcdx*dx;

        u_cr(CRE,k,j,is-i) = pc1/(gc - 1.);
        u_cr(CRF1,k,j,is-i) = fc1/pcr->vmax;
        u_cr(CRF2,k,j,is-i) = u_cr(CRF2,k,j,is);
        u_cr(CRF3,k,j,is-i) = u_cr(CRF3,k,j,is);
        
      }
    }
  }
  return;
}// end OutflowInnerBoundaryCR

void CRStaticOuterBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh)
{
  Field *pf = pmb->pfield;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {

        Real x1 = pco->x1v(ie+i);
        Real x1f = pco->x1f(ie+i);
        Real x2 = pco->x1v(ie+i-1);
        Real dx = x1 - x2;

        Real rho1 = w(IDN,k,j,ie+i);
        Real rho2 = w(IDN,k,j,ie+i-1);
        Real pc1 = u_cr(CRE,k,j,ie+i)*(gc - 1.);
        Real pc2 = u_cr(CRE,k,j,ie+i-1)*(gc - 1.);
        Real fc2 = u_cr(CRF1,k,j,ie+i-1)*pcr->vmax;

        Real r = rho1/rho2;
        Real dpcdx = (pc1 - pc2)/dx;

        Real rho_face = 0.5*(rho1 + rho2);
        Real b_grad_pc = pf->b.x1f(k,j,ie+i)*dpcdx;
        Real dpc_sign = 0.;
        if(b_grad_pc > TINY_NUMBER) dpc_sign = 1.;
        else if(-b_grad_pc > TINY_NUMBER) dpc_sign = -1.;
        Real va = pf->b.x1f(k,j,ie+i)/sqrt(rho_face);
        Real vs = -va*dpc_sign;
        Real fc1 = fc2 + vs*dpcdx*dx;

        u_cr(CRE,k,j,ie+i) = pc1/(gc - 1.);
        u_cr(CRF1,k,j,ie+i) = fc1/pcr->vmax;
        u_cr(CRF2,k,j,ie+i) = u_cr(CRF2,k,j,ie);
        u_cr(CRF3,k,j,ie+i) = u_cr(CRF3,k,j,ie);

      }
    }
  }
  return;
}// end OutflowOuterBoundaryCR

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons)
{
  Hydro *ph = pmb->phydro;
  CosmicRay *pcr = pmb->pcr;

  auto G = [=](Real x) {
    Real x1 = x > 0.0 ? x : -x;
    Real ssum = 0.0;
    for (int i=0; i<=pcr->nu; ++i) {
      ssum += pow(x1/pcr->Hc, i)/boost::math::factorial<double>(i);
    }
    Real term = 1. + 0.5*(3. - gc)*boost::math::factorial<double>(pcr->nu)*(1. - exp(-x1/pcr->Hc)*ssum);
    Real g = pcr->g0*pow(x1/pcr->Hc, pcr->nu)*exp(-x1/pcr->Hc)*(pow(term, -1.) + (0.5*gc*pcr->alpha0)*pow(term, 1./(gc - 3.)));
    g = x > 0.0 ? g : -g;
    return g;
  };

  int num_layer = ph->num_layer;
  ph->layer_grid.ZeroClear();
  ph->layer_heat.ZeroClear();
  Real *xf = &ph->xf(0);
  int *layer_grid = &ph->layer_grid(0);
  Real *layer_heat = &ph->layer_heat(0);

  int kl = pmb->ks, ku = pmb->ke;
  int jl = pmb->js, ju = pmb->je;
  int il = pmb->is, iu = pmb->ie;

  for (int i=il; i<=iu; ++i) {
    Real x1 = pmb->pcoord->x1v(i);
    int grid = 0;
    Real heat = 0.0;
    int ind = 0;
    while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}
    Cooling cooler(pcr->T_floor, pcr->T_ceil, pcr->T0, pcr->L(ind), pcr->epsil);
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
        if ((fabs(x1) >= pcr->x_min_cool_buffer_left) && (fabs(x1) <= pcr->x_max_cool_buffer_right)) {
          temp_new = cooler.townsend(temp, rho, dt);
        } 

        temp_new = std::max(temp_new, cooler.T_floor);
        temp_new = std::min(temp_new, cooler.T_ceil);

        grid += 1;
        heat += -(temp_new - temp)*rho/(gg - 1.);
        
        Real mom_src = -rho*G(x1);
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


void CRSource(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &u_cr)
{
  Hydro *ph = pmb->phydro;
  CosmicRay *pcr = pmb->pcr;
  Real dx = ph->dx;
  int num_layer = ph->num_layer;
  Real *xf = &ph->xf(0);

  int kl = pmb->ks, ku = pmb->ke;
  int jl = pmb->js, ju = pmb->je;
  int il = pmb->is, iu = pmb->ie;

  for (int i=il; i<=iu; ++i) {
    Real x1 = pmb->pcoord->x1v(i);
    int ind = 0;
    while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}
    for (int j=jl; j<=ju; ++j) {
      for (int k=kl; k<=ku; ++k) {

        Real x1 = pmb->pcoord->x1v(i);

        if (time >= pcr->add_qsource_cr_time) {
          if ((fabs(x1) >= pcr->x_min_cr_buffer) && (fabs(x1) <= pcr->x_max_cr_buffer)) {
            u_cr(CRE,k,j,i) += pcr->q(ind)*dt;
          }
          if (fabs(x1) <= 0.5*dx) {
            Real pg = prim(IPR,k,j,i);
            Real pc = u_cr(CRE,k,j,i)*(gc - 1.);
            if (pcr->alpha0*pg > pc) u_cr(CRE,k,j,i) += (pcr->alpha0*pg - pc)/(gc - 1.);
          }
        }

      }
    } 
  }

} // End CR Source Term

// Mean Square density fluctuations
Real MsDensity(MeshBlock *pmb, int iout)
{
  Real drho_ms = 0;

  Hydro *ph = pmb->phydro;
  CosmicRay *pcr = pmb->pcr;
  int num_layer = ph->num_layer;
  int num_grid_sum = ph->num_grid_sum;
  Real *xf = &ph->xf(0);

  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  
  for (int i=is; i<=ie; ++i) {
    Real x1 = pmb->pcoord->x1v(i);
    if ((x1 > ph->x_start_sum) && (x1 < ph->x_end_sum)) {
      int ind = 0;
      while ((ind < num_layer) && (xf[ind+1] < x1)) {ind += 1;}

      for (int j=js; j<=je; ++j) {
        for (int k=ks; k<=ke; ++k) {
          if ((pmb->pmy_mesh->mesh_size.nx2 > 1) || (pmb->pmy_mesh->mesh_size.nx3 > 1)) {
            drho_ms += SQR((ph->u(IDN,k,j,i) - ph->mean_rho(ind))/ph->mean_rho(ind));
          } else {
            drho_ms += SQR((ph->u(IDN,k,j,i) - pcr->rho(ind))/pcr->rho(ind));
          }

        }
      }

    }
  }
  drho_ms /= num_grid_sum;
  return drho_ms;
} // End MsDensity


