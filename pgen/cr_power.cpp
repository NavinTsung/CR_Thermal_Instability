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


//======================================================================================
/*! \file beam.cpp
 *  \brief Beam test for the radiative transfer module
 *
 *====================================================================================*/


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================

static std::string filename = "../analytics/power.hdf5";
// static std::string filename = "./power.hdf5";

static Real gg = 5./3.;
static Real gg1 = gg/(gg - 1.);
static Real gc = 4./3.;
static Real gc1 = gc/(gc - 1.);

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void InnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void OuterBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void InnerBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh);

void OuterBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh);

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons);

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {    
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerBoundaryMHD);
    if(CR_ENABLED){
      EnrollUserCRBoundaryFunction(BoundaryFace::inner_x1, InnerBoundaryCR);
    }
  }
  if (mesh_bcs[BoundaryFace::outer_x1] ==  GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterBoundaryMHD);
    if(CR_ENABLED){
      EnrollUserCRBoundaryFunction(BoundaryFace::outer_x1, OuterBoundaryCR);
    }
  }
  EnrollUserExplicitSourceFunction(SourceTerm);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED){
    pcr->EnrollOpacityFunction(Diffusion);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  Real B; // For storing magnetic field
  Real ldiff;
  Real x[block_size.nx1+2*NGHOST]; // For storing x1 location
  
  // Set up HDF5 structures
  // Make sure the number and size of meshblock are the same in the input file
  // and data to be input
  hsize_t start_file[2]; // For specifying start loc of file dataspace hyperslab 
  hsize_t count_file[2]; // For specifying count num of file dataspace hyperslab 
  hsize_t count_mem[1]; // For specifying count num of memory dataspace 

  start_file[0] = gid;
  start_file[1] = 0;
  count_file[0] = 1;
  count_file[1] = block_size.nx1 + 2*NGHOST;
  count_mem[0] = block_size.nx1 + 2*NGHOST;

  // Open file with mpi-specific property list
#ifdef MPI_PARALLEL
  hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(property_list_file, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, property_list_file);
  H5Pclose(property_list_file);
#else
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
#endif

  // Open files and extract attributes and datasets
  hid_t attr_B = H5Aopen(file, "B", H5P_DEFAULT);
  hid_t attr_kappa = H5Aopen(file, "kappa", H5P_DEFAULT);
  hid_t attr_ldiff = H5Aopen(file, "ldiff", H5P_DEFAULT);
  hid_t dset_x = H5Dopen(file, "x", H5P_DEFAULT);
  hid_t dset_rho = H5Dopen(file, "rho", H5P_DEFAULT);
  hid_t dset_pg = H5Dopen(file, "pg", H5P_DEFAULT);
  hid_t dset_ec = H5Dopen(file, "ec", H5P_DEFAULT);
  hid_t dset_fc = H5Dopen(file, "fc", H5P_DEFAULT);
  hid_t dset_g = H5Dopen(file, "g", H5P_DEFAULT);
  hid_t dset_H = H5Dopen(file, "H", H5P_DEFAULT);

  // Get file dataspace
  hid_t fspace_x = H5Dget_space(dset_x);
  hid_t fspace_rho = H5Dget_space(dset_rho);
  hid_t fspace_pg = H5Dget_space(dset_pg);
  hid_t fspace_ec = H5Dget_space(dset_ec);
  hid_t fspace_fc = H5Dget_space(dset_fc);
  hid_t fspace_g = H5Dget_space(dset_g);
  hid_t fspace_H = H5Dget_space(dset_H);

  // Preparing file dataspace for describing data in memory
  H5Sselect_hyperslab(fspace_x, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_rho, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_pg, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_ec, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_fc, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_g, H5S_SELECT_SET, start_file, NULL, count_file, NULL);
  H5Sselect_hyperslab(fspace_H, H5S_SELECT_SET, start_file, NULL, count_file, NULL);

  // Preparing memory dataspace
  hid_t mspace = H5Screate_simple(1, count_mem, NULL);

  // Transfer data collectively
  hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
#ifdef MPI_PARALLEL
  H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
#else
#endif

  // Read data and store into AthenaArrays initialized in cr/cr.hpp and cr/cr.cpp
  H5Aread(attr_B, H5T_NATIVE_DOUBLE, &B);
  H5Aread(attr_kappa, H5T_NATIVE_DOUBLE, &(pcr->kap));
  H5Aread(attr_ldiff, H5T_NATIVE_DOUBLE, &ldiff);
  H5Dread(dset_x, H5T_NATIVE_DOUBLE, mspace, fspace_x, xfer_plist, &x[0]);
  H5Dread(dset_rho, H5T_NATIVE_DOUBLE, mspace, fspace_rho, xfer_plist, &(pcr->rho_in(0)));
  H5Dread(dset_pg, H5T_NATIVE_DOUBLE, mspace, fspace_pg, xfer_plist, &(pcr->pg_in(0)));
  H5Dread(dset_ec, H5T_NATIVE_DOUBLE, mspace, fspace_ec, xfer_plist, &(pcr->ec_in(0)));
  H5Dread(dset_fc, H5T_NATIVE_DOUBLE, mspace, fspace_fc, xfer_plist, &(pcr->fc_in(0)));
  H5Dread(dset_g, H5T_NATIVE_DOUBLE, mspace, fspace_g, xfer_plist, &(pcr->g_in(0)));
  H5Dread(dset_H, H5T_NATIVE_DOUBLE, mspace, fspace_H, xfer_plist, &(pcr->H_in(0)));
  
  // Close all attributes, datasets, dataspaces, property list, and file
  H5Aclose(attr_B);
  H5Aclose(attr_kappa);
  H5Aclose(attr_ldiff);
  H5Dclose(dset_x);
  H5Dclose(dset_rho);
  H5Dclose(dset_pg);
  H5Dclose(dset_ec);
  H5Dclose(dset_fc);
  H5Dclose(dset_g);
  H5Dclose(dset_H);
  H5Sclose(mspace);
  H5Sclose(fspace_x);
  H5Sclose(fspace_rho);
  H5Sclose(fspace_pg);
  H5Sclose(fspace_ec);
  H5Sclose(fspace_fc);
  H5Sclose(fspace_g);
  H5Sclose(fspace_H);
  H5Pclose(xfer_plist);
  H5Fclose(file);

  // Print out ghost zones to check that values are correctly input
  // for(int i=1; i<=NGHOST; i++){
  //   std::cout << i << " "
  //             << x[ie+i] << " "
  //             << pcr->rho_in(ie+i) << " "
  //             << pcr->v_in(ie+i) << " "
  //             << pcr->pg_in(ie+i) << " "
  //             << pcr->ec_in(ie+i) << " "
  //             << pcr->fc_in(ie+i) << " "
  //             << B << " "
  //             << std::endl;
  // }

  // Boundary perturbation
  pcr->tau = pin->GetOrAddReal("problem","tau",0.02);
  pcr->wvlen = pin->GetOrAddReal("problem","wvlen",1.0)*ldiff;
  pcr->k_wv = 2.*PI/pcr->wvlen;
  pcr->amp_left = pin->GetOrAddReal("problem","amp_left",0.01);
  pcr->amp_right = pin->GetOrAddReal("problem","amp_right",0.01);
  pcr->x_left = pin->GetReal("mesh","x1min");
  pcr->x_right = pin->GetReal("mesh","x1max");

  // Initialize hydro variable
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n,k,j,i) = 0.5*pcr->rho_in(i);
          }
        }
        
        // Initialize hydro variables
        phydro->u(IDN,k,j,i) = pcr->rho_in(i);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = pcr->pg_in(i)/(gg - 1.);
        }

        // Initialize CR variables
        if (CR_ENABLED) {
          pcr->u_cr(CRE,k,j,i) = pcr->ec_in(i);
          pcr->u_cr(CRF1,k,j,i) = pcr->fc_in(i)/pcr->vmax;
        }

      } //end i
    } //end j
  } //end k

  //Need to set opactiy sigma in the ghost zones
  if(CR_ENABLED){

  // Default values are 1/3
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if(nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if(nz3 > 1) nz3 += 2*(NGHOST);
    for(int k=0; k<nz3; ++k){
      for(int j=0; j<nz2; ++j){
        for(int i=0; i<nz1; ++i){
          pcr->sigma_diff(0,k,j,i) = (gc - 1.)*pcr->vmax/pcr->kap;
          pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
        }
      }
    }// end k,j,i

  }// End CR

    // Add horizontal magnetic field lines, to show streaming and diffusion 
  // along magnetic field ines
  if(MAGNETIC_FIELDS_ENABLED){

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
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

void InnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  CosmicRay *pcr = pmb->pcr;

  Real tau = pcr->tau;
  Real growth = 1. - exp(-time/tau);

  Real k_wv = pcr->k_wv;
  Real amp_left = pcr->amp_left;
  Real x_left = pcr->x_left;
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        for (int n=0; n<(NHYDRO); ++n) {
          Real x1 = pco->x1v(is-i);
          Real cs0 = sqrt(gg*pcr->pg_in(is)/pcr->rho_in(is));
          Real vx = amp_left*cs0*growth*sin(k_wv*(x1 - x_left) - k_wv*cs0*time);
          Real rho = pcr->rho_in(is-i)*(1. + vx/cs0);
          Real pg = pcr->pg_in(is-i)*(1. + gg*vx/cs0);
          if (n==(IDN)) {
            prim(IDN,k,j,is-i) = rho;
          } else if (n==(IVX)) {
            prim(IVX,k,j,is-i) = vx;
          } else if (n==(IVY)) {
            prim(IVY,k,j,is-i) = 0.;
          } else if (n==(IEN)) {
            prim(IEN,k,j,is-i) = pg;
          } else {
            prim(n,k,j,is-i) = prim(n,k,j,is);
          }
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
          b.x2f(k,j,(is-i)) =  b.x2f(k,j,is);
        }
      }
    }      
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          b.x3f(k,j,(is-i)) =  b.x3f(k,j,is);
        }
      }
    }
  }
  return;
}// end InnerBoundaryMHD

void OuterBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  CosmicRay *pcr = pmb->pcr;

  Real tau = pcr->tau;
  Real growth = 1. - exp(-time/tau);

  Real k_wv = pcr->k_wv;
  Real amp_right = pcr->amp_right;
  Real x_right = pcr->x_right;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        for (int n=0; n<(NHYDRO); ++n) {
          Real x1 = pco->x1v(ie+i);
          Real cs0 = sqrt(gg*pcr->pg_in(ie)/pcr->rho_in(ie));
          Real vx = amp_right*cs0*growth*sin(k_wv*(x1 - x_right) + k_wv*cs0*time);
          Real rho = pcr->rho_in(ie+i)*(1. - vx/cs0);
          Real pg = pcr->pg_in(ie+i)*(1. - gg*vx/cs0);
          if (n==IDN) {
            prim(IDN,k,j,ie+i) = rho;
          } else if (n==IVX) {
            prim(IVX,k,j,ie+i) = vx;
          } else if (n==IVY) {
            prim(IVY,k,j,ie+i) = 0.0;
          } else if (n==IEN) {
            prim(IEN,k,j,ie+i) = pg;
          } else {
            prim(n,k,j,ie+i) = prim(n,k,j,ie);
          }
        }
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) { 
      for (int j=js; j<=je; ++j) { 
        for (int i=1; i<=(NGHOST+1); ++i) { 
          b.x1f(k,j,ie+i) = b.x1f(k,j,ie); 
        } 
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=1; i<=(NGHOST+1); ++i) {
          b.x2f(k,j,(ie+i)) =  b.x2f(k,j,ie);
        }
      }
    }      
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST+1); ++i) {
          b.x3f(k,j,(ie+i)) =  b.x3f(k,j,ie);
        }
      }
    }
  }
  return;
}// end OuterBoundaryMHD

void InnerBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh)
{
  if (CR_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) { 
        for (int i=1; i<=(NGHOST); ++i) {
          for (int n=0; n<(NCR); ++n) {
            if (n==(CRE)) {
              u_cr(CRE,k,j,is-i) = pcr->ec_in(is-i);
            } else if (n==(CRF1)) {
              u_cr(CRF1,k,j,is-i) = pcr->fc_in(is-i)/pcr->vmax;
            } else if (n==(CRF2)) {
              u_cr(CRF2,k,j,is-i) = 0.0;
            } else {
              u_cr(n,k,j,is-i) = u_cr(n,k,j,is);
            }
          }
        }
      }
    }
  }
  return;
}// end InnerBoundaryCR

void OuterBoundaryCR(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr, 
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie, 
    int js, int je, int ks, int ke, int ngh)
{
  if(CR_ENABLED){
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          for (int n=0; n<(NCR); ++n) {
            if (n==(CRE)) {
              u_cr(CRE,k,j,ie+i) = pcr->ec_in(ie+i);
            } else if (n==(CRF1)) {
              u_cr(CRF1,k,j,ie+i) = pcr->fc_in(ie+i)/pcr->vmax;
            } else if (n==(CRF2)) {
              u_cr(CRF2,k,j,ie+i) = 0.;
            } else {
              u_cr(n,k,j,ie+i) = u_cr(n,k,j,ie);
            }
          }
        }
      }
    }
  }
  return;
}// end OuterBoundaryCR

void SourceTerm(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons)
{
  CosmicRay *pcr = pmb->pcr;

  int kl = pmb->ks, ku = pmb->ke;
  int jl = pmb->js, ju = pmb->je;
  int il = pmb->is, iu = pmb->ie;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {

        Real mom_src = prim(IDN,k,j,i)*pcr->g_in(i);
        Real eng_src = mom_src*prim(IVX,k,j,i) + pcr->H_in(i);

        // Update momentum and energy
        cons(IM1,k,j,i) += mom_src*dt;
        cons(IEN,k,j,i) += eng_src*dt;

      }
    }
  }
  return;
} // End SourceTerm

