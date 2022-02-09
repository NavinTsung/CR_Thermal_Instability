//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file radiation.cpp
//  \brief implementation of functions in class Radiation
//======================================================================================


#include <sstream>  // msg
#include <iostream>  // cout
#include <stdexcept> // runtime erro
#include <stdio.h>  // fopen and fwrite
#include <boost/math/special_functions/factorials.hpp>
#include <hdf5.h>

#ifdef MPI_PARALLEL
#include <mpi.h>   // MPI_COMM_WORLD, MPI_INFO_NULL
#endif

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp" 
#include "cr.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../globals.hpp"
#include "../coordinates/coordinates.hpp"
#include "integrators/cr_integrators.hpp"
#include "../hydro/hydro.hpp"

using namespace boost::math::tools;

// constructor, initializes data structures and parameters

// The default opacity function.

// This function also needs to set the streaming velocity
// This is needed to calculate the work term 
inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
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
        pcr->sigma_diff(0,k,j,i) = pcr->max_opacity;
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
          pcr->sigma_adv(0,k,j,i) = dprdx;
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
          pcr->sigma_adv(1,k,j,i) = dprdy;

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
          pcr->sigma_adv(2,k,j,i) = dprdz;
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

          Real va = sqrt(btot*btot/prim(IDN,k,j,i));
          if(va < TINY_NUMBER){
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          }else{
            Real b_grad_pc = bcc(IB1,k,j,i) * pcr->sigma_adv(0,k,j,i)
                           + bcc(IB2,k,j,i) * pcr->sigma_adv(1,k,j,i)
                           + bcc(IB3,k,j,i) * pcr->sigma_adv(2,k,j,i);
            pcr->sigma_adv(0,k,j,i) = fabs(b_grad_pc)/(btot * va * (1.0 + 1.0/3.0) 
                                               * invlim * u_cr(CRE,k,j,i));
          }
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

        }//end i        

      }// end j
    }// end k

  }// End MHD  
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

}

CosmicRay::CosmicRay(MeshBlock *pmb, ParameterInput *pin):
    pmy_block(pmb), u_cr(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    u_cr1(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
// constructor overload resolution of non-aggregate class type AthenaArray<Real>
    flux{ {NCR, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
      {NCR,pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
       (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)},
      {NCR,pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
       (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)}}, UserSourceTerm_{},
    coarse_cr_(NCR,pmb->ncc3, pmb->ncc2, pmb->ncc1,
             (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)),
    cr_bvar(pmb, &u_cr, &coarse_cr_, flux){

  Mesh *pm = pmy_block->pmy_mesh;
  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u_cr, &coarse_cr_);
  }

  cr_source_defined = false;

  cr_bvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&cr_bvar);
  pmb->pbval->bvars_main_int.push_back(&cr_bvar);      

  vmax = pin->GetOrAddReal("cr","vmax",1.0);
  vlim = pin->GetOrAddReal("cr","vlim",1.0);
  max_opacity = pin->GetOrAddReal("cr","max_opacity",1.e10);
  stream_flag = pin->GetOrAddInteger("cr","vs_flag",1);  
  src_flag = pin->GetOrAddInteger("cr","src_flag",1);

  add_cr_heat_to_gas = pin->GetOrAddInteger("problem", "add_cr_heat_to_gas",1); // added by Navin

  // Get user couple start time, added by Navin
  add_source_cr_time = pin->GetOrAddReal("problem","add_source_cr_time",0.0);
  add_qsource_cr_time = pin->GetOrAddReal("problem","add_qsource_cr_time",0.0);

  // Added by Navin for thermal stability problem
  Real gg = 5./3.;
  Real gc = 4./3.;
  Real gg1 = gg/(gg - 1.);
  Real gc1 = gc/(gc - 1.);

  // Read data from hdf5 file
  int num_grid = pm->mesh_size.nx1;
  rho.NewAthenaArray(num_grid);
  pg.NewAthenaArray(num_grid);
  pc.NewAthenaArray(num_grid);
  fc.NewAthenaArray(num_grid);
  q.NewAthenaArray(num_grid);
  L.NewAthenaArray(num_grid);
  heat.NewAthenaArray(num_grid);
  bx.NewAthenaArray(num_grid+1+2*NGHOST);
  pc_xinner_bval.NewAthenaArray(NGHOST);
  pc_xouter_bval.NewAthenaArray(NGHOST);
  dd.NewAthenaArray(pm->mesh_size.nx3, pm->mesh_size.nx2, pm->mesh_size.nx1);

  // Open file with mpi-specific property list
  std::string filename = "../analytics/init_crprofile.hdf5";

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
  hid_t attr_alpha0 = H5Aopen(file, "alpha0", H5P_DEFAULT);
  hid_t attr_beta0 = H5Aopen(file, "beta0", H5P_DEFAULT);
  hid_t attr_nu = H5Aopen(file, "nu", H5P_DEFAULT);
  hid_t attr_Hc = H5Aopen(file, "Hc", H5P_DEFAULT);
  hid_t attr_g0 = H5Aopen(file, "g0", H5P_DEFAULT);
  hid_t attr_T0 = H5Aopen(file, "T0", H5P_DEFAULT);
  hid_t dset_rho = H5Dopen(file, "rho", H5P_DEFAULT);
  hid_t dset_pg = H5Dopen(file, "pg", H5P_DEFAULT);
  hid_t dset_pc = H5Dopen(file, "pc", H5P_DEFAULT);
  hid_t dset_fc = H5Dopen(file, "fc", H5P_DEFAULT);
  hid_t dset_q = H5Dopen(file, "q", H5P_DEFAULT);
  hid_t dset_L = H5Dopen(file, "L", H5P_DEFAULT);
  hid_t dset_heat = H5Dopen(file, "heat", H5P_DEFAULT);
  hid_t dset_bx = H5Dopen(file, "bx", H5P_DEFAULT);
  hid_t dset_pc_xinner_bval = H5Dopen(file, "pc_xinner_bval", H5P_DEFAULT);
  hid_t dset_pc_xouter_bval = H5Dopen(file, "pc_xouter_bval", H5P_DEFAULT);
  hid_t dset_dd = H5Dopen(file, "dd", H5P_DEFAULT);

  // Ensure the datasapce matches with the mesh
  hid_t fspace_rho = H5Dget_space(dset_rho);
  hsize_t num = H5Sget_simple_extent_npoints(fspace_rho);
  if (num != num_grid) {
    std::stringstream msg;
    msg << "### FATAL ERROR in cr_thermal2d.cpp ProblemGenerator" << std::endl
        << "Number of elements in user input array is different from mesh_size.nx1" << std::endl 
        << "user input size = " << num << ", mesh_size.nx1 = " << num_grid << std::endl;
    ATHENA_ERROR(msg);
  }

  // Transfer data collectively
  hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
#ifdef MPI_PARALLEL
  H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
#else
#endif

  // Read data and store into arrays
  H5Aread(attr_B, H5T_NATIVE_DOUBLE, &B);
  H5Aread(attr_kappa, H5T_NATIVE_DOUBLE, &kap);
  H5Aread(attr_alpha0, H5T_NATIVE_DOUBLE, &alpha0);
  H5Aread(attr_beta0, H5T_NATIVE_DOUBLE, &beta0);
  H5Aread(attr_nu, H5T_NATIVE_INT, &nu);
  H5Aread(attr_Hc, H5T_NATIVE_DOUBLE, &Hc);
  H5Aread(attr_g0, H5T_NATIVE_DOUBLE, &g0);
  H5Aread(attr_T0, H5T_NATIVE_DOUBLE, &T0);
  H5Dread(dset_rho, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(rho(0)));
  H5Dread(dset_pg, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(pg(0)));
  H5Dread(dset_pc, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(pc(0)));
  H5Dread(dset_fc, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(fc(0)));
  H5Dread(dset_q, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(q(0)));
  H5Dread(dset_L, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(L(0)));
  H5Dread(dset_heat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(heat(0)));
  H5Dread(dset_bx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(bx(0)));
  H5Dread(dset_pc_xinner_bval, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(pc_xinner_bval(0)));
  H5Dread(dset_pc_xouter_bval, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(pc_xouter_bval(0)));
  H5Dread(dset_dd, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer_plist, &(dd(0,0,0)));
  
  // Close all attributes, datasets, dataspaces, property list, and file
  H5Aclose(attr_B);
  H5Aclose(attr_kappa);
  H5Aclose(attr_alpha0);
  H5Aclose(attr_beta0);
  H5Aclose(attr_nu);
  H5Aclose(attr_Hc);
  H5Aclose(attr_g0);
  H5Aclose(attr_T0);
  H5Dclose(dset_rho);
  H5Dclose(dset_pg);
  H5Dclose(dset_pc);
  H5Dclose(dset_fc);
  H5Dclose(dset_q);
  H5Dclose(dset_L);
  H5Dclose(dset_heat);
  H5Dclose(dset_bx);
  H5Dclose(dset_pc_xinner_bval);
  H5Dclose(dset_pc_xouter_bval);
  H5Dclose(dset_dd);
  H5Sclose(fspace_rho);
  H5Pclose(xfer_plist);
  H5Fclose(file);
  
  epsil = pin->GetOrAddReal("problem","epsilon",-0.66667);
  T_floor = pin->GetOrAddReal("problem","T_floor",0.01);
  T_ceil = pin->GetOrAddReal("problem","T_ceil",5.0);

  // a = pin->GetOrAddReal("problem","a",0.0)*nu*Hc;
  a = pin->GetOrAddReal("problem","a",0.0)*Hc;

  Real cool_buffer_left = pin->GetOrAddReal("problem","cool_buffer_left", 1.0)*a;
  Real cool_buffer_right = pin->GetOrAddReal("problem","cool_buffer_right", 1.0)*a;

  Real x_min = pmb->pmy_mesh->mesh_size.x1min;
  Real x_max = pmb->pmy_mesh->mesh_size.x1max;
  Real y_min = pmb->pmy_mesh->mesh_size.x2min;
  Real y_max = pmb->pmy_mesh->mesh_size.x2max;
  Real z_min = pmb->pmy_mesh->mesh_size.x3min;
  Real z_max = pmb->pmy_mesh->mesh_size.x3max;

  x_min_cool_buffer_left = 0.0 + cool_buffer_left;
  x_max_cool_buffer_right = x_max - cool_buffer_right;

  Real cr_buffer_min = pin->GetOrAddReal("problem","cr_buffer_min",1.0)*a;
  Real cr_buffer_max = pin->GetOrAddReal("problem","cr_buffer_max",1.0)*a;

  x_min_cr_buffer = 0.0 + cr_buffer_min;
  x_max_cr_buffer = x_max - cr_buffer_max;

  Real add_cr_heat_to_gas_min = pin->GetOrAddReal("problem","add_cr_heat_to_gas_min",0.0)*a;
  Real add_cr_heat_to_gas_max = pin->GetOrAddReal("problem","add_cr_heat_to_gas_max",0.0)*a;

  x_min_add_cr_heat_to_gas = 0.0 + add_cr_heat_to_gas_min;
  x_max_add_cr_heat_to_gas = x_max - add_cr_heat_to_gas_max;

  Real add_perturb_min = pin->GetOrAddReal("problem","add_perturb_min",0.0)*a;
  Real add_perturb_max = pin->GetOrAddReal("problem","add_perturb_max",0.0)*a;

  x_min_add_perturb = 0.0 + add_perturb_min;
  x_max_add_perturb = x_max - add_perturb_max;

  pmb->phydro->x_start_sum = pin->GetOrAddReal("problem","start_sum",0.9)*nu*Hc;
  pmb->phydro->x_end_sum = pin->GetOrAddReal("problem","end_sum",1.1)*nu*Hc;
  
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;

  b_grad_pc.NewAthenaArray(nc3,nc2,nc1);
  b_angle.NewAthenaArray(4,nc3,nc2,nc1);

  cwidth.NewAthenaArray(nc1);
  cwidth1.NewAthenaArray(nc1);
  cwidth2.NewAthenaArray(nc1);
  
  // set a default opacity function
  UpdateOpacity = DefaultOpacity;

  pcrintegrator = new CRIntegrator(this, pin);

  // Added by Navin
  layer_grid_cr.NewAthenaArray(pm->mesh_size.nx1);
  layer_heat_cr.NewAthenaArray(pm->mesh_size.nx1);

}

//Enrol the function to update opacity

void CosmicRay::EnrollOpacityFunction(CROpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction; 
}

void CosmicRay::EnrollUserCRSource(CRSrcTermFunc my_func){
  UserSourceTerm_ = my_func;
  cr_source_defined = true; 
}

