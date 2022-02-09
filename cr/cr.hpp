#ifndef CR_HPP
#define CR_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include <string>



class MeshBlock;
class ParameterInput;
class CRIntegrator;

//! \class CosmicRay
//  \brief CosmicRay data and functions



// Array indices for  moments
enum {CRE=0, CRF1=1, CRF2=2, CRF3=3};


class CosmicRay {
  friend class CRIntegrator;
  friend class BoundaryValues;
public:
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
//  ~CosmicRay();

  // Added by Navin: Auxiliary quantities
  Real add_source_cr_time, add_qsource_cr_time;
  int add_cr_heat_to_gas;

  // Added by Navin: Problem specific quantities
  int nu;
  Real kap, B, T0, g0, Hc, alpha0, beta0, a;
  Real T_floor, T_ceil, epsil, L0;
  Real x_min_cool_buffer_left, x_max_cool_buffer_right;
  Real x_min_cr_buffer, x_max_cr_buffer;
  Real x_min_add_cr_heat_to_gas, x_max_add_cr_heat_to_gas;
  Real x_min_add_perturb, x_max_add_perturb;

  // Added by Navin
  AthenaArray<int> layer_grid_cr;
  AthenaArray<Real> layer_heat_cr;
  AthenaArray<Real> rho, pg, pc, fc, q, L, bx, heat;
  AthenaArray<Real> pc_xinner_bval, pc_xouter_bval;
  AthenaArray<Real> dd;
    
  AthenaArray<Real> u_cr, u_cr1, u_cr2; //cosmic ray energy density and flux

  AthenaArray<Real> coarse_cr_;

  //   diffusion coefficients for both normal diffusion term, and advection term
  AthenaArray<Real> sigma_diff, sigma_adv; 

  AthenaArray<Real> v_adv; // streaming velocity
  AthenaArray<Real> v_diff; // the diffuion velocity, need to calculate the flux
    
  int refinement_idx{-1};

  AthenaArray<Real> flux[3]; // store transport flux, also need for refinement
  
  Real vmax; // the maximum velocity (effective speed of light)
  Real vlim;
  Real max_opacity;

  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Fluid
  CellCenteredBoundaryVariable cr_bvar;
  
  CRIntegrator *pcrintegrator;
  
  //Function in problem generators to update opacity
  void EnrollOpacityFunction(CROpacityFunc MyOpacityFunction);

  void EnrollUserCRSource(CRSrcTermFunc my_func);
  bool cr_source_defined;

  // The function pointer for the diffusion coefficient
  CROpacityFunc UpdateOpacity;

  AthenaArray<Real> cwidth; 
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc; // array to store B\dot Grad Pc
  AthenaArray<Real> b_angle; //sin\theta,cos\theta,sin\phi,cos\phi of B direction

  int stream_flag; // flag to include streaming or not
  int src_flag; // flag to 

private:

  CRSrcTermFunc UserSourceTerm_;

};

#endif // CR_HPP
