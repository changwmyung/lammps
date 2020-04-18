/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(pimd,FixPIMD)

#else

#ifndef FIX_PIMD_H
#define FIX_PIMD_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPIMD : public Fix {
 public:
  FixPIMD(class LAMMPS *, int, char **);

  int setmask();

  void init();
  void setup(int);
  void post_force(int);
  void initial_integrate(int);
  void final_integrate();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int,int,int);
  int pack_exchange(int,double*);
  int unpack_exchange(int,double*);
  int pack_restart(int,double*);
  void unpack_restart(int,int);
  int maxsize_restart();
  int size_restart(int);
  double compute_vector(int);

  int pack_forward_comm(int, int*, double *, int, int*);
  void unpack_forward_comm(int, int, double *);

  int method;
  int np;
  double inverse_np;
  //CM centroid approx. for barostat
  int method_centroid;

  /* ring-polymer model */

  double omega_np, fbond, spring_energy, sp;
  int x_last, x_next;

  void spring_force();

  /* fictious mass */
  // mass: nmpimd mass 
  double fmass, *mass;

  /* inter-partition communication */

  int max_nsend;
  tagint* tag_send;
  double *buf_send;

  int max_nlocal;
  double *buf_recv, **buf_beads;

  int size_plan;
  int *plan_send, *plan_recv;
  double **comm_ptr;

  void comm_init();
  void comm_exec(double **);

  //CM communication for barostat
  void comm_exec_barostat(double );

  /* normal-mode operations */

  double *lam, **M_x2xp, **M_xp2x, **M_f2fp, **M_fp2f;
  int *mode_index;

  void nmpimd_init();
  void nmpimd_fill(double**);
  void nmpimd_transform(double**, double**, double*);

  /* Nose-hoover chain integration */

  int nhc_offset_one_1, nhc_offset_one_2;
  int nhc_size_one_1, nhc_size_one_2;
  int nhc_nchain;
  bool nhc_ready;
  double nhc_temp, dtv, dtf, t_sys;

  double **nhc_eta;        /* coordinates of NH chains for ring-polymer beads */
  double **nhc_eta_dot;    /* velocities of NH chains                         */
  double **nhc_eta_dotdot; /* acceleration of NH chains                       */
  double **nhc_eta_mass;   /* mass of NH chains                               */

  void nhc_init();
  void nhc_update_v();
  void nhc_update_x();

  /* NPT variables  */
  int tstat_flag;
  double t_period;

  int *rfix;                       // indices of rigid fixes
  int pstyle,pcouple;
  int p_flag[6];                   // 1 if control P on this dim, 0 if not
  double p_start[6],p_stop[6];
  double p_freq[6],p_target[6];
  double p_current[6];
  double pressure_vector[6];	   // manual calculation of pressure vector
  double p_freq_max;               // maximum barostat frequency
  double omega[6],omega_dot[6];
  double posexp[3];                // position scaling update by omega_dot
  double velexp[3];                // velocity scaling update by omega_dot
  double cellexp[3];               // cell scaling update by omega_dot
  double omega_mass[6];
  double h0_inv[6];                // h_inv of reference (zero strain) box
  int deviatoric_flag;             // 0 if target stress tensor is hydrostatic 
  double p_hydro;                  // hydrostatic target pressure
  double sigma[6];                 // scaled target stress
  double fdev[6];                  // deviatoric force on barostat
  int allremap;
  int nrigid;                      // number of rigid fixes
  double fixedpoint[3];            // location of dilation fixed-point
  int dilate_group_bit;            // mask for dilation group
  int scaleyz;                     // 1 if yz scaled with lz
  int scalexz;                     // 1 if xz scaled with lz
  int scalexy;                     // 1 if xy scaled with ly
  int kspace_flag;                 // 1 if KSpace invoked, 0 if not
  double ke_target;
  double factor_eta;

  int eta_mass_flag;               // 1 if eta_mass updated, 0 if not.
  int etap_mass_flag;              // 1 if etap_mass updated, 0 if not.
  int omega_mass_flag;             // 1 if omega_mass updated, 0 if not.
  int tcomputeflag,pcomputeflag;   // 1 = compute was created by fix
                                   // 0 = created externally

  int pstat_flag;                   // 1 if control P
  int mpchain;                     // length of chain
  int mtchain;                     // length of chain
  int nc_tchain,nc_pchain;
  double drag, pdrag_factor;             // drag factor on barostat
  int pdim;                        // number of barostatted dims
  double vol0;                      // reference volume
  double dthalf,dt4,dt8,dto;                                               
  double boltz,nktv2p,tdof;  

  char *id_temp,*id_press,*id_pe;

  int mtk_flag;                    // 0 if using Hoover barostat
  double mtk_term1,mtk_term2;      // Martyna-Tobias-Klein corrections

  //fullcell fluctuations related 
  int flipflag;                    // 1 if box flips are invoked as needed
  int pre_exchange_flag;           // set if pre_exchange needed for box flips
  virtual void pre_exchange();
  class Irregular *irregular;      // for migrating atoms after box flips

  //barostat variables
  //this I need to be careful about. they used ** instead of *.
  //CM: it is important to keep in mind that the thermostat variables are nchain*d*N*M !!!
  double *etap;                    // chain thermostat for barostat                
  double *etap_dot;
  double *etap_dotdot;                                                             
  double *etap_mass;  

  //thermostat variables
  double **eta,**eta_dot;            // chain thermostat for particles
  double **eta_dotdot;
  double **eta_mass;
  double t_freq;
  double tdrag_factor;        // drag factor on particle thermostat

//  double compute_scalar();
  void compute_press_target();
  void couple();
  void nh_omega_dot();
  void nh_omega_dot_x();
  void compute_deviatoric();
  void nhc_press_integrate();
  void nhc_temp_integrate();

  virtual void nh_v_press();
  virtual void nh_v_temp();
  virtual void nve_v();
  virtual void nve_x();            // may be overwritten by child classes
  virtual void remap();
  virtual void compute_temp_target();

  //CM test function 
  void remove_spring_force();
  void multiply_post_force();
  void divide_post_force();
  //pressure virial
  void compute_pressure_scalar();
  double pressure_scalar;
  void compute_pressure_vector();
  //temp measure
  //scalar
  double compute_temp_scalar();
  double t_current;
  double *t_current_beads;
  double t_current_avg;
  double compute_avg(double *, int );
  double compute_sum(double *, int );
  void observe_temp_scalar();

  //vector
  double *t_current_vector;
  double **t_current_vector_beads;
  double *t_current_vector_avg;
  void compute_temp_vector();

  //observable
  double vol_current;
  void monitor_observable();
  void initialize_logfile();
  double etot; 
  void observe_etot();

  //pe
  double pe_current; 
  double *pe_current_beads;  
  double pe_current_avg; 
  void observe_pe_avg();

  //vir
  double vir_current;
  double *vir_current_beads; 
  double vir_current_avg;
  void observe_virial_avg();
  double **x_buff;

  //consv energy 
  double E_consv;
  void observe_E_consv();
  void observe_eta_E_sum();
  double eta_E;
  double *eta_E_beads;
  double eta_E_sum;
  void observe_omega_E();
  double omega_E;
  void observe_etap_E_sum();
  double etap_E_sum;

  double *spring_energy_beads;
  double spring_energy_sum;
  void observe_spring_energy_sum();

  class Compute *temperature,*pressure,*pe;

  //CM diagonalization routine
  void dsyevc3(double **, double *);
  void dsyevv3(double **, double **, double *);
  double **hg_dot;
  double **eigv;
  double *omega_dot_eig;

  //CM output file 
  FILE *pimdfile;  // pimd log file

 protected:
  int dimension, which;


};


}

#endif
#endif
