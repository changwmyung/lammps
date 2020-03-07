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

  /* ring-polymer model */

  double omega_np, fbond, spring_energy, sp;
  int x_last, x_next;

  void spring_force();

  /* fictious mass */

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
  int dimension;
  int pstyle,pcouple;
  int p_flag[6];                   // 1 if control P on this dim, 0 if not
  double p_start[6],p_stop[6];
  double p_freq[6],p_target[6];
  double p_current[6];
  double p_freq_max;               // maximum barostat frequency
  double omega[6],omega_dot[6];
  double omega_mass[6];
  double h0_inv[6];                // h_inv of reference (zero strain) box
  int deviatoric_flag;             // 0 if target stress tensor is hydrostatic 
  double p_hydro;                  // hydrostatic target pressure
  double t_current;
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

  char *id_temp,*id_press;

  int mtk_flag;                    // 0 if using Hoover barostat
  double mtk_term1,mtk_term2;      // Martyna-Tobias-Klein corrections

  //barostat variables
  //this I need to be careful about. they used ** instead of *.
  double *etap;                    // chain thermostat for barostat                
  double *etap_dot;
  double *etap_dotdot;                                                             
  double *etap_mass;  

  //thermostat variables
  double *eta,*eta_dot;            // chain thermostat for particles
  double *eta_dotdot;
  double *eta_mass;
  double t_freq;
  double tdrag_factor;        // drag factor on particle thermostat

  double compute_scalar();
  void compute_press_target();
  void couple();
  void nh_omega_dot();
  void compute_deviatoric();
  void nhc_press_integrate();
  void nhc_temp_integrate();

  virtual void nh_v_press();
  virtual void nh_v_temp();
  virtual void nve_v();
  virtual void nve_x();            // may be overwritten by child classes
  virtual void remap();
  virtual void compute_temp_target();

  class Compute *temperature,*pressure;

};


}

#endif
#endif
