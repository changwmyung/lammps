/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Package      FixPIMD
   Purpose      Quantum Path Integral Algorithm for Quantum Chemistry
   Copyright    Voth Group @ University of Chicago
   Authors      Chris Knight & Yuxing Peng (yuxing at uchicago.edu)

   Updated      Oct-01-2011
   Version      1.0
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Copyright	Parrinello Group @ ETHZ & USI, Switzerland
   Authors 	Chang Woo Myung & Barak Hirshberg
   Updated	March, 2020
   Version 	2.0
   Features	* PIMD NPT Parrinello-Rahman barostat(including isotropic cell fluctuations)

   Next Feat.	* Bosonic Exchange PIMD (pimdb) PNAS (2019)
   		* Perturbed PIMD
      		* PIMD enhanced sampling

   REF
   [1] Martyna, Tuckerman, Tobias & Klein, Molecular Physics 87 1117 (1996) 
   [2] Martyna, Hughes, & Tuckerman, J. Chem. Phys. 110 3275 (1999) 
------------------------------------------------------------------------- */

#include "fix_pimd.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "universe.h"
#include "comm.h"
#include "force.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "memory.h"
#include "error.h"

//CM
#include "group.h"
#include "neighbor.h"
#include "irregular.h"
#include "modify.h"
#include "fix_deform.h"
#include "compute.h"
#include "kspace.h"
//#include "respa.h"

using namespace LAMMPS_NS;
using namespace FixConst;

//CM 
#define DELTAFLIP 0.1
#define TILTMAX 1.5

enum{PIMD,NMPIMD,CMD};

//CM 
enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};


/* ---------------------------------------------------------------------- */

FixPIMD::FixPIMD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  method     = PIMD;
  fmass      = 1.0;
  nhc_temp   = 298.15;
  nhc_nchain = 2;
  sp         = 1.0;
  //CM
  boltz = force->boltz;

  //CM variables for NPT
  mtchain = mpchain = 3;
  nc_pchain = 1;
  deviatoric_flag = 0;
  pcouple = NONE;
  mtk_flag = 1;
  allremap = 1;
  nrigid = 0;
  drag = 0.0;
  nc_tchain = nc_pchain = 1;
  dimension=3;

  omega_mass_flag = 0;
  etap_mass_flag = 0;
  eta_mass_flag = 1;

  eta = NULL;
  eta_dot = NULL;
  eta_dotdot = NULL;
  eta_mass = NULL;

  id_temp = NULL;
  id_press = NULL;
  tcomputeflag = 0;
  pcomputeflag = 0;

  // set fixed-point to default = center of cell
  fixedpoint[0] = 0.5*(domain->boxlo[0]+domain->boxhi[0]);
  fixedpoint[1] = 0.5*(domain->boxlo[1]+domain->boxhi[1]);
  fixedpoint[2] = 0.5*(domain->boxlo[2]+domain->boxhi[2]);

  scaleyz = scalexz = scalexy = 0;

  double p_period[6];
  for (int i = 0; i < 6; i++) {
    p_start[i] = p_stop[i] = p_period[i] = p_target[i] = 0.0;
    p_flag[i] = 0;
  }


  for(int i=3; i<narg-1; i+=2)
  {
    if(strcmp(arg[i],"method")==0)
    {
      if(strcmp(arg[i+1],"pimd")==0) method=PIMD;
      else if(strcmp(arg[i+1],"nmpimd")==0) method=NMPIMD;
      else if(strcmp(arg[i+1],"cmd")==0) method=CMD;
      else error->universe_all(FLERR,"Unkown method parameter for fix pimd");
    }
    else if(strcmp(arg[i],"fmass")==0)
    {
      fmass = atof(arg[i+1]);
      if(fmass<0.0 || fmass>1.0) error->universe_all(FLERR,"Invalid fmass value for fix pimd");
    }
    else if(strcmp(arg[i],"sp")==0)
    {
      sp = atof(arg[i+1]);
      if(fmass<0.0) error->universe_all(FLERR,"Invalid sp value for fix pimd");
    }
    else if(strcmp(arg[i],"temp")==0)
    {
      nhc_temp = atof(arg[i+1]);
      if(nhc_temp<0.0) error->universe_all(FLERR,"Invalid temp value for fix pimd");
    }
    else if(strcmp(arg[i],"nhc")==0)
    {
      nhc_nchain = atoi(arg[i+1]);
      if(nhc_nchain<2) error->universe_all(FLERR,"Invalid nhc value for fix pimd");
    }

    //CM pressure(iso) 
    else if(strcmp(arg[i],"iso")==0)
    {
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[i+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[i+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[i+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      //i += 4;
    }

    //else error->universe_all(arg[i],i+1,"Unkown keyword for fix pimd");
  }

  // set pstat_flag and box change and restart_pbc variables

  pstat_flag = 0;
  pstyle = ISO;

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pstat_flag) {
    if (p_flag[0] || p_flag[1] || p_flag[2]) box_change_size = 1;
    if (p_flag[3] || p_flag[4] || p_flag[5]) box_change_shape = 1;
    no_change_box = 1;

    // pstyle = TRICLINIC if any off-diagonal term is controlled -> 6 dof
    // else pstyle = ISO if XYZ coupling or XY coupling in 2d -> 1 dof
    // else pstyle = ANISO -> 3 dof

    if (p_flag[3] || p_flag[4] || p_flag[5]) pstyle = TRICLINIC;
    //else if (pcouple == XYZ || (dimension == 2 && pcouple == XY)) pstyle = ISO;
    else pstyle = ANISO;

  }

  // convert input periods to frequencies
  //CM 
  //need to make it parameter 
  t_freq = 1.0;
  p_freq[0] = p_freq[1] = p_freq[2] = p_freq[3] = p_freq[4] = p_freq[5] = 0.0;

  //if (tstat_flag) t_freq = 1.0 / t_period;
  if (p_flag[0]) p_freq[0] = 1.0 / p_period[0];
  if (p_flag[1]) p_freq[1] = 1.0 / p_period[1];
  if (p_flag[2]) p_freq[2] = 1.0 / p_period[2];
  if (p_flag[3]) p_freq[3] = 1.0 / p_period[3];
  if (p_flag[4]) p_freq[4] = 1.0 / p_period[4];
  if (p_flag[5]) p_freq[5] = 1.0 / p_period[5];

  // Nose/Hoover temp and pressure init                                            
  
  size_vector = 0;

  // thermostat variables initialization
  // CM
  int max = 3 * atom->nlocal;
  int ich;
  
  if(universe->me==0) printf("memory start\n");

  //allocating memory for array
  //eta = new double[max][mtchain];
  memory->grow(eta,        max, mtchain,   "FixPIMD::eta");
  //eta_dot = new double[max][mtchain+1];
  // add one extra dummy thermostat, set to zero
  memory->grow(eta_dot,    max, mtchain+1, "FixPIMD::eta_dot");
  //eta_dotdot = new double[max][mtchain];
  memory->grow(eta_dotdot, max, mtchain,   "FixPIMD::eta_dotdot");
  //eta_mass = new double[max][mtchain];
  memory->grow(eta_mass,   max, mtchain,   "FixPIMD::eta_mass");

  if(universe->me==0) printf("memory complete\n");

  for(int ip=0;ip<max;ip++){
    eta_dot[ip][mtchain] = 0.0;
  }
  for (int ip=0;ip<max;ip++){
    for (ich = 0; ich < mtchain; ich++) {
      eta[ip][ich] = eta_dot[ip][ich] = eta_dotdot[ip][ich] = 0.0;
    }
  }

  size_vector += 2*2*mtchain;

  // barostat variables initialization
  if (pstat_flag) {
    omega[0] = omega[1] = omega[2] = 0.0;
    omega_dot[0] = omega_dot[1] = omega_dot[2] = 0.0;
    omega_mass[0] = omega_mass[1] = omega_mass[2] = 0.0;
    omega[3] = omega[4] = omega[5] = 0.0;
    omega_dot[3] = omega_dot[4] = omega_dot[5] = 0.0;
    omega_mass[3] = omega_mass[4] = omega_mass[5] = 0.0;
    if (pstyle == ISO) size_vector += 2*2*1; 
    else if (pstyle == ANISO) size_vector += 2*2*3;
    else if (pstyle == TRICLINIC) size_vector += 2*2*6;

    if (mpchain) {
      int ich;
      etap = new double[mpchain];

      // add one extra dummy thermostat, set to zero

      etap_dot = new double[mpchain+1];
      etap_dot[mpchain] = 0.0;
      etap_dotdot = new double[mpchain];
      for (ich = 0; ich < mpchain; ich++) {
        etap[ich] = etap_dot[ich] =
          etap_dotdot[ich] = 0.0;
      }
      etap_mass = new double[mpchain];
      size_vector += 2*2*mpchain;
    }

    if (deviatoric_flag) size_vector += 1;
  }

  // create a new compute temp style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  // and thus its KE/temperature contribution should use group all

  int n = strlen(id) + 6;
  id_temp = new char[n];
  strcpy(id_temp,id);
  strcat(id_temp,"_temp");

  char **newarg = new char*[3];
  newarg[0] = id_temp; 
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "temp";
  
  modify->add_compute(3,newarg);
  delete [] newarg;
  tcomputeflag = 1;
  
  // create a new compute pressure style
  // id = fix-ID + press, compute group = all
  // pass id_temp as 4th arg to pressure constructor
  
  n = strlen(id) + 7;
  id_press = new char[n];
  strcpy(id_press,id);
  strcat(id_press,"_press");
  
  newarg = new char*[4];
  newarg[0] = id_press;
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pressure";
  newarg[3] = id_temp;
  modify->add_compute(4,newarg);
  delete [] newarg;
  pcomputeflag = 1;

  /* Initiation */

  max_nsend = 0;
  tag_send = NULL;
  buf_send = NULL;

  max_nlocal = 0;
  buf_recv = NULL;
  buf_beads = NULL;

  size_plan = 0;
  plan_send = plan_recv = NULL;

  M_x2xp = M_xp2x = M_f2fp = M_fp2f = NULL;
  lam = NULL;
  mode_index = NULL;

  mass = NULL;

  array_atom = NULL;
  nhc_eta = NULL;
  nhc_eta_dot = NULL;
  nhc_eta_dotdot = NULL;
  nhc_eta_mass = NULL;

  size_peratom_cols = 12 * nhc_nchain + 3;

  nhc_offset_one_1 = 3 * nhc_nchain;
  nhc_offset_one_2 = 3 * nhc_nchain +3;
  nhc_size_one_1 = sizeof(double) * nhc_offset_one_1;
  nhc_size_one_2 = sizeof(double) * nhc_offset_one_2;

  restart_peratom = 1;
  peratom_flag    = 1;
  peratom_freq    = 1;

  global_freq = 1;
  thermo_energy = 1;
  vector_flag = 1;
  size_vector = 2;
  extvector   = 1;
  comm_forward = 3;

  atom->add_callback(0); // Call LAMMPS to allocate memory for per-atom array
  atom->add_callback(1); // Call LAMMPS to re-assign restart-data for per-atom array

  grow_arrays(atom->nmax);

  // some initilizations

  nhc_ready = false;
}

/* ---------------------------------------------------------------------- */

int FixPIMD::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::init()
{
  if (atom->map_style == 0)
    error->all(FLERR,"Fix pimd requires an atom map, see atom_modify");

  if(universe->me==0 && screen) fprintf(screen,"Fix pimd initializing Path-Integral ...\n");

  // prepare the constants

  np = universe->nworlds;
  inverse_np = 1.0 / np;

  const double Plank     = force->hplanck;
  nktv2p = force->nktv2p;

  double hbar   = Plank / ( 2.0 * M_PI );
  double beta   = 1.0 / (boltz * nhc_temp);
  double _fbond = 1.0 * np / (beta*beta*hbar*hbar) ;

  omega_np = sqrt(np) / (hbar * beta) * sqrt(force->mvv2e);
  fbond = - _fbond * force->mvv2e;

  if(universe->me==0)
    printf("Fix pimd -P/(beta^2 * hbar^2) = %20.7lE (kcal/mol/A^2)\n\n", fbond);


  // CM 
  // setting the time-step for npt as well

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  tdrag_factor = 1.0 - (update->dt * t_freq * drag / nc_tchain);

  p_freq_max = 0.0;
  if (pstat_flag) {
    p_freq_max = MAX(p_freq[0],p_freq[1]);
    p_freq_max = MAX(p_freq_max,p_freq[2]);
    if (pstyle == TRICLINIC) {
      p_freq_max = MAX(p_freq_max,p_freq[3]);
      p_freq_max = MAX(p_freq_max,p_freq[4]);
      p_freq_max = MAX(p_freq_max,p_freq[5]);
    }
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / nc_pchain);
  }

  if (pstat_flag) {
    pdim = p_flag[0] + p_flag[1] + p_flag[2];
    if (vol0 == 0.0) {
      if (dimension == 3) vol0 = domain->xprd * domain->yprd * domain->zprd;
      else vol0 = domain->xprd * domain->yprd;
      h0_inv[0] = domain->h_inv[0];
      h0_inv[1] = domain->h_inv[1];
      h0_inv[2] = domain->h_inv[2];
      h0_inv[3] = domain->h_inv[3];
      h0_inv[4] = domain->h_inv[4];
      h0_inv[5] = domain->h_inv[5];
    }
  }

  int icompute = modify->find_compute(id_temp);
  if(universe->me==0)
    printf("icompute is: %d \n", icompute);

  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix nvt/npt does not exist");
  temperature = modify->compute[icompute];

  if (pstat_flag) {
    icompute = modify->find_compute(id_press);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    pressure = modify->compute[icompute];
  }

  //CM test for NPT
  if(universe->me==0)
    printf("p_start/p_stop/pstyle = %f / %f / %d \n", p_start[0], p_stop[0], pstyle);
    printf("p_flag = %d / %d / %d / %d / %d / %d / %d / %d \n", p_flag[0], p_flag[1], p_flag[2], p_flag[3], p_flag[4], p_flag[5], pdim, pstat_flag);

  kspace_flag = 0;
  //if (force->kspace) kspace_flag = 1;
  //else kspace_flag = 0;

// CM
/* ---------------------------------------------------------------------- */

  // MPI initialization between beads
  comm_init();

  mass = new double [atom->ntypes+1];

  if(method==CMD || method==NMPIMD) nmpimd_init();
  else for(int i=1; i<=atom->ntypes; i++) mass[i] = atom->mass[i] / np * fmass;

  if(!nhc_ready) nhc_init();


}

/* ---------------------------------------------------------------------- */

void FixPIMD::setup(int vflag)
{

  if(universe->me==0 && screen) fprintf(screen,"Setting up Path-Integral ...\n");

  //CM
  //force is updated first 
  post_force(vflag);  //previous post_force function
  remove_spring_force();

  if(universe->me==0 && screen) fprintf(screen,"1. Setting up Path-Integral ...\n");

/* CM ----------------------------------------------------------------------
  Compute T,P before integrator starts

  - It's important that the spring force terms is excluded from the pressure calculations.
------------------------------------------------------------------------- */

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;
  //if(universe->me==0) printf("tdof: %f\n", tdof);

  if (pstat_flag) compute_press_target();

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  //CM 
  spring_force();

  // masses and initial forces on thermostat variables
  double t_target=nhc_temp;
  int max = 3 * atom->nlocal;
  for (int ip=0;ip<max;ip++){
    //Note that eta_mass[0] = kT/(freq**2) 
    //In conventional eom, eta_mass[0] = (d*N)*kT/(freq**2)
    //Martyna, Tuckerman, Tobias, Klein, Molecular Physics 87 1117 (1996) 
    eta_mass[ip][0] = boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++)
      eta_mass[ip][ich] = boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++) {
      eta_dotdot[ip][ich] = (eta_mass[ip][ich-1]*eta_dot[ip][ich-1]*eta_dot[ip][ich-1] -
                       boltz * t_target) / eta_mass[ip][ich];
    }
  }
  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * nhc_temp;
    /* CM
    Note that the barostat mass W becomes 
    (N*np+1)*k_B*T/(p_freq**2) (vs. (N+1)*k_B*T/(p_freq**2))
    However, the approximation of translation mode gives the same barostat mass as before:
    (N+1)*k_B*T/(p_freq**2)
    Eq(3.5) J. Chem. Phys. 110 3275 (1999)
    */

    // CM
    double nkt = (atom->natoms + 1) * kt;

    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }

  // masses and initial forces on barostat thermostat variables

    if (mpchain) {
      etap_mass[0] = boltz * nhc_temp / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * nhc_temp / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * nhc_temp) / etap_mass[ich];
    }
  }

/* ---------------------------------------------------------------------- */
  if(universe->me==0 && screen) fprintf(screen,"Finished setting up Path-Integral ...\n");

}

/* ---------------------------------------------------------------------- */

void FixPIMD::initial_integrate(int /*vflag*/)
{

  if (pstat_flag && mpchain){

  //if(universe->me==0)
  //  printf("NPT running...! %d %d \n", pstat_flag, mpchain);

    nhc_press_integrate();
    compute_temp_target();
    nhc_temp_integrate();

//  // need to recompute pressure to account for change in KE
//  // t_current is up-to-date, but compute_temperature is not
//  // compute appropriately coupled elements of mvv_current

//  CM measure the pressure & temperature again!

//  For test purpose,
    remove_spring_force();
////////

    if (pstat_flag) {
      if (pstyle == ISO) {
        temperature->compute_scalar();
        pressure->compute_scalar();
      } else {
        temperature->compute_vector();
        pressure->compute_vector();
      }
      couple();
      pressure->addstep(update->ntimestep+1);
    }
  
    if (pstat_flag) {
      compute_press_target();
      //CM
      //this should only be done in proc0 and then broadcast
      if(universe->me==0) nh_omega_dot();
      //broadcast
      MPI_Barrier(universe->uworld);
      MPI_Bcast(omega_dot, 6, MPI_DOUBLE, 0, universe->uworld);
      MPI_Barrier(universe->uworld);

      //CM
      //reduced centroid eom.
      if(universe->me==0) nh_v_press();
      
    }

//  For test purpose,
    spring_force();
////////
  
    nve_v();
  
    // remap simulation box by 1/2 step
  
    if (pstat_flag) remap();
  
    nve_x();
  
    // remap simulation box by 1/2 step
    // redo KSpace coeffs since volume has changed
  
    if (pstat_flag) {
      remap();
      if (kspace_flag) force->kspace->setup();
    }

  }else{
  nhc_update_v();
  nhc_update_x();
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMD::final_integrate()
{
  if (pstat_flag && mpchain){
    nve_v();

    //CM
    //scaling is only for centroid
    if(universe->me==0) nh_v_press();
    //nh_v_press();

    // compute new T,P after velocities rescaled by nh_v_press()
    // compute appropriately coupled elements of mvv_current

    t_current = temperature->compute_scalar();
    tdof = temperature->dof;

    // need to recompute pressure to account for change in KE
    // t_current is up-to-date, but compute_temperature is not
    // compute appropriately coupled elements of mvv_current

//  For test purpose,
    remove_spring_force();
////////

    if (pstat_flag) {
      if (pstyle == ISO) pressure->compute_scalar();
      else {
        temperature->compute_vector();
        pressure->compute_vector();
      }
      couple();
      pressure->addstep(update->ntimestep+1);
    }

//  For test purpose,
    spring_force();
////////

    //CM
    //this should only be done in proc0 and then broadcast
    if(universe->me==0) nh_omega_dot();
    //broadcast
    MPI_Barrier(universe->uworld);
    MPI_Bcast(omega_dot, 6, MPI_DOUBLE, 0, universe->uworld);
    MPI_Barrier(universe->uworld);

    // update eta_dot
    // update eta_press_dot
    nhc_temp_integrate();
    nhc_press_integrate();

  } else{
  nhc_update_v();
  }
}

/* ---------------------------------------------------------------------- */
// CM force calculations
void FixPIMD::post_force(int /*flag*/)
{
  for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) atom->f[i][j] /= np;
  //CM if no scaling
  //for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) atom->f[i][j] /= 1.0;

  comm_exec(atom->x);
  spring_force();

  if(method==CMD || method==NMPIMD)
  {
    /* forward comm for the force on ghost atoms */

    nmpimd_fill(atom->f);

    /* inter-partition comm */

    comm_exec(atom->f);

    /* normal-mode transform */

    nmpimd_transform(buf_beads, atom->f, M_f2fp[universe->iworld]);
  }
}

/* ----------------------------------------------------------------------
   Nose-Hoover Chains
------------------------------------------------------------------------- */

void FixPIMD::nhc_init()
{

  double tau = 1.0 / omega_np;
  double KT  = boltz * nhc_temp;

  double mass0 = KT * tau * tau;
  int max = 3 * atom->nlocal;

  for(int i=0; i<max; i++)
  {
    for(int ichain=0; ichain<nhc_nchain; ichain++)
    {
      nhc_eta[i][ichain]        = 0.0;
      nhc_eta_dot[i][ichain]    = 0.0;
      nhc_eta_dot[i][ichain]    = 0.0;
      nhc_eta_dotdot[i][ichain] = 0.0;
      nhc_eta_mass[i][ichain]   = mass0;
      if((method==CMD || method==NMPIMD) && universe->iworld==0) ; else nhc_eta_mass[i][ichain]  *= fmass;
    }

    nhc_eta_dot[i][nhc_nchain]    = 0.0;

    for(int ichain=1; ichain<nhc_nchain; ichain++)
      nhc_eta_dotdot[i][ichain] = (nhc_eta_mass[i][ichain-1] * nhc_eta_dot[i][ichain-1]
        * nhc_eta_dot[i][ichain-1] * force->mvv2e - KT) / nhc_eta_mass[i][ichain];
  }

  // Zero NH acceleration for CMD

  if(method==CMD && universe->iworld==0) for(int i=0; i<max; i++)
    for(int ichain=0; ichain<nhc_nchain; ichain++) nhc_eta_dotdot[i][ichain] = 0.0;

  nhc_ready = true;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::nhc_update_x()
{
  int n = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;

  if(method==CMD || method==NMPIMD)
  {
    nmpimd_fill(atom->v);
    comm_exec(atom->v);

    /* borrow the space of atom->f to store v in cartisian */

    v = atom->f;
    nmpimd_transform(buf_beads, v, M_xp2x[universe->iworld]);
  }

  for(int i=0; i<n; i++)
  {
    x[i][0] += dtv * v[i][0];
    x[i][1] += dtv * v[i][1];
    x[i][2] += dtv * v[i][2];
  }
}

/* ---------------------------------------------------------------------- */
// CM
void FixPIMD::nhc_update_v()
{
  int n = atom->nlocal;
  int *type = atom->type;
  double **v = atom->v;
  double **f = atom->f;

  for(int i=0; i<n; i++)
  {
    double dtfm = dtf / mass[type[i]];
    v[i][0] += dtfm * f[i][0];
    v[i][1] += dtfm * f[i][1];
    v[i][2] += dtfm * f[i][2];
  }

  t_sys = 0.0;
  if(method==CMD && universe->iworld==0) return;

  double expfac;
  int nmax = 3 * atom->nlocal;
  double KT = boltz * nhc_temp;
  double kecurrent, nhc_t_current;

  double dthalf = 0.5   * update->dt;
  double dt4    = 0.25  * update->dt;
  double dt8    = 0.125 * update->dt;

  for(int i=0; i<nmax; i++)
  {
    int iatm = i/3;
    int idim = i%3;

    double *vv = v[iatm];

    kecurrent = mass[type[iatm]] * vv[idim]* vv[idim] * force->mvv2e;
    nhc_t_current = kecurrent / boltz;

    // eta
    double *eta = nhc_eta[i];
    // v_eta
    double *eta_dot = nhc_eta_dot[i];
    // G_k
    double *eta_dotdot = nhc_eta_dotdot[i];

    eta_dotdot[0] = (kecurrent - KT) / nhc_eta_mass[i][0];

    for(int ichain=nhc_nchain-1; ichain>0; ichain--)
    {
      expfac = exp(-dt8 * eta_dot[ichain+1]);
      eta_dot[ichain] *= expfac;
      eta_dot[ichain] += eta_dotdot[ichain] * dt4;
      eta_dot[ichain] *= expfac;
    }

    expfac = exp(-dt8 * eta_dot[1]);
    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * dt4;
    eta_dot[0] *= expfac;

    // Update particle velocities half-step

    double nhc_factor_eta = exp(-dthalf * eta_dot[0]);
    vv[idim] *= nhc_factor_eta;

    nhc_t_current *= (nhc_factor_eta * nhc_factor_eta);
    kecurrent = boltz * nhc_t_current;
    eta_dotdot[0] = (kecurrent - KT) / nhc_eta_mass[i][0];

    for(int ichain=0; ichain<nhc_nchain; ichain++)
      eta[ichain] += dthalf * eta_dot[ichain];

    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * dt4;
    eta_dot[0] *= expfac;

    for(int ichain=1; ichain<nhc_nchain; ichain++)
    {
      expfac = exp(-dt8 * eta_dot[ichain+1]);
      eta_dot[ichain] *= expfac;
      eta_dotdot[ichain] = (nhc_eta_mass[i][ichain-1] * eta_dot[ichain-1] * eta_dot[ichain-1]
                           - KT) / nhc_eta_mass[i][ichain];
      eta_dot[ichain] += eta_dotdot[ichain] * dt4;
      eta_dot[ichain] *= expfac;
    }

    t_sys += nhc_t_current;
  }

  t_sys /= nmax;
}

/* ----------------------------------------------------------------------
   Normal Mode PIMD
------------------------------------------------------------------------- */

void FixPIMD::nmpimd_init()
{
  memory->create(M_x2xp, np, np, "fix_feynman:M_x2xp");
  memory->create(M_xp2x, np, np, "fix_feynman:M_xp2x");
  memory->create(M_f2fp, np, np, "fix_feynman:M_f2fp");
  memory->create(M_fp2f, np, np, "fix_feynman:M_fp2f");

  lam = (double*) memory->smalloc(sizeof(double)*np, "FixPIMD::lam");

  // Set up  eigenvalues

  lam[0] = 0.0;
  if(np%2==0) lam[np-1] = 4.0 * np;

  for(int i=2; i<=np/2; i++)
  {
    lam[2*i-3] = lam[2*i-2] = 2.0 * np * (1.0 - 1.0 *cos(2.0*M_PI*(i-1)/np));
  }

  // Set up eigenvectors for non-degenerated modes

  for(int i=0; i<np; i++)
  {
    M_x2xp[0][i] = 1.0 / np;
    if(np%2==0) M_x2xp[np-1][i] = 1.0 / np * pow(-1.0, i);
  }

  // Set up eigenvectors for degenerated modes

  for(int i=0; i<(np-1)/2; i++) for(int j=0; j<np; j++)
  {
    M_x2xp[2*i+1][j] =   sqrt(2.0) * cos ( 2.0 * M_PI * (i+1) * j / np) / np;
    M_x2xp[2*i+2][j] = - sqrt(2.0) * sin ( 2.0 * M_PI * (i+1) * j / np) / np;
  }

  // Set up Ut

  for(int i=0; i<np; i++)
    for(int j=0; j<np; j++)
    {
      M_xp2x[i][j] = M_x2xp[j][i] * np;
      M_f2fp[i][j] = M_x2xp[i][j] * np;
      M_fp2f[i][j] = M_xp2x[i][j];
    }

  // Set up masses

  int iworld = universe->iworld;

  for(int i=1; i<=atom->ntypes; i++)
  {
    mass[i] = atom->mass[i];

    if(iworld)
    {
      mass[i] *= lam[iworld];
      mass[i] *= fmass;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMD::nmpimd_fill(double **ptr)
{
  comm_ptr = ptr;
  comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */

void FixPIMD::nmpimd_transform(double** src, double** des, double *vector)
{
  int n = atom->nlocal;
  int m = 0;

  for(int i=0; i<n; i++) for(int d=0; d<3; d++)
  {
    des[i][d] = 0.0;
    for(int j=0; j<np; j++) { des[i][d] += (src[j][m] * vector[j]); }
    m++;
  }
}

/* ---------------------------------------------------------------------- */

// CM 
// test function that removes the harmonic terms 

void FixPIMD::remove_spring_force()
{
  double **x = atom->x;
  double **f = atom->f;
  double* _mass = atom->mass;
  int* type = atom->type;
  int nlocal = atom->nlocal;

  double* xlast = buf_beads[x_last];
  double* xnext = buf_beads[x_next];

  for(int i=0; i<nlocal; i++)
  {
    double delx1 = xlast[0] - x[i][0];
    double dely1 = xlast[1] - x[i][1];
    double delz1 = xlast[2] - x[i][2];
    xlast += 3;
    domain->minimum_image(delx1, dely1, delz1);

    double delx2 = xnext[0] - x[i][0];
    double dely2 = xnext[1] - x[i][1];
    double delz2 = xnext[2] - x[i][2];
    xnext += 3;
    domain->minimum_image(delx2, dely2, delz2);

    double ff = fbond * _mass[type[i]];

    double dx = delx1+delx2;
    double dy = dely1+dely2;
    double dz = delz1+delz2;

    f[i][0] += (dx) * ff;
    f[i][1] += (dy) * ff;
    f[i][2] += (dz) * ff;
  }
}

void FixPIMD::spring_force()
{
  spring_energy = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double* _mass = atom->mass;
  int* type = atom->type;
  int nlocal = atom->nlocal;

  double* xlast = buf_beads[x_last];
  double* xnext = buf_beads[x_next];

  for(int i=0; i<nlocal; i++)
  {
    double delx1 = xlast[0] - x[i][0];
    double dely1 = xlast[1] - x[i][1];
    double delz1 = xlast[2] - x[i][2];
    xlast += 3;
    domain->minimum_image(delx1, dely1, delz1);

    double delx2 = xnext[0] - x[i][0];
    double dely2 = xnext[1] - x[i][1];
    double delz2 = xnext[2] - x[i][2];
    xnext += 3;
    domain->minimum_image(delx2, dely2, delz2);

    double ff = fbond * _mass[type[i]];

    double dx = delx1+delx2;
    double dy = dely1+dely2;
    double dz = delz1+delz2;

    f[i][0] -= (dx) * ff;
    f[i][1] -= (dy) * ff;
    f[i][2] -= (dz) * ff;

    spring_energy += (dx*dx+dy*dy+dz*dz);
  }
}

/* ----------------------------------------------------------------------
   Comm operations
------------------------------------------------------------------------- */

void FixPIMD::comm_init()
{
  if(size_plan)
  {
    delete [] plan_send;
    delete [] plan_recv;
  }

  if(method == PIMD)
  {
    size_plan = 2;
    plan_send = new int [2];
    plan_recv = new int [2];
    mode_index = new int [2];

    int rank_last = universe->me - comm->nprocs;
    int rank_next = universe->me + comm->nprocs;
    if(rank_last<0) rank_last += universe->nprocs;
    if(rank_next>=universe->nprocs) rank_next -= universe->nprocs;

    plan_send[0] = rank_next; plan_send[1] = rank_last;
    plan_recv[0] = rank_last; plan_recv[1] = rank_next;

    mode_index[0] = 0; mode_index[1] = 1;
    x_last = 1; x_next = 0;
  }
  else
  {
    size_plan = np - 1;
    plan_send = new int [size_plan];
    plan_recv = new int [size_plan];
    mode_index = new int [size_plan];
    //CM
    /*
    Variables
    - size_plan: the number of beads
    - comm->nprocs: the number of communicating processors, which is basically 1.
    - universe->iworld and universe->me are the same. (not sure)
    - universe->nworlds: the number of total cores
    - x_next, x_last: neighboring bead for spring force

    Basically, it sets up the send-recv operation where adjacent processors can communicate as a ring.

    */

    for(int i=0; i<size_plan; i++)
    {
      plan_send[i] = universe->me + comm->nprocs * (i+1);
      if(plan_send[i]>=universe->nprocs) plan_send[i] -= universe->nprocs;

      plan_recv[i] = universe->me - comm->nprocs * (i+1);
      if(plan_recv[i]<0) plan_recv[i] += universe->nprocs;

      mode_index[i]=(universe->iworld+i+1)%(universe->nworlds);

      if(universe->me==0)
        printf("comm->nprocs / plan_send / plan_recv / mode_index: %d / %d / %d / %d \n", comm->nprocs, plan_send[i], plan_recv[i], mode_index[i]);
        printf("universe->iworld / universe->nworlds: %d / %d \n", universe->iworld, universe->nworlds);
    }

    x_next = (universe->iworld+1+universe->nworlds)%(universe->nworlds);
    x_last = (universe->iworld-1+universe->nworlds)%(universe->nworlds);
  }

  if(buf_beads)
  {
    for(int i=0; i<np; i++) if(buf_beads[i]) delete [] buf_beads[i];
    delete [] buf_beads;
  }

  buf_beads = new double* [np];
  for(int i=0; i<np; i++) buf_beads[i] = NULL;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::comm_exec(double **ptr)
{
  int nlocal = atom->nlocal;

  //CM why do we need this?
  if(nlocal > max_nlocal)
  {
    max_nlocal = nlocal+200;
    int size = sizeof(double) * max_nlocal * 3;
    buf_recv = (double*) memory->srealloc(buf_recv, size, "FixPIMD:x_recv");

    for(int i=0; i<np; i++)
      buf_beads[i] = (double*) memory->srealloc(buf_beads[i], size, "FixPIMD:x_beads[i]");
  }

  // copy local positions

  memcpy(buf_beads[universe->iworld], &(ptr[0][0]), sizeof(double)*nlocal*3);

  // go over comm plans
  // size plan = np-1
  /*
  MPI_SENDRECV(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status)
  [ IN sendbuf] initial address of send buffer (choice)
  [ IN sendcount] number of elements in send buffer (integer)
  [ IN sendtype] type of elements in send buffer (handle)
  [ IN dest] rank of destination (integer)
  [ IN sendtag] send tag (integer)
  [ OUT recvbuf] initial address of receive buffer (choice)
  [ IN recvcount] number of elements in receive buffer (integer)
  [ IN recvtype] type of elements in receive buffer (handle)
  [ IN source] rank of source (integer)
  [ IN recvtag] receive tag (integer)
  [ IN comm] communicator (handle)
  [ OUT status] status object (Status)
  */
  for(int iplan = 0; iplan<size_plan; iplan++)
  {
    // sendrecv nlocal

    int nsend;

    MPI_Sendrecv( &(nlocal), 1, MPI_INT, plan_send[iplan], 0,
                  &(nsend),  1, MPI_INT, plan_recv[iplan], 0, universe->uworld, MPI_STATUS_IGNORE);

    // allocate arrays

    if(nsend > max_nsend)
    {
      max_nsend = nsend+200;
      tag_send = (tagint*) memory->srealloc(tag_send, sizeof(tagint)*max_nsend, "FixPIMD:tag_send");
      buf_send = (double*) memory->srealloc(buf_send, sizeof(double)*max_nsend*3, "FixPIMD:x_send");
    }

    // send tags

    MPI_Sendrecv( atom->tag, nlocal, MPI_LMP_TAGINT, plan_send[iplan], 0,
                  tag_send,  nsend,  MPI_LMP_TAGINT, plan_recv[iplan], 0, universe->uworld, MPI_STATUS_IGNORE);

    // wrap positions

    double *wrap_ptr = buf_send;
    int ncpy = sizeof(double)*3;

    for(int i=0; i<nsend; i++)
    {
      int index = atom->map(tag_send[i]);

      if(index<0)
      {
        char error_line[256];

        sprintf(error_line, "Atom " TAGINT_FORMAT " is missing at world [%d] "
                "rank [%d] required by  rank [%d] (" TAGINT_FORMAT ", "
                TAGINT_FORMAT ", " TAGINT_FORMAT ").\n", tag_send[i],
                universe->iworld, comm->me, plan_recv[iplan],
                atom->tag[0], atom->tag[1], atom->tag[2]);

        error->universe_one(FLERR,error_line);
      }

      memcpy(wrap_ptr, ptr[index], ncpy);
      wrap_ptr += 3;
    }

    // sendrecv x

    MPI_Sendrecv( buf_send, nsend*3,  MPI_DOUBLE, plan_recv[iplan], 0,
                  buf_recv, nlocal*3, MPI_DOUBLE, plan_send[iplan], 0, universe->uworld, MPI_STATUS_IGNORE);

    // copy x

    memcpy(buf_beads[mode_index[iplan]], buf_recv, sizeof(double)*nlocal*3);
  }
}

/*
CM
PIMD Barostat

MPI_Bcast(+MPI_Barrier) unit-cell vectors and pressure tensor to all the beads.
A root proc is proc0 that is the translational mode after the normal mode transf.
*/

void FixPIMD::comm_exec_barostat(double volume)
{
  int num_elements=1;

  //CM
  //Syncronize
  MPI_Barrier(universe->uworld);
  //send volume
  MPI_Bcast(&volume, num_elements, MPI_DOUBLE, 0, universe->uworld);
  MPI_Barrier(universe->uworld);
}

/* ---------------------------------------------------------------------- */

int FixPIMD::pack_forward_comm(int n, int *list, double *buf,
                             int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = comm_ptr[j][0];
    buf[m++] = comm_ptr[j][1];
    buf[m++] = comm_ptr[j][2];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    comm_ptr[i][0] = buf[m++];
    comm_ptr[i][1] = buf[m++];
    comm_ptr[i][2] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   Memory operations
------------------------------------------------------------------------- */

double FixPIMD::memory_usage()
{
  double bytes = 0;
  bytes = atom->nmax * size_peratom_cols * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::grow_arrays(int nmax)
{
  if (nmax==0) return;
  int count = nmax*3;

  memory->grow(array_atom, nmax, size_peratom_cols, "FixPIMD::array_atom");
  memory->grow(nhc_eta,        count, nhc_nchain,   "FixPIMD::nh_eta");
  memory->grow(nhc_eta_dot,    count, nhc_nchain+1, "FixPIMD::nh_eta_dot");
  memory->grow(nhc_eta_dotdot, count, nhc_nchain,   "FixPIMD::nh_eta_dotdot");
  memory->grow(nhc_eta_mass,   count, nhc_nchain,   "FixPIMD::nh_eta_mass");
}

/* ---------------------------------------------------------------------- */

void FixPIMD::copy_arrays(int i, int j, int /*delflag*/)
{
  int i_pos = i*3;
  int j_pos = j*3;

  memcpy(nhc_eta       [j_pos], nhc_eta       [i_pos], nhc_size_one_1);
  memcpy(nhc_eta_dot   [j_pos], nhc_eta_dot   [i_pos], nhc_size_one_2);
  memcpy(nhc_eta_dotdot[j_pos], nhc_eta_dotdot[i_pos], nhc_size_one_1);
  memcpy(nhc_eta_mass  [j_pos], nhc_eta_mass  [i_pos], nhc_size_one_1);
}

/* ---------------------------------------------------------------------- */

int FixPIMD::pack_exchange(int i, double *buf)
{
  int offset=0;
  int pos = i * 3;

  memcpy(buf+offset, nhc_eta[pos],        nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_dot[pos],    nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(buf+offset, nhc_eta_dotdot[pos], nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_mass[pos],   nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols;
}

/* ---------------------------------------------------------------------- */

int FixPIMD::unpack_exchange(int nlocal, double *buf)
{
  int offset=0;
  int pos = nlocal*3;

  memcpy(nhc_eta[pos],        buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(nhc_eta_dot[pos],    buf+offset, nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(nhc_eta_dotdot[pos], buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(nhc_eta_mass[pos],   buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols;
}

/* ---------------------------------------------------------------------- */

int FixPIMD::pack_restart(int i, double *buf)
{
  int offset=0;
  int pos = i * 3;
  buf[offset++] = size_peratom_cols+1;

  memcpy(buf+offset, nhc_eta[pos],        nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_dot[pos],    nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(buf+offset, nhc_eta_dotdot[pos], nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_mass[pos],   nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i=0; i<nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  int pos = nlocal * 3;

  memcpy(nhc_eta[pos],        extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;
  memcpy(nhc_eta_dot[pos],    extra[nlocal]+m, nhc_size_one_2); m += nhc_offset_one_2;
  memcpy(nhc_eta_dotdot[pos], extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;
  memcpy(nhc_eta_mass[pos],   extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;

  nhc_ready = true;
}

/* ---------------------------------------------------------------------- */

int FixPIMD::maxsize_restart()
{
  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

int FixPIMD::size_restart(int /*nlocal*/)
{
  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

double FixPIMD::compute_vector(int n)
{
  if(n==0) { return spring_energy; }
  if(n==1) { return t_sys; }
  return 0.0;
}


/* ----------------------------------------------------------------------
   compute hydrostatic target pressure
-----------------------------------------------------------------------*/

void FixPIMD::compute_press_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  p_hydro = 0.0;
  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);
      p_hydro += p_target[i];
    }
  if (pdim > 0) p_hydro /= pdim;

  if (pstyle == TRICLINIC)
    for (int i = 3; i < 6; i++)
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);

  // if deviatoric, recompute sigma each time p_target changes

  //if (deviatoric_flag) compute_sigma();
}

void FixPIMD::couple()
{
  double *tensor = pressure->vector;

  if (pstyle == ISO)
    p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
  else if (pcouple == XYZ) {
    double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
    p_current[0] = p_current[1] = p_current[2] = ave;
  } else if (pcouple == XY) {
    double ave = 0.5 * (tensor[0] + tensor[1]);
    p_current[0] = p_current[1] = ave;
    p_current[2] = tensor[2];
  } else if (pcouple == YZ) {
    double ave = 0.5 * (tensor[1] + tensor[2]);
    p_current[1] = p_current[2] = ave;
    p_current[0] = tensor[0];
  } else if (pcouple == XZ) {
    double ave = 0.5 * (tensor[0] + tensor[2]);
    p_current[0] = p_current[2] = ave;
    p_current[1] = tensor[1];
  } else {
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
  }

  if (!std::isfinite(p_current[0]) || !std::isfinite(p_current[1]) || !std::isfinite(p_current[2]))
    error->all(FLERR,"Non-numeric pressure - simulation unstable");

  // switch order from xy-xz-yz to Voigt

  if (pstyle == TRICLINIC) {
    p_current[3] = tensor[5];
    p_current[4] = tensor[4];
    p_current[5] = tensor[3];

    if (!std::isfinite(p_current[3]) || !std::isfinite(p_current[4]) || !std::isfinite(p_current[5]))
      error->all(FLERR,"Non-numeric pressure - simulation unstable");
  }
}

/* ----------------------------------------------------------------------
   update omega_dot, omega
-----------------------------------------------------------------------*/

/*CM
This part require a major modification for f_omega term
that should sum all beads' kinetic terms and normalized by N*np

Eq(3.5) J. Chem. Phys. 110 3275 (1999)
*/
void FixPIMD::nh_omega_dot()
{
  double f_omega,volume;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  //CM
  // calculate & broadcast volume of translation mode
  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  //broadcast volume to all 
  //comm_exec_barostat(volume);

  //CM test for mpi communications
  //if(universe->me==0){ 
  //  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  //  else volume = domain->xprd*domain->yprd;}
  //MPI_Barrier(universe->uworld);
  //MPI_Bcast(&volume, 1, MPI_DOUBLE, 0, universe->uworld);
  //MPI_Barrier(universe->uworld);

  //
  //fprintf(universe->ulogfile,"volume: %f \n", volume);

  if(universe->me==0)
    printf("volume 0: %f, %f, %f, %f \n", volume, domain->xprd, domain->yprd, domain->zprd);

  if(universe->me==6)
    printf("volume 6: %f \n", volume);

  if (deviatoric_flag) compute_deviatoric();

  //CM
  //p**2/m term
  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) {
      mtk_term1 = tdof * boltz * t_current;
      mtk_term1 /= pdim * atom->natoms;
    } else {
      double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i])
          mtk_term1 += mvv_current[i];
      mtk_term1 /= pdim * atom->natoms;
    }
  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
        (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
      //CM
      //if(universe->me==0)
      //  printf("omega_dot/p_current/p_hydro: %f / %f / %f \n", omega_dot[0], p_current[0], p_hydro);
    }

  //CM the position update 
  //eq(3.5.1) in [2]
  for (int i = 0; i < 3; i++){
    posexp[i]=exp(dthalf*omega_dot[i]/omega_mass[i]);
    for (int ip = 0; ip < nlocal; ip++) {
      x[ip][i] *= posexp[i];
    }
  }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        mtk_term2 += omega_dot[i];
    if (pdim > 0) mtk_term2 /= pdim * atom->natoms;
  }

  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++) {
      if (p_flag[i]) {
        f_omega = p_current[i]*volume/(omega_mass[i] * nktv2p);
        if (deviatoric_flag)
          f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
        omega_dot[i] += f_omega*dthalf;
        omega_dot[i] *= pdrag_factor;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute deviatoric barostat force = h*sigma*h^t
-----------------------------------------------------------------------*/

void FixPIMD::compute_deviatoric()
{
  // generate upper-triangular part of h*sigma*h^t
  // units of fdev are are PV, e.g. atm*A^3
  // [ 0 5 4 ]   [ 0 5 4 ] [ 0 5 4 ] [ 0 - - ]
  // [ 5 1 3 ] = [ - 1 3 ] [ 5 1 3 ] [ 5 1 - ]
  // [ 4 3 2 ]   [ - - 2 ] [ 4 3 2 ] [ 4 3 2 ]

  double* h = domain->h;

  fdev[0] =
    h[0]*(sigma[0]*h[0]+sigma[5]*h[5]+sigma[4]*h[4]) +
    h[5]*(sigma[5]*h[0]+sigma[1]*h[5]+sigma[3]*h[4]) +
    h[4]*(sigma[4]*h[0]+sigma[3]*h[5]+sigma[2]*h[4]);
  fdev[1] =
    h[1]*(              sigma[1]*h[1]+sigma[3]*h[3]) +
    h[3]*(              sigma[3]*h[1]+sigma[2]*h[3]);
  fdev[2] =
    h[2]*(                            sigma[2]*h[2]);
  fdev[3] =
    h[1]*(                            sigma[3]*h[2]) +
    h[3]*(                            sigma[2]*h[2]);
  fdev[4] =
    h[0]*(                            sigma[4]*h[2]) +
    h[5]*(                            sigma[3]*h[2]) +
    h[4]*(                            sigma[2]*h[2]);
  fdev[5] =
    h[0]*(              sigma[5]*h[1]+sigma[4]*h[3]) +
    h[5]*(              sigma[1]*h[1]+sigma[3]*h[3]) +
    h[4]*(              sigma[3]*h[1]+sigma[2]*h[3]);
}

/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/
//CM
//this scaling is only for centroid mode for reduced scheme [2].
void FixPIMD::nh_v_press()
{
  double factor[3];
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //CM
  //where is the W_{iso} term????
  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2)/omega_mass[0]);
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2)/omega_mass[1]);
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2)/omega_mass[2]);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      v[i][0] *= factor[0];
      v[i][1] *= factor[1];
      v[i][2] *= factor[2];
      if (pstyle == TRICLINIC) {
        v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
        v[i][1] += -dthalf*v[i][2]*omega_dot[3];
      }
      v[i][0] *= factor[0];
      v[i][1] *= factor[1];
      v[i][2] *= factor[2];
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixPIMD::nve_v()
{
  double dtfm;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   change box size
   remap all atoms or dilate group atoms depending on allremap flag
   if rigid bodies exist, scale rigid body centers-of-mass
------------------------------------------------------------------------- */

void FixPIMD::remap()
{
  int i;
  double oldlo,oldhi;
  double expfac;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords

  if (allremap) domain->x2lamda(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->x2lamda(x[i],x[i]);
  }

//  if (nrigid)
//    for (i = 0; i < nrigid; i++)
//      modify->fix[rfix[i]]->deform(0);

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

  }

  domain->yz = h[3];
  domain->xz = h[4];
  domain->xy = h[5];

  // tilt factor to cell length ratio can not exceed TILTMAX in one step

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz > TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz > TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy > TILTMAX*domain->xprd)
    error->all(FLERR,"Fix npt/nph has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state");

  domain->set_global_box();
  domain->set_local_box();

  // convert pertinent atoms and rigid bodies back to box coords

  if (allremap) domain->lamda2x(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->lamda2x(x[i],x[i]);
  }

//  if (nrigid)
//    for (i = 0; i < nrigid; i++)
//      modify->fix[rfix[i]]->deform(1);

}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixPIMD::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if(method==CMD || method==NMPIMD)
  {
    nmpimd_fill(atom->v);
    comm_exec(atom->v);

    /* borrow the space of atom->f to store v in cartisian */

    v = atom->f;
    nmpimd_transform(buf_beads, v, M_xp2x[universe->iworld]);
  }

  // x update by full step only for atoms in group

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables for barostat
   scale barostat velocities
------------------------------------------------------------------------- */

void FixPIMD::nhc_press_integrate()
{
  int ich,i,pdof;
  double expfac,factor_etap,kecurrent;
  double lkt_press;
  
  double t_target=nhc_temp;
  double kt = boltz * t_target;

  // Update masses, to preserve initial freq, if flag set

  if (omega_mass_flag) {
    double nkt = (atom->natoms + 1) * kt;
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }
  }

  if (etap_mass_flag) {
    if (mpchain) {
      etap_mass[0] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * t_target) / etap_mass[ich];
    }
  }

  kecurrent = 0.0;
  pdof = 0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
      pdof++;
    }

  if (pstyle == TRICLINIC) {
    for (i = 3; i < 6; i++)
      if (p_flag[i]) {
        kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
        pdof++;
      }
  }

  if (pstyle == ISO) lkt_press = kt;
  else lkt_press = pdof * kt;
  etap_dotdot[0] = (kecurrent - lkt_press)/etap_mass[0];

  double ncfac = 1.0/nc_pchain;
  for (int iloop = 0; iloop < nc_pchain; iloop++) {

    for (ich = mpchain-1; ich > 0; ich--) {
      expfac = exp(-ncfac*dt8*etap_dot[ich+1]);
      etap_dot[ich] *= expfac;
      etap_dot[ich] += etap_dotdot[ich] * ncfac*dt4;
      etap_dot[ich] *= pdrag_factor;
      etap_dot[ich] *= expfac;
    }

    expfac = exp(-ncfac*dt8*etap_dot[1]);
    etap_dot[0] *= expfac;
    etap_dot[0] += etap_dotdot[0] * ncfac*dt4;
    etap_dot[0] *= pdrag_factor;
    etap_dot[0] *= expfac;

    for (ich = 0; ich < mpchain; ich++)
      etap[ich] += ncfac*dthalf*etap_dot[ich];

    factor_etap = exp(-ncfac*dthalf*etap_dot[0]);
    for (i = 0; i < 3; i++)
      if (p_flag[i]) omega_dot[i] *= factor_etap;

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++)
        if (p_flag[i]) omega_dot[i] *= factor_etap;
    }

    kecurrent = 0.0;
    for (i = 0; i < 3; i++)
      if (p_flag[i]) kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++)
        if (p_flag[i]) kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
    }

    etap_dotdot[0] = (kecurrent - lkt_press)/etap_mass[0];

    etap_dot[0] *= expfac;
    etap_dot[0] += etap_dotdot[0] * ncfac*dt4;
    etap_dot[0] *= expfac;

    for (ich = 1; ich < mpchain; ich++) {
      expfac = exp(-ncfac*dt8*etap_dot[ich+1]);
      etap_dot[ich] *= expfac;
      etap_dotdot[ich] =
        (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] - boltz*t_target) /
        etap_mass[ich];
      etap_dot[ich] += etap_dotdot[ich] * ncfac*dt4;
      etap_dot[ich] *= expfac;
    }
  }
}

/* ----------------------------------------------------------------------
   compute target temperature and kinetic energy
-----------------------------------------------------------------------*/

void FixPIMD::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  double t_target=nhc_temp;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

//  t_target = t_start + delta * (t_stop-t_start);
  t_target = nhc_temp;
  //CM 
  //Now for PIMD thermostat, ke_target = k_B*T rather than d*N*k_B*T
  ke_target = boltz * t_target;
  //ke_target = tdof * boltz * t_target;
}


/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

void FixPIMD::nh_v_temp()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      v[i][0] *= factor_eta;
      v[i][1] *= factor_eta;
      v[i][2] *= factor_eta;
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
   CM
   PIMD thermostat of d*N*M chains
------------------------------------------------------------------------- */

void FixPIMD::nhc_temp_integrate()
{
  double t_target=nhc_temp;
  int ich;
  double expfac;
  double kecurrent;
  //double kecurrent = tdof * boltz * t_current;
  //CM
  int n = atom->nlocal;
  int nmax = 3 * atom->nlocal;  //d*N d.o.f 
  double **v = atom->v;
  int *type = atom->type;  

  //boltz * nhc_temp 

  // Update masses, to preserve initial freq, if flag set
  for(int ip=0; ip<nmax; ip++)
  {
    int iatm = ip/3;
    int idim = ip%3;
    double *vv = v[iatm];    

    kecurrent=mass[type[iatm]] * vv[idim]* vv[idim] * force->mvv2e;

    if (eta_mass_flag) {
      eta_mass[ip][0] = boltz * t_target / (t_freq*t_freq);
      for (int ich = 1; ich < mtchain; ich++)
        eta_mass[ip][ich] = boltz * t_target / (t_freq*t_freq);
    }

    if (eta_mass[ip][0] > 0.0)
      eta_dotdot[ip][0] = (kecurrent - ke_target)/eta_mass[ip][0];
    else eta_dotdot[ip][0] = 0.0;

    //the chains are now coupled to each N particle
    double ncfac = 1.0/nc_tchain;
    for (int iloop = 0; iloop < nc_tchain; iloop++) {

      for (ich = mtchain-1; ich > 0; ich--) {
        expfac = exp(-ncfac*dt8*eta_dot[ip][ich+1]);
        eta_dot[ip][ich] *= expfac;
        eta_dot[ip][ich] += eta_dotdot[ip][ich] * ncfac*dt4;
        eta_dot[ip][ich] *= tdrag_factor;
        eta_dot[ip][ich] *= expfac;
      }

      expfac = exp(-ncfac*dt8*eta_dot[ip][1]);
      eta_dot[ip][0] *= expfac;
      eta_dot[ip][0] += eta_dotdot[ip][0] * ncfac*dt4;
      eta_dot[ip][0] *= tdrag_factor;
      eta_dot[ip][0] *= expfac;

      factor_eta = exp(-ncfac*dthalf*eta_dot[ip][0]);
      //nh_v_temp();
      //CM velocity rescaling.
      vv[idim] *= factor_eta;

      t_current *= factor_eta*factor_eta;
      //CM
      //For PIMD, we couple individual particles and dimension dof. thermostat 
      kecurrent = boltz * t_current;
      //kecurrent = tdof * boltz * t_current;

      if (eta_mass[ip][0] > 0.0)
        eta_dotdot[ip][0] = (kecurrent - ke_target)/eta_mass[ip][0];
      else eta_dotdot[ip][0] = 0.0;

      for (ich = 0; ich < mtchain; ich++)
        eta[ip][ich] += ncfac*dthalf*eta_dot[ip][ich];

      eta_dot[ip][0] *= expfac;
      eta_dot[ip][0] += eta_dotdot[ip][0] * ncfac*dt4;
      eta_dot[ip][0] *= expfac;

      for (ich = 1; ich < mtchain; ich++) {
        expfac = exp(-ncfac*dt8*eta_dot[ip][ich+1]);
        eta_dot[ip][ich] *= expfac;
        eta_dotdot[ip][ich] = (eta_mass[ip][ich-1]*eta_dot[ip][ich-1]*eta_dot[ip][ich-1]
                           - boltz * t_target)/eta_mass[ip][ich];
        eta_dot[ip][ich] += eta_dotdot[ip][ich] * ncfac*dt4;
        eta_dot[ip][ich] *= expfac;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
/*
double FixPIMD::compute_scalar()
{
  int i;
  double volume;
  double energy;
  double t_target=nhc_temp;
  double kt = boltz * t_target;
  double lkt_press = 0.0;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  energy = 0.0;

  // thermostat chain energy is equivalent to Eq. (2) in
  // Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117
  // Sum(0.5*p_eta_k^2/Q_k,k=1,M) + L*k*T*eta_1 + Sum(k*T*eta_k,k=2,M),
  // where L = tdof
  //       M = mtchain
  //       p_eta_k = Q_k*eta_dot[k-1]
  //       Q_1 = L*k*T/t_freq^2
  //       Q_k = k*T/t_freq^2, k > 1

  energy += ke_target * eta[0] + 0.5*eta_mass[0]*eta_dot[0]*eta_dot[0];
  for (ich = 1; ich < mtchain; ich++)
    energy += kt * eta[ich] + 0.5*eta_mass[ich]*eta_dot[ich]*eta_dot[ich];

  // barostat energy is equivalent to Eq. (8) in
  // Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117
  // Sum(0.5*p_omega^2/W + P*V),
  // where N = natoms
  //       p_omega = W*omega_dot
  //       W = N*k*T/p_freq^2
  //       sum is over barostatted dimensions

  if (pstat_flag) {
    for (i = 0; i < 3; i++) {
      if (p_flag[i]) {
        energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i] +
          p_hydro*(volume-vol0) / (pdim*nktv2p);
        lkt_press += kt;
      }
    }

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++) {
        if (p_flag[i]) {
          energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i];
          lkt_press += kt;
        }
      }
    }

    // extra contributions from thermostat chain for barostat

    if (mpchain) {
      energy += lkt_press * etap[0] + 0.5*etap_mass[0]*etap_dot[0]*etap_dot[0];
      for (ich = 1; ich < mpchain; ich++)
        energy += kt * etap[ich] +
          0.5*etap_mass[ich]*etap_dot[ich]*etap_dot[ich];
    }

    // extra contribution from strain energy

    // if (deviatoric_flag) energy += compute_strain_energy();
  }

  if(universe->me==0)
    printf("compute_scalar running...! \n");
  return energy;
}
*/
