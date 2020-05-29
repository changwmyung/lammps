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
   Copyright	Parrinello Group @ ETH Zurich & USI, Lugano, Switzerland
   Authors 	Chang Woo Myung & Barak Hirshberg
   Updated	March, 2020
   Version 	2.0
   Features	* PIMD NPT Parrinello-Rahman barostat(including isotropic & full cell fluctuations) [2].
                * Bosonic Exchange PIMD (pimdb) PNAS (2019) [3].

   Next Feat.	* Fermionic Exchange PIMD (pimdf) (2020) [4].
   		* Perturbed PIMD [5].
      		* PIMD enhanced sampling

   REF
   [1] Martyna, Tuckerman, Tobias & Klein, Molecular Physics 87 1117 (1996).
   [2] Martyna, Hughes, & Tuckerman, J. Chem. Phys. 110 3275 (1999).
   [3] Hirshberg, Rizzi, & Parrinello PNAS 116 21445 (2019).
------------------------------------------------------------------------- */

#include <cmath>
#include "fix_pimd.h"
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "universe.h"
#include "comm.h"
#include "force.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "memory.h"
#include "error.h"

//CM
#include "output.h"
#include <float.h>
#include "group.h"
#include "neighbor.h"
#include "irregular.h"
#include "modify.h"
#include "fix_deform.h"
#include "compute.h"
#include "kspace.h"
#include <stdlib.h>
#include <stdio.h>
//#include "respa.h"

using namespace LAMMPS_NS;
using namespace FixConst;

//CM 
#define DELTAFLIP 0.1
#define TILTMAX 1.5

enum{PIMD,NMPIMD,CMD};

//CM 
//enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};
enum{REDUCE,FULL};
enum{BOLTZMANN,BOSON,FERMION};

//CM diagonalization routine
// Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
// Macros
#define SQR(x)      ((x)*(x))                        // x^2

/* ---------------------------------------------------------------------- */

FixPIMD::FixPIMD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  method     = PIMD;
  method_centroid = FULL;
  method_statistics = BOLTZMANN;
  fmass      = 1.0;
  nhc_temp   = 298.15;
  nhc_nchain = 4;
  sp         = 1.0;
  //CM
  boltz = force->boltz;
  sEl=1.0;

  pe = NULL;

  //CM variables for NPT
  mtchain = mpchain = 3;
  nc_pchain = 1;
  deviatoric_flag = 0;
  pcouple = NONE;
  mtk_flag = 1;
  allremap = 1;
  nrigid = 0;
  drag = 0.0;
  nc_pchain = 1;
  nc_tchain = 1;
  dimension=domain->dimension;
  flipflag = 1;

  omega_mass_flag = 0;
  etap_mass_flag = 0;
  eta_mass_flag = 1;

  p_current_tensor_avg=NULL;
  xc  = NULL;
  fc  = NULL;
  fpre  = NULL;
  eta = NULL;
  eta_dot = NULL;
  eta_dotdot = NULL;
  eta_mass = NULL;

  id_temp = NULL;
  id_press = NULL;
  id_pe = NULL;

  tcomputeflag = 0;
  pcomputeflag = 0;

  // set fixed-point to default = center of cell
  fixedpoint[0] = 0.5*(domain->boxlo[0]+domain->boxhi[0]);
  fixedpoint[1] = 0.5*(domain->boxlo[1]+domain->boxhi[1]);
  fixedpoint[2] = 0.5*(domain->boxlo[2]+domain->boxhi[2]);

  scaleyz = scalexz = scalexy = 0;

  double p_period[6];
  for (int i = 0; i < 6; i++){
    p_start[i] = p_stop[i] = p_period[i] = p_target[i] = 0.0;
    p_flag[i] = 0;
  }

  for(int i=3; i<narg; i+=1)
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
      t_period = atof(arg[i+2]);
      tstat_flag=1;
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

    //CM pressure(parrinello-rahman) 
    else if (strcmp(arg[i],"tri") == 0) {
      pcouple = NONE;
      scalexy = scalexz = scaleyz = 0;
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[i+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[i+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[i+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_start[3] = p_start[4] = p_start[5] = 0.0;
      p_stop[3] = p_stop[4] = p_stop[5] = 0.0;
      p_period[3] = p_period[4] = p_period[5] =
        force->numeric(FLERR,arg[i+3]);
      p_flag[3] = p_flag[4] = p_flag[5] = 1;
      //iarg += 4;
    }

    // centroid approximation for pressure virial & barostat
    else if (strcmp(arg[i],"reduce") == 0) {
      method_centroid=REDUCE;
    }

    else if (strcmp(arg[i],"boson") == 0) {
      method_statistics=BOSON;
    }

    else if (strcmp(arg[i],"fermion") == 0) {
      method_statistics=FERMION;
    }

    else if(strcmp(arg[i],"sEl")==0)
    {
      sEl = atof(arg[i+1]);
      if(sEl<0.0) error->universe_all(FLERR,"Invalid sEl value for fix pimd. The scaling of Elongest should be positive.");
    }

    //else error->universe_all(arg[i],i+1,"Unkown keyword for fix pimd");
  }

  // CM
  // determines the type of barostat iso vs. tri (Parrinello-Rahman)
  // set pstat_flag and box change and restart_pbc variables
  pre_exchange_flag = 0;
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
    else if (pcouple == XYZ || (dimension == 2 && pcouple == XY)) pstyle = ISO;
    else pstyle = ANISO;

    // pre_exchange only required if flips can occur due to shape changes

    if (flipflag && (p_flag[3] || p_flag[4] || p_flag[5]))
      pre_exchange_flag = 1;
    if (flipflag && (domain->yz != 0.0 || domain->xz != 0.0 ||
                     domain->xy != 0.0))
      pre_exchange_flag = 1;
  }

  // convert input periods to frequencies
  //CM 
  //need to make it parameter 
  t_freq =0.01;
  p_freq[0] = p_freq[1] = p_freq[2] = p_freq[3] = p_freq[4] = p_freq[5] = 0.0;

  //CM
  //manual pressure cal. init. 
  for (int i=0; i<6; i++){
    pressure_vector[i]=0.0;
  }

  if (tstat_flag) t_freq = 1.0 / t_period;
  if (p_flag[0]) p_freq[0] = 1.0 / p_period[0];
  if (p_flag[1]) p_freq[1] = 1.0 / p_period[1];
  if (p_flag[2]) p_freq[2] = 1.0 / p_period[2];
  if (p_flag[3]) p_freq[3] = 1.0 / p_period[3];
  if (p_flag[4]) p_freq[4] = 1.0 / p_period[4];
  if (p_flag[5]) p_freq[5] = 1.0 / p_period[5];

//  xc = new double*[180];
//  for(int i = 0; i < 3; i++)
//      xc[i] = new double[3];

  //CM declare 3x3 hg unit-cell
  hg_dot = new double*[3];
  for(int i = 0; i < 3; ++i)
      hg_dot[i] = new double[3];

  //CM declare eig vector 
  eigv = new double*[3];
  for(int i = 0; i < 3; ++i)
      eigv[i] = new double[3];

  omega_dot_eig = new double[3];
  p_current_tensor_avg = new double[3];

  // Nose/Hoover temp and pressure init                                            
  
  size_vector = 0;

  // thermostat variables initialization
  // CM
  //int max = 3 * atom->nlocal;
  int max = 3 * atom->natoms;
  int ich;
  
  //if(universe->me==0) printf("memory start\n");

//  int nlocal0=atom->nlocal;
//  //position copy
//  x_buff = new double*[nlocal0];
//  for(int i = 0; i < 3; ++i)
//      x_buff[i] = new double[3];

  memory->grow(fpre, atom->natoms, 3, "FixPIMD::fpre");
  memory->grow(xc,   atom->natoms, 3, "FixPIMD::xc");
  memory->grow(fc,   atom->natoms, 3, "FixPIMD::fc");

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

  //if(universe->me==0) printf("memory complete\n");

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

  nrigid = 0;
  rfix = NULL;

  if (pre_exchange_flag) irregular = new Irregular(lmp);
  else irregular = NULL;

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

  //from here CM
  n = strlen(id) + 4;
  id_pe = new char[n];
  strcpy(id_pe,id);
  strcat(id_pe,"_pe");

  newarg = new char*[3];
  newarg[0] = id_pe;
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pe";
  modify->add_compute(3,newarg);
  delete [] newarg;

  //id_pe = (char *) "fix_pimd_pe";
  //index_pe = modify->add_compute(id_pe, 1);
  //modify->add_compute(id_pe, 1);

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
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMD::init()
{
  pimdfile = fopen("pimd.log","w");
  initialize_logfile();

  if (atom->map_style == 0)
    error->all(FLERR,"Fix pimd requires an atom map, see atom_modify");

  //if(universe->me==0 && screen) fprintf(screen,"Fix pimd initializing Path-Integral ...\n");

  // prepare the constants
  np = universe->nworlds;
  inverse_np = 1.0 / np;

  //CM 
  //temp array of beads
  t_current_beads = new double[np];
  //pe array of beads
  pe_current_beads = new double[np];
  //vir 
  vir_current_beads = new double[np];
  //eta_E
  eta_E_beads = new double[np];
  //spring energy
  spring_energy_beads = new double[np]; 
  
  //for vector
  t_current_vector = new double[6];
  t_current_vector_avg = new double[6];

  t_current_vector_beads = new double*[np];
  for(int i = 0; i < np; ++i)
    t_current_vector_beads[i] = new double[6];

  const double Plank     = force->hplanck;
  nktv2p = force->nktv2p;

  double hbar   = Plank / ( 2.0 * M_PI );
  double beta   = 1.0 / (boltz * nhc_temp);
  double _fbond = 1.0 * np / (beta*beta*hbar*hbar) ;

  omega_np = sqrt(np) / (hbar * beta) * sqrt(force->mvv2e);
  fbond = - _fbond * force->mvv2e;

  if(universe->me==0) fprintf(pimdfile, "\n********************************************** UNITS **********************************************\n");
  if(universe->me==0){
    fprintf(pimdfile, " * LAMMPS internal units = %s \n", update->unit_style);
    fprintf(pimdfile, " * Boltzmann constant = %20.7lE \n", boltz);
    fprintf(pimdfile, " * -P/(beta^2 * hbar^2) = %20.7lE \n", fbond);
    fprintf(pimdfile, " * Dimension of the system = %d\n", dimension);
  }
  if(universe->me==0) fprintf(pimdfile, "*****************************************************************************************************\n");

//  // CM 
//  // setting the time-step for npt as well
//  if (temperature->tempbias) which = BIAS;
//  else which = NOBIAS;
//
//  if(universe->me==0)
//    printf("BIAS: %d \n", which);

//  which = NOBIAS;

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
  
  vol_current=domain->xprd*domain->yprd*domain->zprd;

  int icompute = modify->find_compute(id_temp);
//  if(universe->me==0)
//    printf("icompute temp is: %d \n", icompute);

  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix nvt/npt does not exist");
  temperature = modify->compute[icompute];

  if (pstat_flag) {
    icompute = modify->find_compute(id_press);
//    if(universe->me==0)
//      printf("icompute press is: %d \n", icompute);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    pressure = modify->compute[icompute];
  }

  //pe for etot
  icompute = modify->find_compute(id_pe);
  pe = modify->compute[icompute];

//  //CM test for NPT
//  if(universe->me==0)
//    printf("p_start/p_stop/pstyle = %f / %f / %d \n", p_start[0], p_stop[0], pstyle);
//    printf("p_flag = %d / %d / %d / %d / %d / %d / %d / %d \n", p_flag[0], p_flag[1], p_flag[2], p_flag[3], p_flag[4], p_flag[5], pdim, pstat_flag);

  kspace_flag = 0;
  //if (force->kspace) kspace_flag = 1;
  //else kspace_flag = 0;

  // MPI initialization between beads
  comm_init();

  mass = new double [atom->ntypes+1];

  if(method==CMD || method==NMPIMD) nmpimd_init();
  else for(int i=1; i<=atom->ntypes; i++) mass[i] = atom->mass[i] / np * fmass;

  if(!nhc_ready) nhc_init();

  //CM total energy measure
  etot=0.0;

  //MPI Comm Group 
  //communication between beads
  int ranks_beads[np]; 
  for(int i=0; i<np; i++)
  {
    ranks_beads[i]=i*comm->nprocs;
  }

//  printf("COMM RANK/SIZE: %d/%d \n",
//    comm->me, comm->nprocs);

  /*
  MPI COMMUNICATORS for fix pimd

  universe: the toal cores 
  universe->uwolrd: MPI communicator for all cores.

  world: MPI communicator between cores within the partition. It reduces nlocals to natoms.

  beads_comm: MPI communicator between beads.
  */

  //CM beads-to-beads MPI Comm
  int color=universe->me%comm->nprocs; 
  MPI_Comm_split(universe->uworld,color,universe->me,&beads_comm);
  MPI_Comm_rank(beads_comm, &beads_rank);
  MPI_Comm_size(beads_comm, &beads_size);

  printf("UNIVERSE RANK/SIZE (total): %d/%d --- WORLD i/n (partitions): %d/%d --- BEADS RANK/SIZE (beads): %d/%d\n",
    universe->me, universe->nprocs, universe->iworld, universe->nworlds, beads_rank, beads_size);

  int world_rank, world_size;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  printf("WORLD RANK/SIZE: %d/%d --- BEADS RANK/SIZE: %d/%d\n",
    world_rank, world_size, beads_rank, beads_size);

  // Get the group of processes in MPI_COMM_WORLD
  MPI_Comm_group(universe->uworld, &world_group);
//  //MPI_Comm_group(MPI_COMM_WORLD, &world_group);
//
//  // Construct a group containing all of the prime ranks for beads communications
//  MPI_Group_incl(world_group, np, ranks_beads, &beads_group);
//
//  // Create a new communicator based on the group
//  MPI_Comm_create_group(universe->uworld, beads_group, 0, &beads_comm);
//  //MPI_Comm_create_group(MPI_COMM_WORLD, beads_group, 0, &beads_comm);
//
//  beads_rank = -1, beads_size = -1;
//  // If this rank isn't in the new communicator, it will be MPI_COMM_NULL
//  // Using MPI_COMM_NULL for MPI_Comm_rank or MPI_Comm_size is erroneous
//  if (MPI_COMM_NULL != beads_comm) {
//    MPI_Comm_rank(beads_comm, &beads_rank);
//    MPI_Comm_size(beads_comm, &beads_size);
//  }
//
//  if(beads_rank!=-1){
//    printf("WORLD RANK/SIZE: %d/%d --- PRIME RANK/SIZE: %d/%d\n",
//      universe->me, universe->nprocs, beads_rank, beads_size);
//  }

  //CM
  //It is important to sure that the particle sorting is disabled. Unless, the simulation is unreliable. 
  if(atom->sortfreq>0&&(method==CMD||method==NMPIMD)){
    if(universe->me==0) fprintf(pimdfile, "\n********************************************* WARNINGS **********************************************\n");
    if(universe->me==0) fprintf(pimdfile, "Particle sort should be disabled. Unless the normal mode transformation becomes non-sense!\n");
    if(universe->me==0) fprintf(pimdfile, "Please insert this line in the LAMMPS input file: atom_modify sort 0 0.0 \n");
    if(universe->me==0) fprintf(pimdfile, "Hope know what you are doing...");
    //error->all(FLERR,"Local particle sort enabled - normal mode transformation becomes nonsense.");
    if(universe->me==0) fprintf(pimdfile, "\n*****************************************************************************************************\n");
  }

  if(universe->me==0) fprintf(pimdfile, "\n***************************************** Nuclei STATISTICS *****************************************\n");
  if(universe->me==0) fprintf(pimdfile, "1. BOLTZMANN (distingiushable) - default 2. BOSON (indistinguishable) 3. FERMION (indistinguishable) - to be implemented\n");
  if(method_statistics==BOLTZMANN)
    if(universe->me==0) fprintf(pimdfile, " - Nuclei follow 1. distinguishable BOLTZMANN statistics (default).\n");
  if(method_statistics==BOSON)
    if(universe->me==0) fprintf(pimdfile, " - Nuclei follow 2. indistinguishable BOSON statistics (check if your nuclei are boson!).\n");
  if(universe->me==0) fprintf(pimdfile, "*****************************************************************************************************\n");

  if (pstat_flag && mpchain){
    if(universe->me==0) fprintf(pimdfile, "\nStep       Temp (K)       Volume       Pressure       E_consv. (K)      E_tot (K/ptcl)    PE (K/ptcl)   KE(K/ptcl)\n");}
  else{
    if(universe->me==0) fprintf(pimdfile, "\nStep       Temp (K)       E_tot (K/ptcl)    PE (K/ptcl)   KE(K/ptcl)   Pc_long\n");}

  Pc_longest=0.0;
  if(method_statistics==BOSON){
    E_kn=std::vector<double>((atom->natoms * (atom->natoms + 1) / 2),0.0);
    V=std::vector<double>((atom->natoms + 1),0.0);
    dV=std::vector<std::vector<double>>(atom->natoms*universe->nworlds, std::vector<double>(3, 0.0));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMD::setup(int vflag)
{

  //if(universe->me==0 && screen) fprintf(screen,"Setting up Path-Integral ...\n");

  //CM
  //force is updated first 
  post_force(vflag);  //previous post_force function

  //post_force(vflag);  //previous post_force function
  //remove_spring_force();

  //if(universe->me==0 && screen) fprintf(screen,"1. Setting up Path-Integral ...\n");

/* CM ----------------------------------------------------------------------
  Compute T,P before integrator starts

  - It's important that the spring force terms is excluded from the pressure calculations.
------------------------------------------------------------------------- */


  //t_current = temperature->compute_scalar();
  t_current = compute_temp_scalar();
  tdof = temperature->dof;
  if(universe->me==0) printf("tdof: %f\n", tdof);

  if (pstat_flag) compute_press_target();

//  comm_exec(atom->x);
//  remove_spring_force();
    if (pstat_flag) {
      if (pstyle == ISO){ 
        //pressure->compute_scalar();
        compute_pressure_scalar();
        //if(universe->me==0) printf("pressure scalar computed. \n");
        if (pcouple == XYZ){
          //pressure->compute_vector();
          compute_pressure_vector();
          //if(universe->me==0) printf("pressure vector computed. \n");
          }
        }
      if (pstyle == TRICLINIC){ 
        //pressure->compute_vector();
        compute_pressure_vector();
      }
      couple();
      pressure->addstep(update->ntimestep+1);
    }

//  else{
//    if (pstat_flag) {
//      if (pstyle == ISO){ 
//        pressure->compute_scalar();
//          if (pcouple == XYZ) pressure->compute_vector();
//        }
//      if (pstyle == TRICLINIC) pressure->compute_vector();
//      couple();
//      pressure->addstep(update->ntimestep+1);
//    }
//  }

//  comm_exec(atom->x);
//  spring_force();

// CM
/* ---------------------------------------------------------------------- */

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
    double npkt = (atom->natoms*np + 1) * kt;

    for (int i = 0; i < 3; i++)
      if (p_flag[i]){
        if(method_centroid==REDUCE)
          omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
        else
          omega_mass[i] = npkt/(p_freq[i]*p_freq[i]);
      }

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]){
          if(method_centroid==REDUCE)
             omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
          else
             omega_mass[i] = npkt/(p_freq[i]*p_freq[i]);
       }
    }

  // masses and initial forces on barostat thermostat variables

    if (mpchain) {
      etap_mass[0] = boltz * nhc_temp / (p_freq_max*p_freq_max);
      // See Appendix D, [2].
      if (pstyle == TRICLINIC) {
        etap_mass[0] = dimension*(dimension+1)/2 * boltz * nhc_temp / (p_freq_max*p_freq_max);
      }
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * nhc_temp / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * nhc_temp) / etap_mass[ich];
    }
  }


/* ---------------------------------------------------------------------- */

  //CM 
  //spring_force();

  if(universe->me==0 && screen) fprintf(screen,"Finished setting up Path-Integral ...\n");

  if(universe->me==0) printf("Setup finished! \n");

}

/* ---------------------------------------------------------------------- */

void FixPIMD::initial_integrate(int /*vflag*/)
{

  //if(universe->me==0) printf("init integ. finished! \n");

  //CM update the centroid xc & fc
  //uncomment later !
  update_x_centroid();

  //CM debug
  //observe_temp_scalar();

//  if (pstat_flag && mpchain)
//  if(update->ntimestep%output->thermo_every==0){
//    monitor_observable();
//  }

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
//    remove_spring_force();
//    multiply_post_force();
////////

//    comm_exec(atom->x);
//    remove_spring_force();

//    if(universe->me==0){
      if (pstat_flag) {
        if (pstyle == ISO) {
          t_current = compute_temp_scalar();
          temperature->compute_scalar();
          //pressure->compute_scalar();
          compute_pressure_scalar();
          if (pcouple == XYZ){
            //pressure->compute_vector();
            compute_pressure_vector();
          }
        } else {
          t_current = compute_temp_scalar();
          compute_temp_vector();
          temperature->compute_vector();
          //pressure->compute_vector();
          compute_pressure_vector();
        }
        couple();
        pressure->addstep(update->ntimestep+1);
      }
//    }

//    comm_exec(atom->x);
//    spring_force();

//  For test purpose,
//    spring_force();
//    divide_post_force();
////////
  
    if (pstat_flag) {
      compute_press_target();
      //CM
      //this should only be done in proc0 and then broadcast
      if(universe->me==0) nh_omega_dot();
      //nh_omega_dot();

      //broadcast
      //CM do we need this ?could be a redundant communication. 
      //CM change it to Allreduce 
      MPI_Barrier(universe->uworld);
      MPI_Bcast(omega_dot, 6, MPI_DOUBLE, 0, universe->uworld);
      if (pstyle == TRICLINIC) {
        MPI_Bcast(omega_dot_eig, 3, MPI_DOUBLE, 0, universe->uworld);
        for(int i=0; i<3; i++){
          MPI_Bcast(eigv[i], 3, MPI_DOUBLE, 0, universe->uworld);}
      }
      MPI_Barrier(universe->uworld);

      //centroid approx.
      if(method_centroid==REDUCE){
        //if(universe->me==0) nh_omega_dot_x();}
        if(universe->iworld==0) nh_omega_dot_x();}
      else if(method_centroid==FULL){
        nh_omega_dot_x();}

      //CM
      //reduced centroid eom.
      //if(universe->me==0) nh_v_press();
      if(method_centroid==REDUCE){
        //if(universe->me==0) nh_v_press();
        if(universe->iworld==0) nh_v_press();
      }
      else if(method_centroid==FULL){
        nh_v_press();
      }
      
    }

  
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

  //NVT
  }else{
  t_current = compute_temp_scalar();
  nhc_update_v();
  nhc_update_x();
  }

  //if(universe->me==0) printf("init integ. finished! \n");
}

/* ---------------------------------------------------------------------- */

void FixPIMD::final_integrate()
{
  if (pstat_flag && mpchain){
    nve_v();

    //CM
    //scaling is only for centroid
    //if(universe->me==0) nh_v_press();
    //nh_v_press();

//    // re-compute temp before nh_v_press()
//    // only needed for temperature computes with BIAS on reneighboring steps:
//    //   b/c some biases store per-atom values (e.g. temp/profile)
//    //   per-atom values are invalid if reneigh/comm occurred
//    //     since temp->compute() in initial_integrate()
//  
//    if (which == BIAS && neighbor->ago == 0)
//      t_current = temperature->compute_scalar();

    if(method_centroid==REDUCE){
      //if(universe->me==0) nh_v_press();
      if(universe->iworld==0) nh_v_press();
    }
    else if(method_centroid==FULL){
      nh_v_press();
    }

    // compute new T,P after velocities rescaled by nh_v_press()
    // compute appropriately coupled elements of mvv_current

    //t_current = temperature->compute_scalar();
    t_current = compute_temp_scalar();
    tdof = temperature->dof;

    // need to recompute pressure to account for change in KE
    // t_current is up-to-date, but compute_temperature is not
    // compute appropriately coupled elements of mvv_current

//  For test purpose,
//    remove_spring_force();
//    multiply_post_force();
////////

//    comm_exec(atom->x);
//    remove_spring_force();
//    if(universe->me==0){
      if (pstat_flag) {
        if (pstyle == ISO){
          t_current = compute_temp_scalar();
          temperature->compute_scalar();
          //pressure->compute_scalar();
          compute_pressure_scalar();
          if (pcouple == XYZ){
            //pressure->compute_vector();
            compute_pressure_vector();
          }
        }
        else {
          t_current = compute_temp_scalar();
          compute_temp_vector();
          temperature->compute_vector();
          //pressure->compute_vector();
          compute_pressure_vector();
        }
        couple();
        pressure->addstep(update->ntimestep+1);
      }
//    }


//    comm_exec(atom->x);
//    spring_force();

//  For test purpose,
//    spring_force();
//    divide_post_force();
////////

    //CM
    //this should only be done in proc0 and then broadcast
    if(universe->me==0) nh_omega_dot();
    //nh_omega_dot();
    //broadcast
    MPI_Barrier(universe->uworld);
    MPI_Bcast(omega_dot, 6, MPI_DOUBLE, 0, universe->uworld);
    if (pstyle == TRICLINIC) {
      MPI_Bcast(omega_dot_eig, 3, MPI_DOUBLE, 0, universe->uworld);
      for(int i=0; i<3; i++){
        MPI_Bcast(eigv[i], 3, MPI_DOUBLE, 0, universe->uworld);}
    }
    MPI_Barrier(universe->uworld);

    if(method_centroid==REDUCE){
      //reduced 
      //if(universe->me==0) nh_omega_dot_x();}
      if(universe->iworld==0) nh_omega_dot_x();}
    else if(method_centroid==FULL){
      nh_omega_dot_x();}


    // update eta_dot
    // update eta_press_dot
    nhc_temp_integrate();
    nhc_press_integrate();

  //NVT
  } else{
  t_current = compute_temp_scalar();
  nhc_update_v();
  }

  //if(universe->me==0) printf("fin integ. finished! \n");
}

/* ---------------------------------------------------------------------- */
// CM force calculations
void FixPIMD::post_force(int /*flag*/)
{
  //CM
  //if(universe->me==0){
  //  fprintf(pimdfile, "post_force!");} 

  //if(universe->me==0) printf("nlocal:%d \n", atom->nlocal);
  //CM store bare forces before transformations
  for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) fpre[i][j]=atom->f[i][j];
  for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) atom->f[i][j] /= np;

  //if(universe->me==0) printf("f centroid! \n");
  //CM store the centroid force
  //need to uncomment later!
  update_f_centroid();

  //if(universe->me==0) printf("nm-transform start! \n");
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
  //if(universe->me==0) printf("nm-transform finished! \n");
}

//CM
//For pressure virial calculations
void FixPIMD::multiply_post_force()
{
  for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) atom->f[i][j] *= np;
}

void FixPIMD::divide_post_force()
{
  for(int i=0; i<atom->nlocal; i++) for(int j=0; j<3; j++) atom->f[i][j] /= np;
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
//CM
//need to understand this properly
void FixPIMD::nmpimd_init()
{
  memory->create(M_x2xp, np, np, "fix_feynman:M_x2xp");
  memory->create(M_xp2x, np, np, "fix_feynman:M_xp2x"); //used in nve_x
  memory->create(M_f2fp, np, np, "fix_feynman:M_f2fp"); //used in force
  memory->create(M_fp2f, np, np, "fix_feynman:M_fp2f");

  lam = (double*) memory->smalloc(sizeof(double)*np, "FixPIMD::lam");

  // CM
  // In this setting, proc0 is the centroid mode.
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
      M_xp2x[i][j] = M_x2xp[j][i]*np;
      M_f2fp[i][j] = M_x2xp[i][j]*np;
      // CM found a bug
      //M_fp2f[i][j] = M_xp2x[i][j];
      M_fp2f[i][j] = M_xp2x[i][j]/np;
    }

  // Set up masses

  int iworld = universe->iworld;

  //normal mode mass 
  for(int i=1; i<=atom->ntypes; i++)
  {
    mass[i] = atom->mass[i];

    if(iworld)
    {
      mass[i] *= lam[iworld];
      mass[i] *= fmass;
    }
  //CM print the nm mass 
  //  if (logfile){
  //    fprintf(logfile,"Normal mode mass for %d mode = "
  //                    "%.4f \n", iworld, mass[i]);}
  }

  //CM 
  //normal mode velocity rescaling 
  if(iworld)
  {
    for(int i=0;i<atom->nlocal;i++){
      for(int j=0;j<3;j++){
      atom->v[i][j]/=sqrt(lam[iworld]);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
//CM
//what is this?
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
  if(method_statistics==BOLTZMANN){
    int nlocal = atom->nlocal;
    double spring_E=0.0;
    spring_energy = 0.0;
  
    double **x = atom->x;
    double **f = atom->f;
    double* _mass = atom->mass;
    int* type = atom->type;
  
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
  
      spring_E += fbond * (dx*dx+dy*dy+dz*dz);
      //spring_energy += (dx*dx+dy*dy+dz*dz);
    }
    MPI_Allreduce(&spring_E,&spring_energy,1,MPI_DOUBLE,MPI_SUM,world);
  }
  else if(method_statistics==BOSON){ 
    V.at(0) = 0.0;
    //energy
    //E_kn = Evaluate_VBn(V, atom->natoms);

    //Prob. of the longest polymer  
//    observe_Pc_longest();

    //force
    //dV = Evaluate_dVBn(V,E_kn,atom->natoms);

    //ke
    //for(int i=0; i<atom->natoms*(atom->natoms+1)/2; i++){ 
    //  if (universe->me ==0){ 
    //    printf("E_kn: %e \n", E_kn.at(i));}
    //}
//    ke_boson_vir=Evaluate_ke_boson(V, E_kn);

    //parallel ver.
    E_kn=Evaluate_VBn_new(V, atom->natoms);
    dV=Evaluate_dVBn_new(V,E_kn,atom->natoms);

  }
  else if(method_statistics==FERMION){

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

//CM
//      if(universe->me==0)
//        printf("comm->nprocs / plan_send / plan_recv / mode_index: %d / %d / %d / %d \n", comm->nprocs, plan_send[i], plan_recv[i], mode_index[i]);
//        printf("universe->iworld / universe->nworlds: %d / %d \n", universe->iworld, universe->nworlds);
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

//CM this issue should be solved
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
  //CM
  double *tensor = pressure->vector;
  
  //CM
  //compute_pressure_scalar();
  //remove_spring_force();
  //compute_pressure_scalar();
  //compute_pressure_vector();
  //double *tensor=pressure_vector;
  //spring_force();

  if (pstyle == ISO){
    //CM
    //p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
    p_current[0] = p_current[1] = p_current[2] = p_current_avg;
//    if(universe->me==0)
//      printf("couple(iso) pressure: %f \n", p_current[0]);
  }
  if (pcouple == XYZ) {
    //double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
    double ave = 1.0/3.0 * (p_current_tensor_avg[0] + p_current_tensor_avg[1] + p_current_tensor_avg[2]);
    //double ave = (p_current_tensor_avg[0] + p_current_tensor_avg[1] + p_current_tensor_avg[2]);
    p_current[0] = p_current[1] = p_current[2] = ave;
    //p_current[0] = p_current[1] = p_current[2] = pressure_scalar;

//  if(universe->me==0)
//    printf("couple(xyz) pressure: %f \n", p_current[0]);

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
//    double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
//    p_current[0] = p_current[1] = p_current[2] = ave;
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
  }

  if (!std::isfinite(p_current[0]) || !std::isfinite(p_current[1]) || !std::isfinite(p_current[2]))
    error->all(FLERR,"Non-numeric pressure - simulation unstable");

  // switch order from xy-xz-yz to Voigt

  if (pstyle == TRICLINIC) {
//    double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
//    p_current[0] = ave;
//    p_current[1] = ave;
//    p_current[2] = ave;
//    p_current[3] = 0.0;
//    p_current[4] = 0.0;
//    p_current[5] = 0.0;
//    p_current[0] = tensor[0];
//    p_current[1] = tensor[1];
//    p_current[2] = tensor[2];
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
//  double **x = atom->x;
  int nlocal = atom->nlocal;

  double h[6];
  //CM eigen transformation
  double **hh, **hh_eigv, *h_eig;

  hh = new double*[3];
  for(int i = 0; i < 3; ++i)
      hh[i] = new double[3];

  hh_eigv = new double*[3];
  for(int i = 0; i < 3; ++i)
      hh_eigv[i] = new double[3];

  h_eig = new double[3];

  if (pstyle == TRICLINIC) {
    h[0]=domain->h[0];
    h[1]=domain->h[1];
    h[2]=domain->h[2];
    h[3]=domain->h[3];
    h[4]=domain->h[4];
    h[5]=domain->h[5];

    hh[0][0]=h[0]; 
    hh[1][1]=h[1]; 
    hh[2][2]=h[2]; 
    hh[1][2]=hh[2][1]=h[3]; 
    hh[0][2]=hh[2][0]=h[4]; 
    hh[0][1]=hh[1][0]=h[5]; 

    //CM diagonalize the unit-cell
    dsyevv3(hh, hh_eigv, h_eig);
  }

  //CM
  // calculate & broadcast volume of translation mode
  //But this is only for isotropic npt!
  if (pstyle == ISO) {
    if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
    else volume = domain->xprd*domain->yprd;
  }
  else if (pstyle == TRICLINIC) {
    volume = h_eig[0]*h_eig[1]*h_eig[2];
    //CM print the nm mass
    //if(universe->me==0){
    //  if (pimdfile){
    //    fprintf(pimdfile, "h, vol(tri) = (%f, %f, %f), %.4f \n", h_eig[0], h_eig[1], h_eig[2], volume);
    //  }
    //}
  }

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

//  if(universe->me==0)
//    printf("volume 0/pressure: %f, %f, %f, %f, %f \n", volume, domain->xprd, domain->yprd, domain->zprd, p_current[0]);

  //if(universe->me==6)
  //  printf("volume 6: %f \n", volume);

  if (deviatoric_flag) compute_deviatoric();

  //CM
  //p**2/m term
  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) {
      if (method_centroid==REDUCE){
        mtk_term1 = tdof * boltz * t_current;
        mtk_term1 /= pdim * atom->natoms;
      }
      else if (method_centroid==FULL){
        mtk_term1 = tdof * boltz * t_current_avg;
        mtk_term1 /= pdim * atom->natoms;
      }
    } else {
      compute_temp_vector();
      //CM is it right?
      double *mvv_current = t_current_vector;
      //double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i])
          mtk_term1 += mvv_current[i];
      mtk_term1 /= pdim * atom->natoms;
    }
  }
 
//  if (method_centroid == FULL){
//    if(universe->me==0){
//      printf("FULL\n");}
//  }
//  else if (method_centroid == REDUCE){
//    if(universe->me==0){
//      printf("REDUCE\n");}
//  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
        (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
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

  vol_current=volume;

  //delete
  for(int i = 0; i < 3; ++i){
    delete[] hh[i];
  }
  delete[] hh;

  for(int i = 0; i < 3; ++i){
    delete[] hh_eigv[i];
  }
  delete[] hh_eigv;

  delete[] h_eig;

}

void FixPIMD::nh_omega_dot_x()
{
  double f_omega,volume;
  double **x = atom->x;
  //CM position in eigenspace
  double **xx;
  int nlocal = atom->nlocal;

  double h[6];
  //CM eigen transformation
  double **hh, **hh_eigv, *h_eig;

  hh = new double*[3];
  for(int i = 0; i < 3; ++i)
      hh[i] = new double[3];

  hh_eigv = new double*[3];
  for(int i = 0; i < 3; ++i)
      hh_eigv[i] = new double[3];

  h_eig = new double[3];

  //CM 
  xx = new double*[nlocal];
  for(int i = 0; i < nlocal; ++i)
    xx[i] = new double[3];

/* position update */

  //CM the position scaling update 
  //eq(3.5.1) in [2]
  //for isotropic NPT
  if (pstyle == ISO) {
    for (int i = 0; i < 3; i++){
      posexp[i]=exp(dthalf*omega_dot[i]);
      for (int ip = 0; ip < nlocal; ip++) {
        x[ip][i] *= posexp[i];
      }
    }
  }

//  if(universe->me==0)
//    printf("eig: %f / %f / %f \n", eig[0], eig[1], eig[2]);

  //for full-cell fluctuations
  //CM is this right? The matrix is one of the whole block by mpi parallelization. 
  else if (pstyle == TRICLINIC) {
    hg_dot[0][0]=omega_dot[0]; 
    hg_dot[1][1]=omega_dot[1]; 
    hg_dot[2][2]=omega_dot[2]; 
    hg_dot[1][2]=hg_dot[2][1]=omega_dot[3]; 
    hg_dot[0][2]=hg_dot[2][0]=omega_dot[4]; 
    hg_dot[0][1]=hg_dot[1][0]=omega_dot[5]; 

    //CM diagonalize the unit-cell momentum
    //dsyevc3(hg, eig);
    dsyevv3(hg_dot, eigv, omega_dot_eig);

//    if(universe->me==0){
//      printf("eig: %f / %f / %f \n", omega_dot_eig[0], omega_dot_eig[1], omega_dot_eig[2]);
//      printf("eigv: %f / %f / %f \n", eigv[0][0], eigv[0][1], eigv[0][2]);
//      printf("eigv: %f / %f / %f \n", eigv[1][0], eigv[1][1], eigv[1][2]);
//      printf("eigv: %f / %f / %f \n", eigv[2][0], eigv[2][1], eigv[2][2]);
//    }

//    if(universe->me==0){
//      printf("x: %f / %f / %f \n", x[0][0], x[0][1], x[0][2]);
//    }
    //CM position transformation by eigv
    //need to use xx so as not to currupt x!!
    //EIGV * X 

    for (int ip = 0; ip < nlocal; ip++) {
      xx[ip][0] = eigv[0][0]*x[ip][0]+eigv[1][0]*x[ip][1]+eigv[2][0]*x[ip][2];
      xx[ip][1] = eigv[0][1]*x[ip][0]+eigv[1][1]*x[ip][1]+eigv[2][1]*x[ip][2];
      xx[ip][2] = eigv[0][2]*x[ip][0]+eigv[1][2]*x[ip][1]+eigv[2][2]*x[ip][2]; 
    }

    for (int i = 0; i < 3; i++){
      posexp[i]=exp(dthalf*omega_dot_eig[i]);
      for (int ip = 0; ip < nlocal; ip++) {
        xx[ip][i] *= posexp[i];
      }
    }

    //CM position transformation by eigv
    //EIGV^T * X 
    for (int ip = 0; ip < nlocal; ip++) {
      x[ip][0] = eigv[0][0]*xx[ip][0]+eigv[0][1]*xx[ip][1]+eigv[0][2]*xx[ip][2];
      x[ip][1] = eigv[1][0]*xx[ip][0]+eigv[1][1]*xx[ip][1]+eigv[1][2]*xx[ip][2];
      x[ip][2] = eigv[2][0]*xx[ip][0]+eigv[2][1]*xx[ip][1]+eigv[2][2]*xx[ip][2]; 
    }
  }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++){
      if (p_flag[i]){
        if (pstyle == ISO) {
          mtk_term2 += omega_dot[i];
        }
        else if (pstyle == TRICLINIC) {
          mtk_term2 += omega_dot_eig[i];
        }
      }
    }
    if (method_centroid==REDUCE){
      if (pdim > 0) mtk_term2 /= pdim * atom->natoms;}
    else if (method_centroid==FULL){
      if (pdim > 0) mtk_term2 /= pdim * atom->natoms * np;}
  }

  //delete
  for(int i = 0; i < 3; ++i){
    delete[] hh[i];
  }
  delete[] hh;

  for(int i = 0; i < 3; ++i){
    delete[] hh_eigv[i];
  }
  delete[] hh_eigv;

  delete[] h_eig;

  for(int i = 0; i < nlocal; ++i){
    delete[] xx[i];
  }
  delete[] xx;

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
//this scaling is only for centroid mode for reduced scheme, Eq 3.7.2 [2].
void FixPIMD::nh_v_press()
{
  double factor[3];
  double **v = atom->v;
  //CM eigen transformation of v
  double **vv;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //CM 
  vv = new double*[nlocal];
  for(int i = 0; i < nlocal; ++i)
    vv[i] = new double[3];

  if (pstyle == ISO) {
    //CM
    factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2)); // exp[-dt4*(omega_dot+1/N*omega_dot)/W]
    factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
    factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));
  
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          v[i][0] *= factor[0];
          v[i][1] *= factor[1];
          v[i][2] *= factor[2];
    //      if (pstyle == TRICLINIC) {
    //        v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
    //        v[i][1] += -dthalf*v[i][2]*omega_dot[3];
    //      }
          v[i][0] *= factor[0];
          v[i][1] *= factor[1];
          v[i][2] *= factor[2];
        }
      }

//      if(universe->me==0)
//        printf("bias v: %f %f %f \n", vbias[0]); 
  }

  else if (pstyle == TRICLINIC) {
    //CM
    factor[0] = exp(-dt4*(mtk_term2));
    factor[1] = exp(-dt4*(mtk_term2));
    factor[2] = exp(-dt4*(mtk_term2));
  
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          v[i][0] *= factor[0];
          v[i][1] *= factor[1];
          v[i][2] *= factor[2];

          v[i][0] *= factor[0];
          v[i][1] *= factor[1];
          v[i][2] *= factor[2];
        }
      }

    //CM velocity transformation by eigv
    //EIGV * V
    for (int ip = 0; ip < nlocal; ip++) {
      vv[ip][0] = eigv[0][0]*v[ip][0]+eigv[1][0]*v[ip][1]+eigv[2][0]*v[ip][2];
      vv[ip][1] = eigv[0][1]*v[ip][0]+eigv[1][1]*v[ip][1]+eigv[2][1]*v[ip][2];
      vv[ip][2] = eigv[0][2]*v[ip][0]+eigv[1][2]*v[ip][1]+eigv[2][2]*v[ip][2]; 
    }

    for (int i = 0; i < 3; i++){
      velexp[i]=exp(-dt4*omega_dot_eig[i]);
      //velexp[i]=1.0;
      for (int ip = 0; ip < nlocal; ip++) {
        vv[ip][i] *= velexp[i];
        vv[ip][i] *= velexp[i];
      }
    }

    //CM velocity transformation by eigv
    //EIGV^T * V 
    for (int ip = 0; ip < nlocal; ip++) {
      v[ip][0] = eigv[0][0]*vv[ip][0]+eigv[0][1]*vv[ip][1]+eigv[0][2]*vv[ip][2];
      v[ip][1] = eigv[1][0]*vv[ip][0]+eigv[1][1]*vv[ip][1]+eigv[1][2]*vv[ip][2];
      v[ip][2] = eigv[2][0]*vv[ip][0]+eigv[2][1]*vv[ip][1]+eigv[2][2]*vv[ip][2]; 
    }

  }

  for(int i = 0; i < nlocal; ++i){
    delete[] vv[i];
  }
  delete[] vv;

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
//  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //CM print the nm mass
  //if (pimdfile){
  //  fprintf(pimdfile,"Normal mode mass for %d mode = "
  //                  "%.4f \n", universe->iworld, mass[type[0]]);}

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
// CM this causes a trouble, but why? probably memory problem. or public private.  
//  double *h = domain->h;
  double h[6];
  //CM eigen transformation
  double **hh, **h2;

  h[0]=domain->h[0];
  h[1]=domain->h[1];
  h[2]=domain->h[2];
  h[3]=domain->h[3];
  h[4]=domain->h[4];
  h[5]=domain->h[5];

//  if(universe->me==0){
//    printf("h: %f / %f / %f \n", h[0], h[1], h[2]);
//    printf("h: %f / %f / %f \n", h[3], h[4], h[5]);
//  }
//
//  if(universe->me==0){
//    printf("eig: %f / %f / %f \n", omega_dot_eig[0], omega_dot_eig[1], omega_dot_eig[2]);
//    printf("eigv: %f / %f / %f \n", eigv[0][0], eigv[0][1], eigv[0][2]);
//    printf("eigv: %f / %f / %f \n", eigv[1][0], eigv[1][1], eigv[1][2]);
//    printf("eigv: %f / %f / %f \n", eigv[2][0], eigv[2][1], eigv[2][2]);
//  }

  hh = new double*[3];
  for(int i = 0; i < 3; ++i)
      hh[i] = new double[3];

  h2 = new double*[3];
  for(int i = 0; i < 3; ++i)
      h2[i] = new double[3];

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords

  if (allremap) domain->x2lamda(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->x2lamda(x[i],x[i]);
  }

  if (nrigid)
    for (i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(0);

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

//  if (pstyle == TRICLINIC) {
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      htmp[4] *= expfac;
//      htmp[4] += dto4*(omega_dot[5]*htmp[3]+omega_dot[4]*htmp[2]);
//      htmp[4] *= expfac;
//    }
//
//    if (p_flag[3]) {
//      expfac = exp(dto4*omega_dot[1]);
//      htmp[3] *= expfac;
//      htmp[3] += dto2*(omega_dot[3]*htmp[2]);
//      htmp[3] *= expfac;
//    }
//
//    if (p_flag[5]) {
//      expfac = exp(dto4*omega_dot[0]);
//      htmp[5] *= expfac;
//      htmp[5] += dto2*(omega_dot[5]*htmp[1]);
//      htmp[5] *= expfac;
//    }
//
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      htmp[4] *= expfac;
//      htmp[4] += dto4*(omega_dot[5]*htmp[3]+omega_dot[4]*htmp[2]);
//      htmp[4] *= expfac;
//    }
//  }

//  if (pstyle == TRICLINIC) {
//
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      h[4] *= expfac;
//      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
//      h[4] *= expfac;
//    }
//
//    if (p_flag[3]) {
//      expfac = exp(dto4*omega_dot[1]);
//      h[3] *= expfac;
//      h[3] += dto2*(omega_dot[3]*h[2]);
//      h[3] *= expfac;
//    }
//
//    if (p_flag[5]) {
//      expfac = exp(dto4*omega_dot[0]);
//      h[5] *= expfac;
//      h[5] += dto2*(omega_dot[5]*h[1]);
//      h[5] *= expfac;
//    }
//
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      h[4] *= expfac;
//      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
//      h[4] *= expfac;
//    }
//  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (pstyle == ISO) {

//    if(universe->me==0)
//      printf("omega_dot: %f / %f / %f \n", omega_dot[0], omega_dot[1], omega_dot[2]);

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
      //if (scalexy) h[5] *= expfac;
    }
  
    if (p_flag[2]) {
      oldlo = domain->boxlo[2];
      oldhi = domain->boxhi[2];
      expfac = exp(dto*omega_dot[2]);
      domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
      domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
      //if (scalexz) h[4] *= expfac;
      //if (scaleyz) h[3] *= expfac;
    }
  }

  if (pstyle == TRICLINIC) {

    //CM diagonalize the unit-cell momentum
    //dsyevv3(hg_dot, eigv, omega_dot_eig);

    //CM unit-cell in 2D array
    h2[0][0]=h[0]; 
    h2[1][1]=h[1]; 
    h2[2][2]=h[2]; 
    h2[1][2]=h2[2][1]=h[3]; 
    h2[0][2]=h2[2][0]=h[4]; 
    h2[0][1]=h2[1][0]=h[5]; 

    //transformation of unit-cell h
    for(int ii=0; ii<3; ii++){
      for(int jj=0; jj<3; jj++){
        hh[ii][jj]=0;
        for(int kk=0; kk<3; kk++){
          hh[ii][jj]+=eigv[ii][kk]*h2[kk][jj];
        }
      }
    }

    //scaling
    for (int i = 0; i < 3; i++){
      cellexp[i]=exp(dto*omega_dot_eig[i]);
      for (int j = 0; j < 3; j++) {
        hh[i][j] *= cellexp[i];
      }
    }

    //re-transformation of unit-cell h
    for(int ii=0; ii<3; ii++){
      for(int jj=0; jj<3; jj++){
        h2[ii][jj]=0;
        for(int kk=0; kk<3; kk++){
          h2[ii][jj]+=eigv[kk][ii]*hh[kk][jj];
        }
      }
    }

    //CM update the unit-cell info.
    h[0]=h2[0][0]; 
    h[1]=h2[1][1]; 
    h[2]=h2[2][2]; 
    h[3]=h2[1][2]; 
    h[4]=h2[0][2]; 
    h[5]=h2[0][1]; 
  
//    if(universe->me==0){
//      printf("h: %f / %f / %f \n", h[0], h[1], h[2]);
//      printf("h: %f / %f / %f \n", h[3], h[4], h[5]);
//    }

    if (p_flag[0]) {
      oldlo = domain->boxlo[0];
      oldhi = domain->boxhi[0];
      expfac = h[0]/(oldhi-oldlo);
      //expfac = exp(dto*omega_dot[0]);
      domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
      domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
    }
  
    if (p_flag[1]) {
      oldlo = domain->boxlo[1];
      oldhi = domain->boxhi[1];
      expfac = h[1]/(oldhi-oldlo);
      //expfac = exp(dto*omega_dot[1]);
      domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
      domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    }
  
    if (p_flag[2]) {
      oldlo = domain->boxlo[2];
      oldhi = domain->boxhi[2];
      expfac = h[2]/(oldhi-oldlo);
      //expfac = exp(dto*omega_dot[0]);
      domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
      domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    }
  }

  // off-diagonal components, second half

//  if (pstyle == TRICLINIC) {
//
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      h[4] *= expfac;
//      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
//      h[4] *= expfac;
//    }
//
//    if (p_flag[3]) {
//      expfac = exp(dto4*omega_dot[1]);
//      h[3] *= expfac;
//      h[3] += dto2*(omega_dot[3]*h[2]);
//      h[3] *= expfac;
//    }
//
//    if (p_flag[5]) {
//      expfac = exp(dto4*omega_dot[0]);
//      h[5] *= expfac;
//      h[5] += dto2*(omega_dot[5]*h[1]);
//      h[5] *= expfac;
//    }
//
//    if (p_flag[4]) {
//      expfac = exp(dto8*omega_dot[0]);
//      h[4] *= expfac;
//      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
//      h[4] *= expfac;
//    }
//
//  }

  domain->yz = h[3];//h[3];
  domain->xz = h[4];//h[4];
  domain->xy = h[5];//h[5];


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

  if (nrigid)
    for (i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(1);

  //delete
  for(int i = 0; i < 3; ++i){
    delete[] hh[i];
  }
  delete[] hh;

  for(int i = 0; i < 3; ++i){
    delete[] h2[i];
  }
  delete[] h2;


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

  //for (int i = 0; i < 3; i++){
  //  posexp[i]=exp(dthalf*omega_dot[i]/omega_mass[i]);
  //  for (int ip = 0; ip < nlocal; ip++) {
  //    x[ip][i] *= posexp[i];
  //  }
  //}

  for (int j=0; j<3; j++){
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][j] += dtv*v[i][j];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables for barostat
   scale barostat velocities
------------------------------------------------------------------------- */
// CM
// Need to revise the code. According to the [2].
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
      // See Appendix D, [2].
      if (pstyle == TRICLINIC) {
        etap_mass[0] = dimension*(dimension+1)/2 * boltz * nhc_temp / (p_freq_max*p_freq_max);
      }
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * t_target) / etap_mass[ich];
    }
  }

  kecurrent = 0.0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
    }

  if (pstyle == TRICLINIC) {
    hg_dot[0][0]=omega_dot[0]; 
    hg_dot[1][1]=omega_dot[1]; 
    hg_dot[2][2]=omega_dot[2]; 
    hg_dot[1][2]=hg_dot[2][1]=omega_dot[3]; 
    hg_dot[0][2]=hg_dot[2][0]=omega_dot[4]; 
    hg_dot[0][1]=hg_dot[1][0]=omega_dot[5]; 

    //CM diagonalize the unit-cell momentum
    dsyevv3(hg_dot, eigv, omega_dot_eig);

    kecurrent = 0.0;
    pdof = 0;
    for (i = 0; i < 3; i++){
        kecurrent += omega_mass[i]*omega_dot_eig[i]*omega_dot_eig[i];
        pdof++;
    }
//    for (i = 3; i < 6; i++)
//      if (p_flag[i]) {
//        kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
//        pdof++;
//      }
  }

  if (pstyle == ISO) lkt_press = kt;
  else if (pstyle == TRICLINIC) lkt_press = pdof * pdof * kt;
  etap_dotdot[0] = (kecurrent - lkt_press)/etap_mass[0];

  double ncfac = 1.0/nc_pchain;
  for (int iloop = 0; iloop < nc_pchain; iloop++) {
    
    //CM
    //ich!=0
    for (ich = mpchain-1; ich > 0; ich--) {
      expfac = exp(-ncfac*dt8*etap_dot[ich+1]);
      etap_dot[ich] *= expfac;
      etap_dot[ich] += etap_dotdot[ich] * ncfac*dt4;
      etap_dot[ich] *= pdrag_factor;
      etap_dot[ich] *= expfac;
    }

    //CM
    //ich=0
    expfac = exp(-ncfac*dt8*etap_dot[1]);
    etap_dot[0] *= expfac;
    etap_dot[0] += etap_dotdot[0] * ncfac*dt4;
    etap_dot[0] *= pdrag_factor;
    etap_dot[0] *= expfac;

    //etap update
    for (ich = 0; ich < mpchain; ich++)
      etap[ich] += ncfac*dthalf*etap_dot[ich];

    //omega_dot update
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
  //CM each degree of freedom is coupled!
  //if (logfile){
  //  fprintf(logfile,"Normal mode mass for %d mode = "
  //                  "%.4f \n", universe->iworld, mass[type[0]]);}

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
      //kecurrent = boltz * t_current;
      //kecurrent = tdof * boltz * t_current;
      kecurrent=mass[type[iatm]] * vv[idim]* vv[idim] * force->mvv2e;

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

//CM 
//measure temperature scalar
double FixPIMD::compute_temp_scalar()
{
  double **v = atom->v;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double temp, temp_p;

  temp_p=0;
  //CM
  //note that we are using the NM mass!
  for (int i=0; i<nlocal; i++){
    temp_p+=mass[type[i]] * (v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
  }

//  if (logfile){
//    printf("Temp. of %d mode in partition %d = "
//                    "%.4f, nlocal: %d \n", universe->iworld, universe->me, temp_p, nlocal);
//    printf("commnprocs: %d, universenprocess: %d \n", comm->nprocs, universe->nprocs);
//  }

  MPI_Allreduce(&temp_p,&temp,1,MPI_DOUBLE,MPI_SUM,world);
  temp*=force->mvv2e/boltz/atom->natoms/dimension;
  
  return temp;
}

//CM
//measure temperature scalar
void FixPIMD::compute_temp_vector()
{
  double **v = atom->v;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double t[6];

  //init
  for (int idim=0; idim<6; idim++){
    t[idim]=0.0;
  }

  //note that we are using the NM mass!
  for (int i=0; i<nlocal; i++){
    t[0]+=mass[type[i]]*v[i][0]*v[i][0];
    t[1]+=mass[type[i]]*v[i][1]*v[i][1];
    t[2]+=mass[type[i]]*v[i][2]*v[i][2];

    t[3]+=mass[type[i]]*v[i][1]*v[i][2];
    t[4]+=mass[type[i]]*v[i][0]*v[i][2];
    t[5]+=mass[type[i]]*v[i][0]*v[i][1];
  }

  MPI_Allreduce(t,t_current_vector,6,MPI_DOUBLE,MPI_SUM,world);

  for (int idim=0; idim<6; idim++){
    //t_current_vector[idim]*=force->mvv2e/boltz/atom->natoms/dimension;
    t_current_vector[idim]*=force->mvv2e/boltz/atom->natoms;
  }

//  if (logfile){
//    fprintf(logfile,"Temp. of %d mode = "
//                    "(%f, %f, %f, %f, %f, %f )\n", universe->iworld, t_current_vector[0], t_current_vector[1], t_current_vector[2], t_current_vector[3], t_current_vector[4], t_current_vector[5]);}
}

//update the centroid forces
void FixPIMD::update_f_centroid()
{
  //method 2 
  nmpimd_fill(atom->f);
  comm_exec(atom->f);
  nmpimd_transform(buf_beads, fc, M_x2xp[0]);

//  //method 1
//  double ff[atom->nlocal][3];
//
//  for (int i = 0; i < atom->nlocal; i++) {
//    for(int j=0; j<3; j++){
//      ff[i][j]=fpre[i][j]/(double)np;
//    }
//  }
//
//  for(int i=0; i<atom->nlocal; i++){
//      MPI_Allreduce(ff[i],fc[i],3,MPI_DOUBLE,MPI_SUM,beads_comm); 
//  }

//  if(universe->me==23){
//    printf("fc1:%d %f %f %f \n", atom->tag[120], fc[120][0], fc[120][1], fc[120][2]);
//  }

//  //method 2
//  nmpimd_fill(atom->f);
//  comm_exec(atom->f);
//  nmpimd_transform(buf_beads, atom->f, M_f2fp[universe->iworld]);
//
//  for (int i = 0; i < atom->nlocal; i++) {
//    for(int j=0; j<3; j++){
//      fc[i][j]=atom->f[i][j];
//    }
//  }
//
//  //broadcast the centroid force
//  MPI_Barrier(universe->uworld);
//  for(int i=0; i<atom->nlocal; i++){
//    MPI_Bcast(fc[i], 3, MPI_DOUBLE, 0, universe->uworld);}
//  MPI_Barrier(universe->uworld);
//
//  nmpimd_fill(atom->f);
//  comm_exec(atom->f);
//  nmpimd_transform(buf_beads, atom->f, M_fp2f[universe->iworld]);
//
//  if(universe->me==23){
//    printf("fc2:%d %f %f %f \n", atom->tag[120], fc[120][0], fc[120][1], fc[120][2]);
//  }

}

//CM
//update the centroid position
void FixPIMD::update_x_centroid()
{
  // method 1 
  /*
  double xx[atom->nlocal][3];

  for (int i = 0; i < atom->nlocal; i++) {
    for(int j=0; j<3; j++){
      xx[i][j]=atom->x[i][j];
    }
  }
  for(int i=0; i<atom->nlocal; i++){
      MPI_Allreduce(xx[i],xc[i],3,MPI_DOUBLE,MPI_SUM,beads_comm); 
//      MPI_Allreduce(xx[i],xc[i],3,MPI_DOUBLE,MPI_SUM,universe->uworld); 
  }
  for(int i=0; i<atom->nlocal; i++){
    for(int j=0; j<3; j++){
      xc[i][j]/=(double)np;
    }
  }
  */

//  if(universe->me==23){
//    printf("xc1:%d %f %f %f \n", atom->tag[120], xc[120][0], xc[120][1], xc[120][2]);
//  }

//  if(universe->me==6){
//    printf("xc1:%d %f %f %f \n", atom->tag[12], atom->x[12][0], atom->x[12][1], atom->x[12][2]);
//  }

  //method 2 
  nmpimd_fill(atom->x);
  comm_exec(atom->x);
  nmpimd_transform(buf_beads, xc, M_x2xp[0]);

//  if(universe->me==6){
//    printf("xc1:%d %f %f %f \n", atom->tag[12], atom->x[12][0], atom->x[12][1], atom->x[12][2]);
//  }
//
//  if(universe->me==6){
//    printf("xc1:%d %f %f %f \n", atom->tag[12], xc[12][0], xc[12][1], xc[12][2]);
//  }

//  for (int i = 0; i < atom->nlocal; i++) {
//    for(int j=0; j<3; j++){
//      xc[i][j]=xx[i][j];
//    }
//  }

//  //broadcast the centroid 
//  MPI_Barrier(universe->uworld);
//  for(int i=0; i<atom->nlocal; i++){
//    MPI_Bcast(xc[i], 3, MPI_DOUBLE, 0, universe->uworld);}
//  MPI_Barrier(universe->uworld);
//
////  if(universe->me==0){
////    printf("xc2: %f %f %f \n", xc[120][0], xc[120][1], xc[120][2]);
////  }
//
//  if(universe->me==23){
//    printf("xc2:%d %f %f %f \n",atom->tag[120], xc[120][0], xc[120][1], xc[120][2]);
//  }

}

//  dvalue = pe->scalar;


// CM 
// is unit ok?
void FixPIMD::observe_etot()
{
  int nlocal = atom->nlocal;
  etot=0.0;
  ketot=0.0;
  petot=0.0;
  //temp contribution
  //etot+=dimension*atom->natoms*boltz*t_current_avg/2.;
  //etot+=3.0*atom->natoms*boltz*t_current_avg/2.;

  if(method_statistics==BOLTZMANN){
    //pe
    observe_pe_avg();
    petot+=pe_current_avg;
    //ke
    ketot+=dimension*atom->natoms*boltz*nhc_temp/2.;
    observe_virial_avg();
    ketot+=vir_current_avg;
  }
  else if(method_statistics==BOSON){
    //pe
    observe_pe_avg();
    petot+=pe_current_avg;
    //ke
    ketot+=dimension*np*atom->natoms*boltz*nhc_temp/2.;
    ketot+=ke_boson_vir;
    //for(int i=0; i<atom->natoms; i++){ 
    //  if (universe->me ==0){ 
    //    printf("V: %e \n", V.at(i));}
    //}
  }
  //etot
  etot=petot+ketot;
}

//CM
//pimd virial avg.
//need to work on this. 
//better to keep centroid and others as separate array.
void FixPIMD::observe_virial_avg() 
{
  double xx[atom->nlocal][3];

  for (int i = 0; i < atom->nlocal; i++) {
    for(int j=0; j<3; j++){
      xx[i][j]=atom->x[i][j]; 
    }
  }

  vir_current=0.0;
  vir_current_avg=0.0;

//CM
//direct summation of virial, but lots of fluctuations...
//CM
//transform forces in real coordinates

//  if(universe->me==0)
//    printf("force0: %f %f %f \n", atom->f[150][0],atom->f[150][1],atom->f[150][2]);

//  nmpimd_fill(atom->f);
//  comm_exec(atom->f);
//  nmpimd_transform(buf_beads, atom->f, M_fp2f[universe->iworld]);
//
//  comm_exec(atom->x);
//  remove_spring_force();

//  if(universe->me==23)
//    printf("xx: %f %f %f \n", xx[150][0],xx[150][1],xx[150][2]);
//  if(universe->me==23)
//    printf("xc: %f %f %f \n", xc[150][0],xc[150][1],xc[150][2]);

  //centroid virial
  double vir_current_p=0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    for(int j=0; j<3; j++){
      vir_current_p -= fpre[i][j]*(xx[i][j] - xc[i][j]);
      //vir_current_p -= fpre[i][j]*(xx[i][j] - xc[i][j]);
      //vir_current_p += abs(fpre[i][j]/(double)np*(xx[i][j] - xc[i][j]));
    }
  }
  vir_current_p*=0.5/(double)np;

//  //primitive virial 
//  for (int i = 0; i < atom->nlocal; i++) {
//    for(int j=0; j<3; j++){
//      vir_current_p -= fpre[i][j]/(double)np*xx[i][j];
//    }
//  }
//  vir_current_p*=0.5;

//  comm_exec(atom->x);
//  spring_force();
//
//  nmpimd_fill(atom->f);
//  comm_exec(atom->f);
//  nmpimd_transform(buf_beads, atom->f, M_f2fp[universe->iworld]);

//  if(universe->me==0)
//    printf("force0: %f %f %f \n", atom->f[150][0],atom->f[150][1],atom->f[150][2]);

////  //CM
////  if(universe->me==0){
////    fprintf(pimdfile, "%f %f %f %f %f %f \n", atom->f[0][0], atom->f[0][1], atom->f[0][2], atom->x[0][0], atom->x[0][1], atom->x[0][2] );
////    fprintf(pimdfile, "%f %f \n", force->mvv2e, vir_current);
////  }
//

  //sum nlocal 
  //MPI_Allreduce(&vir_current_p,&vir_current,1,MPI_DOUBLE,MPI_SUM,world);
  //sum over beads
  //MPI_Allreduce(&vir_current,&vir_current_avg,1,MPI_DOUBLE,MPI_SUM,beads_comm);
  MPI_Allreduce(&vir_current_p,&vir_current_avg,1,MPI_DOUBLE,MPI_SUM,universe->uworld);

//  //gather and sum vir.
//  MPI_Gather(&vir_current, 1, MPI_DOUBLE, vir_current_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//  if(universe->me==0){
//    vir_current_avg=compute_sum(vir_current_beads, np);
//  }
//  MPI_Barrier(universe->uworld);
}

//pimd consv. energy
//this quantity should be conserved throughout the simulations
//we have a drift of the term.
void FixPIMD::observe_E_consv() 
{
  E_consv=0.0;
  //pv
  E_consv+=(p_stop[0]+p_stop[1]+p_stop[2])/3.*vol_current/nktv2p;
  //ke
  E_consv+=dimension/2.*boltz * np * atom->natoms * t_current_avg;
  //pe
  observe_pe_avg();
  E_consv+=pe_current_avg;
  //spring
  observe_spring_energy_sum();
  E_consv+=spring_energy_sum;
  //thermostat energy
  observe_eta_E_sum();
  E_consv+=eta_E_sum;
  //omega
  observe_omega_E();
  E_consv+=omega_E;
  //etap
  observe_etap_E_sum();
  E_consv+=etap_E_sum;
}

void FixPIMD::observe_omega_E()
{
  omega_E=0.0;
  for(int i=0; i<3; i++){
    omega_E+=0.5*omega_mass[i]*omega_dot[i]*omega_dot[i];
  }
}

void FixPIMD::observe_etap_E_sum() 
{
  double t_target=nhc_temp;
  etap_E_sum=0.;
  for (int ich=0; ich<mpchain; ich++){
    etap_E_sum+=0.5*etap_mass[ich]*etap_dot[ich]*etap_dot[ich];
    etap_E_sum+=boltz*t_target*etap[ich];
  }
}

//pimd thermostat energy
void FixPIMD::observe_eta_E_sum() 
{
  int nmax = 3 * atom->nlocal;  //d*N d.o.f
  int *type = atom->type;
  double t_target=nhc_temp;
  double eta_E_p=0.0; 

  for(int ip=0; ip<nmax; ip++)
  {
    int iatm = ip/3;
    int idim = ip%3;
    for (int ich=0; ich<mtchain; ich++){
      //kinetic
      eta_E_p+= 0.5*eta_mass[ip][ich]*eta_dot[ip][ich]*eta_dot[ip][ich]; // *force->mvv2e;
      //eta term
      eta_E_p+= boltz*t_target*eta[ip][ich];
    }
  }
  //sum nlocal 
  MPI_Allreduce(&eta_E_p,&eta_E,1,MPI_DOUBLE,MPI_SUM,world);
  //sum over beads
  MPI_Allreduce(&eta_E,&eta_E_sum,1,MPI_DOUBLE,MPI_SUM,beads_comm);

//  //gather eat_E
//  MPI_Gather(&eta_E, 1, MPI_DOUBLE, eta_E_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//  if(universe->me==0){
//    eta_E_sum=compute_sum(eta_E_beads, np);
//  }
//  MPI_Barrier(universe->uworld);
}

void FixPIMD::observe_spring_energy_sum()
{
  spring_energy_sum=0;
  MPI_Allreduce(&spring_energy,&spring_energy_sum,1,MPI_DOUBLE,MPI_SUM,beads_comm);

//  //gather spring energy 
//  MPI_Gather(&spring_energy, 1, MPI_DOUBLE, spring_energy_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//  if(universe->me==0){
//    spring_energy_sum=compute_avg(spring_energy_beads, np);
//  }
//  MPI_Barrier(universe->uworld);
}

//CM
//pimd potential energy avg.
void FixPIMD::observe_pe_avg() 
{
  pe_current=pe->compute_scalar();

  MPI_Allreduce(&pe_current,&pe_current_avg,1,MPI_DOUBLE,MPI_SUM,beads_comm);
  pe_current_avg/=np;

//  //gather and average pe
//  MPI_Gather(&pe_current, 1, MPI_DOUBLE, pe_current_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//  if(universe->me==0){
//    pe_current_avg=compute_avg(pe_current_beads, np);
//  }
//  MPI_Barrier(universe->uworld);
}

//CM 
//pimd temperature
void FixPIMD::observe_temp_scalar() 
{
//  printf("Temp. current: %f --- BEADS RANK/SIZE: %d/%d\n",
//     t_current, beads_rank, beads_size);

  //beads_comm
  MPI_Allreduce(&t_current,&t_current_avg,1,MPI_DOUBLE,MPI_SUM,beads_comm);

//  printf("Temp. allreduce: %f --- BEADS RANK/SIZE: %d/%d\n",
//     t_current_avg, beads_rank, beads_size);

  t_current_avg/=np;

//  if(beads_rank!=-1){
//    printf("Temp. current/avg: %f/%f --- BEADS RANK/SIZE: %d/%d\n",
//       t_current, t_current_avg, beads_rank, beads_size);
//  }

//  MPI_Gather(&t_current, 1, MPI_DOUBLE, t_current_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//  if(universe->me==0){
//    t_current_avg=compute_avg(t_current_beads, np);
//  }
//  MPI_Barrier(universe->uworld);
}

//CM
//Monitor obervable 
void FixPIMD::monitor_observable()
{
  //NPT
  if (pstat_flag && mpchain){
    //temp observe
    observe_temp_scalar();
    //pressure
    double pressure_current = 1.0/3.0 * (p_current[0] + p_current[1] + p_current[2]);
    //E_consv
    observe_E_consv();
    if(universe->me==0){
      if (pimdfile){
        fprintf(pimdfile, "%d    %f    %f    %f    %f    %f    %f    %f\n", 
                          update->ntimestep, t_current_avg, vol_current, pressure_current, 
                          E_consv, etot/boltz/atom->natoms, petot/boltz/atom->natoms, ketot/boltz/atom->natoms);

        //fprintf(pimdfile, "%d    %f    %f\n", update->ntimestep,  ketot/boltz,  boltz);
        //fprintf(pimdfile, "%d    %f    %f    %f    %f    %f    \n", update->ntimestep, t_current_avg, vol_current, pressure_current, pe_current_avg, E_consv);
//        fprintf(pimdfile, "%f    %f    %f \n", eta_E_sum, etap_E_sum, omega_E);
      }
    }
  }
  //NVT
  else{
    //temp observe
    observe_temp_scalar();
    //etot
    observe_etot();
    if(universe->me==0){
      if (pimdfile){
        //fprintf(pimdfile, "%d    %f    %f    %f    %f    %f    %f\n", update->ntimestep, t_current_avg, vol_current, pressure_current, E_consv, etot/boltz/atom->natoms, vir_current_avg/boltz/atom->natoms);
        //fprintf(pimdfile, "%d    %f    %f  \n", update->ntimestep, t_current_avg, vol_current);
        fprintf(pimdfile, "%d    %f    %f    %f    %f    %f\n", 
                          update->ntimestep, t_current_avg, 
                          etot/boltz/atom->natoms, petot/boltz/atom->natoms, ketot/boltz/atom->natoms, Pc_longest);
//        fprintf(pimdfile, "%f    %f    %f \n", eta_E_sum, etap_E_sum, omega_E);
      }
    }

  }
}

//CM
//Monitor obervable 
void FixPIMD::initialize_logfile()
{
  if(universe->me==0){
    if (pimdfile){
      fprintf(pimdfile,"                                                                     \n"
        "   __ _        ____ ___ __  __ ____                                                 \n" 
        "  / _(_)_  __ |  _ \\_ _|  \\/  |  _ \\                                             \n" 
        " | |_| \\ \\/ / | |_) | || |\\/| | | | |                                            \n" 
        " |  _| |>  <  |  __/| || |  | | |_| |                                               \n" 
        " |_|_|_/_/\_\_|_|_ |___|_|__|_|____/                       _   __  __ ____          \n" 
        " |  _ \\ __ _| |_| |__   |_ _|_ __ | |_ ___  __ _ _ __ __ _| | |  \\/  |  _ \\      \n" 
        " | |_) / _` | __| '_ \\   | || '_ \\| __/ _ \\/ _` | '__/ _` | | | |\\/| | | | |    \n"  
        " |  __/ (_| | |_| | | |  | || | | | ||  __/ (_| | | | (_| | | | |  | | |_| |        \n"  
        " |_|   \\__,_|\\__|_| |_| |___|_| |_|\\__\\___|\\__, |_|  \\__,_|_| |_|  |_|____/   \n" 
        "                                           |___/                                    \n"  
        "                                                                                    \n"
        "                                                                   ver.2.0          \n"); 
    }
  }
}

double FixPIMD::compute_sum(double *array, int n)
{
  double sum = 0.f;
  for (int i = 0; i < n; i++) {
    sum += array[i];
  }
  return sum;
}

double FixPIMD::compute_avg(double *array, int n) 
{
  double sum = 0.f;
  for (int i = 0; i < n; i++) {
    sum += array[i];
  }
  return sum / n;
}

//wrapper function for pimd pressure virial calculation
void FixPIMD::compute_pressure_scalar()
{
  //ISO case
  double p_current_p;
  //CM compute the pressure in NM coordinates
  if(method==CMD || method==NMPIMD)
  {
//      if(universe->me==7)
//        printf("position0: %f %f %f \n", atom->x[150][0],atom->x[150][1],atom->x[150][2]);
  
//Does this affect the pressure calculation?
//      nmpimd_fill(atom->x);
//      comm_exec(atom->x);
//      nmpimd_transform(buf_beads, atom->x, M_x2xp[universe->iworld]);
  
//      if(universe->me==7)
//        printf("position1: %f %f %f \n", atom->x[150][0],atom->x[150][1],atom->x[150][2]);
    p_current_avg=0.0;  
    p_current_p=pressure->compute_scalar();
    MPI_Allreduce(&p_current_p,&p_current_avg,1,MPI_DOUBLE,MPI_SUM,beads_comm);
    p_current_avg/=(double)np;

//      nmpimd_fill(atom->x);
//      comm_exec(atom->x);
//      nmpimd_transform(buf_beads, atom->x, M_xp2x[universe->iworld]);

//      if(universe->me==7)
//        printf("position0: %f %f %f \n", atom->x[150][0],atom->x[150][1],atom->x[150][2]);
  }
  else{
    pressure->compute_scalar();}
}

//wrapper function for pimd pressure virial calculation
//CM
//Pressure centroid estimator
void FixPIMD::compute_pressure_vector()
{
//  pressure->compute_vector();

  compute_temp_vector();
  double *t_tensor_nm=t_current_vector;
  double *t_tensor_md=temperature->vector;
  double p_current_tensor_p[3];
  double virial[3];
  double volume=domain->xprd*domain->yprd*domain->zprd;

  virial[0]=virial[1]=virial[2]=0.0;
  for(int i=0; i<atom->nlocal; i++){
    for(int j=0; j<3; j++){
      virial[j]+=fc[i][j]*xc[i][j];
    }
  }

  p_current_tensor_p[0]=p_current_tensor_p[1]=p_current_tensor_p[2]=0.0;
  for(int i=0; i<3; ++i){
    p_current_tensor_p[i]+=virial[i]/volume*force->nktv2p;
  }
  MPI_Allreduce(p_current_tensor_p,p_current_tensor_avg,3,MPI_DOUBLE,MPI_SUM,world);

  for(int i=0; i<3; ++i){
    //p_current_tensor_avg[i]+=atom->natoms*boltz*t_tensor_nm[i]/volume*force->nktv2p;
    //p_current_tensor_avg[i]+=atom->natoms*boltz*t_current/3.0/volume*force->nktv2p;
    p_current_tensor_avg[i]+=atom->natoms*boltz*nhc_temp/volume*force->nktv2p;
  }

// de/dv
  pressure->compute_vector();
  temperature->compute_scalar();
  double *p_current_tensor_ev=pressure->vector;

  virial[0]=virial[1]=virial[2]=0.0;
  for(int i=0; i<atom->nlocal; i++){
    for(int j=0; j<3; j++){
      virial[j]+=fpre[i][j]*atom->x[i][j];
    }
  }

  for(int i=0; i<3; ++i){
    p_current_tensor_ev[i]-=virial[i]/volume*force->nktv2p;
    p_current_tensor_ev[i]-=atom->natoms*boltz*t_tensor_md[i]/volume*force->nktv2p;
    //p_current_tensor_ev[i]-=atom->natoms*boltz*nhc_temp/volume*force->nktv2p;
  }

  MPI_Allreduce(MPI_IN_PLACE,p_current_tensor_ev,3,MPI_DOUBLE,MPI_SUM,beads_comm);

  for(int i=0; i<3; ++i){
    p_current_tensor_avg[i]+=p_current_tensor_ev[i]/(double)np;
    //p_current_tensor_avg[i]+=p_current_tensor_ev[i];
  }


/*
  double *t_tensor=temperature->vector;
  temperature->compute_vector();
  pressure->compute_vector();
  compute_temp_vector();

  double *p_current_tensor_p=pressure->vector;
  double volume=domain->xprd*domain->yprd*domain->zprd;

  double virial[3];

  virial[0]=virial[1]=virial[2]=0.0;
  for(int i=0; i<atom->nlocal; i++){
    for(int j=0; j<3; j++){
      virial[j]+=fpre[i][j]*atom->x[i][j];
    }
  }

  for(int i=0; i<3; ++i){
    p_current_tensor_p[i]-=virial[i]/volume*force->nktv2p;
    p_current_tensor_p[i]-=atom->natoms*boltz*t_tensor[i]/volume*force->nktv2p;
  }

  //dU/dV
  MPI_Allreduce(MPI_IN_PLACE,p_current_tensor_p,3,MPI_DOUBLE,MPI_SUM,beads_comm);

  for(int i=0; i<3; ++i){
    p_current_tensor_p[i]/=(double)np;
  }

  virial[0]=virial[1]=virial[2]=0.0;
  for(int i=0; i<atom->nlocal; i++){
    for(int j=0; j<3; j++){
      virial[j]+=fc[i][j]*xc[i][j];
    }
  }

  for(int i=0; i<3; ++i){
    p_current_tensor_p[i]+=virial[i]/volume*force->nktv2p;
    p_current_tensor_p[i]+=atom->natoms*boltz*t_tensor_nm[i]/volume*force->nktv2p+100.;
  }
  MPI_Allreduce(p_current_tensor_p,p_current_tensor_avg,3,MPI_DOUBLE,MPI_SUM,world);
*/

/*
//  if(universe->me==0)
//    printf("p_current: %f %f %f \n", p_current_tensor_p[0],p_current_tensor_p[1],p_current_tensor_p[2]);

  //CM compute the pressure in NM coordinates
  if(method==CMD || method==NMPIMD)
  {
    //correct temp term.
    for(int i=0; i<3; ++i){
      p_current_tensor_p[i]-=atom->natoms*boltz*t_tensor[i]/volume;
      p_current_tensor_p[i]+=atom->natoms*boltz*t_tensor_nm[i]/volume;
    }
    MPI_Allreduce(p_current_tensor_p,p_current_tensor_avg,3,MPI_DOUBLE,MPI_SUM,beads_comm);
//    for(int i=0; i<3; ++i){
//      p_current_tensor_avg[i]/=(double)np;
//    }
  }
  else{
    pressure->compute_vector();}
*/
}

/*

//CM
//computes pressure virial
//need to improve
void FixPIMD::compute_pressure_scalar()
{
  double **f = atom->f;
  double **x = atom->x;
  double p_virial=0.0;
  double volume = domain->xprd*domain->yprd*domain->zprd;
  //t_current = temperature->compute_scalar();
  t_current = compute_temp_scalar();

  //pressure_scalar=atom->nlocal*boltz*t_current/volume*nktv2p;
  //scalar = (temperature->dof * boltz * t + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  pressure_scalar = ( atom->nlocal* boltz * t_current ) / volume * nktv2p;

  comm_exec(atom->x);
  remove_spring_force();
  for (int i = 0; i < atom->nlocal; i++) {
      p_virial += np*f[i][0] * x[i][0];
      p_virial += np*f[i][1] * x[i][1];
      p_virial += np*f[i][2] * x[i][2];
  } 
  pressure_scalar += (p_virial/3.0/volume*nktv2p);
  spring_force();

//  if(universe->me==0) printf("pressure: %f \n", pressure_scalar);
}

//CM
//computes pressure virial
//need to improve
void FixPIMD::compute_pressure_vector()
{
  double **f = atom->f;
 // double **v = atom->v;
  double **x = atom->x;
  double p_virial=0.0;
  double volume = domain->xprd*domain->yprd*domain->zprd;
  //t_current = temperature->compute_scalar();
  t_current = compute_temp_scalar();

  //pressure_scalar=atom->nlocal*boltz*t_current/volume*nktv2p;
  //scalar = (temperature->dof * boltz * t + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  //pressure_scalar = ( atom->nlocal* boltz * t_current ) / volume * nktv2p;

  double *mvv_current = temperature->vector;
  for (int i; i<6; i++){
    pressure_vector[i] = ( mvv_current[i] ) / volume * nktv2p;
  }
//  if(universe->me==0) printf("mvv: %f %f %f \n", mvv_current[0], mvv_current[1], mvv_current[2]);
//  if(universe->me==0) printf("mvv: %f %f %f \n", mvv_current[3], mvv_current[4], mvv_current[5]);

  comm_exec(atom->x);
  remove_spring_force();
  for (int i = 0; i < atom->nlocal; i++) {
      //should i multiply somthing here?
      //mvv2e?
      pressure_vector[0] += np*f[i][0] * x[i][0] / volume * nktv2p;
      pressure_vector[1] += np*f[i][1] * x[i][1] / volume * nktv2p;
      pressure_vector[2] += np*f[i][2] * x[i][2] / volume * nktv2p;
      pressure_vector[3] += np*f[i][1] * x[i][2] / volume * nktv2p;
      pressure_vector[4] += np*f[i][0] * x[i][2] / volume * nktv2p;
      pressure_vector[5] += np*f[i][0] * x[i][1] / volume * nktv2p;
  } 
  spring_force();
  //CM
  //how to add volume contributions?

  if(universe->me==0) printf("nktv2p: %f \n", nktv2p);
  if(universe->me==0) printf("pressure: %f %f %f \n", pressure_vector[0], pressure_vector[1], pressure_vector[2]);
  if(universe->me==0) printf("pressure: %f %f %f \n", pressure_vector[3], pressure_vector[4], pressure_vector[5]);
}

*/

// ----------------------------------------------------------------------------
void FixPIMD::dsyevc3(double **A, double *w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;

  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double de = A[0][1] * A[1][2];                                    // d * e
  double dd = SQR(A[0][1]);                                         // d^2
  double ee = SQR(A[1][2]);                                         // e^2
  double ff = SQR(A[0][2]);                                         // f^2
  m  = A[0][0] + A[1][1] + A[2][2];
  c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
          - (dd + ee + ff);
  c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
            - 2.0 * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);

  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;
}

// ----------------------------------------------------------------------------
void FixPIMD::dsyevv3(double **A, double **Q, double *w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
  double norm;          // Squared norm or inverse norm of current eigenvector
  double n0, n1;        // Norm of first and second columns of A
  double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
  double thresh;        // Small number used as threshold for floating point comparisons
  double error;         // Estimated maximum roundoff error in some steps
  double wmax;          // The eigenvalue of maximum modulus
  double f, t;          // Intermediate storage
  int i, j;             // Loop counters

  // Calculate eigenvalues
  dsyevc3(A, w);

  wmax = fabs(w[0]);
  if ((t=fabs(w[1])) > wmax)
    wmax = t;
  if ((t=fabs(w[2])) > wmax)
    wmax = t;
  thresh = SQR(8.0 * DBL_EPSILON * wmax);

  // Prepare calculation of eigenvectors
  n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
  n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
  Q[2][1] = SQR(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
  norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
  n0      = n0tmp + SQR(A[0][0]);
  n1      = n1tmp + SQR(A[1][1]);
  error   = n0 * n1;
  
  if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
  {
    Q[0][0] = 1.0;
    Q[1][0] = 0.0;
    Q[2][0] = 0.0;
  }
  else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
  {
    Q[0][0] = 0.0;
    Q[1][0] = 1.0;
    Q[2][0] = 0.0;
  }
  else if (norm < SQR(64.0 * DBL_EPSILON) * error)
  {                         // If angle between A[0] and A[1] is too small, don't use
    t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];
    if (SQR(A[1][1]) > t)
    {
      t = SQR(A[1][1]);
      f = -A[0][1] / A[1][1];
    }
    if (SQR(A[1][2]) > t)
      f = -A[0][2] / A[1][2];
    norm    = 1.0/sqrt(1 + SQR(f));
    Q[0][0] = norm;
    Q[1][0] = f * norm;
    Q[2][0] = 0.0;
  }
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  
  // Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
  {
    // For non-degenerate eigenvalue, calculate second eigenvector by the formula
    //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
    A[0][0] += t;
    A[1][1] += t;
    Q[0][1]  = Q[0][1] + A[0][2]*w[1];
    Q[1][1]  = Q[1][1] + A[1][2]*w[1];
    Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
    norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
    n0       = n0tmp + SQR(A[0][0]);
    n1       = n1tmp + SQR(A[1][1]);
    error    = n0 * n1;
 
    if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
    {
      Q[0][1] = 1.0;
      Q[1][1] = 0.0;
      Q[2][1] = 0.0;
    }
    else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
    {
      Q[0][1] = 0.0;
      Q[1][1] = 1.0;
      Q[2][1] = 0.0;
    }
    else if (norm < SQR(64.0 * DBL_EPSILON) * error)
    {                       // If angle between A[0] and A[1] is too small, don't use
      t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if (SQR(A[1][1]) > t)
      {
        t = SQR(A[1][1]);
        f = -A[0][1] / A[1][1];
      }
      if (SQR(A[1][2]) > t)
        f = -A[0][2] / A[1][2];
      norm    = 1.0/sqrt(1 + SQR(f));
      Q[0][1] = norm;
      Q[1][1] = f * norm;
      Q[2][1] = 0.0;
    }
    else
    {
      norm = sqrt(1.0 / norm);
      for (j=0; j < 3; j++)
        Q[j][1] = Q[j][1] * norm;
    }
  }
  else
  {
    // For degenerate eigenvalue, calculate second eigenvector according to
    //   v[1] = v[0] x (A - w[1]).e[i]
    //   
    // This would really get to complicated if we could not assume all of A to
    // contain meaningful values.
    A[1][0]  = A[0][1];
    A[2][0]  = A[0][2];
    A[2][1]  = A[1][2];
    A[0][0] += w[0];
    A[1][1] += w[0];
    for (i=0; i < 3; i++)
    {
      A[i][i] -= w[1];
      n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
      if (n0 > thresh)
      {
        Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
        Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
        Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
        norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
        if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
        {                                         // the two vectors was not too small
          norm = sqrt(1.0 / norm);
          for (j=0; j < 3; j++)
            Q[j][1] = Q[j][1] * norm;
          break;
        }
      }
    }
    
    if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
    {
      for (j=0; j < 3; j++)
        if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
        {                                                     // ... and swap it with the next one
          norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
          Q[j][1]       = Q[(j+1)%3][0] * norm;
          Q[(j+1)%3][1] = -Q[j][0] * norm;
          Q[(j+2)%3][1] = 0.0;
          break;
        }
    }
  }
      
  
  // Calculate third eigenvector according to
  //   v[2] = v[0] x v[1]
  Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
}

/* ----------------------------------------------------------------------
  if any tilt ratios exceed limits, set flip = 1 and compute new tilt values
  do not flip in x or y if non-periodic (can tilt but not flip)
    this is b/c the box length would be changed (dramatically) by flip
  if yz tilt exceeded, adjust C vector by one B vector
  if xz tilt exceeded, adjust C vector by one A vector
  if xy tilt exceeded, adjust B vector by one A vector
  check yz first since it may change xz, then xz check comes after
  if any flip occurs, create new box in domain
  image_flip() adjusts image flags due to box shape change induced by flip
  remap() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

void FixPIMD::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;

  // flip is only triggered when tilt exceeds 0.5 by DELTAFLIP
  // this avoids immediate re-flipping due to tilt oscillations

  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy,flipxz,flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd;
      domain->xz += domain->xy;
      flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd;
      domain->xz -= domain->xy;
      flipyz = -1;
    }
  }

  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) {
      domain->xz += xprd;
      flipxz = 1;
    } else if (domain->xz >= xtiltmax) {
      domain->xz -= xprd;
      flipxz = -1;
    }
    if (domain->xy < -xtiltmax) {
      domain->xy += xprd;
      flipxy = 1;
    } else if (domain->xy >= xtiltmax) {
      domain->xy -= xprd;
      flipxy = -1;
    }
  }

  int flip = 0;
  if (flipxy || flipxz || flipyz) flip = 1;

  if (flip) {
    domain->set_global_box();
    domain->set_local_box();

    domain->image_flip(flipxy,flipxz,flipyz);

    double **x = atom->x;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

    domain->x2lamda(atom->nlocal);
    irregular->migrate_atoms();
    domain->lamda2x(atom->nlocal);
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



/*
Particle symmetry: Boson

*/

double FixPIMD::Evaluate_ke_boson(const std::vector<double> &V, const std::vector<double> &save_E_kn)
{
  int n=atom->natoms;
  double numerator;
  double beta   = 1.0 / (boltz * nhc_temp);
  double Emax;
  int beta_n; //partitioning beta for the low temp. limit
  int beta_grid=100; //beta grid
  //double E_kn_tmp;

  //bosonic kinetic energy
  //ke_boson=std::vector<double>((atom->natoms + 1),0.0);
  std::vector<double> ke_boson(n+1, 0.0);

  int count = 0;
  for (int m = 1; m < n+1; ++m) {
    numerator=0.0;
//    Emax = sEl*(Evaluate_Ekn(m,1)+V.at(m-1));
    Emax=std::min((Evaluate_Ekn(m,1)+V.at(m-1)), (Evaluate_Ekn(m,m)+V.at(0)));
    for (int k = m; k > 0; --k) {
      //E_kn_tmp = Evaluate_Ekn(m,k);
      //if (universe->me ==0)  printf("keboson: %e / save_E_kn: %e / V: %e / Emax: %e \n", ke_boson.at(m-k), save_E_kn.at(count), V.at(m-k), Emax);
      numerator += (ke_boson.at(m-k)-save_E_kn.at(count))*exp(-beta*(save_E_kn.at(count) + V.at(m-k) - Emax));
      //if (universe->me ==0)  printf("E_kn(%d,%d): %e, save_E_kn(%d): %e \n", m, k, E_kn_tmp, count, save_E_kn.at(count));
      //if (universe->me ==0)  printf("V(%d-%d): %e, Emax: %e \n", m, k, V.at(m-k), Emax);
      //if (universe->me ==0)  printf("save_E_kn: %e \n", save_E_kn.at(count));
      if(std::isnan(numerator) || std::isinf(numerator)) {
        if (universe->me ==0){
          printf("E_kn(%d,%d): %e, numerator: %e \n", m, k, save_E_kn.at(count), numerator);}
      }
      count++;
    }
    //if (universe->iworld ==0) printf("V(m-1): %e \n", V.at(m));

//    beta_n=(int)(beta*(V.at(m) - Emax))/beta_grid+1;
    beta_n=(int)(abs(beta*(V.at(m) - Emax)))/beta_grid+1;
    double sig_denom_m = exp(-beta*(V.at(m) - Emax)/(double)beta_n);

    //if (universe->iworld ==0) printf("sig_denom_m: %e \n", sig_denom_m);

    if(sig_denom_m ==0 || std::isnan(sig_denom_m) || std::isinf(sig_denom_m) || std::isnan(numerator) || std::isinf(numerator)) {
      if (universe->iworld ==0){
        std::cout << "m is: "<< m << " V.at(m) is " <<V.at(m)<< " beta is: " << beta << " sig_denom_m is: " <<sig_denom_m << std::endl;
      }
      exit(0);
    }

    for(int ib=0; ib<beta_n; ib++){
      if(ib==0){
        ke_boson.at(m) = numerator/sig_denom_m/(double)m;}
      else{
        ke_boson.at(m) /= sig_denom_m;
      }
    }
  //if (universe->iworld ==0) printf("ke_boson: %e\n", ke_boson.at(m));
  }

  return ke_boson.at(n);
}

//std::vector<double> FixPIMD::Evaluate_VBn_new(std::vector <double>& V, const int n)
std::vector<double> FixPIMD::Evaluate_VBn_new(std::vector <double>& V, const int n)
{
  int ngroups;
  int* ranks;
  double* save_E_kn_arr_mk;

  double sig_denom;
  double Elongest;
  double Ekn;
  double beta   = 1.0 / (boltz * nhc_temp);
  std::vector<double> save_E_kn(n*(n+1)/2, 0.0);

  //temp. mearsure for mpi
  double save_E_kn_arr[n*(n+1)/2];

  double V_arr[n+1];
  for (int ii=0;ii<n+1;ii++) V_arr[ii]=0.0;

  int count = 0;
  //for (int m = 1; m < n+1; ++m) {
  for (int m = 1; m < n+1; ++m) {
  //for (int m = 2; m < n+1; ++m) {
  //for (int m = universe->me+1; m < n+1; m+=universe->nprocs) {
    //if (universe->me ==0)  printf("m: %d \n", m);
    sig_denom = 0.0;
    //max of -beta*E
    //Elongest = std::min((Evaluate_Ekn(m,1)+V.at(m-1)), (Evaluate_Ekn(m,m)+V.at(0)));
    Elongest = std::min((Evaluate_Ekn(m,1)+V_arr[m-1]), (Evaluate_Ekn(m,m)+V_arr[0]));
    //for (int k = m; k > 0; --k) {
    //for (int k = m-universe->me; k > 0; k-=universe->nprocs) {

    //set-up communicator 
    int prime_rank = -1, prime_size = -1, i;
    if (m<np){
      ngroups=m;
      ranks=new int[m];
      for (i=0;i<m;i++) ranks[i]=i;
      MPI_Group_incl(world_group, m, ranks, &prime_group);
      MPI_Comm_create_group(universe->uworld, prime_group, 0, &prime_comm);
      
      if (MPI_COMM_NULL != prime_comm) {
        MPI_Comm_rank(prime_comm, &prime_rank);
        MPI_Comm_size(prime_comm, &prime_size);
      }  
      //if (universe->me ==0){
      //  for (i=0;i<m;i++) printf("m(%d): %d\n", m,ranks[i]);}
      //printf("m(%d) UNIVERSE RANK/SIZE (total): %d/%d  PRIME RANK/SIZE: %d/%d\n", m, universe->me, universe->nprocs, prime_rank, prime_size);
    }

    //para-range
    int iwork1=int(m/universe->nprocs);
    int iwork2=int(m%universe->nprocs); 
    int istart=universe->me*iwork1+1+std::min(universe->me, iwork2);
    int iend=istart+iwork1-1;
    if (iwork2>universe->me) iend=iend+1; 

//    if (universe->me ==1)  printf("istart= %d \n", istart);
//    double save_E_kn_arr_tmp[iend-istart+1];

//    if (universe->me ==1)  printf("istart= %d \n", m-universe->me);

    //store mpi Ekn
    double save_E_kn_arr_m[m];
    for (int j=0;j<m;j++) save_E_kn_arr_m[j]=0.0;
    //if (iend-istart+1>0) save_E_kn_arr_mk=new double[iend-istart+1];
    //save_E_kn_arr_mk=new double[m];
   
    for (int k = m-universe->me; k > 0; k-=universe->nprocs) { //fastest
    //for (int k = iend; k >= istart; --k) { //to collect Ekn
      //count_k+=istart-1;
    //for (int k = m; k > 0; --k) {
      //Ekn = Evaluate_Ekn(m,k);
      Ekn = Evaluate_Ekn_new(m,k);
      save_E_kn_arr_m[k-1] = Ekn;
      //printf("Ekn(%d,%d)=%e \n",m,k,Ekn);
      //Ekn = 0.01;
      //save_E_kn.at(count) = Ekn;
      //MPI_Bcast(&save_E_kn_arr[count], 1, MPI_DOUBLE, int(k%universe->nprocs), universe->uworld);
      //MPI_Bcast(&save_E_kn_arr[count], 1, MPI_DOUBLE, 0, universe->uworld);

      //if (universe->me ==0)  printf("after eval ekn \n");

      //sig_denom += exp(-beta*(Ekn + V.at(m-k)-Elongest));
      sig_denom += exp(-beta*(Ekn + V_arr[m-k]-Elongest));

      //if (universe->me ==0)  printf("after sig denom \n");

      if(std::isnan(sig_denom) || std::isinf(sig_denom)) {
        if (universe->me ==0){
          //printf("E_kn(%d,%d): %e, V.at(m-k):%e, Elongest: %e, sig_denom: %e \n", m, k, Ekn, V.at(m-k), Elongest, sig_denom);}
          printf("E_kn(%d,%d): %e, V.at(m-k):%e, Elongest: %e, sig_denom: %e \n", m, k, Ekn, V_arr[m-k], Elongest, sig_denom);}
      }
      count++;
    }
 
    //gather E_kn
    MPI_Allreduce(MPI_IN_PLACE,save_E_kn_arr_m,m,MPI_DOUBLE,MPI_SUM,universe->uworld);
    int count_i=0;
    for (int i=m*(m-1)/2;i<m*(m+1)/2;i++){
      save_E_kn_arr[i]=save_E_kn_arr_m[count_i];
      count_i++;
    }
//    MPI_Allgather(save_E_kn_arr_mk, iend-istart+1, MPI_DOUBLE, pe_current_beads, 1, MPI_DOUBLE, 0, universe->uworld);
//    MPI_Allgatherv(&save_E_kn_arr_mk[0], iend-istart+1, MPI_DOUBLE, &save_E_kn_arr_m[0], recv_counts, MPI_DOUBLE, 0, universe->uworld);

    //MPI_Barrier(universe->uworld);
    //sum sig denom
    MPI_Allreduce(MPI_IN_PLACE,&sig_denom,1,MPI_DOUBLE,MPI_SUM,universe->uworld);

    V.at(m) = Elongest-1.0/beta*log(sig_denom / (double)m);
    //V_arr[m] = Elongest-1.0/beta*log(sig_denom / (double)m);
    //MPI_Bcast(&V_arr[m], 1, MPI_DOUBLE, 0, universe->uworld);

    if(std::isinf(V.at(m)) || std::isnan(V.at(m))) {
	if (universe->me ==0){
          std::cout << "sig_denom is: " << sig_denom << " Elongest is: " << Elongest
                    << std::endl;}
          exit(0);
    }
//    if (m<np){
//      MPI_Group_free(&prime_group);
//      MPI_Comm_free(&prime_comm);
//    }
  }

  delete[] ranks;
  delete[] save_E_kn_arr_mk;

  //copy arr to vector
  unsigned int arr_size=sizeof(save_E_kn_arr)/sizeof(double);
  save_E_kn.insert(save_E_kn.end(), &save_E_kn_arr[0], &save_E_kn_arr[arr_size]);

//  unsigned int arr_size=sizeof(V_arr)/sizeof(double);
//  V.insert(V.end(), &V_arr[0], &V_arr[arr_size]);

  return save_E_kn;
}

std::vector<double> FixPIMD::Evaluate_VBn(std::vector <double>& V, const int n)
{
  double sig_denom;
  double Elongest;
  double Ekn;
  double beta   = 1.0 / (boltz * nhc_temp);
  //save_E_kn=std::vector<double>((atom->natoms*(atom->natoms+1)/2),0.0);
  std::vector<double> save_E_kn(n*(n+1)/2, 0.0);

  int count = 0;
  for (int m = 1; m < n+1; ++m) {
    sig_denom = 0.0;
    //max of -beta*E
//    Elongest = sEl*(Evaluate_Ekn(m,1)+V.at(m-1));
    Elongest = std::min((Evaluate_Ekn(m,1)+V.at(m-1)), (Evaluate_Ekn(m,m)+V.at(0)));
    //for (int k = 1; k <m+1; ++k) {
    for (int k = m; k > 0; --k) {
      Ekn = Evaluate_Ekn(m,k);
      save_E_kn.at(count) = Ekn;
      //if (universe->me ==0)  printf("(origianl) Ekn(%d,%d)=%e \n",m,k,Ekn);
//      if(k==1){
//        //!BH! I had to add 0.5 below in order to not get sigma which is zero or inf for large systems
//        //CM we choose the maximum value (-\beta*E)
//        //Elongest = sEl*E_kn;
//        Elongest = sEl*(E_kn+V.at(m-k));
//        //Elongest = E_kn;
//        //Elongest = 0.5*E_kn;
//        //Elongest = 100.0; //sEl*(std::max(E_kn,V.at(m-1)));
//        //if (universe->me ==0)
//        //  printf("Elongest/sEl: %f, %f \n", Elongest, sEl);
//      }
      sig_denom += exp(-beta*(Ekn + V.at(m-k)-Elongest));

      if(std::isnan(sig_denom) || std::isinf(sig_denom)) {
        if (universe->me ==0){
//          printf("E_kn(%d,%d): %f, sig_denom: %f \n", m, k, Ekn, sig_denom);}
          printf("E_kn(%d,%d): %e, V.at(m-k):%e, Elongest: %e, sig_denom: %e \n", m, k, Ekn, V.at(m-k), Elongest, sig_denom);}
        //std::cout << "m is: "<<m << " k is: " <<k << " E_kn is: " << E_kn << " V.at(m-k) is: " << V.at(m - k) << " Elongest is: " << Elongest
        //          << " V.at(m-1) is " <<V.at(m-1)<< " beta is: " << beta << " sig_denom is: " <<sig_denom << std::endl ;}
      }
      count++;
    }

    V.at(m) = Elongest-1.0/beta*log(sig_denom / (double)m);
    //if (universe->me ==0)
    //  printf("V.at(%d): %f \n", m, V.at(m));

    if(std::isinf(V.at(m)) || std::isnan(V.at(m))) {
	if (universe->me ==0){
          std::cout << "sig_denom is: " << sig_denom << " Elongest is: " << Elongest
                    << std::endl;}
          exit(0);
    }
    //std::cout<< sig_denom << " " <<  log (sig_denom / (double)m) << " " << beta <<std::endl;
  }
  return save_E_kn;
}

double FixPIMD::Evaluate_Ekn_new(const int n, const int k)
{
  double omega_sq = omega_np*omega_np;
//  int bead = universe->iworld;
//  double **x = atom->x;
  double* _mass = atom->mass;
  int* type = atom->type;
  int nlocal = atom->nlocal;

  //sum over the spring energy of (n-k) particles
  spring_energy = 0.0;
  for (int ib=0;ib<np;ib++){
      double* x_0 = buf_beads[ib];
      double* x_1;
    if(ib==np-1){
      x_1 = buf_beads[0];
    }else{
      x_1 = buf_beads[ib+1];
    }
    //E_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
    x_0 += 3*(n-k);
    x_1 += 3*(n-k);

    if(ib==np-1 && k > 1) x_1 += 3;

    for (int i = n-k; i < n ; ++i) {
      double delx = x_1[0] - x_0[0];
      double dely = x_1[1] - x_0[1];
      double delz = x_1[2] - x_0[2];

      domain->minimum_image(delx, dely, delz);
      spring_energy += 0.5*_mass[type[i]]*omega_sq*(delx*delx + dely*dely + delz*delz);

      x_0+=3;
      if (ib==np-1 && i==n-2) {
        x_1 = buf_beads[0];
        x_1 += 3*(n-k);
      } else x_1 += 3;
    }
  }

  double energy_local = spring_energy;
  if(std::isnan(spring_energy)){
    std::cout<< universe->iworld << " " << spring_energy <<" "<<std::endl;
    exit(0);}

  return energy_local;
}

//E_n^(k) is a function of k atoms (R_n-k+1,...,R_n) for a given n and k.
double FixPIMD::Evaluate_Ekn(const int n, const int k)
{
  //bead is the bead number of current replica. bead = 0,...,np-1.
  int bead = universe->iworld;

  double **x = atom->x;
  double* _mass = atom->mass;
  int* type = atom->type;
  int nlocal = atom->nlocal;

  //xnext is a pointer to first element of buf_beads[x_next].
  //See in FixPIMDB::comm_init() for the definition of x_next.
  //x_next is basically (bead + 1) for bead in (0,...,np-2) and 0 for bead = np-1.
  //buf_beads[j] is a 1-D array of length 3*nlocal x0^j,y0^j,z0^j,...,x_(nlocal-1)^j,y_(nlocal-1)^j,z_(nlocal-1)^j.
  double* xnext = buf_beads[x_next];

  //omega^2, could use fbond instead?
  double omega_sq = omega_np*omega_np;

  //E_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
  xnext += 3*(n-k);

  //np is total number of beads
  if(bead == np-1 && k > 1) xnext += 3;

  spring_energy = 0.0;
  for (int i = n-k; i < n ; ++i) {

    /*if(bead==3 && n==2 && k==2) {
        std::cout << "atom " << i + 1 << ", bead" << bead + 1 << ": " << x[i][0] << " " << x[i][1] << " " << x[i][2]
                  << std::endl;
        std::cout << "next " << i + 1 << ", bead" << bead + 1 << ": " << xnext[0] << " " << xnext[1] << " " << xnext[2]
                  << std::endl;
    }*/

    double delx =x[i][0] - xnext[0]; 
    double dely =x[i][1] - xnext[1]; 
    double delz =x[i][2] - xnext[2]; 
    //std::cout<< xnext[0] << std::endl;

    domain->minimum_image(delx, dely, delz);

    if (bead == np - 1 && i == n - 2) {
      /*if(bead==3 && n==2 && k==2) {
          std::cout<<"I AM HERE"<<std::endl;
          std::cout << "next " << i + 1 << ", bead" << bead + 1 << ": " << xnext[0] << " " << xnext[1] << " " << xnext[2]
                    << std::endl;
      }*/
      xnext = buf_beads[x_next];

      /*if(bead==3 && n==2 && k==2) {
          std::cout<<"NOW I AM HERE"<<std::endl;
          std::cout << "next " << i + 1 << ", bead" << bead + 1 << ": " << xnext[0] << " " << xnext[1] << " " << xnext[2]
                    << std::endl;
      }*/
      //std::cout<<bead<<std::endl;
      //std::cout<<  xnext[0] << " " << xnext[1]<< " " << xnext[2] << std::endl;

      xnext += 3*(n - k);
    } else xnext += 3;

    //std::cout << delx << " " <<dely << " " <<  delz << std::endl;
    //std::cout << _mass[type[i]] << " " << omega_sq << " " <<  delx*delx << std::endl;
    spring_energy += 0.5*_mass[type[i]]*omega_sq*(delx*delx + dely*dely + delz*delz);

  }

  double energy_all = 0.0;
  double energy_local = spring_energy;
  //double energy_local = 0.0;
  //if(bead==0 && n==2 && k==2)
      //std::cout<< universe->iworld << " " << spring_energy <<" " << energy_all <<std::endl;

  //MPI_Allreduce(&spring_energy,&energy_local,1,MPI_DOUBLE,MPI_SUM,world);
  //MPI_Allreduce(&energy_local,&energy_all,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
  MPI_Allreduce(MPI_IN_PLACE,&energy_local,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
  if(std::isnan(spring_energy) || std::isnan(energy_all)){
    std::cout<< universe->iworld << " " << spring_energy <<" " << energy_all <<std::endl;
    exit(0);}

  //return energy_all;
  return energy_local;
}

std::vector<std::vector<double>>FixPIMD::Evaluate_dVBn_new(const std::vector<double> &V, const std::vector<double> &save_E_kn, const int n) 
{
  double beta = 1.0 / (boltz * nhc_temp);
  int bead = universe->iworld;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  double sig_denom_m;

  int beta_n; //partitioning beta for the low temp. limit
  int beta_grid=100; //beta grid

  std::vector<std::vector<double>> dV_all(n, std::vector<double>(3,0.0));

  for (int atomnum = 0; atomnum < nlocal; ++atomnum) {
  //for (int k = m-universe->me; k > 0; k-=universe->nprocs) { //fastest
  //for (int atomnum=universe->me; atomnum<nlocal; atomnum+=universe->nprocs) {
    std::vector<std::vector<double>> dV(n+1, std::vector<double>(3,0.0));
    dV.at(0) = {0.0,0.0,0.0};

    for (int m = 1; m < n + 1; ++m) {
      std::vector<double> sig(3,0.0);

      if (atomnum > m-1) {
        dV.at(m) = {0.0,0.0,0.0};
      }else{

        int count = m*(m-1)/2;

        double Emax=std::min((Evaluate_Ekn_new(m,1)+V.at(m-1)), (Evaluate_Ekn_new(m,m)+V.at(0)));
        //we should be careful at parallelizing this part.
        for (int k = m; k > 0; --k) {
        //for (int k = m-universe->me; k > 0; k-=universe->nprocs) { //fastest
          std::vector<double> dE_kn(3,0.0);
          dE_kn = Evaluate_dEkn_on_atom(m,k,atomnum);

          sig.at(0) += (dE_kn.at(0) + dV.at(m - k).at(0))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));
          sig.at(1) += (dE_kn.at(1) + dV.at(m - k).at(1))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));
          sig.at(2) += (dE_kn.at(2) + dV.at(m - k).at(2))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));

          count++;
        }

        beta_n=(int)(abs(beta*(V.at(m)-Emax)))/beta_grid+1;

        sig_denom_m = exp(-beta*(V.at(m)-Emax)/(double)beta_n);

        if(sig_denom_m ==0 || std::isnan(sig_denom_m) || std::isinf(sig_denom_m) || std::isnan(sig.at(0)) || std::isinf(sig.at(0))) {
          if (universe->iworld ==0){
            std::cout << "m is: "<< m << " Emax is: " << Emax << " V.at(m-1) is " <<V.at(m-1)<< " beta is: " << beta << " sig_denom_m is: " <<sig_denom_m << std::endl;
          }
          exit(0);
        }

        for(int ib=0; ib<beta_n; ib++){
          if(ib==0){
            dV.at(m).at(0) = sig.at(0)/sig_denom_m/(double)m;
            dV.at(m).at(1) = sig.at(1)/sig_denom_m/(double)m;
            dV.at(m).at(2) = sig.at(2)/sig_denom_m/(double)m;}
          else{
            dV.at(m).at(0) /= sig_denom_m;
            dV.at(m).at(1) /= sig_denom_m;
            dV.at(m).at(2) /= sig_denom_m;
          }
        }

        if(std::isinf(dV.at(m).at(0)) || std::isnan(dV.at(m).at(0))) {
          if (universe->iworld ==0){
            std::cout << " Elongest is: " << Emax
                      << " V.at(m) is " << V.at(m) << " beta is " << beta << std::endl;}
          exit(0);
        }
      }
    }

    dV_all.at((atomnum)).at(0) = dV.at(n).at(0);
    dV_all.at((atomnum)).at(1) = dV.at(n).at(1);
    dV_all.at((atomnum)).at(2) = dV.at(n).at(2);

    f[atomnum][0] -= dV.at(n).at(0);
    f[atomnum][1] -= dV.at(n).at(1);
    f[atomnum][2] -= dV.at(n).at(2);
  }
  return dV_all;
}


std::vector<std::vector<double>>FixPIMD::Evaluate_dVBn(const std::vector<double> &V, const std::vector<double> &save_E_kn, const int n) 
{
  double beta = 1.0 / (boltz * nhc_temp);
  int bead = universe->iworld;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  double sig_denom_m;

  int beta_n; //partitioning beta for the low temp. limit
  int beta_grid=100; //beta grid

  std::vector<std::vector<double>> dV_all(n, std::vector<double>(3,0.0));
  //std::cout<<dV_all.at(0).size()<<std::endl;

  for (int atomnum = 0; atomnum < nlocal; ++atomnum) {
    std::vector<std::vector<double>> dV(n+1, std::vector<double>(3,0.0));
    dV.at(0) = {0.0,0.0,0.0};

    for (int m = 1; m < n + 1; ++m) {
      std::vector<double> sig(3,0.0);

      if (atomnum > m-1) {
        dV.at(m) = {0.0,0.0,0.0};
      }else{
        int count = m*(m-1)/2;

        //double Elongest = sEl*(save_E_kn.at(m*(m-1)/2)+V.at(m-1));
        //double Elongest = sEl*(save_E_kn.at(0)+V.at(0)+V.at(m));
        //double Elongest = save_E_kn.at(m*(m-1)/2);

//        double Emax = sEl*(Evaluate_Ekn(m,1)+V.at(m-1));
        double Emax=std::min((Evaluate_Ekn(m,1)+V.at(m-1)), (Evaluate_Ekn(m,m)+V.at(0)));
        for (int k = m; k > 0; --k) {
          std::vector<double> dE_kn(3,0.0);
          dE_kn = Evaluate_dEkn_on_atom(m,k,atomnum);

          sig.at(0) += (dE_kn.at(0) + dV.at(m - k).at(0))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));
          sig.at(1) += (dE_kn.at(1) + dV.at(m - k).at(1))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));
          sig.at(2) += (dE_kn.at(2) + dV.at(m - k).at(2))*exp(-beta*(save_E_kn.at(count) + V.at(m - k) - Emax));
 
          //if (universe->me ==0){
          //  printf("E_kn(%d): %e, V(%d-%d): %e, V(%d): %e \n", count, save_E_kn.at(count),m,k, V.at(m - k), m, V.at(m));}
          //if (universe->me ==0) printf("sig: %e \n", sig.at(0));

          count++;
        }

        beta_n=(int)(abs(beta*(V.at(m)-Emax)))/beta_grid+1;
        //if (universe->me ==0)
        //  printf("beta: %d, beta_n: %d, V(m): %f \n", (int)beta, beta_n, V.at(m));

        //partition beta for the sake of numerical stability.
        sig_denom_m = exp(-beta*(V.at(m)-Emax)/(double)beta_n);

        if(sig_denom_m ==0 || std::isnan(sig_denom_m) || std::isinf(sig_denom_m) || std::isnan(sig.at(0)) || std::isinf(sig.at(0))) {
          if (universe->iworld ==0){
            std::cout << "m is: "<< m << " Emax is: " << Emax << " V.at(m-1) is " <<V.at(m-1)<< " beta is: " << beta << " sig_denom_m is: " <<sig_denom_m << std::endl;
          }
          exit(0);
        }

        for(int ib=0; ib<beta_n; ib++){
          if(ib==0){
            dV.at(m).at(0) = sig.at(0)/sig_denom_m/(double)m;
            dV.at(m).at(1) = sig.at(1)/sig_denom_m/(double)m;
            dV.at(m).at(2) = sig.at(2)/sig_denom_m/(double)m;}
          else{
            dV.at(m).at(0) /= sig_denom_m;
            dV.at(m).at(1) /= sig_denom_m;
            dV.at(m).at(2) /= sig_denom_m;
          }
        }

        if(std::isinf(dV.at(m).at(0)) || std::isnan(dV.at(m).at(0))) {
          if (universe->iworld ==0){
            std::cout << " Elongest is: " << Emax
                      << " V.at(m) is " << V.at(m) << " beta is " << beta << std::endl;}
          exit(0);
        }
      }
    }

    /*if(bead==0 &&atomnum==0)
        std::cout <<"atom: " << atomnum+1 <<" bead: "<< bead+1 << " dV: " << dV.at(n).at(0)<<" "<< dV.at(n).at(1)<< " "<<dV.at(n).at(2) <<std::endl;
    */

    //std::cout<<"index: " <<(atomnum)*np + (bead)<<std::endl;
    dV_all.at((atomnum)).at(0) = dV.at(n).at(0);
    dV_all.at((atomnum)).at(1) = dV.at(n).at(1);
    dV_all.at((atomnum)).at(2) = dV.at(n).at(2);

    /*if(bead==0)
        std::cout <<"atom: " << atomnum+1 <<" bead: "<< bead+1 << " fbefore: " << f[atomnum][0]<<" "<< f[atomnum][1]<< " "<<f[atomnum][2] <<std::endl;
    */

    f[atomnum][0] -= dV.at(n).at(0);
    f[atomnum][1] -= dV.at(n).at(1);
    f[atomnum][2] -= dV.at(n).at(2);
    //if (universe->me ==0) printf("spring force: %e \n", dV.at(n).at(0));

    /*if(bead==0)
        std::cout <<"atom: " << atomnum+1 <<" bead: "<< bead+1 << " fafter: " << f[atomnum][0]<<" "<< f[atomnum][1]<< " "<<f[atomnum][2] <<std::endl;
      */
  }
  return dV_all;
}

//dE_n^(k) is a function of k atoms (R_n-k+1,...,R_n) for a given n and k.
std::vector<double> FixPIMD::Evaluate_dEkn_on_atom(const int n, const int k, const int atomnum)
{
  //dE_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
  if (atomnum < n-k or atomnum > n-1 ) { return std::vector<double>(3, 0.0); }
  else {

    //bead is the bead number of current replica. bead = 0,...,np-1.
    int bead = universe->iworld;

    double **x = atom->x;
    double *_mass = atom->mass;
    int *type = atom->type;
    int nlocal = atom->nlocal;

    //xnext is a pointer to first element of buf_beads[x_next].
    //See in FixPIMDB::comm_init() for the definition of x_next.
    //x_next is basically (bead + 1) for bead in (0,...,np-2) and 0 for bead = np-1.
    //buf_beads[j] is a 1-D array of length 3*nlocal x0^j,y0^j,z0^j,...,x_(nlocal-1)^j,y_(nlocal-1)^j,z_(nlocal-1)^j.
    double *xnext = buf_beads[x_next];
    double *xlast = buf_beads[x_last];

    //omega^2, could use fbond instead?
    double omega_sq = omega_np * omega_np;

    //dE_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
    //But derivative if for atom atomnum
    xnext += 3 * (atomnum);
    xlast += 3 * (atomnum);

    //np is total number of beads
    if (bead == np - 1 && k > 1){
      atomnum == n - 1 ? (xnext-= 3*(k - 1)) : (xnext += 3);
    }

    if (bead == 0 && k > 1){
      atomnum == n-k ? (xlast+= 3*(k - 1)) : (xlast -= 3);
    }

    //if (bead == np - 1 && k > 1) xnext += 3;
    //if (bead == 0 && k > 1) xlast -= 3;

    /*
    if(bead==3 && n==1 && k==1) {
      std::cout << "atom " << atomnum + 1 << ", bead" << bead + 1 << ": " << x[atomnum][0] << " " << x[atomnum][1] << " " << x[atomnum][2]
                << std::endl;
      std::cout << "next " << atomnum + 1 << ", bead" << bead + 1 << ": " << xnext[0] << " " << xnext[1] << " " << xnext[2]
                << std::endl;
    }*/

    std::vector<double> res(3);
/*
    std::cout<< "atom " << atomnum+1 << ", bead" << bead + 1 << ": " << x[atomnum][0] << " " << x[atomnum][1] << " " << x[atomnum][2] << std::endl;
    std::cout<< "next " << atomnum+1 << ", bead" << bead + 1 << ": " << xnext[0] << " " << xnext[1] << " " << xnext[2] << std::endl;
    std::cout<< "last " << atomnum+1 << ", bead" << bead + 1 << ": " << xlast[0] << " " << xlast[1] << " " << xlast[2] << std::endl;
*/
    double delx1 = xnext[0] - x[atomnum][0];
    double dely1 = xnext[1] - x[atomnum][1];
    double delz1 = xnext[2] - x[atomnum][2];
    domain->minimum_image(delx1, dely1, delz1);

    double delx2 = xlast[0] - x[atomnum][0];
    //std::cout<<  xnext[0] << std::endl;
    double dely2 = xlast[1] - x[atomnum][1];
    double delz2 = xlast[2] - x[atomnum][2];
    domain->minimum_image(delx2, dely2, delz2);

    double dx = -1.0*(delx1 + delx2);
    double dy = -1.0*(dely1 + dely2);
    double dz = -1.0*(delz1 + delz2);

    //std::cout << delx << " " <<dely << " " <<  delz << std::endl;
    //std::cout << _mass[type[i]] << " " << omega_sq << " " <<  delx*delx << std::endl;
    res.at(0) = _mass[type[atomnum]] * omega_sq * dx;
    res.at(1) = _mass[type[atomnum]] * omega_sq * dy;
    res.at(2) = _mass[type[atomnum]] * omega_sq * dz;

    //std::cout << bead << ": " << res.at(0) << " " << res.at(1) << " " << res.at(2) << std::endl;

    return res;
  }
}

void FixPIMD::observe_Pc_longest()
{
  double beta = 1.0 / (boltz * nhc_temp);
  int n=atom->natoms;
  double degen=1./(double)(n);
  Pc_longest=degen*exp(-beta*(Evaluate_Ekn(n,n)-V.at(n)));
} 
