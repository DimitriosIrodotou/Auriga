
#ifndef ALLVARS_H
#define ALLVARS_H

#include <stdio.h>

/* multi-dim array sizes */

#define NUM_TYPES 7
#define NUM_PHOTO 8

#ifdef SUBFIND_EXTRA_TNG
#define NUM_METALS 10
#else
#define NUM_METALS 9
#endif

#ifndef LONGIDS
typedef unsigned int MyIDType;
#define H5T_MYIDTYPE H5T_NATIVE_UINT
#else
typedef unsigned long long int MyIDType;
#define H5T_MYIDTYPE H5T_NATIVE_UINT64
#endif


/* globals */

extern int  LastSnapShotNr;
extern int  FirstSnapShotNr;
extern int  SnapSkipFac;

extern char TreeOutputDir[512];
extern char RunOutputDir[512];
extern char SnapshotFileBase[512];

extern int TotHalos;
extern int NtreesPerFile;
extern int NhalosPerFile;

extern double ParticleMass; 

extern int  NumberOfOutputFiles;
extern int  MultipleFilesPerSnap; // -1 if unset, 0 if NumFilesPerSnapshot=1, 1 if true

extern int    *FirstHaloInSnap;

#ifdef SKIP_SNAP
#define MAXLEN_OUTPUTLIST 300

extern int OutputListLength;
extern double OutputListTimes[MAXLEN_OUTPUTLIST];
extern int OutputListFlag[MAXLEN_OUTPUTLIST];

extern char OutputList[512];
#endif


/* structs */

extern struct halo_catalogue
{
  int TotNsubhalos;
  int TotNgroups;
  float redshift;
} *Cats;


extern struct halo_data
{
  int Descendant;
  int FirstProgenitor;
  int NextProgenitor;
  int FirstHaloInFOFgroup;
  int NextHaloInFOFgroup;

  int SubhaloSnapFileNr;
  int SnapNum;
  int FileNr;
  int SubhaloIndex;

#ifdef IMPROVE_POINTERS
  float MassInStem;
#endif
  float Group_M_Mean200;
  float Group_M_Crit200;
  float Group_R_Crit200;
  float Group_M_TopHat200;
  int GroupLenType[NUM_TYPES];

  float SubhaloPos[3];
  float SubhaloVel[3];
  float SubhaloVelDisp;
  float SubhaloVmax;
  float SubhaloSpin[3];
  MyIDType SubhaloIDMostBound;
  float SubhaloMassType[NUM_TYPES];
  
  float GroupCM[3];
  float GroupMassType[NUM_TYPES];
  float Group_R_Mean200;
  float Group_R_TopHat200;

  int SubhaloGrNr;

  int SubhaloLen;
  int SubhaloLenType[NUM_TYPES];

  MyIDType GroupOffsetType[NUM_TYPES];
  MyIDType SubhaloOffsetType[NUM_TYPES];

#ifdef SUBFIND_EXTRA
  float SubhaloGasMetallicity;
  float SubhaloGasMetallicitySfr;
  float SubhaloStarMetallicity;
  float SubhaloBHMass;
  float SubhaloBHMdot;
  float SubhaloSFR;
  float SubhaloSFRinRad;
  float SubhaloStellarPhotometrics[NUM_PHOTO];
  float SubhaloMassInRadType[NUM_TYPES]; 
  float SubhaloHalfmassRadType[NUM_TYPES];
#endif

#ifdef SUBFIND_EXTRA_TNG
  float GroupCM[3];
  float GroupMassType[NUM_TYPES];
  float Group_M_Crit500;  
  float Group_R_Crit500;
  float Group_R_Mean200;
  float Group_R_TopHat200;

  //SubhaloGasMetallicityHalfRad {N}
  //SubhaloGasMetallicityMaxRad {N}
  //SubhaloGasMetallicitySfrWeighted {N}
  //SubhaloStellarPhotometricsMassInRad {N}
  //SubhaloStellarPhotometricsRad {N}
  //SubhaloMassInHalfRadType {N, 6}
  //SubhaloMassInMaxRadType {N, 6}
  //SubhaloStarMetallicityHalfRad {N}
  //SubhaloStarMetallicityMaxRad {N}
  //SubhaloSFRinHalfRad {N}
  //SubhaloSFRinMaxRad {N}
  //SubhaloVmaxRad {N}

  float SubhaloHalfmassRad;

  float SubhaloGasMetalFractions[NUM_METALS];
  float SubhaloGasMetalFractionsSfr[NUM_METALS];
  //SubhaloGasMetalFractionsHalfRad {N, 10}
  //SubhaloGasMetalFractionsMaxRad {N, 10}
  //SubhaloGasMetalFractionsSfrWeighted {N, 10}

  float SubhaloStarMetalFractions[NUM_METALS];
  //SubhaloStarMetalFractionsHalfRad {N, 10}
  //SubhaloStarMetalFractionsMaxRad {N, 10}

  float SubhaloBfldDisk;
  float SubhaloBfldHalo;
#endif

} *Halo, *HaloList;


extern struct halo_aux_data
{
  int UsedFlag;
  int FileNr;
  int TargetIndex;
  int Origin;
  int HaloFlag;
#ifdef IMPROVE_POINTERS
  int DoneFlag;
  float MassInStem;
#endif


}  *HaloAux;


enum iofields
{
  IO_DESCENDANT,
  IO_FIRSTPROGENITOR,
  IO_NEXTPROGENITOR,
  IO_FIRSTHALOINFOFGROUP,
  IO_NEXTHALOINFOFGROUP,
  IO_LEN,
  IO_M_MEAN200,
  IO_M_CRIT200,
  IO_M_TOPHAT,
  IO_POS,
  IO_VEL,
  IO_VELDISP,
  IO_VMAX,
  IO_SPIN,
  IO_IDMOSTBOUND,
  IO_SnapNum,
  IO_FILENR,
  IO_GROUPNR,
  IO_SUBHALO_NR,
  IO_SUBHALO_MASS_TYPE,
  IO_SUBHALO_MASSINRAD_TYPE,
  IO_SUBHALO_HALFMASSRAD_TYPE,

  IO_SUBHALO_OFFSET_TYPE,
  IO_SUBHALO_LEN_TYPE,

  /* SUBFIND_EXTRA: */
  IO_SUBHALO_SFR,
  IO_SUBHALO_GAS_METALLICITY,
  IO_SUBHALO_GAS_METALLICITY_SFR,
  IO_SUBHALO_STELLAR_METALLICITY,
  IO_SUBHALO_BH_MASS,
  IO_SUBHALO_BH_MDOT,
  IO_SUBHALO_SFR_IN_RAD,
  IO_SUBHALO_STELLAR_PHOTOMETRICS,

  /* SUBFIND_EXTRA_TNG: */
  IO_GROUP_CM,
  IO_GROUP_MASS_TYPE,
  IO_GROUP_M_CRIT500,
  IO_GROUP_R_MEAN200,
  IO_GROUP_R_CRIT200,
  IO_GROUP_R_CRIT500,
  IO_GROUP_R_TOPHAT200,

  IO_SUBHALO_HALFMASSRAD,
  IO_SUBHALO_GAS_METAL_FRACTIONS,
  IO_SUBHALO_GAS_METAL_FRACTIONS_SFR,
  IO_SUBHALO_STAR_METAL_FRACTIONS,
  IO_SUBHALO_BFLD_DISK,
  IO_SUBHALO_BFLD_HALO,

  IO_LASTENTRY
};

#endif
