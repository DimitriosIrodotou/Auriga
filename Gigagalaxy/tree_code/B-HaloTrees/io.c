#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include "allvars.h"
#include "proto.h"


void my_hdf5_write_tree(void *ptr, int treenr, int countnr, hid_t handle)
{

  char buf[500];
  hid_t hdf5_tree_grp[1], hdf5_datatype=0, hdf5_dataspace_in_file, hdf5_dataset, hdf5_dataspace_memory, hdf5_status;
  hsize_t dims[2], count[2], start[2];
  int rank = 0, reg_datatype=0, blocknr=0, nblocks=1000, * IntCommBuffer;
  float* FltCommBuffer;
  MyIDType* IDCommBuffer;

  sprintf(buf, "/Tree%d", treenr);
  hdf5_tree_grp[0] = H5Gcreate(handle, buf, 0);

  dims[0]  = countnr;
  dims[1]  = 1;     // gets reset inside loop
  start[0] = 0;     // would be pcsum if writing many times
  start[1] = 0;
  count[0] = countnr;
  count[1] = 1;     // gets reset inside loop

  for(blocknr=0; blocknr<nblocks; blocknr++)
  {
    if(blocknr == IO_LASTENTRY)
      break;

    if( !get_dataset_info(blocknr, buf, dims, count, &rank, &hdf5_datatype, &reg_datatype) )
      continue;

    hdf5_dataspace_in_file  = H5Screate_simple(rank, dims, NULL);
    hdf5_dataspace_memory   = H5Screate_simple(rank, dims, NULL);
    hdf5_dataset    = H5Dcreate(hdf5_tree_grp[0], buf, hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT);
    H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

    if(reg_datatype==1)
      {
        IntCommBuffer = malloc(  countnr * sizeof(int) * dims[1] );
        memset(IntCommBuffer, 0, countnr * sizeof(int) * dims[1] );
        get_dataset_int_data(blocknr, IntCommBuffer, countnr);
        hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, IntCommBuffer);
        free(IntCommBuffer);
      }
    else
      {
        if(reg_datatype==2)
          {
            FltCommBuffer = malloc(  countnr * sizeof(float) * dims[1] );
            memset(FltCommBuffer, 0, countnr * sizeof(float) * dims[1] );
            get_dataset_flt_data(blocknr, FltCommBuffer, countnr);
            hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, FltCommBuffer);
            free(FltCommBuffer);
          }
        else
          {
            IDCommBuffer = malloc(  countnr * sizeof(MyIDType) * dims[1] );
            memset(IDCommBuffer, 0, countnr * sizeof(MyIDType) * dims[1] );
            get_dataset_ID_data(blocknr, IDCommBuffer, countnr);
            hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, IDCommBuffer);
            free(IDCommBuffer);
          }
      }
  
    if(hdf5_status < 0)
      printf("write error!!!\n");

    H5Sclose(hdf5_dataspace_memory);
    H5Dclose(hdf5_dataset);
    H5Sclose(hdf5_dataspace_in_file);
    H5Tclose(hdf5_datatype);
  }

  H5Gclose(hdf5_tree_grp[0]);

}



void write_header_attributes_in_hdf5(hid_t handle, int *npertree)
{
  hid_t hdf5_dataspace, hdf5_attribute, atype;

  hdf5_dataspace = H5Screate(H5S_SCALAR);

  atype = H5Tcopy(H5T_C_S1);

  H5Tset_size(atype, strlen(RunOutputDir));
  hdf5_attribute = H5Acreate(handle, "RunOutputDir", atype, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, atype, &RunOutputDir);
  H5Aclose(hdf5_attribute);

  H5Tset_size(atype, strlen(TreeOutputDir));
  hdf5_attribute = H5Acreate(handle, "TreeOutputDir", atype, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, atype, &TreeOutputDir);
  H5Aclose(hdf5_attribute);

  H5Tset_size(atype, strlen(SnapshotFileBase));
  hdf5_attribute = H5Acreate(handle, "SnapshotFileBase", atype, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, atype, &SnapshotFileBase);
  H5Aclose(hdf5_attribute);

  H5Tclose(atype);


  hdf5_attribute = H5Acreate(handle, "FirstSnapshotNr", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &FirstSnapShotNr);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "LastSnapshotNr", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &LastSnapShotNr);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "SnapSkipFac", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &SnapSkipFac);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "NumberOfOutputFiles", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &NumberOfOutputFiles);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "NtreesPerFile", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &NtreesPerFile);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "NhalosPerFile", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &NhalosPerFile);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Acreate(handle, "ParticleMass", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT);
  H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &ParticleMass);
  H5Aclose(hdf5_attribute);

#ifdef SUBFIND_EXTRA_TNG
  hdf5_attribute = H5Acreate(handle, "AlphaExponent", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT);
  double alpha = 0.8; // just the hard-coded value used by dnelson for TNG trees
  H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &alpha);
  H5Aclose(hdf5_attribute);
#endif

  H5Sclose(hdf5_dataspace);

  hsize_t dims[1];
  hid_t hdf5_datatype=0, hdf5_dataspace_in_file, hdf5_dataset, hdf5_dataspace_memory;
  int int_tmp_arr[LastSnapShotNr + 1];
  double dbl_tmp_arr[LastSnapShotNr + 1];
  int i;

  dims[0] = NtreesPerFile;
  hdf5_dataspace_in_file = H5Screate_simple(1, dims, NULL);
  hdf5_dataspace_memory  = H5Screate_simple(1, dims, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
  hdf5_dataset  = H5Dcreate(handle, "TreeNHalos", hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT);

  H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, npertree);

  H5Sclose(hdf5_dataspace_memory);
  H5Dclose(hdf5_dataset);
  H5Sclose(hdf5_dataspace_in_file);
  H5Tclose(hdf5_datatype);

  dims[0] = LastSnapShotNr + 1;
  hdf5_dataspace_in_file = H5Screate_simple(1, dims, NULL);
  hdf5_dataspace_memory  = H5Screate_simple(1, dims, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
  hdf5_dataset  = H5Dcreate(handle, "TotNsubhalos", hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT);

  for (i = 0; i < LastSnapShotNr + 1; i++)
    int_tmp_arr[i] = Cats[i].TotNsubhalos;
  H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmp_arr);

  H5Sclose(hdf5_dataspace_memory);
  H5Dclose(hdf5_dataset);
  H5Sclose(hdf5_dataspace_in_file);
  H5Tclose(hdf5_datatype);

  dims[0] = LastSnapShotNr + 1;
  hdf5_dataspace_in_file = H5Screate_simple(1, dims, NULL);
  hdf5_dataspace_memory  = H5Screate_simple(1, dims, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
  hdf5_dataset  = H5Dcreate(handle, "Redshifts", hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT);

  for (i = 0; i < LastSnapShotNr + 1; i++)
    dbl_tmp_arr[i] = Cats[i].redshift;
  H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, dbl_tmp_arr);

  H5Sclose(hdf5_dataspace_memory);
  H5Dclose(hdf5_dataset);
  H5Sclose(hdf5_dataspace_in_file);
  H5Tclose(hdf5_datatype);
}



/*! This catches I/O errors occuring for fread(). In this case we
 *  better stop.
 */
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nread;

  if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
    {
      if(feof(stream))
        printf("I/O error (fread) has occured: end of file\n");
      else
        printf("I/O error (fread) has occured: %s\n", strerror(errno));
      fflush(stdout);
      exit(778);
    }
  return nread;
}


int get_dataset_info(enum iofields blocknr, char *buf, hsize_t *dims, hsize_t *count, int *rank, hid_t *hdf5_datatype, int *reg_datatype)
{
  strcpy(buf, "default");

  switch (blocknr)
    {
    case IO_DESCENDANT:
      strcpy(buf, "Descendant");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_FIRSTPROGENITOR:
      strcpy(buf, "FirstProgenitor");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_NEXTPROGENITOR:
      strcpy(buf, "NextProgenitor");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_FIRSTHALOINFOFGROUP:
      strcpy(buf, "FirstHaloInFOFGroup");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_NEXTHALOINFOFGROUP:
      strcpy(buf, "NextHaloInFOFGroup");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_LEN:
      strcpy(buf, "SubhaloLen");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_M_MEAN200:
      strcpy(buf, "Group_M_Mean200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_M_CRIT200:
      strcpy(buf, "Group_M_Crit200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_GROUP_R_CRIT200:
      strcpy(buf, "Group_R_Crit200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_M_TOPHAT:
      strcpy(buf, "Group_M_TopHat200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;

    /* SUBFIND_EXTRA_TNG (HALO): */
    case IO_GROUP_CM:
      strcpy(buf, "GroupCM");
      dims[1] = 3;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_GROUP_MASS_TYPE:
      strcpy(buf, "GroupMassType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_GROUP_M_CRIT500:
      strcpy(buf, "Group_M_Crit500");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_GROUP_R_CRIT500:
      strcpy(buf, "Group_R_Crit500");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_GROUP_R_MEAN200:
      strcpy(buf, "Group_R_Mean200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_GROUP_R_TOPHAT200:
      strcpy(buf, "Group_R_TopHat200");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;

    case IO_POS:
      strcpy(buf, "SubhaloPos");
      dims[1] = 3;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_VEL:
      strcpy(buf, "SubhaloVel");
      dims[1] = 3;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_VELDISP:
      strcpy(buf, "SubhaloVelDisp");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_VMAX:
      strcpy(buf, "SubhaloVMax");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_SPIN:
      strcpy(buf, "SubhaloSpin");
      dims[1] = 3;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    case IO_IDMOSTBOUND:
      strcpy(buf, "SubhaloIDMostBound");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_MYIDTYPE);
      *reg_datatype  = 3;
      break;
    case IO_SnapNum:
      strcpy(buf, "SnapNum");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_FILENR:
      strcpy(buf, "FileNr");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_GROUPNR:
      strcpy(buf, "SubhaloGrNr");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_SUBHALO_NR:       // also add subhalo number within the group
      strcpy(buf, "SubhaloNumber");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;

    case IO_SUBHALO_OFFSET_TYPE:
      strcpy(buf, "SubhaloOffsetType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_MYIDTYPE);
      *reg_datatype  = 3;
      break;
    case IO_SUBHALO_LEN_TYPE:
      strcpy(buf, "SubhaloLenType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
      *reg_datatype  = 1;
      break;
    case IO_SUBHALO_MASS_TYPE:
      strcpy(buf, "SubhaloMassType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
      break;
    /* SUBFIND_EXTRA: */
    case IO_SUBHALO_MASSINRAD_TYPE:
      strcpy(buf, "SubhaloMassInRadType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_HALFMASSRAD_TYPE:
      strcpy(buf, "SubhaloHalfmassRadType");
      dims[1] = NUM_TYPES;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_SFR:
      strcpy(buf, "SubhaloSFR");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_GAS_METALLICITY:
      strcpy(buf, "SubhaloGasMetallicity");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_GAS_METALLICITY_SFR:
      strcpy(buf, "SubhaloGasMetallicitySfr");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_STELLAR_METALLICITY:
      strcpy(buf, "SubhaloStarMetallicity");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_BH_MASS:
      strcpy(buf, "SubhaloBHMass");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_BH_MDOT:
      strcpy(buf, "SubhaloBHMdot");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_SFR_IN_RAD: 
      strcpy(buf, "SubhaloSFRinRad");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;
    case IO_SUBHALO_STELLAR_PHOTOMETRICS:
      strcpy(buf, "SubhaloStellarPhotometrics");
      dims[1] = NUM_PHOTO;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA
      return 0;
#endif
      break;

    /* SUBFIND_EXTRA_TNG: */

    case IO_SUBHALO_HALFMASSRAD:
      strcpy(buf, "SubhaloHalfmassRad");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;

    case IO_SUBHALO_GAS_METAL_FRACTIONS:
      strcpy(buf, "SubhaloGasMetalFractions");
      dims[1] = NUM_METALS;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_SUBHALO_GAS_METAL_FRACTIONS_SFR:
      strcpy(buf, "SubhaloGasMetalFractionsSfr");
      dims[1] = NUM_METALS;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_SUBHALO_STAR_METAL_FRACTIONS:
      strcpy(buf, "SubhaloStarMetalFractions");
      dims[1] = NUM_METALS;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_SUBHALO_BFLD_DISK:
      strcpy(buf, "SubhaloBfldDisk");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;
    case IO_SUBHALO_BFLD_HALO:
      strcpy(buf, "SubhaloBfldHalo");
      dims[1] = 1;
      *hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      *reg_datatype  = 2;
#ifndef SUBFIND_EXTRA_TNG
      return 0;
#endif
      break;

    case IO_LASTENTRY:
      break;
    }
  
    count[1] = dims[1]; //get_values_per_blockelement(blocknr);

    if(dims[1] == 1)
      *rank = 1;
    else
      *rank = 2;
   
    return 1; // dataset exists
}


void get_dataset_int_data(enum iofields blocknr, int *buf, int countnr)
{
  int i;
  int k;

  switch (blocknr)
    {
    case IO_DESCENDANT:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Descendant;
      break;
    case IO_FIRSTPROGENITOR:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].FirstProgenitor;
      break;
    case IO_NEXTPROGENITOR:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].NextProgenitor;
      break;
    case IO_FIRSTHALOINFOFGROUP:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].FirstHaloInFOFgroup;
      break;
    case IO_NEXTHALOINFOFGROUP:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].NextHaloInFOFgroup;
      break;
    case IO_LEN:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloLen;
      break;
    case IO_SnapNum:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SnapNum;
      break;
    case IO_FILENR:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].FileNr;
      break;
    case IO_GROUPNR:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloGrNr;
      break;
    case IO_SUBHALO_NR:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloIndex;
      break;
    case IO_SUBHALO_LEN_TYPE:
      for(i=0; i<countnr; i++)
        for(k=0; k<NUM_TYPES; k++)
          buf[i*NUM_TYPES+k] = HaloList[i].SubhaloLenType[k];
      break;

    default:
      break;
    }
}



void get_dataset_flt_data(enum iofields blocknr, float *buf, int countnr)
{
  int i;
  int k;

  switch (blocknr)
    {
    case IO_M_MEAN200:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_M_Mean200;
      break;
    case IO_M_CRIT200:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_M_Crit200;
      break;
    case IO_M_TOPHAT:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_M_TopHat200;
      break;
    case IO_GROUP_R_CRIT200:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_R_Crit200;
      break;

    /* SUBFIND_EXTRA_TNG (HALO): */
    case IO_GROUP_CM:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        for(k = 0; k < 3; k++)
          buf[i*3+k] = HaloList[i].GroupCM[k];
      break;
#endif
    case IO_GROUP_MASS_TYPE:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        for(k = 0; k < NUM_TYPES; k++)
          buf[i*NUM_TYPES+k] = HaloList[i].GroupMassType[k];
      break;
#endif
    case IO_GROUP_M_CRIT500:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_M_Crit500;
      break;
#endif
    case IO_GROUP_R_CRIT500:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_R_Crit500;
      break;
#endif
    case IO_GROUP_R_MEAN200:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_R_Mean200;
      break;
#endif
    case IO_GROUP_R_TOPHAT200:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].Group_R_TopHat200;
      break;
#endif


    case IO_POS:
      for(i=0; i<countnr; i++)
        for(k=0; k<3; k++)
          buf[i*3+k] = HaloList[i].SubhaloPos[k];
      break;
    case IO_VEL:
      for(i=0; i<countnr; i++)
        for(k=0; k<3; k++)
          buf[i*3+k] = HaloList[i].SubhaloVel[k];
      break;
    case IO_VELDISP:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloVelDisp;
      break;
    case IO_VMAX:
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloVmax;
      break;
    case IO_SPIN:
      for(i=0; i<countnr; i++)
        for(k=0; k<3; k++)
          buf[i*3+k] = HaloList[i].SubhaloSpin[k];
      break;
    case IO_SUBHALO_MASS_TYPE:
      for(i=0; i<countnr; i++)
        for(k=0; k<NUM_TYPES; k++)
          buf[i*NUM_TYPES+k] = HaloList[i].SubhaloMassType[k];
      break;

    /* SUBFIND_EXTRA: */

    case IO_SUBHALO_MASSINRAD_TYPE:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        for(k=0; k<NUM_TYPES; k++)
          buf[i*NUM_TYPES+k] = HaloList[i].SubhaloMassInRadType[k];
#endif
      break;
    case IO_SUBHALO_HALFMASSRAD_TYPE:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        for(k=0; k<NUM_TYPES; k++)
          buf[i*NUM_TYPES+k] = HaloList[i].SubhaloHalfmassRadType[k];
#endif
      break;
    case IO_SUBHALO_SFR:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloSFR;
#endif
      break;
    case IO_SUBHALO_GAS_METALLICITY:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloGasMetallicity;
#endif
      break;
    case IO_SUBHALO_GAS_METALLICITY_SFR:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloGasMetallicitySfr;
#endif
      break;
    case IO_SUBHALO_STELLAR_METALLICITY:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloStarMetallicity;
#endif
      break;
    case IO_SUBHALO_BH_MASS: 
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloBHMass;
#endif
      break;
    case IO_SUBHALO_BH_MDOT:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloBHMdot;
#endif
      break;
    case IO_SUBHALO_SFR_IN_RAD:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloSFRinRad;
#endif
      break;
    case IO_SUBHALO_STELLAR_PHOTOMETRICS:
#ifdef SUBFIND_EXTRA
      for(i=0; i<countnr;i++)
        for(k=0;k<NUM_PHOTO;k++)
        buf[i*NUM_PHOTO+k] = HaloList[i].SubhaloStellarPhotometrics[k];
      break;
#endif


    /* SUBFIND_EXTRA_TNG: */

    case IO_SUBHALO_HALFMASSRAD:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloHalfmassRad;
      break;
#endif
    case IO_SUBHALO_GAS_METAL_FRACTIONS:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr;i++)
        for(k=0;k<NUM_METALS;k++)
        buf[i*NUM_METALS+k] = HaloList[i].SubhaloGasMetalFractions[k];
      break;
#endif
    case IO_SUBHALO_GAS_METAL_FRACTIONS_SFR:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr;i++)
        for(k=0;k<NUM_METALS;k++)
        buf[i*NUM_METALS+k] = HaloList[i].SubhaloGasMetalFractionsSfr[k];
      break;
#endif
    case IO_SUBHALO_STAR_METAL_FRACTIONS:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr;i++)
        for(k=0;k<NUM_METALS;k++)
        buf[i*NUM_METALS+k] = HaloList[i].SubhaloStarMetalFractions[k];
      break;
#endif
    case IO_SUBHALO_BFLD_DISK:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloBfldDisk;
      break;
#endif
    case IO_SUBHALO_BFLD_HALO:
#ifdef SUBFIND_EXTRA_TNG
      for(i=0; i<countnr; i++)
        buf[i] = HaloList[i].SubhaloBfldHalo;
      break;
#endif

    default:
      break;

    }
}



void get_dataset_ID_data(enum iofields blocknr, MyIDType *buf, int countnr)
{
  int i, k;

  switch (blocknr)
    {
      case IO_IDMOSTBOUND:
        for(i = 0; i < countnr; i++)
          buf[i] = HaloList[i].SubhaloIDMostBound;
        break;
        
      case IO_SUBHALO_OFFSET_TYPE:
        for(i = 0; i < countnr; i++)
          for(k = 0; k < NUM_TYPES; k++)
            buf[i*NUM_TYPES+k] = HaloList[i].SubhaloOffsetType[k];
        break;

      default:
        break;
    }
}

/*! Returns either a group catalog or snapshot filename, accounting for
 *  the two cases that there are, or are not, multiple files per snapshot.
 *  group_flag=0 for group, group_flag=1 for snap
 */
void get_filename(char *buf, int file_num, int chunk_num, int group_flag)
{
  // if we do not yet known NumFilesPerSnapshot, decide now
  if(MultipleFilesPerSnap < 0)
  {
    sprintf(buf, "%s/fof_subhalo_tab_%03d.hdf5", RunOutputDir, file_num);
  
    if(access(buf, F_OK) == 0)
      MultipleFilesPerSnap = 0;
    else
      MultipleFilesPerSnap = 1;
  }
        
  if(!MultipleFilesPerSnap)
  {
    // one file per group catalog/snapshot
    if(group_flag)
      sprintf(buf, "%s/fof_subhalo_tab_%03d.hdf5", RunOutputDir, file_num);
    else
      sprintf(buf, "%s/%s_%03d.hdf5", RunOutputDir, SnapshotFileBase, file_num);
  }
  else
  {
    // multiple files per group catalog/snapshot
    if(group_flag)
      sprintf(buf, "%s/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5", 
              RunOutputDir, file_num, file_num, chunk_num);
    else
      sprintf(buf, "%s/snapdir_%03d/%s_%03d.%d.hdf5", 
              RunOutputDir, file_num, SnapshotFileBase, file_num, chunk_num);
  }
}
