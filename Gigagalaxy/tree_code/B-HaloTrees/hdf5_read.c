#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"
  
#include <hdf5.h>


void read_hdf5_subhalo_header(const char *fname, int num, struct halo_catalogue *cat , 
	int * nFiles , long long int * totNids, int * nids, int * nsubhalos, int * ngroups)
{
      hid_t hdf5_file, hdf5_grp, hdf5_attribute;

      hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      hdf5_grp = H5Gopen(hdf5_file, "/Header");

      hdf5_attribute = H5Aopen_name(hdf5_grp, "NumFiles");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, nFiles);
      H5Aclose(hdf5_attribute);

      hdf5_attribute = H5Aopen_name(hdf5_grp, "Ngroups_Total");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, &cat[0].TotNgroups);
      H5Aclose(hdf5_attribute);

      hdf5_attribute = H5Aopen_name(hdf5_grp, "Nids_Total");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT64, totNids);
      H5Aclose(hdf5_attribute);

      hdf5_attribute = H5Aopen_name(hdf5_grp, "Nsubgroups_Total");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, &cat[0].TotNsubhalos);
      H5Aclose(hdf5_attribute);
     
      hdf5_attribute = H5Aopen_name(hdf5_grp, "Nids_ThisFile");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, nids);
      H5Aclose(hdf5_attribute);
      
      hdf5_attribute = H5Aopen_name(hdf5_grp, "Nsubgroups_ThisFile");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, nsubhalos);
      H5Aclose(hdf5_attribute);
         
      hdf5_attribute = H5Aopen_name(hdf5_grp, "Ngroups_ThisFile");
      H5Aread(hdf5_attribute, H5T_NATIVE_INT, ngroups);
      H5Aclose(hdf5_attribute);
  
      H5Gclose(hdf5_grp);
      H5Fclose(hdf5_file);
}


void read_hdf5_myidtype_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, MyIDType *data)
{
  hid_t hdf5_file, hdf5_grp;

  int rank;
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];
  MyIDType *tmpptr;

  hdf5_file     = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_grp	= H5Gopen(hdf5_file, buf1);
  hdf5_dataset  = H5Dopen(hdf5_grp,  buf2);

  dims[0] = dim1;
  dims[1] = dim2;
  count[0] = dim1;
  count[1] = dim2;
  start[0] = 0;
  start[1] = 0;

  if(dims[1] == 1)
    rank = 1;
  else
    rank = 2;

  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

#ifdef LONGIDS
  hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
#else
  hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
#endif

  tmpptr    = mymalloc(dims[0] * dims[1] * sizeof(MyIDType));

  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, tmpptr);

  for(rank=0 ; rank < dims[0]*dims[1] ; rank++)
    data[rank] = tmpptr[rank];

  myfree(tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
  H5Gclose(hdf5_grp);

  H5Fclose(hdf5_file);

}





void read_hdf5_float_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, float *data)
{

  hid_t hdf5_file, hdf5_grp;

  int rank;
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];
  double *tmpptr;

  hdf5_file     = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_grp	= H5Gopen(hdf5_file, buf1);
  hdf5_dataset  = H5Dopen(hdf5_grp,  buf2);

  dims[0] = dim1;
  dims[1] = dim2;
  count[0] = dim1;
  count[1] = dim2;
  start[0] = 0;
  start[1] = 0;

  if(dims[1] == 1)
    rank = 1;
  else
    rank = 2;
  
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);
  
  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  
  hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
  tmpptr    = mymalloc(dims[0] * dims[1] * sizeof(double));
  for(rank=0 ; rank < dims[0]*dims[1] ; rank++)
    tmpptr[rank] = 0;
  
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, tmpptr);
      
  for(rank=0 ; rank < dims[0]*dims[1] ; rank++)
      data[rank] = tmpptr[rank];

  myfree(tmpptr);
              
  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
  H5Gclose(hdf5_grp);
                                                                   
  H5Fclose(hdf5_file);
}



void read_hdf5_int_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, int *data)
{ 
  
  hid_t hdf5_file, hdf5_grp;
  
  int rank;                                                        
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];
  int *tmpptr;

  hdf5_file     = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_grp	= H5Gopen(hdf5_file, buf1);
  hdf5_dataset  = H5Dopen(hdf5_grp,  buf2);

  dims[0] = dim1;
  dims[1] = dim2;
  count[0] = dim1;
  count[1] = dim2;
  start[0] = 0;
  start[1] = 0;

  if(dims[1] == 1)
    rank = 1;
  else
    rank = 2;

  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);
  tmpptr    =  mymalloc(dims[0] * dims[1] * sizeof(int));
                 
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, tmpptr);
                  
  for(rank=0 ; rank < dims[0]*dims[1] ; rank++)
    data[rank] = tmpptr[rank];
              
  myfree(tmpptr);
              
  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
  H5Gclose(hdf5_grp);

  H5Fclose(hdf5_file);

}



void read_snap_header_attributes_in_hdf5(int num)
{

  char fname[255];
  hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;

  get_filename(fname, num, 0, 0);

  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_headergrp = H5Gopen(hdf5_file, "/Header");

  double massarr[7];
  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "MassTable");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &massarr[0]);
  H5Aclose(hdf5_attribute);
  
  double redshift;
  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Redshift");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &redshift);
  H5Aclose(hdf5_attribute);
  
  H5Gclose(hdf5_headergrp);
  H5Fclose(hdf5_file);

  ParticleMass = massarr[1];
  Cats[num].redshift = redshift;
}


