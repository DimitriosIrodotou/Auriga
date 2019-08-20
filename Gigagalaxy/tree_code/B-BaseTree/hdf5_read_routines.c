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


#include "allvars.h"
#include "proto.h"

#include <hdf5.h>
int             *int_tmpptr;
long int        *long_int_tmpptr;
long long int   *long_long_int_tmpptr;

MyIDType	* myIDType_tmpptr;


void write_id_translation_table_hdf5( MyIDType * IdSnapTable, long long TotNumPart)
{
      char buf[1000];
      sprintf(buf, "%s/sorted_id_table_%03d.hdf5", TreeOutputDir, LastSnapShotNr);

      hid_t hdf5_file = 0, hdf5_dataspace_in_file, hdf5_dataspace_memory, hdf5_dataset, hdf5_status, hdf5_datatype, hdf5_dataspace, hdf5_attribute;

      hdf5_file = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      hsize_t dims[2], count[2], start[2];
      int rank = 0;


      dims[0]  = TotNumPart;
      dims[1]  = 1;
      start[0] = 0;
      start[1] = 0;
      count[0] = TotNumPart;
      count[1] = 1;

      rank = 1;

      strcpy(buf, "IdSnapTable");
      hdf5_datatype               = H5Tcopy(H5T_NATIVE_UINT64);
      hdf5_dataspace_in_file      = H5Screate_simple(rank, dims, NULL);
      hdf5_dataspace_memory       = H5Screate_simple(rank, dims, NULL);
      hdf5_dataset                = H5Dcreate(hdf5_file, buf, hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT);
      H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

      hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file, H5P_DEFAULT, IdSnapTable);

      if(hdf5_status < 0)
        printf("failure!\n");

      H5Sclose(hdf5_dataspace_memory);
      H5Dclose(hdf5_dataset);
      H5Sclose(hdf5_dataspace_in_file);
      H5Tclose(hdf5_datatype);


      hdf5_dataspace = H5Screate(H5S_SCALAR);
      hdf5_attribute = H5Acreate(hdf5_file, "TotNumPart", H5T_NATIVE_UINT64, hdf5_dataspace, H5P_DEFAULT);
      H5Awrite(hdf5_attribute, H5T_NATIVE_UINT64, &TotNumPart);
      H5Aclose(hdf5_attribute);
      H5Sclose(hdf5_dataspace);


      H5Fclose(hdf5_file);
}

void read_num_part_table_hdf5(const char *fname, long long *TotNumPart )
{
  hid_t hdf5_file, hdf5_attribute;

  hdf5_file 		= H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_attribute 	= H5Aopen_name(hdf5_file, "TotNumPart");
  H5Aread(hdf5_attribute, H5T_NATIVE_UINT64, TotNumPart);
  H5Aclose(hdf5_attribute);
  H5Fclose(hdf5_file);
}





void read_id_translation_table_hdf5(const char *fname, long long TotNumPart, MyIDType *IdSnapTable )
{
  hid_t hdf5_file;
  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  int rank;
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];

  dims[0] = TotNumPart;
  dims[1] = 1;
  start[0] = 0;
  start[1] = 0;
  count[0] = TotNumPart;
  count[1] = 1;
  rank = 1;

  hdf5_file 			= H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_dataset 			= H5Dopen(hdf5_file, "/IdSnapTable");
  hdf5_dataspace_in_file   	= H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory 	= H5Screate_simple(rank, dims, NULL);
  hdf5_datatype 		= H5Tcopy(H5T_NATIVE_UINT64);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, IdSnapTable);

  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);      
  H5Dclose(hdf5_dataset);
  H5Fclose(hdf5_file);
}


void read_subfind_group_hdf5(const char *fname, int i, struct halo_catalogue *cat, int ngroups, int groupcount)
{
  hid_t hdf5_file, hdf5_grp;
  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  int rank;
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];

/* ========================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Group");
  hdf5_dataset = H5Dopen(hdf5_grp, "GroupLenType");

  dims[0] = ngroups;
  dims[1] = 7;
  start[0] = 0;
  start[1] = 0;
  count[0] = ngroups;
  count[1] = 7;
  rank = 2;

  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(7 * ngroups * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);      
  H5Gclose(hdf5_grp);

  int local_count = 0,j;
  for(i = 0 ; i < ngroups  ; i++)
    for(j = 0; j < 7 ; j++ )
      { 
        cat->GroupLenType[j][groupcount+i] = int_tmpptr[local_count]; 	// this could always be reversed... need to double check //

	local_count++;
      }

  for(j = 0; j<7 ; j++)
    for(i = 0 ; i < ngroups  ; i++)
      {
        if (groupcount+i > 0)
          cat->GroupOffsetType[j][groupcount+i] = cat->GroupOffsetType[j][groupcount+i-1] + cat->GroupLenType[j][groupcount+i-1];
        else
          cat->GroupOffsetType[j][groupcount+i] = 0;
      }


  myfree(int_tmpptr);

/* ========================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Group");
  hdf5_dataset = H5Dopen(hdf5_grp, "GroupLen");

  dims[0] = ngroups;
  dims[1] = 1;
  start[0] = 0;
  start[1] = 0;
  count[0] = ngroups;
  count[1] = 1;
  rank = 1;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(ngroups * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);      
  H5Gclose(hdf5_grp);

  for(i = 0 ; i < ngroups  ; i++)
    cat->GroupLen[groupcount+i] = int_tmpptr[i];


/* ========================================================== */
  for(i = 0 ; i < ngroups  ; i++)
    {
      if (groupcount+i > 0)
        cat->GroupOffset[groupcount+i] = cat->GroupOffset[groupcount+i-1] + int_tmpptr[i];
      else
        cat->GroupOffset[groupcount+i] = 0;
    }

  myfree(int_tmpptr);

/* ========================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Group");
  hdf5_dataset = H5Dopen(hdf5_grp, "GroupNsubs");

  dims[0] = ngroups;
  dims[1] = 1;
  start[0] = 0;
  start[1] = 0;
  count[0] = ngroups;
  count[1] = 1;
  rank = 1;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(ngroups * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);      
  H5Gclose(hdf5_grp);

  for(i = 0 ; i < ngroups  ; i++)
    cat->GroupNsubs[groupcount+i] = int_tmpptr[i];

  myfree(int_tmpptr);
/* ========================================================== */


  H5Fclose(hdf5_file);
}










void read_subfind_subhalo_hdf5(const char *fname, int i, struct halo_catalogue *cat, int nsubhalos, int subcount)
{
  hid_t hdf5_file, hdf5_grp;
  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  int rank;
  hid_t  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];

/* ====================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Subhalo");
  hdf5_dataset = H5Dopen(hdf5_grp, "SubhaloGrNr");

  dims[0] = nsubhalos;
  dims[1] = 1;
  start[0] = 0;
  start[1] = 0;
  count[0] = nsubhalos;
  count[1] = 1;
  rank = 1;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(nsubhalos * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
      
  H5Gclose(hdf5_grp);

  for(i = 0 ; i < nsubhalos  ; i++)
      cat->SubhaloGrNr[subcount+i] = int_tmpptr[i];

  myfree(int_tmpptr);
/* ====================================================== */




/* ====================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Subhalo");
  hdf5_dataset = H5Dopen(hdf5_grp, "SubhaloLen");

  dims[0] = nsubhalos;
  dims[1] = 1;
  start[0] = 0;
  start[1] = 0;
  count[0] = nsubhalos;
  count[1] = 1;
  rank = 1;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(nsubhalos * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
      
  H5Gclose(hdf5_grp);

  for(i = 0 ; i < nsubhalos  ; i++)
      cat->SubhaloLen[subcount+i] = int_tmpptr[i];

  myfree(int_tmpptr);
/* ====================================================== */


/* ====================================================== */
  hdf5_grp = H5Gopen(hdf5_file, "/Subhalo");
  hdf5_dataset = H5Dopen(hdf5_grp, "SubhaloLenType");

  dims[0] = nsubhalos;
  dims[1] = 7;
  start[0] = 0;
  start[1] = 0;
  count[0] = nsubhalos;
  count[1] = 7;
  rank = 2;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  hdf5_datatype = H5Tcopy(H5T_NATIVE_INT);

  int_tmpptr =  mymalloc(7 * nsubhalos * sizeof(int));
  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, int_tmpptr);

  H5Dclose(hdf5_dataset);
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);
      
  H5Gclose(hdf5_grp);


  int local_count = 0,j;
  for(i = 0 ; i < nsubhalos  ; i++)
    for(j = 0; j < 7 ; j++ )
      {
       	cat->SubhaloLenType[j][subcount+i] = int_tmpptr[local_count];   // this could always be reversed... need to double check //
        local_count++;
//	printf("i = %d j = %d local_count = %d  subcount = %d\n",i,j,local_count,subcount);
      }
  
  myfree(int_tmpptr);
/* ====================================================== */




  H5Fclose(hdf5_file);
}












void read_particle_ids_in_hdf5(const char *fname, int parttype, MyIDType *localIdSnapTable, MyIDType Nskip)
{

  int rank;
  hid_t hdf5_file=0, hdf5_grp,  hdf5_dataspace_in_file;
  hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
  hsize_t dims[2], count[2], start[2];

  read_snap_header_attributes_in_hdf5(fname);
//  n_particles_to_read = header.npart[parttype];

  if(header.npart[parttype] == 0)
    return;

  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  if (parttype == 0)
    hdf5_grp = H5Gopen(hdf5_file, "/PartType0");
  if (parttype == 1)
    hdf5_grp = H5Gopen(hdf5_file, "/PartType1");
  if (parttype == 4)
    hdf5_grp = H5Gopen(hdf5_file, "/PartType4");

  hdf5_dataset = H5Dopen(hdf5_grp, "ParticleIDs");

  dims[0] = header.npart[parttype];
  dims[1] = 1;

  rank = 1;
  hdf5_dataspace_in_file   = H5Screate_simple(rank, dims, NULL);
  hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

  start[0] = 0;
  start[1] = 0;
  count[0] = header.npart[parttype];
  count[1] = 1;

  H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

#ifdef LONGIDS
  hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
#else
  hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
//  hdf5_datatype = H5Tcopy(H5T_NATIVE_ULONG);
#endif
 
  MyIDType i=0;
  printf("npart %d \n",header.npart[parttype]);
  myIDType_tmpptr =  mymalloc(header.npart[parttype] * sizeof(MyIDType));
  for(i = 0 ; i < header.npart[parttype] ; i ++)
    myIDType_tmpptr[i] = 0;

  H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, myIDType_tmpptr );

  for(i = 0 ; i < header.npart[parttype] ; i++)
    localIdSnapTable[i+Nskip] = myIDType_tmpptr[i];

#ifdef VERBOSE
#ifdef LONGIDS
  printf("first  id is localIdSnapTable[0] = %llu\n", localIdSnapTable[0] );
  printf("second id is localIdSnapTable[1] = %llu\n", localIdSnapTable[1] );
  printf("third  id is localIdSnapTable[2] = %llu\n", localIdSnapTable[2] );
#else
  printf("first  id is localIdSnapTable[0] = %d\n", localIdSnapTable[0] );
  printf("second id is localIdSnapTable[1] = %d\n", localIdSnapTable[1] );
  printf("third  id is localIdSnapTable[2] = %d\n", localIdSnapTable[2] );
#endif
#endif

  myfree(myIDType_tmpptr);

  H5Dclose(hdf5_dataset);
  
  H5Tclose(hdf5_datatype);
  H5Sclose(hdf5_dataspace_in_memory);
  H5Sclose(hdf5_dataspace_in_file);

  H5Gclose(hdf5_grp);
  H5Fclose(hdf5_file);
    
}
    

void read_basic_subfind_header_hdf5(const char *fname, int i, struct halo_catalogue *cat, 
	int *nFiles, 	int *nids, int *nsubhalos, int *ngroups)
{
  hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;
            
  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_headergrp = H5Gopen(hdf5_file, "/Header");

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumFiles");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, nFiles);
  H5Aclose(hdf5_attribute);
           
  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Ngroups_Total");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, &cat->TotNgroups);
  H5Aclose(hdf5_attribute);
      
  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Nids_Total");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT64, &cat->TotNids);             // might give trouble because of type
  H5Aclose(hdf5_attribute);
          
  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Nsubgroups_Total");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, &cat->TotNsubhalos);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Nids_ThisFile");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, nids);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Nsubgroups_ThisFile");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, nsubhalos);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Ngroups_ThisFile");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, ngroups);
  H5Aclose(hdf5_attribute);

  H5Gclose(hdf5_headergrp);
  H5Fclose(hdf5_file);
}


void read_snap_header_attributes_in_hdf5(const char *fname)
{
  hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;

  hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hdf5_headergrp = H5Gopen(hdf5_file, "/Header");

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_ThisFile");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, header.npart);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total");
  H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotal);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total_HighWord");
  H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotalHighWord);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "MassTable");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, header.mass);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Time");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.time);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Redshift");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.redshift);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "BoxSize");
  H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.BoxSize);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumFilesPerSnapshot");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.num_files);
  H5Aclose(hdf5_attribute);

  hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_DoublePrecision");
  H5Aread(hdf5_attribute, H5T_NATIVE_INT32, &header.flag_doubleprecision);
  H5Aclose(hdf5_attribute);

  fflush(stdout);
  
  H5Gclose(hdf5_headergrp);
  H5Fclose(hdf5_file);
}


