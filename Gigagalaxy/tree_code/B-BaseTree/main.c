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
#include <hdf5.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "allvars.h"
#include "proto.h"

#ifndef ALPHA
#define ALPHA (2.0/3)
#endif

#define WEIGHT_FAK (3.0)

MyIDType *IdSnapTable;
MyIDType *tmpptr;

int 		*int_tmpptr;
long int 	*long_int_tmpptr;
long long int 	*long_long_int_tmpptr;

void get_TotNumPart(void);


int main(int argc, char **argv)
{

  if(argc != 3)
    {
      printf("\n  usage: L-BaseTree <parameterfile>  <outputnum>\n");
      printf("  <parameterfile>    see readparmeterfile.c\n");
      printf("  <outputnum>        snapshot number\n\n");
      exit(1);
    }

  read_parameter_file(argv[1]);

#ifdef SKIP_SNAP
  read_outputlist(OutputList);
#endif

  SnapshotNum = atoi(argv[2]);

  int snapA = SnapshotNum + 0 * SnapSkipFac;
  int snapB = SnapshotNum + 1 * SnapSkipFac;
  int snapC = SnapshotNum + 2 * SnapSkipFac;

  printf("\nB-BASETREE [Snapshot: %d]\n\n", SnapshotNum);

#ifdef SKIP_SNAP
  if(OutputListFlag[snapA] == 0)
     {
      printf("broken snapshot, stopping");
      exit(0);
     }
  while(OutputListFlag[snapB] == 0 && snapB < LastSnapShotNr)
    {
      snapB+=SnapSkipFac;
    }
  snapC = snapB + SnapSkipFac;
   while(OutputListFlag[snapC] == 0 && snapC < LastSnapShotNr)
     {
       snapC+=SnapSkipFac;
     }
   printf("Used snapshots: SnapA %d, SnapB %d, SnapC %d\n", snapA,snapB,snapC);
#endif

#if defined(_OPENMP)
  printf("OPENMP: Using max-threads=%d\n", omp_get_max_threads());
  fflush(stdout);
#endif

/*==========================================================================*/
  printf("allocating group catalogues...\n");		fflush(stdout);
  allocate_group_catalogue(snapA, &CatA);
  allocate_group_catalogue(snapB, &CatB);
  if(snapC <= LastSnapShotNr)
    allocate_group_catalogue(snapC, &CatC);
    
  printf("populating group catalogues...\n");		fflush(stdout);
  printf("Used snapshots: SnapA %d, SnapB %d, SnapC %d\n", snapA,snapB,snapC);  fflush(stdout);
  load_group_catalogue(snapA, &CatA);
  load_group_catalogue(snapB, &CatB);
  if(snapC <= LastSnapShotNr)
    load_group_catalogue(snapC, &CatC);
/*==========================================================================*/



/*==========================================================================*/
  printf("allocating subhalo catalogues...\n");		fflush(stdout);
  allocate_subhalo_catalogue(snapA, &CatA);
  allocate_subhalo_catalogue(snapB, &CatB);
  if(snapC <= LastSnapShotNr)
    allocate_subhalo_catalogue(snapC, &CatC);

  printf("populating subhalo catalogues...\n");		fflush(stdout);
  load_subhalo_catalogue(snapA, &CatA);
  load_subhalo_catalogue(snapB, &CatB);
  if(snapC <= LastSnapShotNr)
    load_subhalo_catalogue(snapC, &CatC);
/*==========================================================================*/

  printf("reading/sorting IDs... ");			fflush(stdout);
#ifdef IDS_HAVE_GAPS
  get_id_translation_table();		/* Load IdSnapTable: sorted array of length N_dm with minimum (first) value of Min(ID_dm) and a maximum (last) value of Max(ID_dm) */
#else
  get_TotNumPart();
#endif
  printf("done.\n"); fflush(stdout);

  printf("reassigning ids ... ");			fflush(stdout);

  reassign_ids(CatA.TotNids, CatA.IdList);
  reassign_ids(CatB.TotNids, CatB.IdList);
  reassign_ids(CatC.TotNids, CatC.IdList);

  myfree(IdSnapTable);

  printf("done.\n");					fflush(stdout);
/*==========================================================================*/

  /* set cat->IdToHalo[i] such that each particle can reference the Halo that it is a part of */
  printf("preparing ID-to-halo tables...\n");		fflush(stdout);

  prepare_index_list(&CatA);
  printf(" index A done.\n");				fflush(stdout);

  prepare_index_list(&CatB);
  printf(" index B done.\n");				fflush(stdout);

  if(snapC <= LastSnapShotNr)
    {
      prepare_index_list(&CatC);
      printf(" index C done.\n");				fflush(stdout);
    }
/*==========================================================================*/

  /* get descendants */
  printf("determine_descendants...\n");			fflush(stdout);
  determine_descendants(&CatA, &CatB, 0, snapB);
  printf(" desc AB done.\n");				fflush(stdout);

#ifdef BACKWARD_CHECKING
  determine_descendants(&CatB, &CatA, 1, snapA);
  printf(" desc BA done.\n");
  fflush(stdout);
#endif

  if(snapC <= LastSnapShotNr)
    {
      determine_descendants(&CatB, &CatC, 0, snapC);
      printf(" desc BC done.\n");				fflush(stdout);

      determine_descendants(&CatA, &CatC, 1, snapC);	/* secondary descendant */
      printf(" desc AC done.\n");				fflush(stdout);
    }

/*==========================================================================*/

  if(snapC <= LastSnapShotNr)
    {
      count_progenitors(&CatA, &CatB);
      printf(" progcount AB done\n");
      fflush(stdout);

      count_progenitors(&CatB, &CatC);
      printf(" progcount BC done\n");
      fflush(stdout);

      decide_upon_descendant();
    }

#ifdef BACKWARD_CHECKING
  printf("Doing Backward decision ...\n");
  fflush(stdout);
  if(snapC > LastSnapShotNr)
    {
      count_progenitors(&CatA, &CatB);
      printf(" progcount AB done\n");
      fflush(stdout);
    }
  if(snapC <= LastSnapShotNr)
    {
      decide_backwards(&CatA, &CatB);
      printf(" backward decision for AB done.\n");
      fflush(stdout);
    }
#endif
/*==========================================================================*/
  printf("saving descendants... ");		fflush(stdout);
  save_decendant_list();

  printf("done.\n\n");			fflush(stdout);
/*==========================================================================*/

  return 0;
}

#ifdef BACKWARD_CHECKING

#ifndef HALO_SIZE_INCREASE_FOR_SWITCHING
#define HALO_SIZE_INCREASE_FOR_SWITCHING 1.5
#endif

void decide_backwards(struct halo_catalogue *catA, struct halo_catalogue *catB)
{
  int i, ic = 0, ict = 0, ptA, p, ifound;

  for(i = 0; i < catB->TotNsubhalos; i++) 
    {
      if(catB->CountProgenitors[i] == 0)	/* select halos with no progenitors */
	{
	  ict++;
	  if(catB->Descendant[i].HaloIndex[1] >= 0)	/* But in reality they have one */
	    {
	      ptA = catB->Descendant[i].HaloIndex[1];
	      /* check if the halo without progenitor's most bound
	         id is part of the descendant found at the previous snap. 
	         Only in  this case, continue ... */
	      ifound = 0;
	      for(p = 0; p < catA->SubLen[ptA]; p++)
		if(catB->IdList[catB->SubOffset[i]] == catA->IdList[catA->SubOffset[ptA] + p])
		  ifound++;
	      if(ifound)
		{
		  /* now check if the tow descendent found merge in the next step ... */
		  if(catB->Descendant[i].HaloIndex[0] ==
		     catB->Descendant[catA->Descendant[ptA].HaloIndex[0]].HaloIndex[0])
		    {
		      /* only redirect if missed descendent has more particles ... */
		      if(catB->SubLen[i] >
			 catB->SubLen[catA->Descendant[ptA].HaloIndex[0]] * HALO_SIZE_INCREASE_FOR_SWITCHING)
			{
			  catA->Descendant[ptA].HaloIndex[0] = i;
			  ic++;
			}
		    }
		}
	    }
	}
    }
  printf("Redirected %d of %d descendents ...\n", ic, ict);
  fflush(stdout);

}
#endif

void decide_upon_descendant(void)
{
  int i, index_b, index_c;
  int count_b, count_c, count_w, count_n;
  double sumpart;

#ifdef SKIP_BY_WEIGHT
  int count_s = 0;
#endif

  count_b = count_c = count_w = count_n = 0;
  sumpart = 0.0;


#if defined(_OPENMP)
#pragma omp parallel for private(index_b, index_c) reduction(+:count_b,count_c,count_w,count_n,sumpart) 
#endif
  for(i = 0; i < CatA.TotNsubhalos; i++)
    {
      index_b = CatA.Descendant[i].HaloIndex[0];
      index_c = CatA.Descendant[i].HaloIndex[1];

      if(index_b >= 0)
	count_b++;

      if(index_b >= 0 && index_c >= 0)
	{
	  if(CatB.CountProgenitors[index_b] > 1 && CatC.CountProgenitors[index_c] == 0)
	    {
	      CatB.CountProgenitors[index_b]--;
	      CatC.CountProgenitors[index_c]++;
	      CatA.Descendant[i].HaloIndex[0] = CatA.Descendant[i].HaloIndex[1];
	      CatA.Descendant[i].SnapNum[0] = CatA.Descendant[i].SnapNum[1];
	      count_c++;
	    }
#ifdef SKIP_BY_WEIGHT
	  else
	    {
	      if(CatA.Descendant[i].Weight[1] / WEIGHT_FAK > CatA.Descendant[i].Weight[0])
		{
		  CatB.CountProgenitors[index_b]--;
		  CatC.CountProgenitors[index_c]++;
		  CatA.Descendant[i].HaloIndex[0] = CatA.Descendant[i].HaloIndex[1];
		  CatA.Descendant[i].SnapNum[0] = CatA.Descendant[i].SnapNum[1];
		  count_c++;
		  count_s++;
		}
	    }
#endif
	}

      if(index_b < 0 && index_c >= 0)
	{
	  CatA.Descendant[i].HaloIndex[0] = CatA.Descendant[i].HaloIndex[1];
	  CatA.Descendant[i].SnapNum[0] = CatA.Descendant[i].SnapNum[1];
	  CatC.CountProgenitors[index_c]++;
	  count_w++;
	}

      if(index_b < 0 && index_c < 0)
	{
	  sumpart += CatA.SubLen[i];
	  count_n++;
	}
    }

  printf("Out of %d primary descendants, %d have been rerouted to the secondary descendant.\n",
	 count_b, count_c);

  printf("Additionally, %d have been pointed to the secondary because they had no primary.\n", count_w);

#ifdef SKIP_BY_WEIGHT
  printf("Additionally, %d have been pointed to the secondary because the primary have had low weights.\n",
	 count_s);
#endif

  printf("This leaves %d without descendant, of average size = %g particles.\n", count_n, sumpart / count_n);
  fflush(stdout);
}



void count_progenitors(struct halo_catalogue *catA, struct halo_catalogue *catB)
{
  int i;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for(i = 0; i < catB->TotNsubhalos; i++)
    catB->CountProgenitors[i] = 0;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for(i = 0; i < catA->TotNsubhalos; i++)
    {
      if(catA->Descendant[i].HaloIndex[0] >= 0)
	{
	  catB->CountProgenitors[catA->Descendant[i].HaloIndex[0]]++;
	}
    }
}


struct cand_data
{
  int haloindex;
  float weight;
};


void determine_descendants(struct halo_catalogue *catA, struct halo_catalogue *catB, int entry, int snapnum)
{
  int i, j, ndiff, ncand, haloB, prev, maxlen;
  MyIDType id;
  float weightmax;
  int halomax;
  struct cand_data *candlist, *difflist;

  maxlen = 0;

  for(i = 0; i < catA->TotNsubhalos; i++)
    if(catA->SubLen[i] > maxlen)
      maxlen = catA->SubLen[i];


#if defined(_OPENMP)
#pragma omp parallel private(candlist, difflist, ncand, i, j, id, haloB, ndiff, prev, weightmax, halomax) 
#endif
  {
    candlist = mymalloc(maxlen * sizeof(struct cand_data));
    difflist = mymalloc(maxlen * sizeof(struct cand_data));

#if defined(_OPENMP)
#pragma omp for schedule(dynamic) nowait 
#endif
    for(i = 0; i < catA->TotNsubhalos; i++)			// for each subhalo in Snapshot A ...
      {
	ncand = 0;
	for(j = 0; j < catA->SubLen[i]; j++)			// ... and for each particle in each subhalo
	  {
	    id = catA->IdList[catA->SubOffset[i] + j];		// ... identify the particle's ID

	    if(id >= 0 && id < TotNumPart)			// ... (and as long as it's in the accetable range)
	      {
		haloB = catB->IdToHalo[id];			// ... identify the halo that contains this particle in snapshot B

		if(haloB >= 0)					// all particles are in haloes (they have -1), but if it is in a halo...
		  {
		    candlist[ncand].haloindex = haloB;			// ... set the haloindex accordingly
		    candlist[ncand].weight = 1.0 / pow(j + 1, ALPHA);	// ... and set the weighting based on how bound it was 
		    ncand++;
		  }
	      }
	    else
	      {
		char buf[100];

		long_to_str(buf, id);
		printf("bummer! i=%d  id=%s TotumPart=%d\n", i, buf, (int)TotNumPart);
		exit(4);
	      }
	  }

	qsort(candlist, ncand, sizeof(struct cand_data), sort_candlist);

	for(j = 0, ndiff = 0, prev = -1; j < ncand; j++)
	  {
	    if(candlist[j].haloindex != prev)
	      {
		ndiff++;
		difflist[ndiff - 1].haloindex = candlist[j].haloindex;
		difflist[ndiff - 1].weight = 0;
	      }
	    difflist[ndiff - 1].weight += candlist[j].weight;
	    prev = candlist[j].haloindex;
	  }

	weightmax = 0;
	halomax = -1;

	for(j = 0; j < ndiff; j++)
	  {
	    if(difflist[j].weight > weightmax)
	      {
		weightmax = difflist[j].weight;
		halomax = difflist[j].haloindex;
	      }
	  }


	if(ndiff > 0 && halomax >= 0)
	  {
	    catA->Descendant[i].HaloIndex[entry] = halomax;
	    catA->Descendant[i].SnapNum[entry] = snapnum;
#ifdef SKIP_BY_WEIGHT
	    catA->Descendant[i].Weight[entry] = weightmax;
#endif
	  }
	else
	  {
	    catA->Descendant[i].HaloIndex[entry] = -1;
	    catA->Descendant[i].SnapNum[entry] = -1;
#ifdef SKIP_BY_WEIGHT
	    catA->Descendant[i].Weight[entry] = -1;
#endif
	  }
      }

    myfree(candlist);
    myfree(difflist);
  }



}



int sort_twoids_id(const void *a, const void *b)
{
  if(((struct twoids *) a)->id < ((struct twoids *) b)->id)
    return -1;

  if(((struct twoids *) a)->id > ((struct twoids *) b)->id)
    return +1;

  return 0;
}

int sort_twoids_ord(const void *a, const void *b)
{
  if(((struct twoids *) a)->ord < ((struct twoids *) b)->ord)
    return -1;

  if(((struct twoids *) a)->ord > ((struct twoids *) b)->ord)
    return +1;

  return 0;
}


int sort_candlist(const void *a, const void *b)
{
  if(((struct cand_data *) a)->haloindex < ((struct cand_data *) b)->haloindex)
    return -1;

  if(((struct cand_data *) a)->haloindex > ((struct cand_data *) b)->haloindex)
    return +1;

  return 0;
}

int sort_IDType(const void *a, const void *b)
{
  if(*((MyIDType *) a) < *((MyIDType *) b))
    return -1;

  if(*((MyIDType *) a) > *((MyIDType *) b))
    return +1;

  return 0;
}

void prepare_index_list(struct halo_catalogue *cat)
{
  MyIDType id;
  signed long long ii;
  int i, j;

  cat->IdToHalo = mymalloc(sizeof(int) * TotNumPart);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for(ii = 0; ii < TotNumPart; ii++)		// start by assigning all particles to no halo
    cat->IdToHalo[ii] = -1;

#if defined(_OPENMP)
#pragma omp parallel for private(j,id)
#endif
  for(i = 0; i < cat->TotNsubhalos; i++)		// loop over all subhalos
    for(j = 0; j < cat->SubLen[i]; j++)			// loop over all particles in each subhalo
      {
	id = cat->IdList[cat->SubOffset[i] + j];		// id from the subhalo list 

	if(id >= 0  && id < TotNumPart)
	  cat->IdToHalo[id] = i;
	else
	  {
	    char buf[100];

	    long_to_str(buf, id);

	    printf("bummer! i=%d j=%d id=%s id=%d TotNumPart=%d)\n", i, j, buf, (int)id, (int)TotNumPart);
	    exit(1);
	  }
      }

}


void allocate_group_catalogue(int num, struct halo_catalogue *cat)
{
  int nids, nFiles, nsubhalos, ngroups;
  char buf[1000];

  get_filename(buf, num, 0, 1);

  read_basic_subfind_header_hdf5(buf, 0, cat, &nFiles , &nids , &nsubhalos, &ngroups);

  int jjj;

  cat->GroupNsubs           = mymalloc(sizeof(MyIDType) * cat->TotNgroups);
  cat->GroupLen             = mymalloc(sizeof(MyIDType) * cat->TotNgroups);
  cat->GroupOffset          = mymalloc(sizeof(MyIDType) * cat->TotNgroups);
  cat->GroupLenType         = mymalloc(7 * sizeof(MyIDType *));
  cat->GroupOffsetType      = mymalloc(7 * sizeof(MyIDType *));
  cat->SubhaloLenType       = mymalloc(7 * sizeof(MyIDType *));

  cat->Group 		    = mymalloc(sizeof(struct group_data) * cat->TotNgroups);

  for(jjj=0 ; jjj< 7; jjj++)
    {
      cat->GroupLenType[jjj]    = mymalloc(cat->TotNgroups   * sizeof(MyIDType));
      cat->GroupOffsetType[jjj] = mymalloc(cat->TotNgroups   * sizeof(MyIDType));
    }

  for(jjj=0; jjj< cat->TotNgroups; jjj++)
    cat->Group[jjj].count = 0;
}


void load_group_catalogue(int num, struct halo_catalogue *cat)
{
  int i=0, nids, nFiles, nsubhalos, ngroups, groupcount;
  char buf[1000];

  get_filename(buf, num, i, 1);

  read_basic_subfind_header_hdf5(buf, i, cat, &nFiles , &nids , &nsubhalos, &ngroups);

  groupcount = 0;

  for(i = 0, nFiles = 1; i < nFiles; i++)
    {
      get_filename(buf, num, i, 1);

      read_basic_subfind_header_hdf5(buf, i, cat, &nFiles , &nids , &nsubhalos, &ngroups);

      if(ngroups > 0)
        read_subfind_group_hdf5(buf, i, cat, ngroups, groupcount);

      groupcount += ngroups;
    }

  for(i=0 ; i < cat->TotNgroups ; i++)
    cat->Group[i].count = 0;
}


void allocate_subhalo_catalogue(int num, struct halo_catalogue *cat)
{
  int nids, nFiles, nsubhalos, ngroups;
  char buf[1000];

  get_filename(buf, num, 0, 1);

  read_basic_subfind_header_hdf5(buf, 0, cat, &nFiles , &nids , &nsubhalos, &ngroups);

  int iii,jjj;

  cat->SubLen               = mymalloc(sizeof(int) * cat->TotNsubhalos);
  cat->SubParentHalo        = mymalloc(sizeof(int) * cat->TotNsubhalos);
  cat->CountProgenitors     = mymalloc(sizeof(int) * cat->TotNsubhalos);
  cat->SubOffset            = mymalloc(sizeof(MyIDType) * cat->TotNsubhalos);
  cat->Descendant           = mymalloc(sizeof(struct descendant_data) * cat->TotNsubhalos);
  cat->SubhaloGrNr          = mymalloc(sizeof(MyIDType) * cat->TotNsubhalos);
  cat->SubhaloLen           = mymalloc(sizeof(MyIDType) * cat->TotNsubhalos);
  cat->SubhaloLenType       = mymalloc(7 * sizeof(MyIDType *));

  for(jjj=0 ; jjj< 7; jjj++)
    cat->SubhaloLenType[jjj]  = mymalloc(cat->TotNsubhalos * sizeof(MyIDType));

  for(iii=0 ; iii < cat->TotNgroups ; iii++)
    if(cat->GroupNsubs[iii] > 0)
      cat->Group[iii].Subhalo = mymalloc(sizeof(struct subhalo_data) * cat->GroupNsubs[iii]);
}



void load_subhalo_catalogue(int num, struct halo_catalogue *cat)
{
  int i=0, nids, nFiles, nsubhalos, subcount, ngroups;
  char buf[1000];
  MyIDType * local_id_array, ndm = 0 , Nskip = 0;

  get_filename(buf, num, i, 1);
  read_basic_subfind_header_hdf5(buf, i, cat, &nFiles , &nids , &nsubhalos, &ngroups);

  subcount = 0;

  for(i = 0, nFiles = 1; i < nFiles; i++)
    {
      get_filename(buf, num, i, 1);

      read_basic_subfind_header_hdf5(buf, i, cat, &nFiles , &nids , &nsubhalos, &ngroups);

      if(nsubhalos > 0)
        read_subfind_subhalo_hdf5(buf, i, cat, nsubhalos, subcount);

      subcount += nsubhalos;
    }

  long_to_str(buf, cat->TotNids);

  int iii,jjj,j;
  long int subfind_dm_ids=0;
  for(i=0 ; i < cat->TotNsubhalos ; i++)
    {
      cat->Group[cat->SubhaloGrNr[i]].Subhalo[cat->Group[cat->SubhaloGrNr[i]].count].SubhaloLen =  cat->SubhaloLen[i];
      for(j=0 ; j < 7 ; j++)
          cat->Group[cat->SubhaloGrNr[i]].Subhalo[cat->Group[cat->SubhaloGrNr[i]].count].SubhaloLenType[j] =  cat->SubhaloLenType[j][i];

      cat->Group[cat->SubhaloGrNr[i]].count = cat->Group[cat->SubhaloGrNr[i]].count + 1;
    }

  for(iii = 0; iii < cat->TotNgroups; iii++)                                                  // for each group
    for(jjj = 0; jjj < cat->Group[iii].count ; jjj++)                                         // and each subhalo within the group
      subfind_dm_ids += cat->Group[iii].Subhalo[jjj].SubhaloLenType[1];

  cat->TotNids = subfind_dm_ids;
  cat->IdList = mymalloc( subfind_dm_ids * sizeof(MyIDType));


  i = 0;
  get_filename(buf, num, i, 0);
  read_snap_header_attributes_in_hdf5(buf);
  
  ndm = header.npartTotal[1]+ ((long long) header.npartTotalHighWord[1] << 32);
  local_id_array = mymalloc(ndm * sizeof(MyIDType));

  for(i = 0; i < nFiles; i++)
    {
      get_filename(buf, num, i, 0);

      read_snap_header_attributes_in_hdf5(buf);
      read_particle_ids_in_hdf5(buf, 1, local_id_array, Nskip);				// loads all dm particle ids

      Nskip += header.npart[1];
    }

  int k;
  MyIDType local_idcount=0, local_galaxycount=0;
  
  i = j = k = 0;

  printf("starting the assignment loop... "); 		fflush(stdout);

  MyIDType cumulative_subhalo_offset = 0, local_offset = 0;

  for(i = 0; i < cat->TotNgroups; i++)							// for each group
    {
      printf("Starting Group %d \n",i);
      cat->GroupOffset[i] = cumulative_subhalo_offset;
      local_offset = 0;
      for(j = 0; j < cat->Group[i].count ; j++) 					// and each subhalo within the group
        {
	  printf("Starting Group %d, Subhalo %d of %d with Len %d\n",i,j,cat->Group[i].count,cat->Group[i].Subhalo[j].SubhaloLenType[1]);
	  for(k = 0; k < cat->Group[i].Subhalo[j].SubhaloLenType[1] ; k++)		// and each DM particle within the subhalo 
	    {
	      //if(i == 0 && j == 492)
	      //printf("local_idcount %d, Group %d, Subhalo %d of %d\n",local_idcount,i,j,cat->Group[i].count);
	      cat->IdList[local_idcount] = local_id_array[cat->GroupOffsetType[1][i] + local_offset + k ];	// can't trust this group offset
	      local_idcount++;

#ifdef VERBOSE
#ifdef LONGIDS
	      if (i < 2 && j < 2 && k < 2)
                {
	 	  printf("cat->GroupOffsetType[1][i] = %lu, local_offset = %d, k = %d, local_id_array[%lu] = %llu\n",
			cat->GroupOffsetType[1][i], local_offset, k, cat->GroupOffsetType[1][i] + local_offset + k , local_id_array[cat->GroupOffsetType[1][i] + local_offset + k ]);
		  printf("Group %d, Subhalo %d, Particle %d, ID = %llu\n",i,j,k,local_id_array[cat->GroupOffsetType[1][i] + local_offset + k ]);
	        }
#else
	      if (i < 10 && j < 10 && k < 10)
		printf("Group %d, Subhalo %d, Particle %d, ID = %d\n",i,j,k,local_id_array[cat->GroupOffsetType[1][i] + local_offset + k ]);
#endif
#endif
	    }
	  
	  printf("Finished particle loop of Group %d, Subhalo %d of %d\n",i,j,cat->Group[i].count);
	  cat->SubOffset[local_galaxycount] = cumulative_subhalo_offset;
	  cat->SubLen[local_galaxycount] = cat->Group[i].Subhalo[j].SubhaloLenType[1];

	  cumulative_subhalo_offset += cat->Group[i].Subhalo[j].SubhaloLenType[1];
	  local_offset += cat->Group[i].Subhalo[j].SubhaloLenType[1];	  
	  local_galaxycount++;
	  printf("Ending Group %d, Subhalo %d of %d\n",i,j,cat->Group[i].count);
	}
#ifdef VERBOSE
      if(i < 10)
	{
          printf("\nFirst ID of Group %d can be indexed as:\n",i);
	  printf("    local_id_array[cat->GroupOffsetType[1][%d]] = %llu   where  cat->GroupOffsetType[1][%d] = %llu\n",
			i,local_id_array[cat->GroupOffsetType[1][i]],i,cat->GroupOffsetType[1][i]);
	  printf("    cat->IdList[cat->GroupOffset[%d]]           = %llu   where cat->GroupOffset[%d]         = %llu\n\n",
			i,cat->IdList[cat->GroupOffset[i]], i, cat->GroupOffset[i]);
	}
#endif
    }

  printf("done.\n"); 		fflush(stdout);

  myfree(local_id_array);
}




void save_decendant_list(void)
{
  int i, *data;
  char buf[1000];
  FILE *fd;

  sprintf(buf, "%s/", TreeOutputDir);
  mkdir(buf, 0755);

  sprintf(buf, "%s/sub_desc_sf%d_%03d", TreeOutputDir, SnapSkipFac, SnapshotNum);
  if(!(fd = fopen(buf, "w")))
    {
      printf("can't open file `%s'\n", buf);
      exit(1);
    }

  fwrite(&CatA.TotNsubhalos, sizeof(int), 1, fd);

  data = mymalloc(sizeof(int) * CatA.TotNsubhalos);

  for(i = 0; i < CatA.TotNsubhalos; i++)
    data[i] = CatA.Descendant[i].HaloIndex[0];

  fwrite(data, sizeof(int), CatA.TotNsubhalos, fd);

  for(i = 0; i < CatA.TotNsubhalos; i++)
    data[i] = CatA.Descendant[i].SnapNum[0];

  fwrite(data, sizeof(int), CatA.TotNsubhalos, fd);
  fclose(fd);  

  myfree(data);
}


#define SKIP   {my_fread(&blksize1,sizeof(int),1,fd);}
#define SKIP2  {my_fread(&blksize2,sizeof(int),1,fd);}


void get_TotNumPart(void)
{
  char buf[1000], bufA[100];

  get_filename(buf, LastSnapShotNr, 0, 0);

  read_snap_header_attributes_in_hdf5(buf);

  TotNumPart =
    header.npartTotal[1] + (((long long) header.npartTotalHighWord[1]) << (long long) 32);

  long_to_str(bufA, TotNumPart);
}


void get_id_translation_table(void)
{
  FILE *fd;
  char buf[1000], bufA[100], bufB[100];
  int filenr, numfiles;

  MyIDType i, minID, maxID, Nskip = 0;

  printf("reading IDs from last snapshot\n");
  fflush(stdout);
  sprintf(buf, "%s/sorted_id_table_%03d.hdf5", TreeOutputDir, LastSnapShotNr);

  if((fd = fopen(buf, "r")))
    {
      fclose(fd);

      read_num_part_table_hdf5(buf, &TotNumPart );
      IdSnapTable = mymalloc(TotNumPart * sizeof(MyIDType));
      read_id_translation_table_hdf5(buf, TotNumPart, IdSnapTable );

      printf(" finished reading sorted id table, TotNumPart = %llu.\n",TotNumPart);
      fflush(stdout);
    }
  else
    {
      numfiles = 1;

      for(filenr = 0; filenr < numfiles; filenr++)
	{
	  if(filenr == 0)
	    printf("Starting to read...\n");

        get_filename(buf, LastSnapShotNr, filenr, 0);
	  printf("  %s\n",buf);

	  read_snap_header_attributes_in_hdf5(buf);


	  if(filenr == 0)
	    {
	      numfiles = header.num_files;

	      TotNumPart =
		header.npartTotal[1] + (((long long) header.npartTotalHighWord[1]) << (long long) 32);

	      long_to_str(bufA, TotNumPart);
#ifdef VERBOSE
	      printf("Allocating IdSnapTable...\n");
	      printf("  header.npartTotal[0] = %d\n",header.npartTotal[0]);
	      printf("  header.npartTotal[1] = %d\n",header.npartTotal[1]);
	      printf("  header.npartTotal[4] = %d\n",header.npartTotal[4]);
	      printf("  TotNumPart	       = %llu\n\n",TotNumPart);
#endif

	      IdSnapTable = mymalloc(TotNumPart * sizeof(MyIDType));
	    }


	  int parttype;

	  parttype = 1;
	  read_particle_ids_in_hdf5(buf, parttype , IdSnapTable , Nskip);
	  Nskip += header.npart[parttype];
#ifdef VERBOSE
	  if(filenr == 0)
	    {
  	      printf("\n\n Check that ids are being loaded properly...\n");
	      printf("  First 10 DM particle ids in %s are:\n");
	      int id_check;
	      for(id_check=0 ; id_check < 10 ; id_check ++)
#ifdef LONGIDS	      
		printf("    ID[%d] = %llu\n",id_check,IdSnapTable[id_check+Nskip - header.npart[0]]);
#else
		printf("    ID[%d] = %d\n",id_check,IdSnapTable[id_check+Nskip - header.npart[0]]);
#endif		// LONGIDS
	    }
#endif		// VERBOSE
	}

      printf("TotNumPart=%s. IDs read.\n", bufA); fflush(stdout);

      for(i = 1, minID = maxID = IdSnapTable[0]; i < TotNumPart; i++)
	{
	  if(minID > IdSnapTable[i])
	    minID = IdSnapTable[i];

	  if(maxID < IdSnapTable[i])
	    maxID = IdSnapTable[i];
	}

      long_to_str(bufA, minID);
      long_to_str(bufB, maxID);

      printf(" (min-ID=%s max-ID=%s)\n", bufA, bufB); 
      printf("sorting IDs... "); fflush(stdout);

      qsort(IdSnapTable, Nskip, sizeof(MyIDType), sort_IDType); 

      printf("done.\n"); fflush(stdout);

      printf("writing sorted id table...\n");
      fflush(stdout);

      write_id_translation_table_hdf5(IdSnapTable, TotNumPart);
    }
}



void reassign_ids(MyIDType N, MyIDType * ids)
{
#ifdef IDS_HAVE_GAPS

  long long i, j, offset, NN;
#if defined(_OPENMP)
  int tid;
  int nthreads;
#endif
  struct twoids *TwoIDs;

#if defined(_OPENMP)
#pragma omp parallel private(tid, nthreads, offset, NN, i, j, TwoIDs) shared(IdSnapTable)
#endif
  {
#if defined(_OPENMP)
    tid = omp_get_thread_num();
    nthreads = omp_get_max_threads();
    
    offset = tid * (N / nthreads);
    NN = (N / nthreads);
    
    if(nthreads > 1 && tid == (nthreads - 1))
      {
	NN = N - offset;
      }
#else
    NN = N;
    offset = 0;
#endif

    TwoIDs = mymalloc(NN * sizeof(struct twoids));

    for(i = 0; i < NN; i++)			// load all ids into the TwoID array 
      {
	TwoIDs[i].id = ids[i + offset];		// the ids at each location
	TwoIDs[i].ord = i;			// the index at each location
      }

    qsort(TwoIDs, NN, sizeof(struct twoids), sort_twoids_id);		// sort them by id! -> Min id first

    /* now assign */
    j = 0;
    for(i = 0; i < NN; i++)
      {
	while(IdSnapTable[j] < TwoIDs[i].id && j < (TotNumPart - 1))	// this breaks when IdSnapTable[j] == TwoIDs[i].id
	  j++;

	if(IdSnapTable[j] != TwoIDs[i].id)		// if this occurs, should imply that
	  {						//  - j reached TotNumPart without finding a match...
	    printf("\nID mismatch found?\n");		//  -> this means there is a particle in the subfind catalog not in the snapshot (IdSnapTable)
	    printf("IdSnapTable[%llu] = %llu    TwoIDs[%llu].id = %llu   TotNumPart = %llu \n",j,IdSnapTable[j], i,TwoIDs[i].id, TotNumPart);
	    exit(1);
	  }
	else
	  TwoIDs[i].id = j;			// THIS IS THE KEY POINT -- THE NEW ID IS THE INDEX IN IdSnapTable!!! min=0; max=N_dm
      }

    /* sort back */
    qsort(TwoIDs, NN, sizeof(struct twoids), sort_twoids_ord);  // Sort them by orig order -> old first entry is again first

    for(i = 0; i < NN; i++)
	ids[i + offset] = TwoIDs[i].id;		// repackage them back into the origional array

    myfree(TwoIDs);

  }

#else
  
  signed long long i;
 
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for(i=0; i< N; i++)
    ids[i] -= 1;

#endif

}




void long_to_str(char *s, long long n)
{
  if(n >= 1000000000)
    sprintf(s, "%d%09d", (int) (n / 1000000000), (int) (n % 1000000000));
  else
    sprintf(s, "%d", (int) n);
}





