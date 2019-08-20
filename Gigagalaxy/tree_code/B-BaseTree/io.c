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
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "allvars.h"
#include "proto.h"




/*! This catches I/O errors occuring for my_fwrite(). In this case we
 *  better stop.
 */
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nwritten;

  if(size * nmemb > 0)
    {
      if((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
	{
	  printf("I/O error (fwrite) has occured: %s\n", strerror(errno));
	  fflush(stdout);
	  exit(777);
	}
    }
  else
    nwritten = 0;

  return nwritten;
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
