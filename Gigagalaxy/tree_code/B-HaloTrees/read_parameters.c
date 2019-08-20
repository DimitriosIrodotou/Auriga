#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"



#define DOUBLE 1
#define STRING 2
#define INT 3
#define MAXTAGS 300



void read_parameter_file(char *fname)
{
  MultipleFilesPerSnap = -1;
#ifndef PROCCOMPATIBLE
  FILE *fd;
  char buf[400], buf1[400], buf2[400], buf3[400];
  int i, j, nt = 0;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int errorFlag = 0;

  strcpy(tag[nt], "TreeOutputDir");
  addr[nt] = TreeOutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "RunOutputDir");
  addr[nt] = RunOutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "SnapshotFileBase");
  addr[nt] = SnapshotFileBase;
  id[nt++] = STRING;

  strcpy(tag[nt], "FirstSnapShotNr");
  addr[nt] = &FirstSnapShotNr;
  id[nt++] = INT;

  strcpy(tag[nt], "LastSnapShotNr");
  addr[nt] = &LastSnapShotNr;
  id[nt++] = INT;

  strcpy(tag[nt], "SnapSkipFac");
  addr[nt] = &SnapSkipFac;
  id[nt++] = INT;

  strcpy(tag[nt], "NumberOfOutputFiles");
  addr[nt] = &NumberOfOutputFiles;
  id[nt++] = INT;

#ifdef SKIP_SNAP
  strcpy(tag[nt], "OutputList");
  addr[nt] = OutputList;
  id[nt++] = STRING;
#endif

  if((fd = fopen(fname, "r")))
    {
      while(!feof(fd))
	{
	  *buf = 0;
	  fgets(buf, 200, fd);
	  if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
	    continue;

	  if(buf1[0] == '%')
	    continue;

	  for(i = 0, j = -1; i < nt; i++)
	    if(strcmp(buf1, tag[i]) == 0)
	      {
		j = i;
		tag[i][0] = 0;
		break;
	      }

	  if(j >= 0)
	    {
	      switch (id[j])
		{
		case DOUBLE:
		  *((double *) addr[j]) = atof(buf2);
		  break;
		case STRING:
		  strcpy(addr[j], buf2);
		  break;
		case INT:
		  *((int *) addr[j]) = atoi(buf2);
		  break;
		}
	    }
	  else
	    {
	      printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
	      errorFlag = 1;
	    }
	}
      fclose(fd);

      i = strlen(TreeOutputDir);
      if(i > 0)
	if(TreeOutputDir[i - 1] != '/')
	  strcat(TreeOutputDir, "/");

      i = strlen(RunOutputDir);
      if(i > 0)
	if(RunOutputDir[i - 1] != '/')
	  strcat(RunOutputDir, "/");
    }
  else
    {
      printf("Parameter file %s not found.\n", fname);
      errorFlag = 1;
    }


  for(i = 0; i < nt; i++)
    {
      if(*tag[i])
	{
	  printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
	  errorFlag = 1;
	}
    }
#endif

  if(errorFlag)
    exit(1);

}
#ifdef SKIP_SNAP

int read_outputlist(char *fname)
{
  FILE *fd;
  int count, flag;
  char buf[512], msg[512];

  if(!(fd = fopen(fname, "r")))
    {
      printf("can't read output list in file '%s'\n", fname);
      exit(1);
    }

  OutputListLength = 0;

  while(1)
    {
      if(fgets(buf, 500, fd) != buf)
        break;

      count = sscanf(buf, " %lg %d ", &OutputListTimes[OutputListLength], &flag);

      if(count == 1)
        flag = 1;

      if(count == 1 || count == 2)
        {
          if(OutputListLength >= MAXLEN_OUTPUTLIST)
            {
              printf(msg, "\ntoo many entries in output-list. You should increase MAXLEN_OUTPUTLIST=%d.\n", (int) MAXLEN_OUTPUTLIST);
              exit(1);
            }

          OutputListFlag[OutputListLength] = flag;
          OutputListLength++;
        }
    }

  fclose(fd);

  printf("\nfound %d times in output-list.\n", OutputListLength);

  return 0;
}

#endif
