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


int CountUsed, CountSumUsed;


void walk_it(int i, int flag, int filenr)
{
  int p;

  HaloAux[i].UsedFlag = 1;
  HaloAux[i].FileNr = filenr;
  HaloAux[i].TargetIndex = CountUsed;
  HaloAux[CountUsed].Origin = i;

  if(flag == 1)
    {
      HaloList[CountUsed] = Halo[i];
      HaloAux[i].FileNr = -1; /* to prevent that it is used again in another file */
    }

  CountUsed++;

  if(Halo[i].Descendant >= 0)
    {
      if(HaloAux[Halo[i].Descendant].UsedFlag == 0)
        walk_it(Halo[i].Descendant, flag, filenr);
    }

  p = Halo[i].FirstProgenitor;
  while(p >= 0)
    {
      if(HaloAux[p].UsedFlag == 0)
        walk_it(p, flag, filenr);

      p = Halo[p].NextProgenitor;
    }

  p = Halo[i].FirstHaloInFOFgroup;
  if(HaloAux[p].HaloFlag == 0)
    {
      HaloAux[p].HaloFlag = 1;
      while(p >= 0)  
        {
          if(HaloAux[p].UsedFlag == 0)
            walk_it(p, flag, filenr);
          p = Halo[p].NextHaloInFOFgroup;
        }
    }
}


