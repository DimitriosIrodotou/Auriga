#include "allvars.h"

int SnapSkipFac;
int LastSnapShotNr;
int FirstSnapShotNr;
int MultipleFilesPerSnap;

char TreeOutputDir[512];
char RunOutputDir[512];
char SnapshotFileBase[512];

int TotHalos;
int NumberOfOutputFiles;
int *FirstHaloInSnap;
int NtreesPerFile;
int NhalosPerFile;

double ParticleMass;

struct halo_catalogue *Cats;
struct halo_aux_data *HaloAux;
struct halo_data *Halo, *HaloList;


#ifdef SKIP_SNAP
int OutputListLength;
double OutputListTimes[MAXLEN_OUTPUTLIST];
int OutputListFlag[MAXLEN_OUTPUTLIST];
char OutputList[512];
#endif
