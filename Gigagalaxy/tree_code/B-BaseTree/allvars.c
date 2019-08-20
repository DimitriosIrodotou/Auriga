
#include "allvars.h"

int SnapSkipFac;
int SnapshotNum;
long long TotNumPart;

char RunOutputDir[512];
char TreeOutputDir[512];
char SnapshotFileBase[512];

int LastSnapShotNr;
int MultipleFilesPerSnap;

struct halo_catalogue CatA, CatB, CatC;


struct io_header header;

#ifdef SKIP_SNAP
int OutputListLength;
double OutputListTimes[MAXLEN_OUTPUTLIST];
int OutputListFlag[MAXLEN_OUTPUTLIST];
char OutputList[512];
#endif
