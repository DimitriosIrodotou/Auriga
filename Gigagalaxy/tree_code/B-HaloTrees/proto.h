
#include "allvars.h"
#include <hdf5.h>

void walk_it(int i, int flag, int filenr);

void generate_trees(void);

void load_subhalo_catalogue(int num);
void count_halos(void);

void read_parameter_file(char *fname);
int read_outputlist(char* fname);
void get_filename(char *buf, int file_num, int chunk_num, int group_or_snap);

void *mymalloc(size_t n);
void myfree(void *p);
void set_progenitor_pointers(void);
#ifdef IMPROVE_POINTERS
void modify_tree(void);
void walk(int nr);
#endif

int peano_hilbert_key(int x, int y, int z, int bits);
int whichfile(float *pos);

void write_header_attributes_in_hdf5(hid_t handle, int *npertree);			/* my new hdf5 header write routine */
void my_hdf5_write_tree(void *ptr, int treenr, int countnr, hid_t handle); /* my new hdf5 write routine */


int get_dataset_info(enum iofields blocknr, char *buf, hsize_t *dims, hsize_t *count, int *rank, hid_t *hdf5_datatype, int *reg_datatype);


void get_dataset_int_data(enum iofields blocknr, int *buf    , int countnr);
void get_dataset_flt_data(enum iofields blocknr, float *buf , int countnr);
void get_dataset_ID_data(enum iofields blocknr, MyIDType *buf, int countnr);



size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream);


void read_hdf5_int_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, int *data);
void read_hdf5_float_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, float *data);
void read_hdf5_myidtype_dataset(const char *fname, const char *buf1, const char *buf2, int dim1, int dim2, MyIDType *data);

void read_hdf5_subhalo_header(const char *fname, int num, struct halo_catalogue *cat ,
	 int * nFiles , long long int * totNids, int * nids, int * nsubhalos, int * ngroups);




void read_snap_header_attributes_in_hdf5(int num);
