
#include "allvars.h"



void determine_descendants(struct halo_catalogue *catA, struct halo_catalogue *catB, int entry, int snapnum);

int sort_candlist(const void *a, const void *b);
void prepare_index_list(struct halo_catalogue *cat);
void load_subhalo_catalogue(int num, struct halo_catalogue *cat);

void read_parameter_file(char *fname);
int read_outputlist(char* fname);
void get_filename(char *buf, int file_num, int chunk_num, int group_or_snap);

void *mymalloc(size_t n);
void myfree(void *p);
#ifdef BACKWARD_CHECKING
void decide_backwards(struct halo_catalogue *catA, struct halo_catalogue *catB);
#endif
void decide_upon_descendant(void);
void count_progenitors(struct halo_catalogue *catA, struct halo_catalogue *catB);

void save_decendant_list(void);

size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream);
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);
void get_id_translation_table(void);
void long_to_str(char *s, long long n);
int sort_IDType(const void *a, const void *b);

int sort_twoids_ord(const void *a, const void *b);
int sort_twoids_id(const void *a, const void *b);
void reassign_ids(MyIDType N, MyIDType *ids); 
void myfree(void *ptr);


void read_snap_header_attributes_in_hdf5(const char *fname);
void read_particle_ids_in_hdf5(const char *fname, int parttype, MyIDType *IdSnapTable, MyIDType Nskip);

void read_subfind_header_hdf5(const char *fname, int i, struct halo_catalogue *cat, int *nFiles, int *nids, int *nsubhalos, int *ngroups);
void read_subfind_subhalo_hdf5(const char *fname, int i, struct halo_catalogue *cat, int nsubhalos, int subcount);
void read_subfind_group_hdf5(const char *fname, int i, struct halo_catalogue *cat, int ngroups, int groupcount);

void read_basic_subfind_header_hdf5(const char *fname, int i, struct halo_catalogue *cat, int *nFiles, int *nids, int *nsubhalos, int *ngroups);


//void allocate_cat(struct halo_catalogue *cat);
void allocate_subhalo_catalogue(int num, struct halo_catalogue *cat);

void allocate_group_catalogue(int num, struct halo_catalogue *cat);
void load_group_catalogue(int num, struct halo_catalogue *cat);


void write_id_translation_table_hdf5( MyIDType * IdSnapTable, long long TotNumPart);


void read_id_translation_table_hdf5(const char *fname, long long TotNumPart, MyIDType *IdSnapTable );
void read_num_part_table_hdf5(const char *fname, long long *TotNumPart );

