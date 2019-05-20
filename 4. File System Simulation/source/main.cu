#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define SUPERBLOCK_SIZE 4096 //4KB
#define FCB_SIZE 32 //32 bytes per FCB
#define FCB_ENTRIES 1024
#define STORAGE_SIZE 1085440 //1060KB
#define STORAGE_BLOCK_SIZE 32

#define MAX_FILENAME_SIZE 20 //20 bytes
#define MAX_FILE_NUM 1024
#define MAX_FILE_SIZE 1048576 //1024KB

#define G_WRITE 0  // operation code for write
#define G_READ 1  // operation code for read
#define LS_D 2  // operation code for listing all files sorted by modified time
#define LS_S 3  // operation code for listing all files sorted by file size
#define RM 4  // operation code for remove
#define INVALID 0xffffffff

#define FILE_CONTENT_START (STORAGE_SIZE - MAX_FILE_SIZE)  // file content stores from here
#define NUM_BLOCKS (MAX_FILE_SIZE / STORAGE_BLOCK_SIZE)  // number of total storage blocks, 32768
#define NUM_BIT_VECTOR_ELEMENT (SUPERBLOCK_SIZE / 4)  // number of bit_vector's elements, 1024

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

__device__ __managed__ uchar *volume_d;  // the volume size is STORAGE_SIZE bit
// __device__ uchar volume_d[STORAGE_SIZE];

typedef struct FileControlBlock
{  /* size of an FCB = 20 + 1 + 1 + 2 + 4 + 2 + 2 = 32 bytes */
	char file_name[MAX_FILENAME_SIZE];  // file name, 20 bytes
	u8 used;  // 1 byte
	u8 mode;  // G_WRITE or G_READ, 1 byte
	u16 block_idx;  // 2 bytes
	u32 file_size;  // size of a file, 4 bytes
	u16 created_time;  // create time of a file, 2 bytes
	u16 modified_time;  // modified time of a file, 2 bytes
} FCB;

__device__ __managed__ u32 *bit_vector;  // num of total blocks is 32768, so we need 32768 bits as the bit vector
__device__ __managed__ FCB *fcbs;  // file control blocks
__device__ __managed__ u16 sys_clock;  // system clock (global), use 2 clock to avoid overflow


void init_volume()
{
	int i, j;
	bit_vector = (u32*)volume_d;  // allocate the head space in volume_d for bit_vector (4KB)
	fcbs = (FCB*)(volume_d + SUPERBLOCK_SIZE);  // allocate the following space in volume_d for the table of FCBs (32KB)

	sys_clock = 0;

	for (i = 0; i < NUM_BIT_VECTOR_ELEMENT; i++)
	{  /* initialize bit_vector */
		bit_vector[i] = 0;
	}

	for (i = 0; i < FCB_ENTRIES; i++)
	{  /* initialize the table of FCBs */
		for (j = 0; j < MAX_FILENAME_SIZE; j++)
		{
			fcbs[i].file_name[j] = '\0';
		}
		fcbs[i].used = 0;
		fcbs[i].block_idx = INVALID;
		fcbs[i].mode = G_READ;
		fcbs[i].file_size = 0;
		fcbs[i].created_time = sys_clock;
		fcbs[i].modified_time = sys_clock;
	}
}


__device__ u32 get_bit_vector_at(u32 idx_in_storage_block)
{
	/* Return the bit value at given index in the storage block.
	 * Note it stores in Littel Endian.
	 * ---------------------------------------------------------
	 * idx_in_storage_block: index in the storage block, from 0 to 32767
	 * bit_num: the index in the bit_vector to find idx_in_storage_block, from 0 to 1023
	 * offset: the offset from most right to the target bit
	 */
	u32 bit_num = idx_in_storage_block / STORAGE_BLOCK_SIZE;
	u32 offset = idx_in_storage_block % STORAGE_BLOCK_SIZE;
	return ((bit_vector[bit_num] >> offset) & 1);
}


__device__ u32 set_bit_vector_at(u32 idx_in_storage_block, u32 value)
{
	/** Set the bit value at given index in the storage blcok.
	 * Note it stores in Little Endian.
	 * ---------------------------------------------------------
	 * idx_in_storage_block: index in the storage block, from 0 to 32767
	 * bit_num: the index in the bit_vector to find idx_in_storage_block, from 0 to 1023
	 * offset: the offset from most right to the target bit
	 */
	u32 bit_num = idx_in_storage_block / STORAGE_BLOCK_SIZE;
	u32 offset = idx_in_storage_block % STORAGE_BLOCK_SIZE;
	if (value == 0)
		bit_vector[bit_num] &= (0x0 << offset);
	else
		bit_vector[bit_num] |= (0x1 << offset);
}


__device__ void str_copy(const char *source, char *destination, int size)
{
	/* copy characters from source to destination */
	int i;
	for (i = 0; source[i] != '\0' && i < size; i++)
	{
		destination[i] = source[i];
	}
}


__device__ bool strs_are_matched(const char *str1, const char *str2)
{
	/* return true if two strings are indentical */
	int i = -1;
	do {
		i++;
		if (str1[i] != str2[i])
		{
			return false;
		}
	} while (str1[i] != '\0' && str2[i] != '\0');
	return true;
}


__device__ void swap_FCB(int FCB1_idx, int FCB2_idx)
{
	/* swap FCB1 with FCB2
	 * --------------------
	 * FCB1_idx: index of FCB1 in fcbs
	 * FCB2_idx: index of FCB2 in fcbs
	 */

	FCB temp;
	temp = fcbs[FCB1_idx];
	fcbs[FCB1_idx] = fcbs[FCB2_idx];
	fcbs[FCB2_idx] = temp;
}


__device__ u32 find_file_by_name(const char *file_name)
{
	/* If the file exist, return its index in FCB,
	 * otherwise return INVALID.
	 * -------------------------
	 * file_name: the name of required file
	 */

	int i;
	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].file_name[0] == '\0') continue;
		if (strs_are_matched(fcbs[i].file_name, file_name) && fcbs[i].used == 1)
		{
			return i;
		}
	}
	return (u32)INVALID;
}


__device__ void clear_file_content(u16 idx_in_FCBs)
{
	int i;
	u32 offset = fcbs[idx_in_FCBs].block_idx * STORAGE_BLOCK_SIZE;
	u32 required_num_blocks = (fcbs[idx_in_FCBs].file_size % STORAGE_BLOCK_SIZE == 0) ? (fcbs[idx_in_FCBs].file_size / STORAGE_BLOCK_SIZE) : (fcbs[idx_in_FCBs].file_size / STORAGE_BLOCK_SIZE + 1);
	for (i = 0; i < fcbs[idx_in_FCBs].file_size; i++)
	{
		volume_d[FILE_CONTENT_START + offset + i] = '\0';
	}
	for (i = 0; i < required_num_blocks; i++)
	{
		set_bit_vector_at(fcbs[idx_in_FCBs].block_idx + i, 0);
	}

	fcbs[idx_in_FCBs].used = 0;
}

__device__ void avoid_modified_time_overflow()
{

	int i, j = 0;

	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].used != 1) continue;
		for (j = 0; j < FCB_ENTRIES; j++)
		{
			if (fcbs[i].used != 1 || i == j) continue;
			if (fcbs[i].modified_time < fcbs[j].modified_time)
			{
				swap_FCB(i, j);
			}
		}
	}

	sys_clock = 1;

	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].used == 0)
		{
			fcbs[i].modified_time = sys_clock;
			sys_clock++;
		}
	}
}

__device__ void sort_by_modifed_time()
{
	/* sort files by modifed time */
	int i, j;
	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].used != 1) continue;
		// if (fcbs[i].file_name[0] == '\0') continue;
		for (j = 0; j < FCB_ENTRIES; j++)
		{
			if (fcbs[j].used != 1 || i == j) continue;
			// if (fcbs[j].file_name[0] == '\0' || i == j) continue;
			if (fcbs[i].modified_time > fcbs[j].modified_time)
			{
				swap_FCB(i, j);
			}
		}
	}
}


__device__ void sort_by_file_size()
{
	/* sort files by file size, if */
	int i, j;
	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].used != 1) continue;
		// if (fcbs[i].file_name[0] == '\0') continue;
		for (j = 0; j < FCB_ENTRIES; j++)
		{
			if (fcbs[j].used != 1 || i == j) continue;
			// if (fcbs[j].file_name[0] == '\0' || i == j) continue;
			if (fcbs[i].file_size > fcbs[j].file_size)
			{
				swap_FCB(i, j);
			}
			else if (fcbs[i].file_size == fcbs[j].file_size)
			{  /* sort by created time if two files have the same size */
				if (fcbs[i].created_time < fcbs[j].created_time)
				{
					swap_FCB(i, j);
				}
			}
		}
	}
}


__device__ u32 open(char *s, int op)
{
	/* Open a given file according to file_name s with operation op,
	 * Return the index of file in FCB.
	 * --------------------------------
	 * s: file name
	 * op: operation code
	 */

	int i;
	u32 idx_in_FCBs;
	idx_in_FCBs = find_file_by_name(s);

	//printf("Get idx_in_FCBs: %d\n\n", idx_in_FCBs);

	if (idx_in_FCBs != INVALID)  // find an existing file with the given file name
	{
		fcbs[idx_in_FCBs].mode = op;
	}
	else
	{
		if (op == G_WRITE)
		{  /* create a new file */
			for (i = 0; i < FCB_ENTRIES; i++)
			{
				if (fcbs[i].used != 1)  // find a free block
				{
					str_copy(s, fcbs[i].file_name, MAX_FILENAME_SIZE);
					fcbs[i].used = 1;
					fcbs[i].block_idx = INVALID;
					fcbs[i].mode = op;
					fcbs[i].file_size = 0;
					fcbs[i].created_time = ++sys_clock;
					idx_in_FCBs = i;  // update return value
					break;
				}
			}
		}
		else
		{  /* try to read an unexist file */
			printf("Fail to open! Please check your filename or operation code.\n");
		}
	}
	//printf("Going to return %d\n\n", idx_in_FCBs);
	return idx_in_FCBs;
}


__device__ void read(uchar *output, u32 size, u32 fp)
{
	/* Read a file according to fp and then load the content to output
	 * -------------------------------------------------------------------------
	 * output: the buffer that stores output content
	 * size: size in terms of bit
	 * fp: index in the table of FCBs
	 */

	if (fp >= FCB_ENTRIES)  // try to access an non-existing FCB
	{
		printf("Cannot read a file at %d, operation abort.\n", fp);
		return;
	}

	int i;
	u32 offset = fcbs[fp].block_idx * STORAGE_BLOCK_SIZE;
	for (i = 0; i < size; i++)
	{
		output[i] = volume_d[FILE_CONTENT_START + offset + i];
	}
}


__device__ u32 write(uchar* input, u32 size, u32 fp)
{
	/* Write a file according to fp from the content of input
	 * -------------------------------------------------------------------------
	 * input: the buffer that stores input content
	 * size: size in terms of bit
	 * fp: index in the table of FCBs
	 */

	if (fp >= FCB_ENTRIES)  // try to access an non-existing FCB
	{
		printf("Cannot write a file at %d, operation abort.\n", fp);
		return INVALID;
	}

	// fp = (u16)fp;
	clear_file_content(fp);  // file exists, clearup the older content
	//printf("File cleaned.\n\n");

	int i;
	u32 index = 0, count = 0;
	u32 required_num_blocks = (size % STORAGE_BLOCK_SIZE == 0) ? (size / STORAGE_BLOCK_SIZE) : (size / STORAGE_BLOCK_SIZE + 1);
	while (index < NUM_BLOCKS)
	{
		if (get_bit_vector_at(index) == 1)
		{  /* reset if the number of free blocks is not enough */
			count = 0;
			index++;
			continue;
		}
		else
		{
			count++;
			index++;
			if (count == required_num_blocks) break;
		}
	}
	if (count < required_num_blocks && index >= NUM_BLOCKS)  // there is no space for this file
	{
		printf("required_num_blocks: %d, count: %d at %d\n", required_num_blocks, count, index);
		printf("There is no contiguous space to store a(n) %d-bit file.\n", size);
		fcbs[fp].used = 0;
		return INVALID;
	}
	//printf("Find enough space.\n\n");

	fcbs[fp].used = 1;
	fcbs[fp].block_idx = index - required_num_blocks;  // update the block index of the file
	fcbs[fp].file_size = size;
	fcbs[fp].modified_time = ++sys_clock;
	for (i = 0; i < required_num_blocks; i++)
	{
		set_bit_vector_at(fcbs[fp].block_idx + i, 1);
	}
	//printf("Bit vecotr set.\n\n");
	u32 offset = fcbs[fp].block_idx * STORAGE_BLOCK_SIZE;
	//printf("The offset is %d.\n\n", offset);
	for (i = 0; i < size; i++)
	{
		volume_d[FILE_CONTENT_START + offset + i] = input[i];
	}

	if (sys_clock >= (u16)INVALID) {
		avoid_modified_time_overflow();
	}
	return 0;
}


__device__ void gsys(int op)
{
	/* Implement LS_D and LS_S operation here */
	int i;
	switch (op)
	{
	case LS_D:
		printf("=== sort by modified time ===\n");
		sort_by_modifed_time();
		break;
	case LS_S:
		printf("=== sort by file size ===\n");
		sort_by_file_size();
		break;
	default:
		printf("Operation code error, please check your input!\n");
		return;
	}

	/* print sorted result */
	for (i = 0; i < FCB_ENTRIES; i++)
	{
		if (fcbs[i].file_name[0] == '\0' || fcbs[i].used != 1) continue;
		if (op == LS_S)
			printf("File name: %s , file size: %d\n", fcbs[i].file_name, fcbs[i].file_size);
		else
			printf("File name: %s, file modified at clock %d\n", fcbs[i].file_name, fcbs[i].modified_time);
	}
}


__device__ void gsys(int op, char *s)
{
	/* Implement rm operation here */
	int i;
	if (op != RM)
	{
		printf("Operation code should be RM, operation abort.\n");
		return;
	}
	u32 idx_in_FCBs = find_file_by_name(s);
	if (idx_in_FCBs != INVALID)
	{
		clear_file_content(idx_in_FCBs);
		for (i = 0; i < MAX_FILENAME_SIZE; i++)
		{
			fcbs[idx_in_FCBs].file_name[i] = '\0';
		}
		//fcbs[idx_in_FCBs].used = 0;
		fcbs[idx_in_FCBs].mode = G_READ;
		fcbs[idx_in_FCBs].created_time = 0;
		fcbs[idx_in_FCBs].modified_time = 0;
		fcbs[idx_in_FCBs].file_size = 0;
		fcbs[idx_in_FCBs].block_idx = INVALID;
	}
}


__global__ void mykernel(uchar *input, uchar *output)
{
	// kernel test1 start  

	//u32 fp = open("t.txt\0", G_WRITE);
	//write(input, 64, fp);

	//fp = open("b.txt\0", G_WRITE);
	//write(input + 32, 32, fp);

	//fp = open("t.txt\0", G_WRITE);
	//write(input + 32, 32, fp);

	//fp = open("t.txt\0", G_READ);
	//read(output, 32, fp);

	//gsys(LS_D);

	//gsys(LS_S);

	//fp = open("b.txt\0", G_WRITE);
	//write(input + 64, 12, fp);

	//gsys(LS_S);
	//gsys(LS_D);

	//gsys(RM, "t.txt\0");
	//gsys(LS_S);

	// kernel test1 end

	//kernel test2 start   

	/*u32 fp = open("t.txt\0", G_WRITE);
	write(input, 64, fp);

	fp = open("b.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	fp = open("t.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	fp = open("t.txt\0", G_READ);
	read(output, 32, fp);

	gsys(LS_D);

	gsys(LS_S);

	fp = open("b.txt\0", G_WRITE);
	write(input + 64, 12, fp);

	gsys(LS_S);
	gsys(LS_D);

	gsys(RM, "t.txt\0");
	gsys(LS_S);

	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = open(fname[i], G_WRITE);
		write(input + i, 24 + i, fp);
	}

	gsys(LS_S);

	for (int i = 0; i < 5; i++)
		gsys(RM, fname[i]);

	gsys(LS_D);*/

	// kernel test2 end

	//kernel test3 start  

	u32 fp = open("t.txt\0", G_WRITE);
	write(input, 64, fp);

	fp = open("b.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	fp = open("t.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	fp = open("t.txt\0", G_READ);
	read(output, 32, fp);

	gsys(LS_D);
	gsys(LS_S);

	fp = open("b.txt\0", G_WRITE);
	write(input + 64, 12, fp);

	gsys(LS_S);
	gsys(LS_D);

	gsys(RM, "t.txt\0");

	gsys(LS_S);

	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = open(fname[i], G_WRITE);
		write(input + i, 24 + i, fp);
	}

	gsys(LS_S);

	for (int i = 0; i < 5; i++)
		gsys(RM, fname[i]);

	gsys(LS_D);

	//

	char fname2[1018][20];
	int p = 0;

	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++)
		{
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}

	for (int i = 0; i < 1001; i++)
	{
		fp = open(fname2[i], G_WRITE);
		write(input + i, 24 + i, fp);
	}

	gsys(LS_S);

	//

	fp = open(fname2[1000], G_READ);
	read(output + 1000, 1024, fp);

	char fname3[17][3];
	for (int i = 0; i < 17; i++)
	{
		fname3[i][0] = 97 + i;
		fname3[i][1] = 97 + i;
		fname3[i][2] = '\0';
		fp = open(fname3[i], G_WRITE);
		write(input + 1024 * i, 1024, fp);
	}

	fp = open("EA\0", G_WRITE);
	write(input + 1024 * 100, 1024, fp);

	gsys(LS_S);

	//kernel test3 end

}


__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}


__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}


int main()
{
	cudaMallocManaged(&volume_d, STORAGE_SIZE);
	init_volume();
	printf("init_volume() ok.\n");

	uchar *input_h;
	uchar *input;

	uchar *output_h;
	uchar *output;

	input_h = (uchar *)malloc(sizeof(uchar)*MAX_FILE_SIZE);
	output_h = (uchar *)malloc(sizeof(uchar)*MAX_FILE_SIZE);


	cudaMalloc(&input, sizeof(uchar)*MAX_FILE_SIZE);
	cudaMalloc(&output, sizeof(uchar)*MAX_FILE_SIZE);

	// load binary file from data.bin
	load_binaryFile(DATAFILE, input_h, MAX_FILE_SIZE);


	cudaMemcpy(input, input_h, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(output, output_h, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyHostToDevice);

	mykernel << <1, 1 >> > (input, output);

	cudaMemcpy(output_h, output, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyDeviceToHost);

	// dump output array to snapshot.bin 
	write_binaryFile(OUTFILE, output_h, MAX_FILE_SIZE);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
