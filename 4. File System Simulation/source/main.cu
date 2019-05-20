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

typedef unsigned char uchar;
typedef uint32_t u32;

__device__ uchar volume_d[STORAGE_SIZE];


__device__ u32 open(char *s, int op)
{
	/* Implement open operation here */
}


__device__ void read(uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
}

__device__ u32 write(uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
}

__device__ void gsys(int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void gsys(int op, char *s)
{
	/* Implement rm operation here */
}


__global__ void mykernel(uchar *input, uchar *output)
{
	//kernel test start

	/* Complete your test case for file opereation here! */

	// kernel test end
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

	mykernel << <1, 1 >> >(input, output);

	cudaMemcpy(output_h, output, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyDeviceToHost);

	// dump output array to snapshot.bin 
	write_binaryFile(OUTFILE, output_h, MAX_FILE_SIZE);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
