#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>


//page size is 32bytes
#define PAGESIZE 32  

//32 KB in shared memory
#define PHYSICAL_MEM_SIZE 32768 

//128 KB in global memory
#define STORAGE_SIZE 131072 

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;


__device__ int PAGE_ENTRIES = 0;
__device__ uchar storage[STORAGE_SIZE];
__device__ int PAGEFAULT_NUM_d[1];
extern __shared__ u32 pt[];




__device__ void init_pageTable(int entries)
{
	PAGE_ENTRIES = entries;

	for (int i = 0; i < entries; i++)
	{
		pt[i] = 0x80000000; //invalid
		pt[i + PAGE_ENTRIES] = i;
	}
}



__device__ uchar Gread(uchar *buffer, u32 addr)
{
	/* Complate Gread function to read value from data buffer */
}

__device__ void Gwrite(uchar *buffer, u32 addr, uchar value)
{
	/* Complete Gwrite function to write value to data buffer */
}

__device__ void snapshot(uchar *results, uchar* buffer, int offset, int input_size)
{
	/* Complete snapshot function to load elements from data to result */
}


__global__ void mykernel(int input_size, uchar *input, uchar *results, int *PAGEFAULT_NUM)
{

	//take shared memory as physical memory   
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	//get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE / PAGESIZE;

	//before first Gwrite or Gread 
	init_pageTable(pt_entries);

	// Gread, Gwrite and snapshot are the access pattern for testing page replacement 
	for (int i = 0; i < input_size; i++)
		Gwrite(data, i, input[i]);

	for (int i = input_size - 1; i >= input_size - 32769; i--)
		int value = Gread(data, i);

	snapshot(results, data, 0, input_size);

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
	cudaError_t cudaStatus;

	uchar *input_h;
	uchar *input;

	uchar *results_h;
	uchar *results;

	int *PAGEFAULT_NUM_h;
	int *PAGEFAULT_NUM;

	cudaMalloc(&input, sizeof(uchar)*STORAGE_SIZE);
	cudaMalloc(&results, sizeof(uchar)*STORAGE_SIZE);
	cudaMalloc(&PAGEFAULT_NUM, sizeof(int));

	input_h = (uchar *)malloc(sizeof(uchar)*STORAGE_SIZE);
	results_h = (uchar *)malloc(sizeof(uchar)*STORAGE_SIZE);
	PAGEFAULT_NUM_h = (int *)malloc(sizeof(int) * 1);
	PAGEFAULT_NUM_h[0] = 0;

	int input_size;

	input_size = load_binaryFile(DATAFILE, input_h, STORAGE_SIZE);

	cudaMemcpy(input, input_h, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(results, results_h, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(PAGEFAULT_NUM, PAGEFAULT_NUM_h, sizeof(int) * 1, cudaMemcpyHostToDevice);

	/* Launch kernel function in GPU, with single thread
	and dynamically allocate 16384 bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel << <1, 1, 16384 >> > (input_size, input, results, PAGEFAULT_NUM);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}


	cudaMemcpy(input_h, input, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(results_h, results, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(PAGEFAULT_NUM_h, PAGEFAULT_NUM, sizeof(int), cudaMemcpyDeviceToHost);

	write_binaryFile(OUTFILE, input_h, input_size);

	printf("pagefault number is %d\n", PAGEFAULT_NUM_h[0]);

	cudaFree(input);
	cudaFree(results);
	cudaFree(PAGEFAULT_NUM);

	free(input_h);
	free(results_h);
	free(PAGEFAULT_NUM_h);

	cudaDeviceSynchronize();
	cudaDeviceReset();


	return 0;
}