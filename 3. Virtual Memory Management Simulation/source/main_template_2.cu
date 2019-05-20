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

//page table entries 
__device__ __managed__ int PAGE_ENTRIES = 0;
//count the pagefault times
__device__ __managed__ int PAGEFAULT_NUM = 0;
//__shared__ int PAGEFAULT_NUM;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//page table
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




__global__ void mykernel(int input_size)
{

	//take shared memory as physical memory   
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	//get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE / PAGESIZE;

	//before first Gwrite or Gread 
	init_pageTable(pt_entries);


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
	//printf("fileLen: %ld\n", fileLen);

	

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
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);


	mykernel << <1, 1, 16384 >> > (input_size);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	printf("input size: %d\n", input_size);



	cudaDeviceSynchronize();
	cudaDeviceReset();



	write_binaryFile(OUTFILE, results, input_size);


	printf("pagefault number is %d\n", PAGEFAULT_NUM);


	return 0;
}