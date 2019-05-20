#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>


// page size is 32bytes
#define PAGESIZE 32  
// 32 KB in shared memory
#define PHYSICAL_MEM_SIZE 32768 
// 128 KB in global memory
#define STORAGE_SIZE 131072

/* value of invalid in valid-invalid bit */
#define INVALID 0x80000000
/* value of valid in valid-invalid bit */
#define VALID 0x76

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

// page table entries 
__device__ __managed__ int PAGE_ENTRIES = 0;
// count the pagefault times
__device__ __managed__ int PAGEFAULT_NUM = 0;

// secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

// data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

/* LRU clock */
__device__ __managed__ u32 LRU_clock;
/* total pages of the storage */
__device__ __managed__ u32 TOTAL_PAGES = STORAGE_SIZE / PAGESIZE;

// page table
extern __shared__ u32 pt[];


__device__ void init_pageTable(int entries)
{
	PAGE_ENTRIES = entries;
	PAGEFAULT_NUM = 0;
	LRU_clock = 0;

	for (int i = 0; i < entries; i++)
	{  // i represents the i-th frame in the physical memory
		pt[i] = INVALID;  // pt[i] stores valid-invalid bit, here set all valid-invalid bit to invalid
		pt[i + PAGE_ENTRIES] = i;  // pt[i + PAGE_EMTROES] stores the page number
		pt[i + 2 * PAGE_ENTRIES] = 0;  // pt[i + 2 * PAGE_ENTRIES] stores the counter
	}

}

__device__ void migrate_block(uchar *from_buffer, uchar *to_buffer, u32 from_page_num, u32 to_page_num)
{
	/* Migrate a whole block from from_buffer to to_buffer
	 * ---------------------------------------------------
	 * from_buffer: the memory of source block
	 * to_buffer: the memory of destination blcok
	 * from_page_num: the index of source block in from_buffer
	 * to_page_num: the index of destination block in to_buffer
	 */
	int i, from_idx = from_page_num * PAGESIZE, to_idx = to_page_num * PAGESIZE;
	for (i = 0; i < PAGESIZE; i++, from_idx++, to_idx++)
	{
		to_buffer[to_idx] = from_buffer[from_idx];
	}
}

__device__ void LRU_replacement(uchar *buffer, u32 page_num)
{
	/* Find the least recently used page in physical memory and then 
	 * swap it out to storage and swap in the needed page from storage
	 * -----------------------------------------------------------------------------------------------------
	 * buffer: physical memory (in shared memory)
	 * page_num: the page number of referenced (needed) page
	 * victim_frame: the block index in physical memory (from 0 to PAGE_ENTRIES-1)
	 * victim_page: the block index in logical memory (from 0 to TOTAL_PAGES-1)
	 */

	/* find least recently used page, whose counter field's value is minimum */
	u32 min = buffer[PAGE_ENTRIES], victim_frame, victim_page;
	for (int i = 0; i < PAGE_ENTRIES; i++)
	{
		if (buffer[i + 2 * PAGE_ENTRIES] < min)
		{
			min = buffer[i + 2 * PAGE_ENTRIES];
			victim_frame = i;
		}
	}
	victim_page = pt[victim_frame + PAGE_ENTRIES];  // get victim page number
	migrate_block(buffer, storage, victim_frame, victim_page);  // migrate victim page from physical memory to storage
	migrate_block(storage, buffer, page_num, victim_frame);  // migrate referenced page from storage to physical memory
	pt[victim_frame + PAGE_ENTRIES] = page_num;  // update page number in the frame
}

__device__ u32 paging(uchar *buffer, u32 page_num, u32 offset)
{
	/* Return the index in buffer of the referenced page,
	 * paging if the referenced page is not on the physical memory
	 * -----------------------------------------------------------
	 * buffer: physical memory (in shared memory)
	 * page_num: the referenced page number (from 0 to TOTAL_PAGES-1)
	 * offset: the offset from the beginning of one page (from 0 to PAGESIZE-1)
	 */

	int i;
	u32 addr_in_phy_mem;

	/* check if the reference is valid */
	if (page_num > TOTAL_PAGES - 1)
	{
		printf("Segmentation fault! The valid page number should within 0 and %d", TOTAL_PAGES - 1);
		PAGEFAULT_NUM++;
		return EXIT_FAILURE;
	}

	/* check if the referenced page is on the physical memory */
	for (i = 0; i < PAGE_ENTRIES; i++)
	{
		if (pt[i] == VALID && pt[i + PAGE_ENTRIES] == page_num)  // referenced page is on the physical memory
		{
			pt[i + 2 * PAGE_ENTRIES] = ++LRU_clock;  // update LRU counter
			addr_in_phy_mem = i * PAGESIZE + offset;
			return addr_in_phy_mem;
		}
	}

	/* handling the situation if referenced page is not on the physical memory */
	PAGEFAULT_NUM++;
	/* check if there is free frame */
	for (i = 0; i < PAGE_ENTRIES; i++)
	{
		if (pt[i] == INVALID)  // find free frame
		{
			migrate_block(storage, buffer, page_num, i);  // swap desired page into physical memory
			pt[i] = VALID;  // set this frame's valid-invalid bit as valid
			pt[i + PAGE_ENTRIES] = page_num;  // update page number in the frame
			pt[i + 2 * PAGE_ENTRIES] = ++LRU_clock;  // update LRU counter
			addr_in_phy_mem = i * PAGESIZE + offset;
			return addr_in_phy_mem;
		}
	}
	/* all the frames are occupied, need to swap out a victim page and swap in the referenced page */
	LRU_replacement(buffer, page_num);  // use LRU algorithm to do page replacement
	return paging(buffer, page_num, offset);
}


__device__ uchar Gread(uchar *buffer, u32 addr)
{
	/* Read value from data buffer
	 * ------------------------------------------
	 * buffer: physical memory (in shared memory)
	 * addr: address in logical memory (from 0 to STORAGE_SIZE-1)
	 */

	u32 page_num = addr / PAGESIZE;
	u32 offset = addr % PAGESIZE;

	u32 idx = paging(buffer, page_num, offset);
	return buffer[idx];
}

__device__ void Gwrite(uchar *buffer, u32 addr, uchar value)
{
	/* Write value to data buffer 
	 * ------------------------------------------
	 * buffer: physical memory (in shared memory)
	 * addr: address in logical memory (from 0 to STORAGE_SIZE-1)
	 * value: the value that is being written
	 */

	u32 page_num = addr / PAGESIZE;
	u32 offset = addr % PAGESIZE;

	u32 idx = paging(buffer, page_num, offset);
	buffer[idx] = value;
}

__device__ void snapshot(uchar *results, uchar* buffer, int input_size)
{
	/* Load elements from buffer to results
	 * ------------------------------------------
	 * results: secondary storage (in global memory)
	 * buffer: physical memory (in shared memory)
	 * inputsize: the size of input
	 */

	for (int i = 0; i < input_size; i++)
	{
		results[i] = Gread(buffer, i);
	}
}



__global__ void mykernel(int input_size)
{
	// take shared memory as physical memory   
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	// get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE / PAGESIZE;

	// before first Gwrite or Gread 
	init_pageTable(pt_entries);

	for (int i = 0; i < input_size; i++)  // it will cause 4096 page faults in this case
		Gwrite(data, i, input[i]);

	for (int i = input_size - 1; i >= input_size - 32769; i--)  // cause 1 page fault
		int value = Gread(data, i);

	snapshot(results, data, input_size);  // cause 4096 page faults
}


__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}


/* load a file according to fileName into the buffer and return the size of this file */
__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)	
	{
		printf("***Unable to open file %s***\n", fileName);	
		exit(1);
	}

	// Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	
	fseek(fp, 0, SEEK_SET);
	// printf("fileLen: %ld\n", fileLen);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");	
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	// Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	
	fclose(fp);
	
	return fileLen;
}


int main()
{
	cudaError_t cudaStatus;
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	/* Launch kernel function in GPU, with single thread
	and dynamically allocate 16384 bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel <<<1, 1, 16384 >>> (input_size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	//printf("input size: %d\n", input_size);  // in this case, input_size is 131072

	cudaDeviceSynchronize();
	cudaDeviceReset();

	write_binaryFile(OUTFILE, results, input_size);

	printf("pagefault number is %d\n", PAGEFAULT_NUM);

	return 0;
}