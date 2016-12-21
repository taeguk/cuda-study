

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include <assert.h>

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1

#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START() { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START
#define CHECK_TIME_END(a)
#endif

#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL( cudaEventDestroy( cuda_timer_start ) );
	CUDA_CALL( cudaEventDestroy( cuda_timer_stop ) );
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		//wait_exit();
		exit(1);
	}
}


/**********************************************************************************************/
/***************************************** Constants ******************************************/
/**********************************************************************************************/

const unsigned REPEAT_COUNT = 1U;
const unsigned CUDA_WARP_SIZE = 32U;
const unsigned ELEM_PER_VECTOR = 32U;


/**********************************************************************************************/
/*************************************** Some Functions ***************************************/
/**********************************************************************************************/

inline float absIEEE754(float f)
{
	return (float&)((int&)f &= 0x7fffffff);
}

float GetErrorRate(float* A, float* B, int size)
{
	int cnt = 0;
	const float epsilon = 0.000005f;
	for (int i = 0; i < size; ++i)
	{
		if (absIEEE754(A[i] - B[i]) > epsilon)
		{
			//printf("[%d][%d]: %f != %f\n", i / ELEM_PER_VECTOR, i % ELEM_PER_VECTOR, cpu_result[i], gpu_result[i]);
			cnt++;
		}
		else {
			//printf("[%d][%d]: %f == %f\n", i / ELEM_PER_VECTOR, i % ELEM_PER_VECTOR, cpu_result[i], gpu_result[i]);
		}
	}
	return float(cnt) / size * 100.f;
}

float GetErrorRateTranspose(float* AOS, float* SOA, int size)
{
	int N = size / ELEM_PER_VECTOR;
	int cnt = 0;
	const float epsilon = 0.000005f;
	for (int r = 0; r < N; ++r){
		for (int c = 0; c<ELEM_PER_VECTOR; ++c) {
			if (absIEEE754(AOS[r * ELEM_PER_VECTOR + c] - SOA[c * N + r]) > epsilon)
			{
				//printf("************ [%d][%d]: %f != %f\n", r, c, AOS[r * ELEM_PER_VECTOR + c], SOA[c * N + r]);
				cnt++;
			}
			else {
				//printf("[%d][%d]: %f == %f\n", r, c, AOS[r * ELEM_PER_VECTOR + c], SOA[c * N + r]);
			}
		}
	}
	return float(cnt) / size * 100.f;
}

__global__ void init_cudaY(float *y, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_thread = gridDim.x * blockDim.x;
	for (int i = tid; i < size; i += num_thread)
		y[i] = 0;
}


/**********************************************************************************************/
/****************************************** CPU ***********************************************/
/**********************************************************************************************/

void CPU_AOS(float *Y, float *M, float *X, int N)
{
	for (int i = 0; i<N; ++i) {
		for (int j = 0; j<ELEM_PER_VECTOR; ++j) {
			float result = 0;
			for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
				result +=
					X[i * ELEM_PER_VECTOR + k] *
					M[k * ELEM_PER_VECTOR + j];
			}
			Y[i * ELEM_PER_VECTOR + j] = result;
		}
	}
}

/*
	CPU_AOS���� inner 2�� loop����, X�� row 1���� access pattern�� �����ؼ� cache�� �̵��� ����.
	CPU_AOS_2���� inner 2�� loop����, M�� column 1���� access pattern�� �����ؼ� cache�� �̵��� ����.
	�ٵ� CPU_AOS_2�� CPU_AOS���� ���־��� �ణ ������ �� ������, 
	CPU_AOS_2������ X�� ������ row wise access�� cache�� ������ ���⼭�� �޴µ��� ����,
	CPU_AOS������ M�� column wise access�� cache�� ������ ���ޱ� �������� �����ȴ�.
*/
void CPU_AOS_2(float *Y, float *M, float *X, int N)
{
	for (int j = 0; j<ELEM_PER_VECTOR; ++j) {
		for (int i = 0; i<N; ++i) {
			float result = 0;
			for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
				result +=
					X[i * ELEM_PER_VECTOR + k] *
					M[k * ELEM_PER_VECTOR + j];
			}
			Y[i * ELEM_PER_VECTOR + j] = result;
		}
	}
}

// transpose version (SOA) 
/*
	CPU_SOA�� CPU_AOS�� ���� �ӵ��� 10���̻� ������.
	�� ������ X�� ���� access pattern�� column wise�̱� �����̴�.
	AOS������ M�����ؼ� column wise access�� ������, M�� ũ�Ⱑ �۱⶧���� ����(?)�� ����.
	�׷��� X�� ũ�Ⱑ ŭ���� access pattern�� column wise�̸� ���ذ� Ŀ�� �ӵ��� ����� ������ ���°����� ���δ�.
*/
void CPU_SOA(float *Y, float *M, float *X, int N)
{
	for (int i = 0; i<ELEM_PER_VECTOR; ++i) {
		for (int j = 0; j<N; ++j) {
			float result = 0;
			for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
				result +=
					M[i * ELEM_PER_VECTOR + k] *
					X[k * N + j];
			}
			Y[i * N + j] = result;
		}
	}
}

void CPU_SOA_2(float *Y, float *M, float *X, int N)
{
	for (int j = 0; j<N; ++j) {
		for (int i = 0; i<ELEM_PER_VECTOR; ++i) {
			float result = 0;
			for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
				result +=
					M[i * ELEM_PER_VECTOR + k] *
					X[k * N + j];
			}
			Y[i * N + j] = result;
		}
	}
}


/************************************************************************************************/
/****************************************** GPU 1 ***********************************************/
/************************************************************************************************/

__constant__ float c_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
__constant__ float c_M_trans[ELEM_PER_VECTOR * ELEM_PER_VECTOR];

// thread 1���� vector 1�� ���
// �� thread ���� : N
/*
	������ �ſ� ������. 
	X�� ���� ������ coalescing �� �ƴ϶� warp������ X�� ���� �����͸� �б� ���� 32���� memory transaction�� �ʿ��ϴ�.
	Warp������ M�� ���� ������ �׻� same address��� ���� ������, (constant memory�� ����� ����ȭ�� ȿ������ ���̴�)
	X�� ���� ������ ġ�������� ��ȿ�����̶� ��ü���� ������ �ſ� ������.
*/
__global__ void GPU_1_AOS__VECTOR_PER_THREAD(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				X[vid * ELEM_PER_VECTOR + k] *
				M[k * ELEM_PER_VECTOR + eid];
		}
		Y[vid * ELEM_PER_VECTOR + eid] = result;
	}
}

// thread 1���� element 1�� ���
// �� thread ���� : N * ELEM_PER_VECTOR
/*
	GPU_1_AOS__VECTOR_PER_THREAD ���� �ξ� ����.
	Warp������ X�� ���� ������ �׻� same address��, broadcast�� �Ͼ �ſ� ȿ�����̴�.
	(X�� ũ�Ⱑ Ŀ�� constant memory�� �̿��� ����ȭ�� ���������� bank conflict�� ���� ���̹Ƿ� shared memory�� Ȱ���ϸ� ȿ������ ���̴�.)
	���� Warp������ M�� ���� ������ ���� ������ address�� coalescing�� �Ͼ �ſ� ȿ�����̴�.
	(���� bank conflict�� ���� ���̹Ƿ� shared memory�� Ȱ���ϸ� ȿ������ ���̴�.)
*/
__global__ void GPU_1_AOS__ELEM_PER_THREAD(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	float result = 0;
	for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
		result +=
			X[vid * ELEM_PER_VECTOR + k] *
			M[k * ELEM_PER_VECTOR + eid];
	}
	Y[vid * ELEM_PER_VECTOR + eid] = result;
}

// thread 1���� vector 1��
// �� thread ���� : N
/*
	GPU_1_SOA__ELEM_PER_THREAD ���� �ξ� ����.
	Warp������ X�� ���� ������ ���� ������ address�� coalescing�� �Ͼ �ſ� ȿ�����̴�. 
	(bank conflict�� ���� ���̹Ƿ� shared memory�� Ȱ���ϸ� ȿ������ ���̴�.
	��� �����ϱ� ������, ��������� shared memory�� ����ϸ� �� ������. �� ���� �ؿ� GPU3���� �ٷ� ���̴�.)
	�׸��� Warp������ M�� ���� ������ �׻� same address��, broadcast�� �Ͼ �ſ� ȿ�����̴�. (constant memory�� �̿��� ����ȭ�� �ϸ� ȿ������ ���̴�.)
*/
__global__ void GPU_1_SOA__VECTOR_PER_THREAD(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x; // = tid

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				M[eid * ELEM_PER_VECTOR + k] *
				X[k * N + vid];
		}
		Y[eid * N + vid] = result;
	}
}

// thread 1���� element 1��
// �� thread ���� : N * ELEM_PER_VECTOR
/*
	������ ����� ������.
	M�� ���� ������ coalescing �� �ƴ϶� warp������ X�� ���� �����͸� �б� ���� 32���� memory transaction�� �ʿ��ϴ�.
	X�� ���� ������ �׻� same address��� ���� ������,
	M�� ���� ������ ġ�������� ��ȿ�����̶� ��ü���� ������ �ſ� ������.
*/
__global__ void GPU_1_SOA__ELEM_PER_THREAD(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	float result = 0;
	for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
		result +=
			M[eid * ELEM_PER_VECTOR + k] *
			X[k * N + vid];
	}
	Y[eid * N + vid] = result;
}


/******************************************************************************************************/
/****************************************** GPU 2 (AOS) ***********************************************/
/******************************************************************************************************/

// thread 1���� element 1�� ���
// �� thread ���� : N * ELEM_PER_VECTOR
// X�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
/*
	������ ����� GPU_1_AOS__ELEM_PER_THREAD�� access pattern��θ�, 
	warp������ X�� ���� ������ �׻� same address�� shared memory�� �̿��ϸ� broadcast�� �Ͼ �ſ� ȿ������ ������ �����ȴ�. 
	������ ������ ���, �ణ ��������� �� ���� Ȯ���� �� �־���. 
	�׷��� �׷��� ū ��������� ���� ���ߴµ�, �� ������ ������ X�� ��� global memory�󿡼��� broadcast�� ���� ������ ���Ұ�, 
	X�� ���� ������ row wise�̱� ������ special localityƯ���� ���� cache�� ���� �ñ� �������� �����ȴ�. 
	���� shared memory�� ����������� shared memory�� ���� cost�� ���� ��û���� ū ��������� ��� ��������� �ణ�� ������ ���� ������ ���δ�.
*/
__global__ void GPU_2_AOS__SHARED_X(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	extern __shared__ float s_X[/*blockDim.x*/];

	// s_X �ʱ�ȭ
	s_X[threadIdx.x] = X[tid];	// tid == vid * ELEM_PER_VECTOR + eid

	__syncthreads();

	int svid = (tid % blockDim.x) / ELEM_PER_VECTOR;

	float result = 0;
	for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
		result +=
			s_X[svid * ELEM_PER_VECTOR + k] *
			M[k * ELEM_PER_VECTOR + eid];
	}
	Y[vid * ELEM_PER_VECTOR + eid] = result;
}

// thread 1���� element 1�� ���
// �� thread ���� : N * ELEM_PER_VECTOR
// X, M�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
/*
	GPU_2_AOS__SHARED_X���� �߰������� M�� ���� shared memory�� ����� �����̴�. 
	GPU_2_AOS__SHARED_X�� ����  ���� 2�迡 ����� ��������� �����ش�. 
	X�� shared memory�� ����ص� ū ��������� ���µ��� ����, M�� ���� ū ��������� �����ش�. 
	�� ������ M�� ���� access�� column wise�̱� ������ spacial locality �� �����ʾƼ�
	�̿� ���� cache�� �̵��� ���� M�� shared memory�� �ø��� ��û�� ��������� �Ͼ�� ���̶�� �����ȴ�.
*/
__global__ void GPU_2_AOS__SHARED_X_M(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	__shared__ float s_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
	extern __shared__ float s_X[/*blockDim.x*/];

	// s_M �ʱ�ȭ.
	for (int idx = threadIdx.x; idx < ELEM_PER_VECTOR * ELEM_PER_VECTOR; idx += blockDim.x)
		s_M[idx] = M[idx];

	// s_X �ʱ�ȭ.
	s_X[threadIdx.x] = X[tid];	// tid == vid * ELEM_PER_VECTOR + eid

	__syncthreads();

	int svid = (tid % blockDim.x) / ELEM_PER_VECTOR;

	float result = 0;
	for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
		result +=
			s_X[svid * ELEM_PER_VECTOR + k] *
			s_M[k * ELEM_PER_VECTOR + eid];
	}
	Y[vid * ELEM_PER_VECTOR + eid] = result;
}

// thread 1���� element 1�� ���
// �� thread ���� : N * ELEM_PER_VECTOR
// X�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
// M�� ���� constant memory ���. (�׷��� �̰��� warp���� thread���� ���� �ٸ� address�� �����ϴ� ��������� Bad Usage.)
/*
	GPU_2_AOS__SHARED_X���� �߰������� M�� ���� constant memory�� ����� �����̴�.
	GPU_2_AOS_SHARED_X�� ���� �� 10�谡���� ������ '�϶�'�ϴµ�,
	�̴� M�� ���� access pattern�� same address�� �ƴ϶� memory access�� 32������ ������ �Ͼ ������ ����� �϶��ϱ� �����̴�.
*/
__global__ void GPU_2_AOS__SHARED_X_CONSTANT_M(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	extern __shared__ float s_X[/*blockDim.x*/];

	// s_X �ʱ�ȭ.
	s_X[threadIdx.x] = X[tid];	// tid == vid * ELEM_PER_VECTOR + eid

	__syncthreads();

	int svid = (tid % blockDim.x) / ELEM_PER_VECTOR;

	float result = 0;
	for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
		result +=
			s_X[svid * ELEM_PER_VECTOR + k] *
			c_M[k * ELEM_PER_VECTOR + eid];
	}
	Y[vid * ELEM_PER_VECTOR + eid] = result;
}


/*****************************************************************************************************************/
/******************************************* GPU 3 (SOA, transpose) **********************************************/
/*****************************************************************************************************************/

// thread 1���� vector 1��
// �� thread ���� : N
// X�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
/*
	GPU_1_SOA__VECTOR_PER_THREAD���� �߰������� X�� ���� shared memory�� ����� �����̴�. 
	GPU_1_SOA__VECTOR_PER_THREAD�� ���ؼ� ������ 50%���� ������ '�϶�'�ϴµ� 
	�� ������ ������ X�� ���� access�� coalescing�� ���� ȿ�����̿���, 
	(CUDA_WARP_SIZE * ELEM_PER_VECTOR) (=1024) ũ��. �� 4KB�� �ش��ϴ� X�� �κ� ������ �� ELEM_PER_VECTOR��ŭ iterate�Ǹ� �ݺ��ؼ� ���ٵǱ⿡ 
	temporal locality�� ���� cache�� �̵��� ���� ���� ���̴�. 
	�׷��� X�� shared memory�� �ø����μ� thread�� 256���� ��, 32768bytes��ŭ�� shared memory�� �Ҵ������ ����, 
	SM���� resident�ϴ� block�� ������ 1���� ���ѵǰ� (����ȯ�濡�� shared memory�� ũ��� 48KB��) 
	��, resident warp�� ������ 8���� ���װ�, �̷� ���� M�� global memory���� ���� ��, 
	stall�� �߻��ϸ� scheduling�� warp�� ���� �̷� ���� ������ �϶��Ǵ� ���� �ƴұ��� �����ȴ�.
*/
__global__ void GPU_3_SOA__SHARED_X(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_X �ʱ�ȭ.
	{
		int col = threadIdx.x;
		int col_offset = blockIdx.x * blockDim.x;
		for (int row = 0; row < ELEM_PER_VECTOR; ++row)
			s_X[row * blockDim.x + col] = X[row * N + col_offset + col];
	}

	__syncthreads();
	
	int svid = threadIdx.x;

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				M[eid * ELEM_PER_VECTOR + k] *
				s_X[k * blockDim.x + svid];
		}
		Y[eid * N + vid] = result;
	}
}

// thread 1���� vector 1��
// �� thread ���� : N
// X, M�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
/*
	GPU_3_SOA__SHARED_X���� �߰������� M�� ���� shared memory�� ����� �����̴�.
	�� ��� GPU_3_SOA__SHARED_X�� ���� ������ 3������ ���Ǵµ�, 
	�� ������ �Ƹ� M�� shared memory�� ���������μ� M�� ���� stall�� �߻����� �ʾ� 
	GPU_3_SOA__SHARED_X���� ����� ������ �ذ�Ǿ� ������ ���� ���Ǵ� ������ �����ȴ�.
*/
__global__ void GPU_3_SOA__SHARED_X_M(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	__shared__ float s_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_M �ʱ�ȭ.
	for (int idx = threadIdx.x; idx < ELEM_PER_VECTOR * ELEM_PER_VECTOR; idx += blockDim.x)
		s_M[idx] = M[idx];

	// s_X �ʱ�ȭ.
	{
		int col = threadIdx.x;
		int col_offset = blockIdx.x * blockDim.x;
		for (int row = 0; row < ELEM_PER_VECTOR; ++row)
			s_X[row * blockDim.x + col] = X[row * N + col_offset + col];
	}

	__syncthreads();

	int svid = threadIdx.x;

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				s_M[eid * ELEM_PER_VECTOR + k] *
				s_X[k * blockDim.x + svid];
		}
		Y[eid * N + vid] = result;
	}
}

// thread 1���� vector 1��
// �� thread ���� : N
// X�� ���� shared memory ���. ELEM_PER_VECTOR�� ������� no bank conflict.
// M�� ���� constant memory ���. (warp���� thread���� ���ÿ� ���� �޸𸮸� �����ϹǷ� Good Usage.)
/*
	GPU_3_SOA__SHARED_X���� �߰������� M�� ���� constant memory�� ����� �����̴�. 
	�� ��� GPU_3_SOA__SHARED_X�� ���� �� 7~8��, GPU_3_SOA__SHARED_X_M�� ���� �� 2~3��,  
	GPU_1_SOA__VECTOR_PER_THREAD�� ���� �� 4~5�� ���� ��������.
	GPU_3_SOA__SHARED_X_M�� ���ؼ� ���� ������ M�� shared memory�� ���� ���, block���� �ʱ�ȭ�� �Ҵ��� �ʿ�������, 
	constant memory�� ��ü���� ������ �����ϱ� �����̴�.
*/
__global__ void GPU_3_SOA__SHARED_X_CONSTANT_M(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_X �ʱ�ȭ.
	{
		int col = threadIdx.x;
		int col_offset = blockIdx.x * blockDim.x;
		for (int row = 0; row < ELEM_PER_VECTOR; ++row)
			s_X[row * blockDim.x + col] = X[row * N + col_offset + col];
	}

	__syncthreads();

	int svid = threadIdx.x;

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				c_M_trans[eid * ELEM_PER_VECTOR + k] *
				s_X[k * blockDim.x + svid];
		}
		Y[eid * N + vid] = result;
	}
}

// thread 1���� vector 1��
// �� thread ���� : N
// M�� ���� constant memory ���. (warp���� thread���� ���ÿ� ���� �޸𸮸� �����ϹǷ� Good Usage.)
/*
	GPU_1_SOA__VECTOR_PER_THREAD���� �߰������� M�� ���� constant memory�� ����� �����̴�. 
	������ ���� ���ٰ� ���� �� ������ ���� �̼��ϰ� ����Ͽ��µ�, 
	�Ƹ��� GPU_1_SOA__VECTOR_PER_THREAD���� �̹� M�� ���� cache hit�� ���� ����, 
	Ȥ�� cache miss�� ���� stall�ǵ� GPU_3_SOA__SHARED_X�� ���� resident warp�� ���Ƽ� scheduling�� �Ǳ� ������ 
	constant memory�� ����Ͽ��� ��������� ���� ���� �ƴұ��� �����ȴ�.
*/
__global__ void GPU_3_SOA__CONSTANT_M(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	for (int eid = 0; eid<ELEM_PER_VECTOR; ++eid) {
		float result = 0;
		for (int k = 0; k<ELEM_PER_VECTOR; ++k) {
			result +=
				c_M_trans[eid * ELEM_PER_VECTOR + k] *
				X[k * N + vid];
		}
		Y[eid * N + vid] = result;
	}
}


/****************************************************************************************************/
/***************************************** Main() ***************************************************/
/****************************************************************************************************/

int main()
{
	int n;
	float M[ELEM_PER_VECTOR * ELEM_PER_VECTOR], trans_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
	float *x, *trans_x;
	float *y_cpu_SOA, *y_cpu_AOS, *y_gpu;

	FILE* fp = fopen( "gen.bin", "rb" );
	fread(&n, sizeof(int), 1, fp);
	x = new float[ELEM_PER_VECTOR * n];
	trans_x = new float[ELEM_PER_VECTOR * n];
	y_cpu_SOA = new float[ELEM_PER_VECTOR * n];
	y_cpu_AOS = new float[ELEM_PER_VECTOR * n];
	y_gpu = new float[ELEM_PER_VECTOR * n];
	fread(x, sizeof(float), n * ELEM_PER_VECTOR, fp);
	for (int r = 0; r < n; ++r)
		for (int c = 0; c < ELEM_PER_VECTOR; ++c)
			trans_x[c * n + r] = x[r * ELEM_PER_VECTOR + c];
	fread(M, sizeof(float), ELEM_PER_VECTOR * ELEM_PER_VECTOR, fp);
	for (int r = 0; r < ELEM_PER_VECTOR; ++r)
		for (int c = 0; c < ELEM_PER_VECTOR; ++c)
			trans_M[c * ELEM_PER_VECTOR + r] = M[r * ELEM_PER_VECTOR + c];
	fclose(fp);

	printf("N = %d\n\n", n);

#define CPU_FUNC_TEST(funcname, is_transpose) { \
		float *arg_Y = y_cpu_AOS, *arg_M = M, *arg_X = x; \
		if (is_transpose) { arg_Y = y_cpu_SOA; arg_M = trans_M; arg_X = trans_x; } \
		float totalCPUtime = 0, cpuTime; \
		for( int i = 0; i < REPEAT_COUNT; ++i ) \
		{ \
			CHECK_TIME_START(); \
			funcname(arg_Y, arg_M, arg_X, n); \
			CHECK_TIME_END(cpuTime); \
			totalCPUtime += cpuTime; \
		} \
		printf("Finish " #funcname " calculation\n"); \
		printf(" - Elapsed time: %f (msec)\n\n", totalCPUtime / REPEAT_COUNT); \
	}
#define CPU_FUNC_TEST__MACRO_END

	puts("-------------------------------------------------\n");

	CPU_FUNC_TEST(CPU_AOS, false);
	CPU_FUNC_TEST(CPU_AOS_2, false);
	CPU_FUNC_TEST(CPU_SOA, true);
	CPU_FUNC_TEST(CPU_SOA_2, true);

	printf("AOS and SOA(transpose) compare -> Error rate: %.2f%%\n\n", GetErrorRateTranspose(y_cpu_AOS, y_cpu_SOA, n * ELEM_PER_VECTOR));

	/***********************************CUDA*****************************************/

	CHECK_TIME_INIT_GPU();

	float *cudaM, *cudaM_trans, *cudaX, *cudaX_trans, *cudaY;
	cudaError_t cudaStatus;

	{
		///////////// cudaM
		cudaStatus = cudaMalloc((void**)&cudaM, ELEM_PER_VECTOR * ELEM_PER_VECTOR * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(cudaM, M, ELEM_PER_VECTOR * ELEM_PER_VECTOR * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		///////////// cudaM_trans
		cudaStatus = cudaMalloc((void**)&cudaM_trans, ELEM_PER_VECTOR * ELEM_PER_VECTOR * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(cudaM_trans, trans_M, ELEM_PER_VECTOR * ELEM_PER_VECTOR * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		//////////////// cudaX
		cudaStatus = cudaMalloc((void**)&cudaX, n * ELEM_PER_VECTOR * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(cudaX, x, n * ELEM_PER_VECTOR * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		//////////////// cudaX_trans
		cudaStatus = cudaMalloc((void**)&cudaX_trans, n * ELEM_PER_VECTOR * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(cudaX_trans, trans_x, n * ELEM_PER_VECTOR * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		//////////////////// cudaY
		cudaStatus = cudaMalloc((void**)&cudaY, n * ELEM_PER_VECTOR * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
	}

	cudaMemcpyToSymbol(c_M, M, sizeof(float) * ELEM_PER_VECTOR * ELEM_PER_VECTOR);
	cudaMemcpyToSymbol(c_M_trans, trans_M, sizeof(float) * ELEM_PER_VECTOR * ELEM_PER_VECTOR);

#define GPU_FUNC_TEST_SHARED(funcname, num_block, num_thread, size_shared, is_transpose) { \
	float *compareY = y_cpu_AOS, *arg_M = cudaM, *arg_X = cudaX; \
	if (is_transpose) { compareY = y_cpu_SOA; arg_M = cudaM_trans; arg_X = cudaX_trans; } \
	float totalGPUtime = 0, gpuTime; \
	for( int i = 0; i < REPEAT_COUNT; ++i ) \
			{ \
		init_cudaY <<< n*ELEM_PER_VECTOR/1024, 1024 >>> (cudaY, n * ELEM_PER_VECTOR); \
		cuda_error_check( "ERROR: ", " when init_cudaY() was launched.\n" ); \
		CHECK_TIME_START_GPU(); \
		funcname <<< num_block, num_thread, size_shared >>> ( cudaY, arg_M, arg_X, n); \
		cuda_error_check( "ERROR: ", " when " #funcname "() was launched.\n" ); \
		CHECK_TIME_END_GPU( gpuTime ); \
		totalGPUtime += gpuTime; \
			} \
	printf( "Finish " #funcname " <<< %d, %d >>> calculation\n", num_block, num_thread ); \
	if( size_shared != 0 ) printf( " - Dynamic shared memory size: %d bytes\n", size_shared ); \
	printf( " - Elapsed time: %f (msec)\n", totalGPUtime / REPEAT_COUNT ); \
	CUDA_CALL( cudaMemcpy( y_gpu, cudaY, n * ELEM_PER_VECTOR * sizeof( float ), cudaMemcpyDeviceToHost ) ); \
	CUDA_CALL( cudaDeviceSynchronize() ); \
	float error_rate; \
	if (is_transpose) error_rate = GetErrorRateTranspose(y_cpu_AOS, y_gpu, n * ELEM_PER_VECTOR); \
	else error_rate = GetErrorRate(y_cpu_AOS, y_gpu, n * ELEM_PER_VECTOR); \
	printf(" - Error rate: %.2f%%\n\n", error_rate); \
	}
#define GPU_FUNC_TEST_SHARED__MACRO_END
#define GPU_FUNC_TEST(funcname, num_block, num_thread, is_transpose) GPU_FUNC_TEST_SHARED(funcname, num_block, num_thread, 0, is_transpose)

	puts("-------------------------------------------------\n");

	int num_thread = 256;

	GPU_FUNC_TEST(GPU_1_AOS__VECTOR_PER_THREAD, n / num_thread, num_thread, false);
	GPU_FUNC_TEST(GPU_1_AOS__ELEM_PER_THREAD, n * ELEM_PER_VECTOR / num_thread, num_thread, false);
	GPU_FUNC_TEST(GPU_1_SOA__VECTOR_PER_THREAD, n / num_thread, num_thread, true);
	GPU_FUNC_TEST(GPU_1_SOA__ELEM_PER_THREAD, n * ELEM_PER_VECTOR / num_thread, num_thread, true);

	puts("-------------------------------------------------\n");

	GPU_FUNC_TEST_SHARED(GPU_2_AOS__SHARED_X, n * ELEM_PER_VECTOR / num_thread, num_thread, sizeof(float) * num_thread, false);
	GPU_FUNC_TEST_SHARED(GPU_2_AOS__SHARED_X_M, n * ELEM_PER_VECTOR / num_thread, num_thread, sizeof(float) * num_thread, false);
	GPU_FUNC_TEST_SHARED(GPU_2_AOS__SHARED_X_CONSTANT_M, n * ELEM_PER_VECTOR / num_thread, num_thread, sizeof(float) * num_thread, false);

	puts("-------------------------------------------------\n");

	GPU_FUNC_TEST_SHARED(GPU_3_SOA__SHARED_X, n / num_thread, num_thread, sizeof(float) * num_thread * ELEM_PER_VECTOR, true);
	GPU_FUNC_TEST_SHARED(GPU_3_SOA__SHARED_X_M, n / num_thread, num_thread, sizeof(float) * num_thread * ELEM_PER_VECTOR, true);
	GPU_FUNC_TEST_SHARED(GPU_3_SOA__SHARED_X_CONSTANT_M, n / num_thread, num_thread, sizeof(float) * num_thread * ELEM_PER_VECTOR, true);
	GPU_FUNC_TEST(GPU_3_SOA__CONSTANT_M, n / num_thread, num_thread, true);
	

	printf("- Finish\n\n");
	
	CHECK_TIME_DEST_GPU();

	cudaFree(cudaM);
	cudaFree(cudaM_trans);
	cudaFree(cudaX);
	cudaFree(cudaX_trans);
	cudaFree(cudaY);
	
	CUDA_CALL(cudaDeviceReset());

    return 0;
}