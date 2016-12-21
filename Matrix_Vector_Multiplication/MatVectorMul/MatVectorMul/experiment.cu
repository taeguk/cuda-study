

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
	CPU_AOS에서 inner 2중 loop동안, X의 row 1개가 access pattern이 일정해서 cache의 이득을 본다.
	CPU_AOS_2에서 inner 2중 loop동안, M의 column 1개가 access pattern이 일정해서 cache의 이득을 본다.
	근데 CPU_AOS_2이 CPU_AOS보다 아주아주 약간 빠른데 그 이유는, 
	CPU_AOS_2에서는 X가 어차피 row wise access라 cache의 이익을 여기서도 받는데에 반해,
	CPU_AOS에서는 M이 column wise access라 cache의 이익을 못받기 때문으로 추측된다.
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
	CPU_SOA는 CPU_AOS에 비해 속도가 10배이상 느리다.
	그 이유는 X에 대한 access pattern이 column wise이기 때문이다.
	AOS에서도 M에대해서 column wise access를 하지만, M의 크기가 작기때문에 피해(?)가 적다.
	그러나 X는 크기가 큼으로 access pattern이 column wise이면 피해가 커서 속도가 상당히 느리게 나온것으로 보인다.
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

// thread 1개가 vector 1개 계산
// 총 thread 개수 : N
/*
	성능이 매우 안좋다. 
	X에 대한 접근이 coalescing 이 아니라서 warp내에서 X에 대한 데이터를 읽기 위해 32번의 memory transaction이 필요하다.
	Warp내에서 M에 대한 접근이 항상 same address라는 점이 좋지만, (constant memory를 사용한 최적화가 효과적일 것이다)
	X에 대한 접근이 치명적으로 비효율적이라서 전체적인 성능은 매우 안좋다.
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

// thread 1개가 element 1개 계산
// 총 thread 개수 : N * ELEM_PER_VECTOR
/*
	GPU_1_AOS__VECTOR_PER_THREAD 보다 훨씬 좋다.
	Warp내에서 X에 대한 접근이 항상 same address라서, broadcast가 일어나 매우 효율적이다.
	(X의 크기가 커서 constant memory를 이용한 최적화는 힘들테지만 bank conflict가 없을 것이므로 shared memory를 활용하면 효과적일 것이다.)
	또한 Warp내에서 M에 대한 접근이 서로 인접한 address라서 coalescing이 일어나 매우 효율적이다.
	(따라서 bank conflict가 없을 것이므로 shared memory를 활용하면 효과적일 것이다.)
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

// thread 1개가 vector 1개
// 총 thread 개수 : N
/*
	GPU_1_SOA__ELEM_PER_THREAD 보다 훨씬 좋다.
	Warp내에서 X에 대한 접근이 서로 인접한 address라서 coalescing이 일어나 매우 효율적이다. 
	(bank conflict가 없을 것이므로 shared memory를 활용하면 효과적일 것이다.
	라고 생각하기 쉽지만, 실제결과는 shared memory를 사용하면 더 느리다. 이 것은 밑에 GPU3에서 다룰 것이다.)
	그리고 Warp내에서 M에 대한 접근이 항상 same address라서, broadcast가 일어나 매우 효율적이다. (constant memory를 이용한 최적화를 하면 효과적일 것이다.)
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

// thread 1개가 element 1개
// 총 thread 개수 : N * ELEM_PER_VECTOR
/*
	성능이 상당히 안좋다.
	M에 대한 접근이 coalescing 이 아니라서 warp내에서 X에 대한 데이터를 읽기 위해 32번의 memory transaction이 필요하다.
	X에 대한 접근이 항상 same address라는 점이 좋지만,
	M에 대한 접근이 치명적으로 비효율적이라서 전체적인 성능은 매우 안좋다.
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

// thread 1개가 element 1개 계산
// 총 thread 개수 : N * ELEM_PER_VECTOR
// X를 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
/*
	위에서 언급한 GPU_1_AOS__ELEM_PER_THREAD의 access pattern대로면, 
	warp내에서 X에 대한 접근이 항상 same address라서 shared memory를 이용하면 broadcast가 일어나 매우 효율적일 것으로 추측된다. 
	실제로 적용한 결과, 약간 성능향상이 된 것을 확인할 수 있었다. 
	그러나 그렇게 큰 성능향상은 얻지 못했는데, 그 이유는 어차피 X의 경우 global memory상에서도 broadcast에 의해 성능이 좋았고, 
	X에 대한 접근이 row wise이기 때문에 special locality특성에 따라 cache의 득을 봤기 때문으로 생각된다. 
	따라서 shared memory를 사용했음에도 shared memory에 대한 cost에 비해 엄청나게 큰 성능향상은 없어서 결과적으로 약간만 성능이 향상된 것으로 보인다.
*/
__global__ void GPU_2_AOS__SHARED_X(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	extern __shared__ float s_X[/*blockDim.x*/];

	// s_X 초기화
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

// thread 1개가 element 1개 계산
// 총 thread 개수 : N * ELEM_PER_VECTOR
// X, M을 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
/*
	GPU_2_AOS__SHARED_X에서 추가적으로 M에 대해 shared memory를 사용한 버전이다. 
	GPU_2_AOS__SHARED_X에 비해  거의 2배에 가까운 성능향상을 보여준다. 
	X는 shared memory를 사용해도 큰 성능향상이 없는데에 반해, M의 경우는 큰 성능향상을 보여준다. 
	그 이유는 M에 대한 access가 column wise이기 때문에 spacial locality 를 띄지않아서
	이에 대한 cache의 이득이 적어 M을 shared memory에 올리면 엄청난 성능향상이 일어나는 것이라고 생각된다.
*/
__global__ void GPU_2_AOS__SHARED_X_M(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	__shared__ float s_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
	extern __shared__ float s_X[/*blockDim.x*/];

	// s_M 초기화.
	for (int idx = threadIdx.x; idx < ELEM_PER_VECTOR * ELEM_PER_VECTOR; idx += blockDim.x)
		s_M[idx] = M[idx];

	// s_X 초기화.
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

// thread 1개가 element 1개 계산
// 총 thread 개수 : N * ELEM_PER_VECTOR
// X를 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
// M을 위해 constant memory 사용. (그러나 이경우는 warp내의 thread들이 서로 다른 address에 접근하는 경우임으로 Bad Usage.)
/*
	GPU_2_AOS__SHARED_X에서 추가적으로 M에 대해 constant memory를 사용한 버전이다.
	GPU_2_AOS_SHARED_X에 비해 약 10배가까히 성능이 '하락'하는데,
	이는 M에 대한 access pattern이 same address가 아니라서 memory access가 32번으로 나누어 일어나 성능이 상당히 하락하기 때문이다.
*/
__global__ void GPU_2_AOS__SHARED_X_CONSTANT_M(float *Y, float *M, float *X, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vid = tid / ELEM_PER_VECTOR;
	int eid = tid % ELEM_PER_VECTOR;

	extern __shared__ float s_X[/*blockDim.x*/];

	// s_X 초기화.
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

// thread 1개가 vector 1개
// 총 thread 개수 : N
// X를 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
/*
	GPU_1_SOA__VECTOR_PER_THREAD에서 추가적으로 X에 대해 shared memory를 사용한 버전이다. 
	GPU_1_SOA__VECTOR_PER_THREAD에 비해서 오히려 50%정도 성능이 '하락'하는데 
	그 이유는 어차피 X에 대한 access가 coalescing에 의해 효율적이였고, 
	(CUDA_WARP_SIZE * ELEM_PER_VECTOR) (=1024) 크기. 즉 4KB에 해당하는 X의 부분 영역이 총 ELEM_PER_VECTOR만큼 iterate되며 반복해서 접근되기에 
	temporal locality에 의해 cache의 이득을 많이 봤을 것이다. 
	그런데 X를 shared memory에 올림으로서 thread가 256개일 때, 32768bytes만큼의 shared memory가 할당됨으로 인해, 
	SM내에 resident하는 block의 개수가 1개로 제한되고 (실험환경에서 shared memory의 크기는 48KB라서) 
	즉, resident warp의 개수는 8개가 될테고, 이로 인해 M을 global memory에서 읽을 때, 
	stall이 발생하면 scheduling될 warp가 적어 이로 인해 성능이 하락되는 것이 아닐까라고 추측된다.
*/
__global__ void GPU_3_SOA__SHARED_X(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_X 초기화.
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

// thread 1개가 vector 1개
// 총 thread 개수 : N
// X, M을 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
/*
	GPU_3_SOA__SHARED_X에서 추가적으로 M에 대해 shared memory를 사용한 버전이다.
	이 경우 GPU_3_SOA__SHARED_X에 비해 성능이 3배정도 향상되는데, 
	그 이유는 아마 M이 shared memory에 상주함으로서 M에 의한 stall이 발생하지 않아 
	GPU_3_SOA__SHARED_X에서 언급한 문제가 해결되어 성능이 많이 향상되는 것으로 생각된다.
*/
__global__ void GPU_3_SOA__SHARED_X_M(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	__shared__ float s_M[ELEM_PER_VECTOR * ELEM_PER_VECTOR];
	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_M 초기화.
	for (int idx = threadIdx.x; idx < ELEM_PER_VECTOR * ELEM_PER_VECTOR; idx += blockDim.x)
		s_M[idx] = M[idx];

	// s_X 초기화.
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

// thread 1개가 vector 1개
// 총 thread 개수 : N
// X를 위해 shared memory 사용. ELEM_PER_VECTOR에 상관없이 no bank conflict.
// M을 위해 constant memory 사용. (warp내의 thread들이 동시에 같은 메모리를 접근하므로 Good Usage.)
/*
	GPU_3_SOA__SHARED_X에서 추가적으로 M에 대해 constant memory를 사용한 버전이다. 
	이 경우 GPU_3_SOA__SHARED_X에 비해 약 7~8배, GPU_3_SOA__SHARED_X_M에 비해 약 2~3배,  
	GPU_1_SOA__VECTOR_PER_THREAD에 비해 약 4~5배 정도 빨라진다.
	GPU_3_SOA__SHARED_X_M에 비해서 빠른 이유는 M을 shared memory로 놓을 경우, block마다 초기화와 할당이 필요하지만, 
	constant memory는 전체에서 접근이 가능하기 때문이다.
*/
__global__ void GPU_3_SOA__SHARED_X_CONSTANT_M(float *Y, float *M, float *X, int N)
{
	int vid = blockIdx.x * blockDim.x + threadIdx.x;	// = tid

	extern __shared__ float s_X[/*blockDim.x * ELEM_PER_VECTOR*/];

	// s_X 초기화.
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

// thread 1개가 vector 1개
// 총 thread 개수 : N
// M을 위해 constant memory 사용. (warp내의 thread들이 동시에 같은 메모리를 접근하므로 Good Usage.)
/*
	GPU_1_SOA__VECTOR_PER_THREAD에서 추가적으로 M에 대해 constant memory를 사용한 버전이다. 
	성능은 거의 같다고 봐도 될 정도로 아주 미세하게 향상하였는데, 
	아마도 GPU_1_SOA__VECTOR_PER_THREAD에서 이미 M에 대한 cache hit이 많이 떠서, 
	혹은 cache miss로 인해 stall되도 GPU_3_SOA__SHARED_X에 비해 resident warp가 많아서 scheduling이 되기 때문에 
	constant memory를 사용하여도 성능향상이 적은 것이 아닐까라고 생각된다.
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