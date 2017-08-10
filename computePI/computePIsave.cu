#include <stdio.h>
#include <iostream>
#include <time.h>
//#include <cutil_inline.h>
using namespace std;

template<typename T> __global__ void reducePI1(T* __restrict__ d_sum, int num){
//printf("blockIdx.x is %d\n",blockIdx.x);
//printf("blockDim.x is %d\n",blockDim.x);
//printf("threadIdx.x is %d\n",threadIdx.x);
int id = blockIdx.x*blockDim.x + threadIdx.x;

T temp;
T pSum = 0;
extern T __shared__ s_pi[];
T rnum = 1.0/num;

for(int i=id;i<num;i +=blockDim.x*gridDim.x){
temp = (i+0.5f)*rnum;
pSum += 4.0f/(1+temp*temp);
}

s_pi[threadIdx.x] = pSum*rnum;
__syncthreads();

for(int i = (blockDim.x>>1);i >0;i >>= 1){
//for(int i = (blockDim.x>>1);i >= 0;i >>=1){
	if (threadIdx.x<i){
		s_pi[threadIdx.x] += s_pi[threadIdx.x+i];
	}
	__syncthreads();
}

if (threadIdx.x==0)
{
	d_sum[blockIdx.x]=s_pi[0];
}

//if (warpSize>63){
//	if (threadIdx.x<32){
//		s_pi[threadIdx.x] += s_pi[threadIdx.x +32];
//	}
//}
//if (threadIdx.x<16){
//	s_pi[threadIdx.x] += s_pi[threadIdx.x +16];
//printf("threadIdx.x 16 is %d\n",threadIdx.x);
//}
//if (threadIdx.x<8){
//	s_pi[threadIdx.x] += s_pi[threadIdx.x +8];
//printf("threadIdx.x 8 is %d\n",threadIdx.x);
//}
//if (threadIdx.x<4){
//	s_pi[threadIdx.x] += s_pi[threadIdx.x +4];
//printf("threadIdx.x 4 is %d\n",threadIdx.x);
//}
//if (threadIdx.x<2){
//	s_pi[threadIdx.x] += s_pi[threadIdx.x +2];
//printf("threadIdx.x 2 is %d\n",threadIdx.x);
//}
//if (threadIdx.x<1){
//	d_sum[blockIdx.x] = s_pi[0]+s_pi[1];
//printf("threadIdx.x 1 is %d\n",threadIdx.x);
//}



}

template<typename T> __global__ void reducePI2(T* __restrict__ d_sum, int num, T* __restrict__ d_pi){
int id = threadIdx.x;
extern T __shared__ s_sum[];
s_sum[id]=d_sum[id];
__syncthreads();

for(int i = (blockDim.x>>1);i>0;i >>=1)
//for(int i = (blockDim.x>>1);i>=0;i >>=1)
{
	if (id<i){
		s_sum[id] += s_sum[id+i];	
	}
	__syncthreads();
}

if(threadIdx.x==0)
{
	*d_pi =s_sum[0];
}
//if (warpSize>63){
//	if (threadIdx.x<32){
//		s_sum[threadIdx.x] += s_sum[threadIdx.x +32];
//	}
//}
//if (threadIdx.x<16){
//	s_sum[threadIdx.x] += s_sum[threadIdx.x +16];
//}//
//if (threadIdx.x<8){
//	s_sum[threadIdx.x] += s_sum[threadIdx.x +8];
//}
//if (threadIdx.x<4){
//	s_sum[threadIdx.x] += s_sum[threadIdx.x +4];
//}
//if (threadIdx.x<2){
//	s_sum[threadIdx.x] += s_sum[threadIdx.x +2];
//}
//if (threadIdx.x<1){
//	*d_pi = s_sum[0]+s_sum[1];
//}

}

template <typename T> T reducePI(int num){

int grid = 1024;

T *tmp;
cudaMalloc((void**)&tmp,grid*sizeof(T));
reducePI1<<<grid,256,256*sizeof(T)>>>(tmp,num);


//printf("%d\n",__LINE__);
T *d_PI;
cudaMalloc((void**)&d_PI,sizeof(T));

reducePI2<<<1,grid,grid*sizeof(T)>>>(tmp,grid,d_PI);
T pi;
cudaMemcpy(&pi,d_PI,sizeof(T),cudaMemcpyDeviceToHost);
cudaFree(tmp);
cudaFree(d_PI);

return pi;
}

template <typename T> T cpuPI(int num){

T sum = 0.0f;
T temp;
for (int i=0;i<num;i++)
{
	temp =(i+0.5f)/num;
	sum += 4/(1+temp*temp);
}
return sum/num;

}


int main(){
printf("test for compell \n");
clock_t start, finish;
float costtime;
start = clock(); 
//************
printf("cpu pi is  %f\n",cpuPI<float>(1000000));
//*************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
printf("costtime of CPU is %f\n",costtime);

start = clock();
//************
printf("gpu pi is %f\n",reducePI<float>(1000000));
//************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
printf("costtime of GPU is %f\n",costtime);
return 0;
}
