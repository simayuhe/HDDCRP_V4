#include <stdio.h>
#include <iostream>
#include <time.h>
//#include <cutil_inline.h>
using namespace std;

//*****************************************//
//以下两部分将在设备上编译 由__global__标识；
template<typename T> __global__ void reducePI1(T* __restrict__ d_sum, int num){
//__restrict__ 是说从只读缓存中读取该数据，会有什么优势呢？
//printf("blockIdx.x is %d\n",blockIdx.x);//线程块索引，0~grid-1
//printf("blockDim.x is %d\n",blockDim.x);//线程块包含的线程数，这里就是<<<grid,block,size>>>中的block
//printf("threadIdx.x is %d\n",threadIdx.x);//每个线程块中线程的标号，0~block-1
int id = blockIdx.x*blockDim.x + threadIdx.x;//为每个线程构建唯一标号，0~grid*block-1

T temp;
T pSum = 0;
extern T __shared__ s_pi[];//数据存放在共享存储上，只有本线程块内的线程可以访问
T rnum = 1.0/num;

for(int i=id;i<num;i +=blockDim.x*gridDim.x){
	//每个线程计算的次数是总的次数（num）除以总的线程数（grid*block）
	temp = (i+0.5f)*rnum;
	pSum += 4.0f/(1+temp*temp);
}

s_pi[threadIdx.x] = pSum*rnum;//每个线程块中的线程会把自己计算得到的s_pi独立存储在本块的共享存储上
__syncthreads();//等待本块所有线程计算完毕

for(int i = (blockDim.x>>1);i >0;i >>= 1){
//将本块内的 计算结果 进行累加
	if (threadIdx.x<i){
		s_pi[threadIdx.x] += s_pi[threadIdx.x+i];
	}
	__syncthreads();
}
//将加和的结果写到本块对应的显存中，以备reducePI2使用
if (threadIdx.x==0)
{
	d_sum[blockIdx.x]=s_pi[0];
}

//下面这段代码应该是在执行类似的算法但是结果会有很大偏差，并未找到原因^_^
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
int id = threadIdx.x;//这个函数的线程块只有一个，线程数是grid，这里依然用id作为索引名
extern T __shared__ s_sum[];//这个是共享内存中的，只有块内可见
s_sum[id]=d_sum[id];//把显存中的数据装载进来
__syncthreads();//等待装载完成

for(int i = (blockDim.x>>1);i>0;i >>=1)
//仍然采用半对半折和的方法对本块内所有线程中的s_sum进行求和
{
	if (id<i){
		s_sum[id] += s_sum[id+i];	
	}
	__syncthreads();//等待求和完成
}
//将求和结果写入显存，使得cpu主机端可见
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

//**********************************************//
//以下代码在主机上编译

template <typename T> T reducePI(int num){

int grid = 1024;//用来调整线程块的数量

T *tmp;
cudaMalloc((void**)&tmp,grid*sizeof(T));//在设备存储器（显存）上开辟grid*sizeof(T)大小的空间，主机上的指针tmp指向该空间
reducePI1<<<grid,256,256*sizeof(T)>>>(tmp,num);//调用reducePI1
//参数表示有grid个线程块，每个线程块有256个线程，每个线程块使用256*size大小的共享存储器（只有块内可以访问）

//执行之后，会在tmp为首的显存中存储grid 个中间结果
//printf("%d\n",__LINE__);//显示代码所在行号，不知会有什么用
T *d_PI;
cudaMalloc((void**)&d_PI,sizeof(T));//显存中为π的计算结果开辟空间

reducePI2<<<1,grid,grid*sizeof(T)>>>(tmp,grid,d_PI);//只有一个线程块，有grid个线程
//执行后在显存中d_PI的位置存放最后结果
T pi;//这是在主机内存上的空间
cudaMemcpy(&pi,d_PI,sizeof(T),cudaMemcpyDeviceToHost);//从显存中将数据拷贝出来
cudaFree(tmp);//释放相应的显存空间
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
clock_t start, finish;//用来计时
float costtime;
start = clock(); 
//************
printf("cpu pi is  %f\n",cpuPI<float>(1000000));//调用普通的串行循环计算 π
//*************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; //单位是秒
printf("costtime of CPU is %f\n",costtime);

start = clock();
//************
printf("gpu pi is %f\n",reducePI<float>(1000000));//调用主机上的并行计算函数
//************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
printf("costtime of GPU is %f\n",costtime);
return 0;
}
