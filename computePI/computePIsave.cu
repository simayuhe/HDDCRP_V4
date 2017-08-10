#include <stdio.h>
#include <iostream>
#include <time.h>
//#include <cutil_inline.h>
using namespace std;

//*****************************************//
//���������ֽ����豸�ϱ��� ��__global__��ʶ��
template<typename T> __global__ void reducePI1(T* __restrict__ d_sum, int num){
//__restrict__ ��˵��ֻ�������ж�ȡ�����ݣ�����ʲô�����أ�
//printf("blockIdx.x is %d\n",blockIdx.x);//�߳̿�������0~grid-1
//printf("blockDim.x is %d\n",blockDim.x);//�߳̿�������߳������������<<<grid,block,size>>>�е�block
//printf("threadIdx.x is %d\n",threadIdx.x);//ÿ���߳̿����̵߳ı�ţ�0~block-1
int id = blockIdx.x*blockDim.x + threadIdx.x;//Ϊÿ���̹߳���Ψһ��ţ�0~grid*block-1

T temp;
T pSum = 0;
extern T __shared__ s_pi[];//���ݴ���ڹ���洢�ϣ�ֻ�б��߳̿��ڵ��߳̿��Է���
T rnum = 1.0/num;

for(int i=id;i<num;i +=blockDim.x*gridDim.x){
	//ÿ���̼߳���Ĵ������ܵĴ�����num�������ܵ��߳�����grid*block��
	temp = (i+0.5f)*rnum;
	pSum += 4.0f/(1+temp*temp);
}

s_pi[threadIdx.x] = pSum*rnum;//ÿ���߳̿��е��̻߳���Լ�����õ���s_pi�����洢�ڱ���Ĺ���洢��
__syncthreads();//�ȴ����������̼߳������

for(int i = (blockDim.x>>1);i >0;i >>= 1){
//�������ڵ� ������ �����ۼ�
	if (threadIdx.x<i){
		s_pi[threadIdx.x] += s_pi[threadIdx.x+i];
	}
	__syncthreads();
}
//���Ӻ͵Ľ��д�������Ӧ���Դ��У��Ա�reducePI2ʹ��
if (threadIdx.x==0)
{
	d_sum[blockIdx.x]=s_pi[0];
}

//������δ���Ӧ������ִ�����Ƶ��㷨���ǽ�����кܴ�ƫ���δ�ҵ�ԭ��^_^
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
int id = threadIdx.x;//����������߳̿�ֻ��һ�����߳�����grid��������Ȼ��id��Ϊ������
extern T __shared__ s_sum[];//����ǹ����ڴ��еģ�ֻ�п��ڿɼ�
s_sum[id]=d_sum[id];//���Դ��е�����װ�ؽ���
__syncthreads();//�ȴ�װ�����

for(int i = (blockDim.x>>1);i>0;i >>=1)
//��Ȼ���ð�԰��ۺ͵ķ����Ա����������߳��е�s_sum�������
{
	if (id<i){
		s_sum[id] += s_sum[id+i];	
	}
	__syncthreads();//�ȴ�������
}
//����ͽ��д���Դ棬ʹ��cpu�����˿ɼ�
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
//���´����������ϱ���

template <typename T> T reducePI(int num){

int grid = 1024;//���������߳̿������

T *tmp;
cudaMalloc((void**)&tmp,grid*sizeof(T));//���豸�洢�����Դ棩�Ͽ���grid*sizeof(T)��С�Ŀռ䣬�����ϵ�ָ��tmpָ��ÿռ�
reducePI1<<<grid,256,256*sizeof(T)>>>(tmp,num);//����reducePI1
//������ʾ��grid���߳̿飬ÿ���߳̿���256���̣߳�ÿ���߳̿�ʹ��256*size��С�Ĺ���洢����ֻ�п��ڿ��Է��ʣ�

//ִ��֮�󣬻���tmpΪ�׵��Դ��д洢grid ���м���
//printf("%d\n",__LINE__);//��ʾ���������кţ���֪����ʲô��
T *d_PI;
cudaMalloc((void**)&d_PI,sizeof(T));//�Դ���Ϊ�еļ��������ٿռ�

reducePI2<<<1,grid,grid*sizeof(T)>>>(tmp,grid,d_PI);//ֻ��һ���߳̿飬��grid���߳�
//ִ�к����Դ���d_PI��λ�ô�������
T pi;//�����������ڴ��ϵĿռ�
cudaMemcpy(&pi,d_PI,sizeof(T),cudaMemcpyDeviceToHost);//���Դ��н����ݿ�������
cudaFree(tmp);//�ͷ���Ӧ���Դ�ռ�
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
clock_t start, finish;//������ʱ
float costtime;
start = clock(); 
//************
printf("cpu pi is  %f\n",cpuPI<float>(1000000));//������ͨ�Ĵ���ѭ������ ��
//*************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; //��λ����
printf("costtime of CPU is %f\n",costtime);

start = clock();
//************
printf("gpu pi is %f\n",reducePI<float>(1000000));//���������ϵĲ��м��㺯��
//************
finish = clock();
costtime = (float)(finish - start) / CLOCKS_PER_SEC; 
printf("costtime of GPU is %f\n",costtime);
return 0;
}
