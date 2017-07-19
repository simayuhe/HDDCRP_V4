// LoadDocsByOpenMP.cpp 
//
#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h> 
#include <time.h> 
#include <string>
using namespace std;

int main()
{
	char filename[100],resultname[100];
	int d;
	FILE *fileptr[360];
	int word;
	int ff[360][25] = { 0 };
	//vector< vector<int> > corpus;
	clock_t start,finish;
	int f[360]={0};

	start=clock();
	printf("load data ...\n");
#pragma omp parallel for //num_threads(1)
	for (d = 1; d < 361; d++){
		printf("Hello world, I am %d, docs index %d.\n",omp_get_thread_num(),d);
		sprintf(filename, "..//data/doc_%d.txt", d);
		fileptr[d-1] = fopen(filename, "r");
		//int ff[25]={0};
		////vector<int> doc;
		

		while (fscanf(fileptr[d-1], "%d", &word) != EOF)
		{
			ff[d-1][word - 1] = ff[d-1][word - 1] + 1;
			//ff[word-1]=ff[word-1]+1;
		//	//doc.push_back(word);
		}
		////corpus.push_back(doc);
		fclose(fileptr[d-1]);
		sprintf(resultname, "..//result/freqByOpenMP_%d.txt", d);//Be CAREFUL!For the name "filename" has been used before, we must name the string differently here.
		fileptr[d-1] = fopen(resultname, "w");
		for (f[d-1] = 0; f[d-1] < 25; f[d-1]++)
		{
			
			fprintf(fileptr[d-1], "%d ", ff[f[d-1]]);
		}
		fclose(fileptr[d-1]);
	}
	

	//cout <<"corpus.size()="<< corpus.size() << endl;
	finish=clock();
	cout<<"time cost : "<< (double)(finish-start)/ CLOCKS_PER_SEC<<endl;
	return 0;
}

