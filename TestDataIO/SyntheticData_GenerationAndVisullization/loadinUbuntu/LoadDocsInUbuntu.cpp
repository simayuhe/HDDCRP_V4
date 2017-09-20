// LoadDocsInUbuntu.cpp 
//

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h> 
#include <time.h> 

using namespace std;

int main()
{
	char filename[100];
	size_t d;
	FILE *fileptr;
	int word;
	vector< vector<int> > corpus;
	clock_t start,finish;

	start=clock();
	printf("load data ...\n");

	for (d = 1; d < 37; d++){
		
		sprintf(filename, "..//data/doc_%d.txt", d);
		fileptr = fopen(filename, "r");
		
		vector<int> doc;
		int ff[25] = { 0 };

		while (fscanf(fileptr, "%d", &word) != EOF)
		{
			ff[word - 1] = ff[word - 1] + 1;
			doc.push_back(word);
		}
		corpus.push_back(doc);
		fclose(fileptr);
		sprintf(filename, "..//result/freqUbuntuSerial_%d.txt", d);
		fileptr = fopen(filename, "w");
		for (int f = 0; f < 25; f++)
		{
			
			fprintf(fileptr, "%d ", ff[f]);
		}
		fclose(fileptr);
	}

	cout <<"corpus.size()="<< corpus.size() << endl;
	finish=clock();
	cout<<"time cost : "<< (double)(finish-start)/ CLOCKS_PER_SEC<<endl;
	return 0;
}

