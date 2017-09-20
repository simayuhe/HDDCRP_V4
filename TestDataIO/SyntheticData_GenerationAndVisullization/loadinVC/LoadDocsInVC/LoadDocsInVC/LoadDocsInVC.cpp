// LoadDocsInVC.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <list>
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	char filename[100];
	size_t d;
	
	FILE *fileptr;
	int word;
	vector<vector<int>> corpus;

	printf("load data ...\n");

	for (d = 1; d < 37; d++){
		
		//sprintf(filename, "D:/code/mycode/HDDCRP_V4/TestDataIO/SyntheticData_GenerationAndVisullization/data/doc_%d.txt", d);
		sprintf(filename, "..//..//..//data/doc_%d.txt", d);
		fileptr = fopen(filename, "r");
		vector<int> freq(25,0);
		int ff[25] = { 0 };
		vector<int> doc;
		while (fscanf(fileptr, "%d", &word) != EOF)
		{
			freq[word-1]= freq.at(word-1) + 1;//对应的行列是从标号0开始的，所以多出个[d-1]
			ff[word - 1] = ff[word - 1] + 1;
			doc.push_back(word);
			cout << ff[word-1] << endl;
			cout << word << endl;
		}
		corpus.push_back(doc);
		fclose(fileptr);
		sprintf(filename, "..//result/freqVCSerial_%d.txt", d);
		fileptr = fopen(filename, "w");
		for (int f = 0; f < 25; f++)
		{
			
			fprintf(fileptr, "%d ", freq[f]);
		}
		fclose(fileptr);
	}
	cout <<"corpus.size()="<< corpus.size() << endl;
	
	return 0;
}

