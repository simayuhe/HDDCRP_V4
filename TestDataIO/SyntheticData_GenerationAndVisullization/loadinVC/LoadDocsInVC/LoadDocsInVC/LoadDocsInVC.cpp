// LoadDocsInVC.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <vector>

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
		
		vector<int> doc;
		while (fscanf(fileptr, "%d", &word) != EOF)
		{
			
			doc.push_back(word);
		}
		corpus.push_back(doc);
	}
	cout <<"corpus.size()="<< corpus.size() << endl;
	return 0;
}

