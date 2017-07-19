// LoadDocsInUbuntu.cpp 
//

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;

int main()
{
	char filename[100];
	size_t d;
	FILE *fileptr;
	int word;
	vector< vector<int> > corpus;

	printf("load data ...\n");

	for (d = 1; d < 37; d++){
		
		//sprintf(filename, "D:/code/mycode/HDDCRP_V4/TestDataIO/SyntheticData_GenerationAndVisullization/data/doc_%d.txt", d);
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
		sprintf(filename, "..//result/freqVCSerial_%d.txt", d);
		fileptr = fopen(filename, "w");
		for (int f = 0; f < 25; f++)
		{
			
			fprintf(fileptr, "%d ", ff[f]);
		}
		fclose(fileptr);
	}
	cout <<"corpus.size()="<< corpus.size() << endl;
	return 0;
}

