#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <string>
#include <string.h>
#include <fnmatch.h>
using namespace std;
int main(int argc,char *argv[])
{

DIR *dp;
struct dirent *dirp;
int n=0;
string inputfile="/home1/yxkang_data/HDDCRP_v4/TestDataIO/SyntheticData_GenerationAndVisullization/data";//"./t*.h";

char * filename=&inputfile[0];
/*if (argc!=2)
{
printf("a single argument is required\n");
return 0;
}
if((dp=opendir(argv[1]))==NULL)
printf("can't open %s",argv[1]);
while (((dirp=readdir(dp))!=NULL) && (n<=50))
{
n++;
printf("%s\n",dirp->d_name);
}
printf("\n");*/
dp=opendir(filename);
while ((dirp=readdir(dp))!=NULL  )
{
if(!fnmatch("doc_*.txt",dirp->d_name,FNM_PATHNAME|FNM_PERIOD ) )
{
n++;
printf("%s\n",dirp->d_name);
}
}
printf("n = %d",n);
closedir(dp);

return 0;
}
