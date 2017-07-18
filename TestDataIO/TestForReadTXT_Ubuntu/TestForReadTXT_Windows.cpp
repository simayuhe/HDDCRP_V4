// TestForReadTXT_Windows.cpp : 定义控制台应用程序的入口点。
//借鉴RFT中的load data,读取doc.txt中的n行数据，并写入vector中,这里不进行运算，直接将xy坐标写回名为result_1.txt的文件中。
//数据在txt中的存放格式是 
//5 [3,0](448,26,11)(447,26,13)(447,27,16)(447,28,17)(448,28,20)
//9 [0, 2](282, 101, 1436)(282, 102, 1437)(283, 102, 1440)(283, 101, 1442)(284, 101, 1445)(285, 101, 1447)(284, 101, 1448)(283, 101, 1451)(282, 101, 1455)
//3 [0,0](407,37,678)(406,37,682)(405,37,684)
//其中第一个代表元素个数，[]代表起止点，（x,y,t）代表轨迹信息

#include <iostream>
#include <vector>

using namespace std;

typedef struct{

	int x;
	int y;
	size_t t;

} Point;

class Trajectory{
public:
	//Trajectory();
	size_t length;
	char source;
	char sink;
	vector<Point> d_point;

};
int main(int argc, _TCHAR* argv[])
{
	size_t d, i;
	int length, x, y, t, s1, s2, numTrk = 0;
	Point p;
	Trajectory* traj;
	vector<Trajectory> TD;
	char filename[100];
	FILE *fileptr;

	printf("load data ...\n");

	for (d = 1; d < 2; d++){
		//sprintf(filename, "trks_grand_ss8.txt");
		sprintf(filename, "doc_%d.txt", d);
		//sprintf(filename, "parkinglot_trk.txt");
		fileptr = fopen(filename, "r");

		while ((fscanf(fileptr, "%d ", &length) != EOF))//原始数据中的每行第一个数代表轨迹的长度
		{
		
			traj = new Trajectory();
			fscanf(fileptr, "[%d,%d]", &s1, &s2);//原始数据中的每行第2个代表source,sink点是否被观察
			/* traj->d_source=s1;
			traj->d_sink=s2;*/
			
			for (i = 0; i < length; i++){
				fscanf(fileptr, "(%d,%d,%d)", &x, &y, &t);
				p.x = x; p.y = y; p.t = t;

				traj->d_point.push_back(p);// (i, p, s1, s2);

			}
			
			fscanf(fileptr, "\n");
			traj->sink = s1;
			traj->source = s2;
			traj->length = length;
			TD.push_back(*traj);
			//d_trajSet.push_back(*traj);
			//d_trajSet[numTrk].trkSource = s1;
			//d_trajSet[numTrk].trkSink = s2;
			//printf("cur TrkSource is %d, curTrkSink is %d. \n", d_trajSet[numTrk].trkSource,d_trajSet[numTrk].trkSink);
			//system( "pause ");
			
			numTrk++;
			delete traj;
		}
		fclose(fileptr);
	}
	
	printf("%d trajectories are loaded. \n", numTrk);

	fileptr = fopen("result_1.txt", "w");
	for (int k = 0; k < numTrk; k++){
	
	
		for (int w = 0; w < TD.at(k).length; w++){
		
			fprintf(fileptr, "(%d)", TD.at(k).d_point.at(w).x);
		}
		
		fprintf(fileptr, "\n");
		for (int w = 0; w < TD.at(k).length; w++){
			
			fprintf(fileptr, "(%d)", TD.at(k).d_point.at(w).y);
		}
		fprintf(fileptr, "\n");

	}
	fclose(fileptr);
	return 0;
}

