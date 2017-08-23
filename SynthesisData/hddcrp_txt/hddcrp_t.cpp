// hddcrp_t.cpp : 定义控制台应用程序的入口点。
//
//MY_DEBUG用来展示调试过程，在正常运行时要在stdafx.h中注释掉

#include "stdafx.h"
#include "type_def.h"
#include "HddCRP.hpp"

#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>
#include "mat.h"
#include "rand_utils.h"

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef MY_DEBUG
	printf("MY_DEBUG Model.\n");
#else
	printf("Running Model.\n");
#endif

////*******可调参数**********//
	//在Linux中可以有参数列表进行传输调参
//两层ddcrp 用来处理合成数据集
	int size_voc_word = 25;//字典长度
//各层中餐馆模型的离散度参数
//注 ：1e-10 = 0.1
	double alpha0 = 1;
	double alpha1 = 0.01;//1e-200;
//从num_burn_in开始对起止点标记进行Gibbs采样，总的采样次数是num_samples，间隔num_space 保存中间结果 
	int num_burn_in = 1, num_samples = 100, num_space = 1;
#ifdef LOAD_FROM_TXT
//输入文件夹D:\code\mycode\HDDCRP_T\SyntheticData_GenerationAndVisullization\data
	string inputfile = { "D:/code/mycode/HDDCRP_T/SyntheticData_GenerationAndVisullization/data/" };
#else
	string inputfile = { "D:/code/mycode/HDDCRP_T/SyntheticData_GenerationAndVisullization/data/synthetic_dat_76.mat" };

#endif
//输入文件夹D:\code\mycode\HDDCRP_T\result
	string outputfile = {"D:/code/mycode/HDDCRP_T/result/"};


////*************************//


////*********************************//
//初始化数据和结构体
	double eta_i_word = 1.0 / size_voc_word;

	vector<double> alphas;
	alphas.clear();
	alphas.push_back(alpha0);
	alphas.push_back(alpha1);

	string trainss_file = inputfile;
	string link_file = inputfile;

	HH hh(size_voc_word, eta_i_word);

	HddCRP_Model<DIST> model(hh, alphas, num_burn_in, num_samples, num_space, outputfile);
#ifdef LOAD_FROM_TXT
	model.load_txt_trainss(trainss_file);
	model.load_txt_link(link_file);
#else
	model.load_matlab_trainss(trainss_file);
	model.load_matlab_link(link_file);
#endif
	
	model.initialize();
	Layer<DIST>::base.init_log_pos_vals(Layer<DIST>::trainss.size());
	model.run_sampler();


	return 0;
}

