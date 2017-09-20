// hddcrp_t.cpp : �������̨Ӧ�ó������ڵ㡣
//
//MY_DEBUG����չʾ���Թ��̣�����������ʱҪ��stdafx.h��ע�͵�
#pragma once
#include "stdafx.h"
#include "type_def.h"
#include "HDDCRP.hpp"

#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
//#include <direct.h>
//#include "mat.h"
#include "rand_utils.h"

int main(int argc, char * argv[])
{
#ifdef MY_DEBUG
	printf("MY_DEBUG Model.\n");
#else
	printf("Running Model.\n");
#endif

////*******�ɵ�����**********//
	//��Linux�п����в����б���д������
//����ddcrp ��������ϳ����ݼ�
	int size_voc_word = 25;//�ֵ䳤��
//�����в͹�ģ�͵���ɢ�Ȳ���
//ע ��1e-10 = 0.1
	double alpha0 = 0.0001;
	double alpha1 = 1e-250;//1e-200;
	double alpha2 = 1e-200;
//��num_burn_in��ʼ����ֹ���ǽ���Gibbs�������ܵĲ���������num_samples�����num_space �����м��� 
	int num_burn_in = 1, num_samples = 1000, num_space = 1;
#ifdef LOAD_FROM_TXT
//�����ļ���D:\code\mycode\HDDCRP_T\SyntheticData_GenerationAndVisullization\data;/home1/yxkang_data/HDDCRP_v4/TestDataIO/SyntheticData_GenerationAndVisullization/data
	string inputfile = { "/home1/yxkang_data/HDDCRP_v4/TestDataIO/SyntheticData_GenerationAndVisullization/data" };
#else
	string inputfile = { "D:/code/mycode/HDDCRP_T/SyntheticData_GenerationAndVisullization/data/synthetic_dat_76.mat" };

#endif
//�����ļ���D:\code\mycode\HDDCRP_T\result
	string outputfile = {"./result_360_1e4_1e250_1e200/"};

mkdir("./result_360_1e4_1e250_1e200/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 

////*************************//


////*********************************//
//��ʼ�����ݺͽṹ��
	double eta_i_word = 1.0 / size_voc_word;

	vector<double> alphas;
	alphas.clear();
	alphas.push_back(alpha0);
	alphas.push_back(alpha1);
	alphas.push_back(alpha2);
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

