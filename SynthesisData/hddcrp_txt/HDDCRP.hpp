//#pragma once
//#include "StdAfx.h"
#include "Layer.hpp"
#include <time.h>
#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <sstream>
#include <fstream>
#include<omp.h>
//#include <parallel/algorithm>
using namespace std;

template<typename D> class HddCRP_Model
{
public:
	HddCRP_Model(void){}

	~HddCRP_Model(void){}

	HddCRP_Model(HH &_eta, vector<double> &_alphas, int _num_burn_in, int _num_samples, int _num_space, string& _save_dir) :
		alphas(_alphas), log_alphas(_alphas), layers(_alphas.size(), Layer<D>()), //num_groups(_alphas.size(), 0),
		num_burn_in(_num_burn_in), num_samples(_num_samples), num_space(_num_space), save_dir(_save_dir)
#ifdef TRI_MULT_DIST
		,num_burnin_ss(10)//用来控制采样起点终点的次数，为什么不是每次都采样呢？
#endif
	{
		for (vector<double>::iterator it = log_alphas.begin(); it != log_alphas.end(); it++)
		{
			*it = log(*it);
		}
		Layer<D>::base.initialize_eta(_eta);//得到了一个私有变量eta,长度是1001 最后一个用来统计词典分布的总和
	}

private:
	int num_burn_in, num_samples, num_space; // currently num_layers should only be 2 or 3
#ifdef TRI_MULT_DIST
	int num_burnin_ss;
#endif
	vector< Layer<D> > layers;
	vector<double> alphas;
	vector<double> log_alphas;
	vector<double> liks;
	string save_dir;

public:
	Layer<D>& get_layer(int i);
	void load_matlab_trainss(string& trainss_file);
#ifdef TRI_MULT_DIST
	void load_matlab_initss(string& initss_file);
#endif
	void load_matlab_link(string& link_file);
	void load_txt_trainss(string& trainss_file);//20170821
	void load_txt_link(string& link_file);//20170821
	void initialize();
	void run_sampler();
	void save_classqq(int iter);
	void save_liks();
	void save_run_time(double run_time);
};

/*********************************************/
/****** implement class of HddCRP_Model ******/
/*********************************************/
template<class D> inline Layer<D>& HddCRP_Model<D>::get_layer(int i)
{
	return layers.at(i);
}

template<class D> void HddCRP_Model<D>::load_matlab_trainss(string& trainss_file)
{
	layers.at(0).load_matlab_trainss(trainss_file);
	for (int i = 1; i != layers.size(); i++)
	{
		string empty_file;
		layers.at(i).load_matlab_trainss(empty_file);
	}
}
template<class D> void HddCRP_Model<D>::load_txt_trainss(string& trainss_file)//20170821
{
	layers.at(0).load_txt_trainss(trainss_file);
	for (int i = 1; i != layers.size(); i++)
	{
		string empty_file;
		layers.at(i).load_txt_trainss(empty_file);
	}
}
#ifdef TRI_MULT_DIST
template<class D> void HddCRP_Model<D>::load_matlab_initss(string& initss_file)
{
	MATFile *pmat = matOpen(initss_file.c_str(), "r");
	mxArray *pmx = matGetVariable(pmat, "init_ss");
	double *pd = mxGetPr(pmx);
	vector<SS>& trainss = Layer<D>::trainss;
	vector<int>& inds_groups = layers.front().inds_groups;
	int num_groups = layers.front().num_groups;
	for (int i = 0; i != trainss.size(); i++)
	{
		int idx_group = inds_groups.at(i);
		if (trainss.at(i).gibbs_source)
		{
			trainss.at(i).source = pd[idx_group] - 1;
		}
		if (trainss.at(i).gibbs_sink)
		{
			trainss.at(i).sink = pd[num_groups + idx_group] - 1;
		}
	}
	matClose(pmat);
}
#endif
template<class D> void HddCRP_Model<D>::load_matlab_link(string& link_file)
{
	layers.at(0).load_matlab_link(link_file, alphas.at(1));
	/*for (int i = 1; i < alphas.size() - 1; i++)
	{
		string empty_file;
		layers.at(i).load_matlab_link(empty_file, alphas.at(i + 1));
	}*/
}
template<class D> void HddCRP_Model<D>::load_txt_link(string& link_file)//20170821
{
	layers.at(0).load_txt_link(link_file, alphas.at(1));

}

template<class D> void HddCRP_Model<D>::initialize()
{
	int i;
	int tot_num_words = layers.at(0).trainss.size();
	//vector<Layer>::iterator it_l;

	/* initialize customers, tables and links */
	// in the first and second layer, each customer links to itself
	// in the top layer, all the customers are linked into one cluster
	layers.at(0).parent = &layers.at(1);
	layers.at(0).child = NULL;
	layers.at(0).alpha_item = alphas.at(0);
	layers.at(0).log_alpha_item = log_alphas.at(0);
	layers.at(0).alpha_group = alphas.at(1);
	layers.at(0).log_alpha_group = log_alphas.at(1);
	//layers.at(0).initialize_link(layers.at(0).num_groups, tot_num_words);
	cout << "initialize at the layer(0) : " << endl;
	layers.at(0).initialize_link(layers.at(0).num_groups, 1);
	for (i = 1; i != layers.size() - 1; i++)
	{
		layers.at(i).parent = &layers.at(i + 1);
		layers.at(i).child = &layers.at(i - 1);
		layers.at(i).alpha_item = alphas.at(i);
		layers.at(i).log_alpha_item = log_alphas.at(i);
		layers.at(i).alpha_group = alphas.at(i + 1);
		layers.at(i).log_alpha_group = log_alphas.at(i + 1);
		cout << "intialize at the layer(" << i << ") : " << endl;
		layers.at(i).initialize_link(1, tot_num_words);
	}
	layers.back().parent = NULL;
	layers.back().child = &layers.at(layers.size() - 2);
	layers.back().alpha_item = alphas.back();
	layers.back().log_alpha_item = log_alphas.back();
	layers.back().alpha_group = 1.0;
	layers.back().log_alpha_group = 0.0;
	cout << "intialize at the layer(back) : " << endl;
	layers.back().initialize_link(1, 1);
//cout<<"LINK OVER";
	//layers.back().initialize_link(100, 1);
	Layer<D>::initialize_base();
	Layer<D>::bottom = &layers.at(0);
	Layer<D>::top = &layers.back();

}

template<class D> void HddCRP_Model<D>::run_sampler()
{
	int iter, i, j, iter_diff;
	int tot_num_iters = num_burn_in + num_space*num_samples;
	clock_t  time_start, timer, timer_t;//timer_t 是用来计算程序中耗时的 
	double elp_time, est_time;
	time_start = clock();
	for (i = 0; i != layers.size(); i++)//(i = 0; i !=1; i++)
	{
		cout << "at layer " << i << "..." << endl;
		layers.at(i).traverse_links();
	}
	for (iter = 0; iter != tot_num_iters; iter++)//iter !=0
	{
		cout << "iteration " << iter << " ..." << endl;
		for (i = 0; i != layers.size(); i++)
		{
			cout << "at layer " << i << endl;
			timer_t = clock();
			layers.at(i).run_sampler();
			cout << "the cost of layers.at(" << i << ").run_sampler() is " << (double)(clock() - timer_t) / CLOCKS_PER_SEC << endl;
			for (j = i; j != layers.size(); j++)
			{
				timer_t = clock();
				layers.at(j).traverse_links();
				cout << "the cost of layers.at(" << j << ").traverse_links(); is " << (double)(clock() - timer_t) / CLOCKS_PER_SEC << endl;
			}
		}
#ifdef TRI_MULT_DIST
		if (iter > num_burnin_ss - 1)//控制采样起终点的次数
		{
			Layer<D>::sample_source_sink();
			for (i = 0; i != layers.size(); i++)
			{
				layers.at(i).update_ss_stats();
			}
		}
#endif
		liks.push_back(Layer<D>::compute_tot_lik());//整体的似然值

		iter_diff = iter - num_burn_in + 1;//39-20+1
		if (iter_diff > 0 && (iter_diff % num_space == 0))
		{
			save_classqq(iter + 1);
#ifdef TRI_MULT_DIST
			Layer<D>::label_instances();
			Layer<D>::save_labels(save_dir, iter + 1);
			Layer<D>::save_topic_labels(save_dir, iter + 1);
#endif
		}
		timer = clock() - time_start;
		elp_time = (double)timer / CLOCKS_PER_SEC;
		est_time = (double)timer / (iter + 1)*tot_num_iters / CLOCKS_PER_SEC;
		cout << "elapsed time: " << elp_time << "s" << endl;
		cout << "estimated time: " << est_time << "s" << endl;
		for (int i = 1; i < layers.size(); i++)
		{
			cout << "number of tables in " << "layer[" << i << "]:" << layers.at(i).uni_tables_vec.size() << endl;
		}
		cout << endl;
	}
	save_liks();
	save_run_time(elp_time);
}

template<class D> void HddCRP_Model<D>::save_classqq(int iter)
{
	Layer<D>::base.output(save_dir, iter);

}

template<class D> void HddCRP_Model<D>::save_liks()
{
	if (save_dir.back() != '/')
	{
		save_dir += '/';
	}
	string lik_file_name = save_dir + "likelihoods.txt";
	ofstream output_stream(lik_file_name);
	cout << "saving likelihoods  ..." << endl;
	vector<double>::iterator it_lik;
	for (it_lik = liks.begin(); it_lik != liks.end(); it_lik++)
	{
		output_stream << *it_lik << " ";
	}
	output_stream.close();
}

template<class D> void HddCRP_Model<D>::save_run_time(double run_time)
{
	if (save_dir.back() != '/')
	{
		save_dir += '/';
	}
	string run_time_file = save_dir + "run_time.txt";
	ofstream output_stream(run_time_file);
	cout << "saving run time ..." << endl;
	output_stream << run_time;
	output_stream.close();
}
