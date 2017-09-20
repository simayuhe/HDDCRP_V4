#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;

class Stat_Mult
{
public:

	vector<int> uni_ss;//����Ŵ��еĵ�����������{9��4��7��3}//�����ֵ䳤��Ϊ10
	vector<int> uni_qq;//ÿ�����ʳ��ֵĴ�������{2��1��1��4}
	int tot;
	vector<int> inds;//ÿ�����ʵ�һ�γ��ֵĴ��򣬳���Ϊ�ʵ䳤��{-1��-1��3��1��-1��-1��2��-1��0��-1}
	//���inds ����ͳ�ƺ͸���qqʱʹ�ã������ҵ��õ����Ƿ���ֹ����Լ���qq�г��ֵ�λ�ã���ȥ����uni_ss��������ʹ�࣬���˸о����������������ݽṹʵ�֣���set<double>
	Stat_Mult():tot(0){}
	Stat_Mult(int voc_size):tot(0), inds(voc_size, -1){}
	Stat_Mult& init(int ss);
	Stat_Mult& init(int ss, vector<double>& _eta);
	Stat_Mult& update(int ss);
	Stat_Mult& init(Stat_Mult& stat, vector<double>& _eta);
	Stat_Mult& update(Stat_Mult& stat);
	void part_update(Stat_Mult& stat);
	//Stat_Mult& operator=(const Stat_Mult& stat);
	void clear_inds();
	bool check_stat(int _tot);

private:
	void part_update(int ss, int cnt);
};

class Multinomial
{
public:

	Multinomial(void){}

	~Multinomial(void){}

	Multinomial(vector<double>& _eta);

	void initialize_eta(vector<double>& _eta);

	static void initialize_eta(vector<double>& _eta, vector<double>& eta);

	list< vector<int> >& get_classqq(){return classqq;}

	vector<double>& get_eta(){return eta;}

	double marg_likelihood(vector<int>& qq, int ss);

	double marg_likelihood(vector<int>& qq, Stat_Mult& stat);

	static double marg_likelihood(vector<int>& qq, Stat_Mult& stat, vector< vector<double> >& log_pos_vals, int tot);

	void marg_likelihoods(vector<double>& clik, int ss);

	static void add_data(vector<int>& qq, int ss);

	static void add_data(vector<int>& qq, Stat_Mult& stat);

	static void add_data(vector<int>& qq, Stat_Mult& stat, int tot);

	double add_data_lik(vector<int>& qq, int ss);

	static void del_data(vector<int>& qq, int ss);

	static void del_data(vector<int>& qq, Stat_Mult& stat);

	static void del_data(vector<int>& qq, Stat_Mult& stat, int tot);

	list< vector<int> >::iterator add_class();

	void del_class(list< vector<int> >::iterator it);
	
	void output(string& save_dir, int iter);

	void reset_class(vector<int>& qq);

	void init_log_pos_vals(int tot_num_words);

	static void init_log_pos_vals(int tot_num_words, vector< vector<double> >& log_pos_vals, vector<double>& eta);

	static bool check_qq(vector<int>& qq);
private:
	vector<double> eta;
	list< vector<int> > classqq;
	vector< vector<double> > log_pos_vals;	// all possible log values, we assume all eta[i] are equal
	
};

