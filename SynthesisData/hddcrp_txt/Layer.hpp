#pragma once
#include "StdAfx.h"
#include "type_def.h"
#include "Multinomial.h"
//#include "Tri_Mult.h"
#include "rand_utils.h"
#include <iostream>
#include <math.h>
#include <mat.h>
#include <vector>
#include <list>
#include <algorithm>
#include <functional>
#include <time.h>
#include <cstdlib>
#include <io.h>//20170821
using namespace std;

template<typename D> class Layer
{
public:
	static D base;//代表了本层的基本统计信息，为啥用static
	static vector< list<QQ>::iterator > pos_classqq;//在出现新类别的位置存储指向这个新类的classqq的指针
	static vector< SS > trainss;
	static vector<int> labels;//这个label是轨迹的标签

	static Layer<D> *top;
	static Layer<D> *bottom;//在外层进行指定的

	static double tot_lik;//用来计算所有团簇分布的似然值，最后比较各个iteration 的采样结果时用到

	static Layer<D> *cur_layer;//在runsampler中用来标记当前层
	static vector<bool> is_computed;//用来标记是否计算过以某一点为标记的团簇的似然，在runsampler中会重复清零
	//定义关于对link采样的相关变量
	static Layer<D> *cur_layer_sfl;//在runsampler中用来标记当前层
	static vector<bool> is_computed_sfl;//用来标记是否计算过以某一点为标记的团簇的似然，在runsampler中会重复清零
	//
	static void initialize_base();//把所有的trianss都初始化到一个classqq中去
	static double compute_tot_lik();//计算顶层所有餐桌似然值的总和

	Layer<D> *parent;
	Layer<D> *child;

	int num_groups;
	double alpha_group;//本层餐馆之间的链接先验
	double alpha_item;
	double log_alpha_group;
	double log_alpha_item;

	vector< vector<int> > group_candidates;
	vector< vector<double> > log_group_priors;
	vector< vector<double> > group_priors;
	vector< vector< vector<int> > > item_candidates;//在什么地方赋值的？
	vector< vector< vector<double> > > log_item_priors;
	vector< vector< vector<double> > > item_priors;

	vector<int> customers; // customers to which each member links
	vector<int> tables; // tables each member belongs to
	vector<int> old_tables; // tables each member belongs to
	//vector<int> clusters;	
	vector< vector<int> > links; // customers from which each member is visited
	vector< list<int> > uni_tables; // unique tables in this restaurant为什么是list元素组成的向量，每一层中可以有不同的餐馆
	vector<int> uni_tables_vec;
	vector< list<int>::iterator > pos_uni_tables; // positions of each table in "uni_tables"
	vector< vector<int> > inds_items; // indices of all customers in this restaurant
	vector<int> inds_groups; // indices of documents each word belongs to

	bool is_self_linked;
	int idx_group_cur;
	int old_customer_cur;
	int new_customer_cur;
	int old_table_cur;
	int new_table_cur;
	int old_cls_cur;
	int new_cls_cur;
	vector<int>::const_iterator connection_start;
	vector<int>::const_iterator connection_end;

	static int cur_item;
	static vector< STAT >::iterator it_stat_cur;
	static QQ qq_temp;
	static vector<double> log_pred_liks;
	static vector<double> pred_liks;
	vector<int> item_cands_cur;
	vector<double> item_priors_cur;
	vector<double> log_item_priors_cur;
	vector<int> cls_cands_cur;
	vector<int> uni_cls_cands_cur;
	double log_self_link_lik;
	vector<double> log_probs_sampling;
	double max_log_prob;

	vector< STAT >::iterator cur_stat;//定义每一层中cur_item 所对应的子树的统计量，在非根节点的时候与it_stat_cur相同
	//在根节点时，it_stat_cur 指示的是最上层的统计，包含了所有的link，而希望使用本层的cur_stat来记录本层中该节点的统计值，不包含上层的link
	//这样做的目的是在对根节点进行重新采样时，能够将根节点处的统计值与上层中link处的统计值分开加入不同的团簇中去

	//定义关于对link采样的相关变量
	bool is_self_linked_sfl;
	int idx_group_cur_sfl;
	int old_customer_cur_sfl;
	int new_customer_cur_sfl;
	int old_table_cur_sfl;
	int new_table_cur_sfl;
	int old_cls_cur_sfl;
	int new_cls_cur_sfl;
	vector<int>::const_iterator connection_start_sfl;
	vector<int>::const_iterator connection_end_sfl;


	static int cur_link;
	static vector< STAT >::iterator it_stat_cur_sfl;
	static QQ qq_temp_sfl;
	static vector<double> log_pred_liks_sfl;
	static vector<double> pred_liks_sfl;
	vector<int> item_cands_cur_sfl;
	vector<double> item_priors_cur_sfl;
	vector<double> log_item_priors_cur_sfl;
	vector<int> cls_cands_cur_sfl;
	vector<int> uni_cls_cands_cur_sfl;
	double log_self_link_lik_sfl;
	vector<double> log_probs_sampling_sfl;
	double max_log_prob_sfl;
	//

	vector< vector<int> > trees;//每层的长度为节点的总数，在每个餐桌节点上记录整个树的节点分布
	vector< vector<int> > orders;//计算stats过程中，先后顺序，这里从叶节点到根节点
	vector< vector<STAT> > stats;//与tree中的节点相对应，统计了该节点所有的子树的观测值
	vector< int > inds_start;//长度为节点数目，表示了该节点的子树是从tree.at(i)的那个位置开始
	vector< int > inds_end;//要与tree配合使用
							
	Layer(void){}
	~Layer(void){}
	void load_matlab_trainss(string& trainss_file);
	int check_txt_number(string & filepath);//20170821
	void load_txt_trainss(string& trainss_file);//20170821
	void load_matlab_link(string& link_file, double _group_alpha);
	void load_txt_link(string& link_file, double _group_alpha);//20170821
	void collect_customers();
	void initialize_link(int num_groups, int num_init_cls);

	void get_candidates();
	int get_cluster(int _cur_item);
	void collect_clusters();
	void collect_connections();//得到相应的子树节点标号，及相关统计量
	void check_link_status();
	void compute_marg_lik(int c);
	void compute_marg_liks();
	double compute_log_self_link_lik();
	void compute_log_probs_sampling();
	void sample_customer();
	void change_table(int _new_table);//改变connection_start end索引的节点的餐桌标号为_new_table
	void update_link();//改变cur_item 的customer，和table,并change_table（new_table_cure）
	void delete_table(int _old_table);//在idx_group_cur当前餐馆中把原餐桌标号old_talble erase掉，
	void change_customer(int _new_customer);//近似 ，这个函数用来处理当节点消失时，把原先指向cur_item的节点都指向了新的餐桌的标号节点，这是一种近似处理
	void merge_customers();
	void add_table();//利用uni_table向量中list排列的有序性，在该餐馆idx_group_cur的某个固定位置插入一个餐桌标号
	void sample_new_customer();//在上层为顾客采样新的连接
	void sample_for_single();
	void get_cands_point(int _cur_point);//1201
	void check_link_status_point(int _cur_point);//1201
	void collect_connections_points(int _cur_point);//1201
	void compute_marg_lik_root(int c);
	void compute_marg_liks_root();//1201
	void update_point_link(int _cur_point);
	void sample_link_points(int _cur_point);
	void sample_and_traverse(int _cur_point);
	void add_table_at_point(int _cur_point);
	void sample_new_customer_point(int _cur_point);
	void sample_for_leaf(int _cur_point);//1201
	void sample_for_root(int _cur_point);//1201
	void sample_for_point();//1201
	void run_sampler();
	//定义关于对link采样的相关函数 sample_for_link  :sfl
	void get_candidates_sfl();
	int get_cluster_sfl(int _cur_link);
	void collect_clusters_sfl();
	void collect_connections_sfl();//得到相应的子树节点标号，及相关统计量
	void check_link_status_sfl();
	void compute_marg_lik_sfl(int c);
	void compute_marg_liks_sfl();
	double compute_log_self_link_lik_sfl();
	void compute_log_probs_sampling_sfl();
	void sample_customer_sfl();
	void change_table_sfl(int _new_table);//改变connection_start end索引的节点的餐桌标号为_new_table
	void update_link_sfl();//改变cur_item 的customer，和table,并change_table（new_table_cure）
	void delete_table_sfl(int _old_table);//在idx_group_cur当前餐馆中把原餐桌标号old_talble erase掉，
	void change_customer_sfl(int _new_customer);//近似 ，这个函数用来处理当节点消失时，把原先指向cur_item的节点都指向了新的餐桌的标号节点，这是一种近似处理
	void merge_customers_sfl();
	void add_table_sfl();//利用uni_table向量中list排列的有序性，在该餐馆idx_group_cur的某个固定位置插入一个餐桌标号
	void sample_new_customer_sfl();//在上层为顾客采样新的连接
	void sample_for_single_sfl();
	void run_sampler_sfl();
	//
	void traverse_single_table(int idx_table);
	void traverse_links();
#ifdef TRI_MULT_DIST
	static int sample_ss(vector<int>& qq, vector<double>& eta, int w);
	void sample_source_sink_c(int t, int c);
	static void sample_source_sink();
	void update_ss_stats_c(int idx_table);//通过order的指引从叶子节点开始对stat进行更新
	void update_ss_stats();

	static void label_instances();
	static void save_labels(string& save_dir, int iter);
	static void save_topic_labels(string& save_dir, int iter);
#endif
};

template<typename D> D Layer<D>::base;
template<typename D> vector< list<QQ>::iterator > Layer<D>::pos_classqq;//指向class——qq的指针
template<typename D> vector< SS > Layer<D>::trainss;
template<typename D> vector<int> Layer<D>::labels;
template<typename D> Layer<D>* Layer<D>::top = NULL;
template<typename D> Layer<D>* Layer<D>::bottom = NULL;
template<typename D> double Layer<D>::tot_lik;
template<typename D> int Layer<D>::cur_item;
template<typename D> vector< STAT >::iterator Layer<D>::it_stat_cur;
template<typename D> QQ Layer<D>::qq_temp;
template<typename D> vector<bool> Layer<D>::is_computed;
template<typename D> vector<double> Layer<D>::log_pred_liks;
template<typename D> vector<double> Layer<D>::pred_liks;
template<typename D> Layer<D>* Layer<D>::cur_layer;
//采样链接
template<typename D> int Layer<D>::cur_link;
template<typename D> vector< STAT >::iterator Layer<D>::it_stat_cur_sfl;
//template<typename D> QQ Layer<D>::qq_temp;
template<typename D> vector<bool> Layer<D>::is_computed_sfl;
template<typename D> vector<double> Layer<D>::log_pred_liks_sfl;
template<typename D> vector<double> Layer<D>::pred_liks_sfl;
template<typename D> Layer<D>* Layer<D>::cur_layer_sfl;
template<typename D> void Layer<D>::initialize_base()
{
	/* add one class and initialize it */
	int tot_num_words = trainss.size();
	pos_classqq.assign(tot_num_words, list<QQ>::iterator());
	pos_classqq.at(0) = base.add_class();//返回一个指向classqq末尾的指针
	for (vector<SS>::iterator it_w = trainss.begin(); it_w != trainss.end(); it_w++)
	{
		//	base.add_data(base.get_classqq().front(), trainss.at(i));
		base.add_data(base.get_classqq().front(), *it_w);//定义在Multinomial.cpp 或者 Tri_Mult.cpp中，取决于base的类型作用是在长度为1000词典qq中对应的位置统计每个单词出现的次数
	}
	is_computed.assign(tot_num_words, false);
	log_pred_liks.assign(tot_num_words, 0.0);
	pred_liks.assign(tot_num_words, 0.0);
	//为link采样做准备
	is_computed_sfl.assign(tot_num_words, false);
	log_pred_liks_sfl.assign(tot_num_words, 0.0);
	pred_liks_sfl.assign(tot_num_words, 0.0);
}

template<typename D> double Layer<D>::compute_tot_lik()
{
	list<int> &uni_tables_0 = top->uni_tables.at(0);
	list<int>::iterator it_t = uni_tables_0.begin();
	tot_lik = 0.0;
	for (; it_t != uni_tables_0.end(); it_t++)
	{
		base.reset_class(qq_temp);
		tot_lik += base.marg_likelihood(qq_temp, top->stats.at(*it_t).front());
	}
	return tot_lik;
}

#ifdef TRI_MULT_DIST
template<typename D> void Layer<D>::load_matlab_trainss(string& trainss_file)
{
	if (trainss_file.empty())
	{
		num_groups = 1;//除了layer（0）都初始化为1个团簇
		//num_groups = 100;//除了layer（0）都初始化为100个团簇
		inds_groups.assign(trainss.size(), 0);
	}
	else
	{
		MATFile *pmat = matOpen(trainss_file.c_str(), "r");
		mxArray *pmx_trainss = matGetVariable(pmat, "trainss");
		mxArray *pmx_ss = matGetVariable(pmat, "source_sink");
		mxArray *pmx_doc_i;
		double *pd, *pd_ss = mxGetPr(pmx_ss);
		int i, j, num_words_i, cnt = 0, source, sink;
		const int voc_size_source = base.get_eta().eta_source.size() - 1;
		const int voc_size_sink = base.get_eta().eta_sink.size() - 1;


		/* load data from .mat file */
		num_groups = mxGetNumberOfElements(pmx_trainss);
		inds_items.reserve(num_groups);
		labels.assign(num_groups, 0);
		int num_words = 0;
		for (i = 0; i != num_groups; i++)
		{
			pmx_doc_i = mxGetCell(pmx_trainss, i);
			num_words += mxGetNumberOfElements(pmx_doc_i);
		}
		trainss.reserve(num_words);
		inds_groups.reserve(num_words);

		for (i = 0; i != num_groups; i++)
		{
			pmx_doc_i = mxGetCell(pmx_trainss, i);
			pd = mxGetPr(pmx_doc_i);
			num_words_i = mxGetNumberOfElements(pmx_doc_i);
			source = (int)(pd_ss[i]) - 1;
			sink = (int)(pd_ss[num_groups + i]) - 1;

			inds_items.push_back(vector<int>());
			vector<int> &inds_items_i = inds_items.back();
			inds_items_i.reserve(num_words_i);

			for (j = 0; j != num_words_i; j++)
			{
				trainss.push_back(SS());
				trainss.back().word = (int)(pd[j] - 1);

				if (source >= 0)
				{
					trainss.back().source = source;
					trainss.back().gibbs_source = false;
					//if (source > 7){
					//	cout << "invalid" << endl;
					//	int a = 0;
					//}
				}
				else
				{
					trainss.back().source = (int)(genrand_real2()*(double)voc_size_source);
					trainss.back().gibbs_source = true;
					//if (trainss.back().source > 7 || trainss.back().source < 0){
					//	cout << "invalid" << endl;
					//	int a = 0;
					//}
				}

				if (sink >= 0)
				{
					trainss.back().sink = sink;
					trainss.back().gibbs_sink = false;
					//if (sink > 7){
					//	cout << "invalid" << endl;
					//	int a = 0;
					//}
				}
				else
				{
					trainss.back().sink = (int)(genrand_real2()*(double)voc_size_sink);
					trainss.back().gibbs_sink = true;
					//if (trainss.back().sink > 7 || trainss.back().sink < 0){
					//	cout << "invalid" << endl;
					//	int a = 0;
					//}
				}
				inds_groups.push_back(i);
				inds_items_i.push_back(cnt++);
			}
		}
		matClose(pmat);
	}
}
#else
template<typename D> void Layer<D>::load_matlab_trainss(string& trainss_file)
{
	if (trainss_file.empty())
	{
		num_groups = 1;
		inds_groups.assign(trainss.size(), 0);
	}
	else
	{
		MATFile *pmat = matOpen(trainss_file.c_str(), "r");
		mxArray *pmx_trainss = matGetVariable(pmat, "trainss");
		mxArray *pmx_doc_i;
		double *pd;
		int i, j, num_words_i, cnt = 0;


		/* load data from .mat file */
		num_groups = mxGetNumberOfElements(pmx_trainss);
		inds_items.reserve(num_groups);
		labels.assign(num_groups, 0);
		int num_words = 0;
		for (i = 0; i != num_groups; i++)
		{
			pmx_doc_i = mxGetCell(pmx_trainss, i);
			num_words += mxGetNumberOfElements(pmx_doc_i);
		}
		trainss.reserve(num_words);
		inds_groups.reserve(num_words);

		for (i = 0; i != num_groups; i++)
		{
			pmx_doc_i = mxGetCell(pmx_trainss, i);
			pd = mxGetPr(pmx_doc_i);
			num_words_i = mxGetNumberOfElements(pmx_doc_i);

			inds_items.push_back(vector<int>());
			vector<int> &inds_items_i = inds_items.back();
			inds_items_i.reserve(num_words_i);

			for (j = 0; j != num_words_i; j++)
			{
				trainss.push_back((int)(pd[j] - 1));//为什么要减一呢？原始是77，现在是76，会有什么好处呢？
				inds_groups.push_back(i);
				inds_items_i.push_back(cnt++);
			}
		}
	}
}
#endif
template<typename D> int Layer<D>::check_txt_number(string& filepath)//20170821
{
	int num{ 0 };
	string filename = filepath + "/doc_*.txt";
	char * dir = &filename[0];
	_finddata_t fileDir;
	long lfDir;
	if ((lfDir = _findfirst(dir, &fileDir)) == -1l)
		printf("No file is found\n");
	else{

		do{
			num++;

		} while (_findnext(lfDir, &fileDir) == 0);
		//printf("number of docs : %d \n", num);
	}
	_findclose(lfDir);
	return num;

}
template<typename D> void Layer<D>::load_txt_trainss(string& trainss_file)//20170821
{
	if (trainss_file.empty())
	{
		num_groups = 1;
		inds_groups.assign(trainss.size(), 0);
	}
	else
	{
		int i, j, num_words_i, cnt = 0;
		num_groups = check_txt_number(trainss_file);
#ifdef MY_DEBUG
		printf("num_groups (number of docs in this inputfile) : %d\n", num_groups);
#endif
		inds_items.reserve(num_groups);
		labels.assign(num_groups, 0);
		int num_words = 0;
		int word;
		//这里省略了很多reverse的环节，避免反复读写文档
		//for (i = 0; i != num_groups; i++)
		//{
		//	pmx_doc_i = mxGetCell(pmx_trainss, i);
		//	num_words += mxGetNumberOfElements(pmx_doc_i);
		//}
		//trainss.reserve(num_words);
		//inds_groups.reserve(num_words);
		for (i = 0; i != num_groups; i++)
		{
			inds_items.push_back(vector<int>());
			vector<int> &inds_items_i = inds_items.back();
			//inds_items_i.reserve(num_words_i);
	/*		char txtNo[10];
			sprintf(txtNo, "%d", i+1);*/

			ostringstream s1;
			s1 << i+1;
			string s2 = s1.str();

			string txtname = trainss_file + "doc_" + s2 + ".txt";
			//cout << txtname << endl;
			char * dir = &txtname[0];
			FILE *fileptr;
			fileptr = fopen(dir, "r");
			while (fscanf(fileptr, "%d", &word) != EOF)//这个对于自制文档可以预先知道数量
			{
				trainss.push_back(word-1);//word 是从1到25，后面统计词频的时候要从0统计
				inds_groups.push_back(i);
				inds_items_i.push_back(num_words++);
				//printf("num_words ：%d", word - 1);
			}
			fclose(fileptr);
		}
		//cin >> j;
	}
}
template<typename D> void Layer<D>::load_matlab_link(string& link_file, double _alpha_group)
{
	int i, j;
	alpha_group = _alpha_group;
	log_alpha_group = log(_alpha_group);
	if (link_file.empty())
	{
		/* create document links and prior for ordinary lda */
		vector<int> group_candidates_i;
		vector<double> group_priors_i;
		vector<double> log_group_priors_i;
		group_candidates_i.push_back(0);
		group_priors_i.push_back(alpha_group);
		log_group_priors_i.push_back(log_alpha_group);
		group_candidates.push_back(group_candidates_i);
		group_priors.push_back(group_priors_i);
		log_group_priors.push_back(log_group_priors_i);
		for (i = 1; i != num_groups; i++)
		{
			//group_candidates:0 |0 1|0 1 2|0 1 2 3|...
			//group_prior:	   a |1 a|1 1 a|1 1 1 a|...
			group_candidates_i.push_back(i);
			group_candidates.push_back(group_candidates_i);
			group_priors_i.back() = 1.0;
			log_group_priors_i.back() = 0.0;
			group_priors_i.push_back(alpha_group);
			log_group_priors_i.push_back(log_alpha_group);
			group_priors.push_back(group_priors_i);
			log_group_priors.push_back(log_group_priors_i);
		}
	}
	else
	{
		MATFile *pmat = matOpen(link_file.c_str(), "r");
		mxArray *pmx_cand_links = matGetVariable(pmat, "cand_links");
		mxArray *pmx_log_priors = matGetVariable(pmat, "log_priors");
		mxArray *pmx_cands_i, *pmx_log_priors_i;
		double *pd_cands_i, *pd_log_priors_i;
		int num_cands_i, cnt = 0;


		/* load data from .mat file */

		for (i = 0; i != num_groups; i++)
		{
			pmx_cands_i = mxGetCell(pmx_cand_links, i);
			pmx_log_priors_i = mxGetCell(pmx_log_priors, i);
			pd_cands_i = mxGetPr(pmx_cands_i);
			pd_log_priors_i = mxGetPr(pmx_log_priors_i);
			num_cands_i = mxGetNumberOfElements(pmx_cands_i);

			
			group_candidates.push_back(vector<int>());
			group_priors.push_back(vector<double>());
			log_group_priors.push_back(vector<double>());
			vector<int> &group_candidates_i = group_candidates.back();
			vector<double> &group_priors_i = group_priors.back();
			vector<double> &log_group_priors_i = log_group_priors.back();

			for (j = 0; j != num_cands_i - 1; j++)
			{
				group_candidates_i.push_back((int)(pd_cands_i[j]) - 1);//对所有标号都减了一个
				log_group_priors_i.push_back(pd_log_priors_i[j]);
				group_priors_i.push_back(exp(pd_log_priors_i[j]));
			}

			group_candidates_i.push_back(i);//设置自连接的值是alpha 1
			log_group_priors_i.push_back(log_alpha_group);
			group_priors_i.push_back(alpha_group);
		}
		matClose(pmat);
	}
}
template<typename D> void Layer<D>::load_txt_link(string& link_file, double _alpha_group)//20170821
{
	int i, j;
	alpha_group = _alpha_group;
	log_alpha_group = log(_alpha_group);
	if (link_file.empty())
	{
		/* create document links and prior for ordinary lda */
		vector<int> group_candidates_i;
		vector<double> group_priors_i;
		vector<double> log_group_priors_i;
		group_candidates_i.push_back(0);
		group_priors_i.push_back(alpha_group);
		log_group_priors_i.push_back(log_alpha_group);
		group_candidates.push_back(group_candidates_i);
		group_priors.push_back(group_priors_i);
		log_group_priors.push_back(log_group_priors_i);
		for (i = 1; i != num_groups; i++)
		{
			//group_candidates:0 |0 1|0 1 2|0 1 2 3|...
			//group_prior:	   a |1 a|1 1 a|1 1 1 a|...
			group_candidates_i.push_back(i);
			group_candidates.push_back(group_candidates_i);
			group_priors_i.back() = 1.0;
			log_group_priors_i.back() = 0.0;
			group_priors_i.push_back(alpha_group);
			log_group_priors_i.push_back(log_alpha_group);
			group_priors.push_back(group_priors_i);
			log_group_priors.push_back(log_group_priors_i);
		}
	}
	else
	{
		int  cnt = 0;
		for (i = 0; i != num_groups; i++)
		{
			//对于每篇文档
			int num_cands_i, word;
			double logprior;
			group_candidates.push_back(vector<int>());
			group_priors.push_back(vector<double>());
			log_group_priors.push_back(vector<double>());
			vector<int> &group_candidates_i = group_candidates.back();
			vector<double> &group_priors_i = group_priors.back();
			vector<double> &log_group_priors_i = log_group_priors.back();

			ostringstream s1;
			s1 << i + 1;
			string s2 = s1.str();

			string linkname = link_file + "cand_link" + s2 + ".txt";
			string logpriorname = link_file + "log_prior" + s2 + ".txt";
			//cout << linkname << endl;
			char * dirlink = &linkname[0];
			char * dirlogprior = &logpriorname[0];
			FILE *fileptr1; 
			FILE *fileptr2;
			fileptr1 = fopen(dirlink, "r");
			//cout << "here";
			fileptr2 = fopen(dirlogprior, "r");
			fscanf(fileptr1, "[%d]", &num_cands_i);
			//cout << "num_cands_i" << num_cands_i;
			fscanf(fileptr2, "[%d]", &num_cands_i);
			//cout << "num_cands_i" << num_cands_i;
			for (j = 0; j != num_cands_i - 1; j++)
			{
				fscanf(fileptr1, "%d", &word);// != EOF;
				group_candidates_i.push_back(word - 1);//对所有标号都减了一个
				fscanf(fileptr2, "%d", &logprior);
				log_group_priors_i.push_back(logprior);
				group_priors_i.push_back(exp(logprior));
				//cout << word  << ";";
			}
			//while (fscanf(fileptr, "%d", &word) != EOF)//这个对于自制文档可以预先知道数量
			//{
			//	trainss.push_back(word - 1);//word 是从1到25，后面统计词频的时候要从0统计
			//	inds_groups.push_back(i);
			//	inds_items_i.push_back(num_words++);
			//	//printf("num_words ：%d", word - 1);
			//}
			fclose(fileptr1);
			fclose(fileptr2);
			
			group_candidates_i.push_back(i);//设置自连接的值是alpha 1
			log_group_priors_i.push_back(log_alpha_group);
			group_priors_i.push_back(alpha_group);
			///cout << endl;
		}
	

	}
}

template<typename D> void Layer<D>::initialize_link(int num_groups, int num_init_cls)
{//初始化团簇
	int i;
	int tot_num_words = trainss.size();
	customers.assign(tot_num_words, 0);//每一层的customer，table
	tables.assign(tot_num_words, 0);
	if (num_groups == 1)
	{
		inds_groups.assign(tot_num_words, 0);//每个轨迹单词所属的餐馆
		inds_items.push_back(vector<int>());//word 在数据集中的排序
		if (child)
		{
			collect_customers();
		}
		else
		{
			inds_items.back().reserve(tot_num_words);
			for (i = 0; i != tot_num_words; i++)
			{
				inds_items.back().push_back(i);
			}
		}
	}

	uni_tables.assign(num_groups, list<int>());
	pos_uni_tables.assign(tot_num_words, list<int>::iterator());
	//以下两段有什么本质区别，为啥中间层要用的和layer0，back要用的有区别
	if (num_init_cls == tot_num_words)
	{
		//每个节点自成一桌
		for (i = 0; i != inds_items.size(); i++)
		{
			list<int> &uni_tables_i = uni_tables.at(i);
			vector<int> &inds_items_i = inds_items.at(i);
			vector<int>::iterator it_i;
			int idx_j, idx_0 = inds_items_i.front();
			for (it_i = inds_items_i.begin(); it_i != inds_items_i.end(); it_i++)
			{
				idx_j = *it_i;
				customers.at(idx_j) = idx_j;//每个顾客的链接指向自己，即每个顾客自成一桌
				tables.at(idx_j) = idx_j;//d但餐桌和顾客仍然是对应关系
				uni_tables_i.push_back(idx_j);//每个餐馆内的餐桌标号
				pos_uni_tables.at(idx_j) = --uni_tables_i.end();
			}
		}
	}
	else
	{
		//每个餐馆中的顾客全部初始化到一桌上
		for (i = 0; i != inds_items.size(); i++)
		{

			list<int> &uni_tables_i = uni_tables.at(i);
			vector<int> &inds_items_i = inds_items.at(i);
			vector<int>::iterator it_i;
			int idx_j, idx_0 = inds_items_i.front();
			for (it_i = inds_items_i.begin(); it_i != inds_items_i.end(); it_i++)
			{//为每一个单词做标签
				idx_j = *it_i;
				customers.at(idx_j) = idx_0;//每个餐馆中的顾客初始化为1桌
				tables.at(idx_j) = idx_0;//这餐桌的标号就是第一个进入该餐馆的顾客
			}
			uni_tables_i.push_back(idx_0);//把该餐馆中的所有餐桌推到一个list中去（初始化中每个餐馆只有一个餐桌）
			pos_uni_tables.at(idx_0) = --uni_tables_i.end();//他是uni_tables 的迭代器
		}
	}
}

template<typename D> void Layer<D>::collect_customers()
{//从 child-> uni_table 中统计本层中每个餐馆的顾客标号（也就是child中的餐桌标号）
#ifdef PRINT_FUNC_NAME
	cout << "collect_customers()" << endl;
#endif
	vector< vector<int> >::iterator it_item_i;
	vector<int>::iterator it_item_j;
	vector< list<int> >::iterator it_tab_i;
	list<int>::iterator it_tab_j;
	if (child != NULL)
	{
		for (it_item_i = inds_items.begin(); it_item_i != inds_items.end(); it_item_i++)
		{
			it_item_i->clear();
		}

		vector< list<int> > &uni_tables_child = child->uni_tables;
		for (it_tab_i = uni_tables_child.begin(); it_tab_i != uni_tables_child.end(); it_tab_i++)
		{
			for (it_tab_j = it_tab_i->begin(); it_tab_j != it_tab_i->end(); it_tab_j++)
			{
				int idx_table = *it_tab_j;//childlayer中餐桌的标号
				//cout << "idx_table " << idx_table << endl;
				int idx_group = inds_groups.at(idx_table);//本层中在那个餐桌上的group标号
				//cout << "idx_group " << idx_group << endl;
				inds_items.at(idx_group).push_back(idx_table);//本层的inds_items 把餐桌标号作为单词放在对应的group中，这个group可能代表餐馆信息
			}
		}
	}
}

template<typename D> void Layer<D>::get_candidates()
{
#ifdef PRINT_FUNC_NAME
	cout << "get_candidates()" << endl;
#endif
	if (child == NULL)
	{//layer(0)
		int i = inds_groups.at(cur_item);//找到这个word对应的group
		vector<int> &inds_items_cur = inds_items.at(i);//找到这个group 当前所有word的索引值
		int j = cur_item - inds_items_cur.front();//当前的word距离这一组标签位置的多远
		if (item_candidates.empty())//用item_candidates 做判断，确没有对它进行赋值操作？
		{//如果item_candidates是空的，就对item_cands_cur赋inds_items_cur中的值，也就是这这一组中那些word 的索引
			vector<int>::iterator start = inds_items_cur.begin();
			vector<int>::iterator end = inds_items_cur.begin() + j + 1;
			item_cands_cur.assign(start, end);//item_priors_cur.assign(item_cands_cur.size(), 1.0);
			log_item_priors_cur.assign(item_cands_cur.size(), 0.0);//log(1)=0;//item_priors_cur.back() = alpha_item;
			log_item_priors_cur.back() = log_alpha_item;//这两个东西的维度可能不一致
		}
		else
		{//如果有值，使用item_candidate中的值对当前的item_cands进行赋值
			item_cands_cur = item_candidates.at(i).at(j);//item_priors_cur = item_priors.at(i).at(j);
			log_item_priors_cur = log_item_priors.at(i).at(j);
		}
	}
	else
	{//layer（1），layer（2）
		item_cands_cur.clear();
		item_priors_cur.clear();
		log_item_priors_cur.clear();
		int idx_table_child = cur_item;//当前word
		int idx_group_child = child->inds_groups.at(idx_table_child);//对应的child中的group索引
		vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//都依赖于child 的group candidate
		vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
		vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
		vector<int>::iterator it_c = group_candidates_i.begin();
		vector<double>::iterator it_p = group_priors_i.begin();
		vector<double>::iterator it_log_p = log_group_priors_i.begin();
		for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//和所在的餐馆有链接的餐馆
		{
			list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer中 与该餐馆链接的餐馆中的所有餐桌
			list<int>::iterator it_t = uni_tables_i.begin();
			for (; it_t != uni_tables_i.end(); it_t++)
			{
				item_cands_cur.push_back(*it_t);
			}
			item_priors_cur.insert(item_priors_cur.end(), uni_tables_i.size(), *it_p);
			log_item_priors_cur.insert(log_item_priors_cur.end(), uni_tables_i.size(), *it_log_p);
		}
		list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//所有在它前面的组都纳入候选
		list<int>::iterator it_t = uni_tables_i.begin();
		while (it_t != uni_tables_i.end() && cur_item > *it_t)
		{//cout << "uni_tables_i for item_cands_cur" << *it_t << endl;
			item_cands_cur.push_back(*it_t);
			item_priors_cur.push_back(1.0);
			log_item_priors_cur.push_back(0.0);
			it_t++;
		}
		item_cands_cur.push_back(idx_table_child);//给自己留的一个位置和相应的概率值，
		item_priors_cur.push_back(alpha_item);
		log_item_priors_cur.push_back(log_alpha_item);
	}
}

template<typename D> int Layer<D>::get_cluster(int _cur_item)
{
	int cluster_cur = _cur_item;
	Layer<D> *layer = this;
	do
	{
		cluster_cur = layer->tables.at(cluster_cur);
	} while (layer = layer->parent);
	return cluster_cur;
}

template<typename D> void Layer<D>::collect_clusters()
{
#ifdef PRINT_FUNC_NAME
	cout << "collect_clusters()" << endl;
#endif
	cls_cands_cur.clear();
	uni_cls_cands_cur.clear();

	vector<int>::iterator it, end = item_cands_cur.end() - 1;
	int cluster_i;
	vector<bool> flag(trainss.size(), false);
	for (it = item_cands_cur.begin(); it != end; it++)
	{
		cluster_i = get_cluster(*it);//得到最顶层的餐桌号
		cls_cands_cur.push_back(cluster_i);
		if (!flag.at(cluster_i))//保证存在uni_cls_cands_cur 中的标号是没有重复的
		{
			uni_cls_cands_cur.push_back(cluster_i);
			flag.at(cluster_i) = true;
		}
	}
}

template<typename D> void Layer<D>::collect_connections()
{
#ifdef PRINT_FUNC_NAME
	cout << "collect_connections()" << endl;
#endif
	it_stat_cur = stats.at(old_tables.at(cur_item)).begin() + inds_start.at(cur_item);
	/*cout << "cur_item:" << cur_item << endl;*/
	for (Layer<D>* layer = this; layer != cur_layer->child; layer = layer->child)
	{
		layer->cur_stat = layer->stats.at(old_tables.at(cur_item)).begin() + layer->inds_start.at(cur_item);
		layer->connection_start = layer->trees.at(layer->old_tables.at(cur_item)).begin() + layer->inds_start.at(cur_item);
		/*cout << *(layer->connection_start) << endl;*/
		layer->connection_end = layer->trees.at(layer->old_tables.at(cur_item)).begin() + layer->inds_end.at(cur_item) + 1;
		/*cout << *(layer->connection_start) << endl;*/
	}
#ifdef PRINT_FUNC_NAME
	cout << "collect_connections() was finished" << endl;
#endif
}

template<class D> void Layer<D>::check_link_status()
{
#ifdef PRINT_FUNC_NAME
	cout << "check_link_staturs()" << endl;
#endif
	old_customer_cur = customers.at(cur_item);
	old_table_cur = tables.at(old_customer_cur);
	is_self_linked = (old_customer_cur == cur_item) ? true : false;
	if (is_self_linked)
	{
		if (parent)
		{
			parent->check_link_status();
		}
		else
		{
			collect_connections();
		}
	}
	else
	{
		collect_connections();
	}
}

template<typename D> inline void Layer<D>::compute_marg_lik(int c)
{
	if (!is_computed.at(c))
	{
		double temp = base.marg_likelihood(*pos_classqq.at(c), *it_stat_cur);//这里似乎也应该根据是否为根节点使用it_stat_cur 或者是 cur_stat
		//double temp = base.marg_likelihood(*pos_classqq.at(c), *cur_stat);//1201这里似乎也应该根据是否为根节点使用it_stat_cur 或者是 cur_stat
		log_pred_liks.at(c) = temp;	
		pred_liks.at(c) = exp(log_pred_liks.at(c));
		is_computed.at(c) = true;
	}
}

template<typename D> void Layer<D>::compute_marg_liks()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_marg_liks()" << endl;
#endif
	/*#pragma omp parallel
		{
	#pragma omp for*/
			for (int i = 0; i < uni_cls_cands_cur.size(); i++)
			{
				compute_marg_lik(uni_cls_cands_cur.at(i));
			}
			
		/*}*/
	//clock_t t_strat = clock();
	//for_each(uni_cls_cands_cur.begin(), uni_cls_cands_cur.end(), bind1st(mem_fun(&Layer<D>::compute_marg_lik), this));
	//cout << "time costed : " << clock() - t_strat << endl;
}

template<typename D> double Layer<D>::compute_log_self_link_lik()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_log_self_link_lik()" << endl;
#endif
	get_candidates();
	collect_clusters();//uni_cls_cands_cur,存储了这些候选链接的顶层团簇标号
	compute_marg_liks();

	if (parent)
	{
		log_self_link_lik = parent->compute_log_self_link_lik();//是一个递归，直到算出log_self_link_lik的值，并更新qq_temp
	}
	else
	{
		base.reset_class(qq_temp);//重置为零
		log_self_link_lik = base.marg_likelihood(qq_temp, *it_stat_cur);
		//log_self_link_lik = base.marg_likelihood(qq_temp, *cur_stat);//1201
	}

	double child_self_link_lik = 0.0;
	double sum_priors = 0.0;
	log_probs_sampling.assign(cls_cands_cur.size(), 0.0);

	vector<int>::iterator it_c = cls_cands_cur.begin();
	vector<double>::iterator it_p = item_priors_cur.begin();
	vector<double>::iterator it_log_p = log_item_priors_cur.begin();
	vector<double>::iterator it_prob = log_probs_sampling.begin();
	double log_prob_i;
	for (; it_c != cls_cands_cur.end(); it_c++, it_p++, it_log_p++, it_prob++)
	{
		*it_prob = (*it_log_p) + log_pred_liks.at(*it_c);
		if (*it_prob > max_log_prob)
		{
			max_log_prob = *it_prob;
		}
		child_self_link_lik += pred_liks.at(*it_c) * (*it_p);
		sum_priors += *it_p;
	}
	log_probs_sampling.push_back(log_alpha_item + log_self_link_lik);
	if (log_probs_sampling.back() > max_log_prob)
	{
		max_log_prob = log_probs_sampling.back();
	}
	child_self_link_lik += exp(log_probs_sampling.back());
	sum_priors += alpha_item;
	child_self_link_lik /= sum_priors;
	return log(child_self_link_lik);
}

template<typename D> void Layer<D>::compute_log_probs_sampling()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_log_progs_sampling()" << endl;
#endif
	log_probs_sampling.clear();
	max_log_prob = -DBL_MAX;
	vector<int>::iterator it_c = cls_cands_cur.begin();
	vector<double>::iterator it_log_p = log_item_priors_cur.begin();//在寻找候选的时候就已经算好了
	double log_prob_i;
	for (; it_c != cls_cands_cur.end(); it_c++, it_log_p++)//这里用到的不是uni_cls_cands_cur 
	{
		log_prob_i = (*it_log_p) + log_pred_liks.at(*it_c);
		log_probs_sampling.push_back(log_prob_i);
		if (log_prob_i > max_log_prob)
		{
			max_log_prob = log_prob_i;
		}
	}
	log_prob_i = log_alpha_item + log_self_link_lik;
	log_probs_sampling.push_back(log_prob_i);
	if (log_prob_i > max_log_prob)
	{
		max_log_prob = log_prob_i;
	}
}

template<typename D> void Layer<D>::sample_customer()
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_customer()" << endl;
#endif
	double sum = 0.0;
	for (vector<double>::iterator it_p = log_probs_sampling.begin(); it_p != log_probs_sampling.end(); it_p++)
	{
		//double * it_p = &log_probs_sampling.at(i);
		*it_p = exp(*it_p - max_log_prob);
		sum += *it_p;
	}
	log_probs_sampling.push_back(sum);

	int idx_ci = rand_mult_1(log_probs_sampling);//产生一个随机的位置
	new_customer_cur = item_cands_cur.at(idx_ci);
}

template<typename D> inline void Layer<D>::change_table(int _new_table)
{
	for (vector<int>::const_iterator it = connection_start + 1; it != connection_end; it++)
	{
		tables.at(*it) = _new_table;
	}
}

template<typename D> void Layer<D>::merge_customers()
{//先采样再删除餐桌
#ifdef PRINT_FUNC_NAME
	cout << "merge_customer()" << endl;
#endif
	this->is_self_linked = (this->customers.at(cur_item) == cur_item) ? true : false;
	
	if (this->is_self_linked)
	{
		idx_group_cur = inds_groups.at(cur_item);
		delete_table(cur_item);
	}
	if (links.at(cur_item).size())
	{
		run_sampler_sfl();
		
		
	}
	
	/*change_customer(new_customer_cur);
	new_table_cur = tables.at(new_customer_cur);
	change_table(new_table_cur);*/
	if (is_self_linked)
	{
		if (parent)
		{
			//parent->new_customer_cur = new_table_cur;
			parent->merge_customers();
		}
		else
		{
			base.del_class(pos_classqq.at(cur_item));
			Layer<D> *layer_iter = cur_layer;
			do
			{
				layer_iter->traverse_links();
			} while (layer_iter = layer_iter->parent);
		}
	}
	else
	{
		Layer<D> *layer_iter = cur_layer;
		do
		{
			layer_iter->traverse_links();
		} while (layer_iter = layer_iter->parent);
	}

	
	/*if (is_self_linked)
	{
		idx_group_cur = inds_groups.at(cur_item);
		if (parent)
		{
			parent->new_customer_cur = new_table_cur;
			parent->merge_customers();
		}
		else
		{
			base.del_class(pos_classqq.at(cur_item));
		}
		delete_table(cur_item);
	}
	if (links.at(cur_item).size())
	{

		run_sampler_sfl();

		Layer<D> *layer_iter = this;
		do
		{
			layer_iter->traverse_links();
		} while (layer_iter = layer_iter->parent);
	}*/
	/*change_customer(new_customer_cur);
	new_table_cur = tables.at(new_customer_cur);
	change_table(new_table_cur);//该不该保留的问题*/
}

template<typename D> void Layer<D>::update_link()
{
#ifdef PRINT_FUNC_NAME
	cout << "update_link()" << endl;
#endif
	customers.at(cur_item) = new_customer_cur;
	tables.at(cur_item) = new_table_cur;
	change_table(new_table_cur);
}

template<typename D> inline void Layer<D>::delete_table(int _old_table)
{
	uni_tables.at(idx_group_cur).erase(pos_uni_tables.at(_old_table));
}

template<typename D> void Layer<D>::change_customer(int _new_customer)
{
	vector<int> &links_cur = links.at(cur_item);
	for (vector<int>::iterator it = links_cur.begin(); it != links_cur.end(); it++)
	{
		customers.at(*it) = _new_customer;
	}
}

template<typename D> void Layer<D>::add_table()
{
	list<int> &uni_tables_cur = uni_tables.at(idx_group_cur);
	list<int>::iterator it = uni_tables_cur.begin();
	while (it != uni_tables_cur.end() && cur_item > *it)
	{
		it++;
	}
	pos_uni_tables.at(cur_item) = uni_tables_cur.insert(it, cur_item);
}

template<typename D> void Layer<D>::sample_new_customer()
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_new_customer()" << endl;
#endif
	sample_customer();//这里面用的log_值要取自本层的后验计算，在计算log_self_lin_lik的时候算过，由于是针对不同层的计算，所以不会被更改
	customers.at(cur_item) = new_customer_cur;
	if (new_customer_cur != cur_item)
	{
		new_table_cur = tables.at(new_customer_cur);
		tables.at(cur_item) = new_table_cur;
		new_cls_cur = get_cluster(new_customer_cur);
		base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);
		//base.add_data(*pos_classqq.at(new_cls_cur), *cur_stat);//1201
	}
	else
	{
		new_table_cur = cur_item;//customer上面已经赋值了
		tables.at(cur_item) = cur_item;
		idx_group_cur = inds_groups.at(cur_item);
		add_table();
		if (parent)
		{
			parent->sample_new_customer();
		}
		else
		{
			pos_classqq.at(cur_item) = base.add_class();
			base.add_data(*pos_classqq.at(cur_item), *it_stat_cur);
			//base.add_data(*pos_classqq.at(cur_item), *cur_stat);//1201
		}
	}
}
template<typename D> void Layer<D>::get_cands_point(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "get_cands_point()" << endl;
#endif
	if (child == NULL)
	{//layer(0)
		int i = inds_groups.at(_cur_point);//找到这个word对应的group
		vector<int> &inds_items_cur = inds_items.at(i);//找到这个group 当前所有word的索引值
		int j = _cur_point - inds_items_cur.front();//当前的word距离这一组标签位置的多远
		if (item_candidates.empty())//用item_candidates 做判断，确没有对它进行赋值操作？
		{//如果item_candidates是空的，就对item_cands_cur赋inds_items_cur中的值，也就是这这一组中那些word 的索引
			vector<int>::iterator start = inds_items_cur.begin();
			vector<int>::iterator end = inds_items_cur.begin() + j + 1;
			item_cands_cur.assign(start, end);//item_priors_cur.assign(item_cands_cur.size(), 1.0);
			log_item_priors_cur.assign(item_cands_cur.size(), 0.0);//log(1)=0;//item_priors_cur.back() = alpha_item;
			log_item_priors_cur.back() = log_alpha_item;
		}
		else
		{//如果有值，使用item_candidate中的值对当前的item_cands进行赋值
			item_cands_cur = item_candidates.at(i).at(j);//item_priors_cur = item_priors.at(i).at(j);
			log_item_priors_cur = log_item_priors.at(i).at(j);
		}
	}
	else
	{//layer（1），layer（2）
		item_cands_cur.clear();
		item_priors_cur.clear();
		log_item_priors_cur.clear();
		int idx_table_child = _cur_point;//当前word
		int idx_group_child = child->inds_groups.at(idx_table_child);//对应的child中的group索引
		vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//都依赖于child 的group candidate
		vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
		vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
		vector<int>::iterator it_c = group_candidates_i.begin();
		vector<double>::iterator it_p = group_priors_i.begin();
		vector<double>::iterator it_log_p = log_group_priors_i.begin();
		for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//和所在的餐馆有链接的餐馆
		{
			list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer中 与该餐馆链接的餐馆中的所有餐桌
			list<int>::iterator it_t = uni_tables_i.begin();
			for (; it_t != uni_tables_i.end(); it_t++)
			{
				item_cands_cur.push_back(*it_t);
			}
			item_priors_cur.insert(item_priors_cur.end(), uni_tables_i.size(), *it_p);
			log_item_priors_cur.insert(log_item_priors_cur.end(), uni_tables_i.size(), *it_log_p);
		}
		list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//所有在它前面的组都纳入候选
		list<int>::iterator it_t = uni_tables_i.begin();
		while (it_t != uni_tables_i.end() && _cur_point > *it_t)
		{//cout << "uni_tables_i for item_cands_cur" << *it_t << endl;
			item_cands_cur.push_back(*it_t);
			item_priors_cur.push_back(1.0);
			log_item_priors_cur.push_back(0.0);
			it_t++;
		}
		item_cands_cur.push_back(idx_table_child);//给自己留的一个位置和相应的概率值，
		item_priors_cur.push_back(alpha_item);
		log_item_priors_cur.push_back(log_alpha_item);
	}
	
}
template<typename D> void Layer<D>::collect_connections_points(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "collect_connection_pionts()" << endl;
#endif
	it_stat_cur = stats.at(old_tables.at(_cur_point)).begin() + inds_start.at(_cur_point);
	for (Layer<D>* layer = this; layer != cur_layer->child; layer = layer->child)
	{
		layer->cur_stat = layer->stats.at(layer->old_tables.at(_cur_point)).begin() + layer->inds_start.at(_cur_point);
		layer->connection_start = layer->trees.at(layer->old_tables.at(_cur_point)).begin() + layer->inds_start.at(_cur_point);
		layer->connection_end = layer->trees.at(layer->old_tables.at(_cur_point)).begin() + layer->inds_end.at(_cur_point) + 1;
	}
}
template<typename D> void Layer<D>::check_link_status_point(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "check_link_status_point()" << endl;
#endif
	old_customer_cur = customers.at(_cur_point);
	old_table_cur = tables.at(old_customer_cur);
	is_self_linked = (old_customer_cur == _cur_point) ? true : false;
	if (is_self_linked)
	{
		if (parent)
		{
			parent->check_link_status_point(_cur_point);
		}
		else
		{
			collect_connections_points(_cur_point);
		}
	}
	else
	{
		collect_connections_points(_cur_point);
	}
}
template<typename D> void Layer<D>::compute_marg_lik_root(int c)
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_marg_lik_root()" << endl;
#endif
	if (!is_computed.at(c))
	{
		log_pred_liks.at(c) = base.marg_likelihood(*pos_classqq.at(c), *cur_stat);//1201
		pred_liks.at(c) = exp(log_pred_liks.at(c));
		is_computed.at(c) = true;
	}
}

template<typename D> void Layer<D>::compute_marg_liks_root()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_marg_liks_root()" << endl;
#endif
	for (int i = 0; i < uni_cls_cands_cur.size(); i++)
	{
		compute_marg_lik_root(uni_cls_cands_cur.at(i));
	}
}
template<typename D> void Layer<D>::update_point_link(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "update_point_link()" << endl;
#endif
	customers.at(_cur_point) = new_customer_cur;
	tables.at(_cur_point) = new_table_cur;
	change_table(new_table_cur);//这个没改 1201
}
template<typename D> void Layer<D>::sample_link_points(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "run_sampler_sfl()" << endl;
#endif
	cur_layer_sfl = this;
	vector<int> &links_cur = links.at(_cur_point);
	for (vector<int>::iterator it = links_cur.begin(); it != links_cur.end(); it++)
	{
		cur_link = *it;//不能让item 的采样和link的采样混在一起，这样会跳过很多item的采样
		sample_for_single_sfl();//1201之后就没有改动了
	}
}
template<typename D> void Layer<D>::sample_and_traverse(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_and_traverse()" << endl;
#endif
	 

	is_self_linked = (customers.at(_cur_point) == _cur_point) ? true : false;

	if (is_self_linked)
	{
		idx_group_cur = inds_groups.at(_cur_point);
		delete_table(_cur_point);
		if (parent)
		{
			parent->sample_and_traverse(_cur_point);
		}
		else
		{
			base.del_class(pos_classqq.at(_cur_point));
			//sample_link_points(_cur_point);
		}
	}
	sample_link_points(_cur_point);
	//把traverse放在外面



	//if (links.at(_cur_point).size())
	//{
	//	sample_link_points(_cur_point);
	//}
	//if (is_self_linked)
	//{
	//	if (parent)
	//	{
	//		//parent->new_customer_cur = new_table_cur;
	//		parent->merge_customers();
	//	}
	//	else
	//	{
	//		base.del_class(pos_classqq.at(cur_item));
	//		Layer<D> *layer_iter = cur_layer;
	//		do
	//		{
	//			layer_iter->traverse_links();
	//		} while (layer_iter = layer_iter->parent);
	//	}
	//}
	//else
	//{
	//	Layer<D> *layer_iter = cur_layer;
	//	do
	//	{
	//		layer_iter->traverse_links();
	//	} while (layer_iter = layer_iter->parent);
	//}

	
}
template<typename D> void Layer<D>::sample_for_root(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_for_root()" << endl;
#endif
	idx_group_cur = inds_groups.at(_cur_point);
	get_cands_point(_cur_point);
	if (item_cands_cur.size() == 1)
	{
		return;//如果只有自己就退出循环了
	}
	collect_clusters();
	if (parent)
	{
		parent->check_link_status_point(_cur_point);//如果不是最顶层（parent不是null）,就向上追溯，直到最顶层然后collect_connections.得到上一层层connection_start 和 connection_end
	}
	else
	{
		collect_connections_points(_cur_point);//得到本层connection_start 和 connection_end
	}
	old_cls_cur = get_cluster(_cur_point);
	base.del_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	is_computed.assign(trainss.size(), false);
	compute_marg_liks_root();
	if (parent)
	{
		log_self_link_lik = base.marg_likelihood(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	}
	else
	{
		base.reset_class(qq_temp);
		log_self_link_lik = base.marg_likelihood(qq_temp, *cur_stat);//对于顶层来说，按理cur_stat 应该和 it_stat_cur相等
	}
	compute_log_probs_sampling();
	sample_customer();
	if (new_customer_cur != old_customer_cur)
	{//对于根节点的讨论中，old_customer_cur==_cur_point,若new_customer_cur != old_customer_cur则会产生节点消失的情况
		new_table_cur = tables.at(new_customer_cur);
		update_point_link(_cur_point);
		new_cls_cur = get_cluster(new_customer_cur);
		base.add_data(*pos_classqq.at(new_cls_cur), *cur_stat);//1201
		delete_table(_cur_point);
		if (parent)
		{
			parent->sample_and_traverse(_cur_point);// sample_link_points(_cur_point);
		}
		else
		{
			base.del_class(pos_classqq.at(_cur_point));
		}
		Layer<D> *layer_iter = cur_layer;
				do
				{
					layer_iter->traverse_links();
				} while (layer_iter = layer_iter->parent);
	}
	else
	{
		base.add_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	}

}
template<typename D> void Layer<D>::add_table_at_point(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "add_table_at_point()" << endl;
#endif
	list<int> &uni_tables_cur = uni_tables.at(idx_group_cur);
	list<int>::iterator it = uni_tables_cur.begin();
	while (it != uni_tables_cur.end() && _cur_point > *it)
	{
		it++;
	}
	pos_uni_tables.at(_cur_point) = uni_tables_cur.insert(it, _cur_point);
}
template<typename D> void Layer<D>::sample_new_customer_point(int _cur_point)
{
	sample_customer();//这里面用的log_值要取自本层的后验计算，在计算log_self_lin_lik的时候算过，由于是针对不同层的计算，所以不会被更改
	customers.at(_cur_point) = new_customer_cur;
	if (new_customer_cur != _cur_point)
	{
		new_table_cur = tables.at(new_customer_cur);
		tables.at(_cur_point) = new_table_cur;
		new_cls_cur = get_cluster(new_customer_cur);
		base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);
	}
	else
	{
		new_table_cur = _cur_point;//customer上面已经赋值了
		tables.at(_cur_point) = _cur_point;
		idx_group_cur = inds_groups.at(_cur_point);
		add_table_at_point(_cur_point);
		if (parent)
		{
			parent->sample_new_customer_point(_cur_point);
		}
		else
		{
			pos_classqq.at(_cur_point) = base.add_class();
			base.add_data(*pos_classqq.at(_cur_point), *it_stat_cur);
		}
	}
}
template<typename D> void Layer<D>::sample_for_leaf(int _cur_point)
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_for_leaf()" << endl;
#endif
	idx_group_cur = inds_groups.at(_cur_point);
	get_cands_point(_cur_point);
	collect_clusters();
	collect_connections_points(_cur_point);//得到本层connection_start 和 connection_end
	old_cls_cur = get_cluster(_cur_point);
	base.del_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	is_computed.assign(trainss.size(), false);
	compute_marg_liks();
	if (parent)
	{
		log_self_link_lik = parent->compute_log_self_link_lik();
	}
	else
	{
		base.reset_class(qq_temp);
		log_self_link_lik = base.marg_likelihood(qq_temp, *it_stat_cur);//对于顶层来说，按理cur_stat 应该和 it_stat_cur相等
	}
	compute_log_probs_sampling();
	sample_customer();
	if (new_customer_cur != old_customer_cur)
	{//对于叶子节点的讨论中,可能存在自连接，但是不会有上层节点消失的情况
		if (new_customer_cur != _cur_point)
		{
			new_table_cur = tables.at(new_customer_cur);
			update_point_link(_cur_point);
			new_cls_cur = get_cluster(_cur_point);
			base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);
		}
		else
		{
			new_table_cur = new_customer_cur;
			update_point_link(_cur_point);
			add_table_at_point(_cur_point);
			if (parent)
			{
				parent->sample_new_customer_point(_cur_point);
			}
			else
			{
				pos_classqq.at(_cur_point) = base.add_class();
				base.add_data(*pos_classqq.at(_cur_point), *it_stat_cur);
			}
		}
	}
	else
	{
		base.add_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	}
}
template<typename D> void Layer<D>::sample_for_point()
{
	cout << "cur_item" << cur_item <<endl;
	old_customer_cur = customers.at(cur_item);
	old_table_cur = tables.at(old_customer_cur);
	is_self_linked = (old_customer_cur == cur_item) ? true : false;
	if (is_self_linked)
	{
		sample_for_root(cur_item);
	}
	else
	{
		sample_for_leaf(cur_item);
	}
}
template<typename D> void Layer<D>::sample_for_single()
{
	cout << cur_item << endl;
	idx_group_cur = inds_groups.at(cur_item);//当前节点所在的团簇标号
	get_candidates();//item_cand_cur 收集在本餐馆中所有在cur_item之前出现的顾客，以及在子层中所有通过距离D定义的有链接的餐馆中的餐桌（在本层中叫做顾客）
	if (item_cands_cur.size() == 1)
	{
		return;//如果只有自己就退出循环了
	}
	collect_clusters();//uni_cls_cands_cur,存储了这些候选链接的顶层团
	old_customer_cur = customers.at(cur_item);
	old_table_cur = tables.at(old_customer_cur);
	is_self_linked = (old_customer_cur == cur_item) ? true : false;
	if (is_self_linked)
	{//如果在这一层与自己连接
		if (parent)
		{
			parent->check_link_status();//如果不是最顶层（parent不是null）,就向上追溯，直到最顶层然后collect_connections.得到上一层层connection_start 和 connection_end
		}
		else
		{
			collect_connections();//得到本层connection_start 和 connection_end
		}
	}
	else
	{
		collect_connections();
	}
	old_cls_cur = get_cluster(cur_item);//只是得到自己的团簇标号
	base.del_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);//把整个团簇中与cur_item相连的word 的统计值删掉
	is_computed.assign(trainss.size(), false);
	compute_marg_liks();//计算团簇uni_cls_cands_cur的似然值然后存储在pred_links.at(c)中,(因为剪掉一些之后会有改变)
	/*cout << "we are here" << endl;*/
	//上面是算链接到其他的地方的似然，后面是算自连接的似然

	if (parent)
	{
		if (is_self_linked)
		{
			
			log_self_link_lik = base.marg_likelihood(*pos_classqq.at(old_cls_cur), *it_stat_cur);//计算如果还是链接到old_cls_cur上的似然
			/*cout << "we are here " << endl;*/
		}
		else
		{//不是自连接
			log_self_link_lik = parent->compute_log_self_link_lik();//构造公式5.8
		}
	}
	else
	{//对于顶层
		base.reset_class(qq_temp);
		log_self_link_lik = base.marg_likelihood(qq_temp, *it_stat_cur);
	}
	compute_log_probs_sampling();//这里寻找最大的存在max_log_prob
	sample_customer();
	//已经得到新的采样链接，开始计算当前链接会对餐桌配置产生什么影响
	if (new_customer_cur != old_customer_cur)
	{
		//cout << "got a new customer :" << endl;
		clock_t t_start = clock();
		if (cur_item != new_customer_cur)
		{
			new_table_cur = tables.at(new_customer_cur);
			update_link();
			new_cls_cur = get_cluster(new_customer_cur);
			//base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);//1130,这里如果进行上层的采样的话，就不能把所有的link所带的子树的统计量都加进去
			
			//可以把这一步放到判断里面
			if (is_self_linked)
			{//先到上层去采样，在删除餐桌
				base.add_data(*pos_classqq.at(new_cls_cur), *cur_stat);//1201
				delete_table(cur_item);
				if (parent)
				{
					//parent->new_customer_cur = new_table_cur;//1201
					parent->merge_customers();//注意这里是在上层中做的处理
				}
				else
				{
					base.del_class(pos_classqq.at(cur_item));
				}
				//1128
				//if (parent)
				//{
				//	parent->new_customer_cur = new_table_cur;
				//	parent->merge_customers();//注意这里是在上层中做的处理
				//}
				//else
				//{
				//	base.del_class(pos_classqq.at(cur_item));
				//}
				//delete_table(cur_item);
			}
			else
			{
				base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);//1201
			}
		}
		else
		{
			new_table_cur = new_customer_cur;
			update_link();
			add_table();
			if (parent)
			{
				parent->sample_new_customer();
			}
			else
			{
				pos_classqq.at(cur_item) = base.add_class();
				base.add_data(*pos_classqq.at(cur_item), *it_stat_cur);
				//base.add_data(*pos_classqq.at(cur_item), *cur_stat);//1201
			}
		}
		//cout << "time costed :" << clock() - t_start << endl;
	}
	else
	{
		base.add_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);
	} 
}

template<typename D> void Layer<D>::run_sampler()
{
	clock_t timer_t, timer_a;
	cur_layer = this;
	if (!child)
	{
		timer_t = clock();
		//随机等间隔采样
		//int a = 1, b = 100;
		//srand((unsigned)clock());
		//int space = (rand() % (b - a + 1)) + a;
		//cout << "space = " << space << endl;
		//for (cur_item = 0; cur_item < trainss.size(); cur_item = cur_item + space)
	
//#pragma omp parallel
//		{
//#pragma omp for
			for (int group_i = 0; group_i < this->num_groups; group_i++)
			{
				for (auto iter = inds_items.at(group_i).begin(); iter != inds_items.at(group_i).end(); iter++)
				{
					cur_item = *iter;
					//sample_for_single();
					sample_for_point();//1201
				}
			}
		/*}*/
		/*for (cur_item = 0; cur_item < trainss.size(); cur_item = cur_item + 1)
		{
			sample_for_single();
		}*/
		cout << "the cost is " << (double)(clock() - timer_t) / CLOCKS_PER_SEC << endl;

	}
	else
	{
		timer_t = clock();
		//
		vector<int>& uni_tables_vec_child = child->uni_tables_vec;
		//for (vector<int>::iterator it = uni_tables_vec_child.begin(); it != uni_tables_vec_child.end(); it++)
		for (int i = 0; i < uni_tables_vec_child.size(); i = i + 1)
			//for (vector<int>::iterator it = uni_tables_vec_child.begin(); it < uni_tables_vec_child.end(); it=it+space)
		{
			cur_item = uni_tables_vec_child.at(i);
			//sample_for_single();
			sample_for_point();//1201
		}
		cout << "the cost is " << (double)(clock() - timer_t) / CLOCKS_PER_SEC << endl;
	}
}

template<typename D> void Layer<D>::get_candidates_sfl()
{
	item_cands_cur_sfl.clear();
	item_priors_cur_sfl.clear();
	log_item_priors_cur_sfl.clear();
	int idx_table_child = cur_link;//当前word
	int idx_group_child = child->inds_groups.at(idx_table_child);//对应的child中的group索引
	vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//都依赖于child 的group candidate
	vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
	vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
	vector<int>::iterator it_c = group_candidates_i.begin();
	vector<double>::iterator it_p = group_priors_i.begin();
	vector<double>::iterator it_log_p = log_group_priors_i.begin();
	for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//和所在的餐馆有链接的餐馆
	{
		list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer中 与该餐馆链接的餐馆中的所有餐桌
		list<int>::iterator it_t = uni_tables_i.begin();
		for (; it_t != uni_tables_i.end(); it_t++)
		{
			item_cands_cur_sfl.push_back(*it_t);
		}
		item_priors_cur_sfl.insert(item_priors_cur_sfl.end(), uni_tables_i.size(), *it_p);
		log_item_priors_cur_sfl.insert(log_item_priors_cur_sfl.end(), uni_tables_i.size(), *it_log_p);
	}
	list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//所有在它前面的组都纳入候选
	list<int>::iterator it_t = uni_tables_i.begin();
	while (it_t != uni_tables_i.end() && cur_link > *it_t )
	{//cout << "uni_tables_i for item_cands_cur" << *it_t << endl;
		if (cur_link == cur_item)
		{
			continue;
		}
		item_cands_cur_sfl.push_back(*it_t);
		item_priors_cur_sfl.push_back(1.0);
		log_item_priors_cur_sfl.push_back(0.0);
		it_t++;
	}
	item_cands_cur_sfl.push_back(idx_table_child);//给自己留的一个位置和相应的概率值，
	item_priors_cur_sfl.push_back(alpha_item);
	log_item_priors_cur_sfl.push_back(log_alpha_item);

}

template<typename D> int Layer<D>::get_cluster_sfl(int _cur_link)
{
	int cluster_cur = _cur_link;
	Layer<D> *layer = this;
	do
	{
		cluster_cur = layer->tables.at(cluster_cur);//1129
	} while (layer = layer->parent);
	return cluster_cur;

}

template<typename D> void Layer<D>::collect_clusters_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "collect_clusters_sfl()" << endl;
#endif
	cls_cands_cur_sfl.clear();
	uni_cls_cands_cur_sfl.clear();
	vector<int>::iterator it, end = item_cands_cur_sfl.end() - 1;//这里没包括自连接
	int cluster_i;
	vector<bool> flag(trainss.size(), false);
	for (it = item_cands_cur_sfl.begin(); it != end; it++)
	{
		cluster_i = get_cluster_sfl(*it);//得到最顶层的餐桌号
		cls_cands_cur_sfl.push_back(cluster_i);
		if (!flag.at(cluster_i))//保证存在uni_cls_cands_cur 中的标号是没有重复的
		{
			uni_cls_cands_cur_sfl.push_back(cluster_i);
			flag.at(cluster_i) = true;
		}
	}
}

template<typename D> void Layer<D>::collect_connections_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "collect_connetion_sfl()" << endl;
#endif
	it_stat_cur_sfl = stats.at(old_tables.at(cur_item)).begin() + inds_start.at(cur_link);
	Layer<D>* layer = this;
	/*for (Layer<D>* layer = this; layer != cur_layer->child; layer = layer->child)
	{*/
		layer->connection_start_sfl = layer->trees.at(layer->old_tables.at(cur_link)).begin() + layer->inds_start.at(cur_link);
		layer->connection_end_sfl = layer->trees.at(layer->old_tables.at(cur_link)).begin() + layer->inds_end.at(cur_link) + 1;
	/*}*/

}

template<typename D> void Layer<D>::compute_marg_lik_sfl(int c)
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_marg_lik_sfl()" << endl;
#endif
	if (!is_computed_sfl.at(c))
	{
		/*cout << "c:" << c << endl;
		cout << "(*pos_classqq.at(c)).qq_word.back():" << (*pos_classqq.at(c)).qq_word.back() << endl;
		cout << "(*it_stat_cur_sfl).tot" << (*it_stat_cur_sfl).tot << endl;*/
		double temp =base.marg_likelihood(*pos_classqq.at(c), *it_stat_cur_sfl);
		/*cout << "temp :" << temp << endl;*/
		log_pred_liks_sfl.at(c) = temp;
		/*cout << "we are here" << endl;*/
		pred_liks_sfl.at(c) = exp(log_pred_liks_sfl.at(c));
		is_computed_sfl.at(c) = true;
	}
}


template<typename D> void Layer<D>::compute_marg_liks_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_marg_liks_sfl()" << endl;
#endif
	for (int i = 0; i < uni_cls_cands_cur_sfl.size(); i++)
	{
		compute_marg_lik_sfl(uni_cls_cands_cur_sfl.at(i));
	}
}

template<typename D> double Layer<D>::compute_log_self_link_lik_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_log_self_link_lik_sfl()" << endl;
#endif
	get_candidates_sfl();
	collect_clusters_sfl();
	compute_marg_liks_sfl();
	if (parent)
	{
		log_self_link_lik_sfl = parent->compute_log_self_link_lik_sfl();//是一个递归，直到算出log_self_link_lik的值，并更新qq_temp
	}
	else
	{
		base.reset_class(qq_temp);//重置为零
		log_self_link_lik_sfl = base.marg_likelihood(qq_temp, *it_stat_cur_sfl);
	}
	double child_self_link_lik_sfl = 0.0;
	double sum_priors_sfl = 0.0;
	log_probs_sampling_sfl.assign(cls_cands_cur_sfl.size(), 0.0);
	vector<int>::iterator it_c = cls_cands_cur_sfl.begin();
	vector<double>::iterator it_p = item_priors_cur_sfl.begin();
	vector<double>::iterator it_log_p = log_item_priors_cur_sfl.begin();
	vector<double>::iterator it_prob = log_probs_sampling_sfl.begin();
	double log_prob_i;
	for (; it_c != cls_cands_cur_sfl.end(); it_c++, it_p++, it_log_p++, it_prob++)
	{
		*it_prob = (*it_log_p) + log_pred_liks_sfl.at(*it_c);
		if (*it_prob > max_log_prob_sfl)
		{
			max_log_prob_sfl = *it_prob;
		}
		child_self_link_lik_sfl += pred_liks_sfl.at(*it_c) * (*it_p);
		sum_priors_sfl += *it_p;
	}
	log_probs_sampling_sfl.push_back(log_alpha_item + log_self_link_lik_sfl);
	if (log_probs_sampling_sfl.back() > max_log_prob_sfl)
	{
		max_log_prob_sfl = log_probs_sampling_sfl.back();
	}
	child_self_link_lik_sfl += exp(log_probs_sampling_sfl.back());
	sum_priors_sfl += alpha_item;
	child_self_link_lik_sfl /= sum_priors_sfl;
	return log(child_self_link_lik_sfl);
}

template<typename D> void Layer<D>::compute_log_probs_sampling_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "compute_log_probs_sampling_sfl()" << endl;
#endif
	log_probs_sampling_sfl.clear();
	max_log_prob_sfl = -DBL_MAX;
	vector<int>::iterator it_c = cls_cands_cur_sfl.begin();
	vector<double>::iterator it_log_p = log_item_priors_cur_sfl.begin();
	double log_prob_i;
	for (; it_c != cls_cands_cur_sfl.end(); it_c++, it_log_p++) 
	{//这里用到的不是uni_cls_cands_cur
		log_prob_i = (*it_log_p) + log_pred_liks_sfl.at(*it_c);
		log_probs_sampling_sfl.push_back(log_prob_i);
		if (log_prob_i > max_log_prob_sfl)
		{
			max_log_prob_sfl = log_prob_i;
		}
	}
	log_prob_i = log_alpha_item + log_self_link_lik_sfl;
	log_probs_sampling_sfl.push_back(log_prob_i);
	if (log_prob_i > max_log_prob_sfl)
	{
		max_log_prob_sfl = log_prob_i;
	}
}

template<typename D> void Layer<D>::sample_customer_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "sample_customer_sfl()" << endl;
#endif
	double sum = 0.0;
	for (vector<double>::iterator it_p = log_probs_sampling_sfl.begin(); it_p != log_probs_sampling_sfl.end(); it_p++)
	{
		//double * it_p = &log_probs_sampling.at(i);
		*it_p = exp(*it_p - max_log_prob_sfl);
		sum += *it_p;
	}
	log_probs_sampling_sfl.push_back(sum);

	int idx_ci = rand_mult_1(log_probs_sampling_sfl);//产生一个随机的位置
	new_customer_cur_sfl = item_cands_cur_sfl.at(idx_ci);
}
template<typename D> inline void Layer<D>::change_table_sfl(int _new_table)
{
#ifdef PRINT_FUNC_NAME
	cout << "change_table_sfl()" << endl;
#endif
	for (vector<int>::const_iterator it = connection_start_sfl + 1; it != connection_end_sfl; it++)
	{
		tables.at(*it) = _new_table;
	}
}
template<typename D> void Layer<D>::update_link_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "update_link_sfl()" << endl;
#endif
	customers.at(cur_link) = new_customer_cur_sfl;
	tables.at(cur_link) = new_table_cur_sfl;
	change_table_sfl(new_table_cur_sfl);
}

template<typename D> void Layer<D>::add_table_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "add_table_sfl()" << endl;
#endif
	list<int> &uni_tables_cur = uni_tables.at(idx_group_cur_sfl);
	list<int>::iterator it = uni_tables_cur.begin();
	while (it != uni_tables_cur.end() && cur_link > *it)
	{
		it++;
	}
	pos_uni_tables.at(cur_link) = uni_tables_cur.insert(it, cur_link);
}

template<typename D> void Layer<D>::sample_new_customer_sfl()
{
	sample_customer_sfl();//这里面用的log_值要取自本层的后验计算，在计算log_self_lin_lik的时候算过，由于是针对不同层的计算，所以不会被更改
	customers.at(cur_link) = new_customer_cur_sfl;
	if (new_customer_cur_sfl != cur_link)
	{
		new_table_cur_sfl = tables.at(new_customer_cur_sfl);
		tables.at(cur_link) = new_table_cur_sfl;
		new_cls_cur_sfl = get_cluster_sfl(new_customer_cur_sfl);
		base.add_data(*pos_classqq.at(new_cls_cur_sfl), *it_stat_cur_sfl);
	}
	else
	{
		new_table_cur_sfl = cur_link;//customer上面已经赋值了
		tables.at(cur_link) = cur_link;
		idx_group_cur_sfl = inds_groups.at(cur_link);
		add_table_sfl();
		if (parent)
		{
			parent->sample_new_customer_sfl();
		}
		else
		{
			pos_classqq.at(cur_link) = base.add_class();
			base.add_data(*pos_classqq.at(cur_link), *it_stat_cur_sfl);
		}
	}
}

template<typename D> void Layer<D>::sample_for_single_sfl()
{
	cout << "cur_link"<<cur_link << endl;
	idx_group_cur_sfl= inds_groups.at(cur_link);//当前节点所在的团簇标号
	get_candidates_sfl();
	//if (item_cands_cur_sfl.size() == 1)
	//{//如果只有自己就退出循环了,这里区别于对item的采样，如果只有自己也就只能是自连接的情况了,这种情况在下面有考虑
	//	//这种情况下面有考虑，所以不必单独列出来
	//	cout << "zi ji " << endl;
	//	int iil;
	//	cin >> iil;
	//	collect_connections_sfl();//收集所有的子树

	//	new_customer_cur_sfl = cur_link;
	//	new_table_cur_sfl = cur_link;
	//	update_link_sfl();
	//	add_table_sfl();
	//	if (parent)
	//	{
	//		log_self_link_lik_sfl = parent->compute_log_self_link_lik_sfl();
	//		parent->sample_new_customer_sfl();//
	//	}
	//	else
	//	{
	//		pos_classqq.at(cur_link) = base.add_class();
	//		base.add_data(*pos_classqq.at(cur_link), *it_stat_cur_sfl);
	//	}
	//	return;
	//}
	collect_clusters_sfl();//uni_cls_cands_cur,存储了这些候选链接的顶层团
	collect_connections_sfl();//收集所有的子树
	//old_cls_cur_sfl = get_cluster_sfl(cur_item);//只是得到自己的团簇标号
	////已经见过一回来不能再减了
	//base.del_data(*pos_classqq.at(old_cls_cur_sfl), *it_stat_cur_sfl);//把整个团簇中与cur_item相连的word 的统计值删掉
	is_computed_sfl.clear();
	is_computed_sfl.assign(trainss.size(), false);
	compute_marg_liks_sfl();//计算团簇uni_cls_cands_cur的似然值然后存储在pred_links.at(c)中,(因为剪掉一些之后会有改变)
	if (parent)
	{
		log_self_link_lik_sfl = parent->compute_log_self_link_lik_sfl();
	}
	else
	{
		base.reset_class(qq_temp);
		log_self_link_lik_sfl = base.marg_likelihood(qq_temp, *it_stat_cur_sfl);
	}
	compute_log_probs_sampling_sfl();//这里寻找最大的存在max_log_prob
	sample_customer_sfl();
	if (new_customer_cur_sfl == cur_link)//如果新采样的链接自成一桌
	{
		new_table_cur_sfl = new_customer_cur_sfl;
		update_link_sfl();
		add_table_sfl();
		if (parent)
		{
			parent->sample_new_customer_sfl();//
		}
		else
		{
			pos_classqq.at(cur_link) = base.add_class();
			base.add_data(*pos_classqq.at(cur_link), *it_stat_cur_sfl);
		}
	}
	else
	{
		new_table_cur_sfl = tables.at(new_customer_cur_sfl);
		update_link_sfl();
		new_cls_cur_sfl = get_cluster_sfl(new_customer_cur_sfl);
		base.add_data(*pos_classqq.at(new_cls_cur_sfl), *it_stat_cur_sfl);
	}
}

template<typename D> void Layer<D>::run_sampler_sfl()
{
#ifdef PRINT_FUNC_NAME
	cout << "run_sampler_sfl()" << endl;
#endif
	cur_layer_sfl = this;
	vector<int> &links_cur = links.at(cur_item);
	for (vector<int>::iterator it = links_cur.begin(); it != links_cur.end(); it++)
	{
		cur_link = *it;//不能让item 的采样和link的采样混在一起，这样会跳过很多item的采样
		sample_for_single_sfl();
		//customers.at(*it) = _new_customer;
	}
	//delete_table
}

template<typename D> void Layer<D>::traverse_single_table(int idx_table)
{//输入是餐桌的标号，对餐桌的树及其子树进行梳理存储在tree.at(idx_table)，并统计其观测值存在相应的stats中，
	vector<int>& tree_i = trees.at(idx_table);//
	vector<int>& order_i = orders.at(idx_table);
	vector<STAT>& stat_i = stats.at(idx_table);
	vector<int> to_visit, to_visit_father, father_i;
	int cnt = 0;
	tree_i.push_back(idx_table);//起始元素是这个餐桌的标号点
	father_i.push_back(-1);//根节点处标记为-1 应该是没有父节点的意思
	stat_i.push_back(STAT());
	inds_start.at(idx_table) = cnt++;//为根节点处的状态夹1
	vector<int>& links_t = links.at(idx_table);//在一个餐桌标号处的link值
	vector<int>::iterator p = links_t.begin();
	if (p != links_t.end())//这里并非循环只是判断是否为空
	{
		to_visit.push_back(*p);//to_visit 装填的是下一个要访问的位置
		to_visit_father.push_back(idx_table);//装填的是本次所在的节点
		p++;
	}
	else
	{//如果这个餐桌的link值是空的 
		inds_end.at(idx_table) = 0;
		child ?//child 为true计算expr1；为false计算expr2
			(stat_i.front() = child->stats.at(idx_table).front()) : //如果有子节点
			(stat_i.front().init(trainss.at(idx_table), base.get_eta()));//如果是最底层,初始化idx_table那一点的uni_ss；uni_qq;的值

		order_i.push_back(0);
		return;
	}

	for (; p != links_t.end(); p++)
	{
		to_visit.push_back(*p);//指向下一个值
		to_visit_father.push_back(-1);//父节点已经保存过了，这里都标记成-1
	}
	int curr;
	while (!to_visit.empty())//如果经过上面的折腾之后to_visit不是空的，则？？
	{
		curr = to_visit.back();//从最后一个开始
		inds_start.at(curr) = cnt++;//这个cnt是跟着上面的来的
		tree_i.push_back(curr);
		stat_i.push_back(STAT());
		father_i.push_back(to_visit_father.back());
		to_visit.pop_back();
		to_visit_father.pop_back();//把tovisit和tovisitfater的值给tree_i ,father_i之后释放
		vector<int> & links_curr = links.at(curr);
		if (!links_curr.empty())
		{//如果在curr点的link不是空的
			p = links_curr.begin();
			to_visit.push_back(*p);
			to_visit_father.push_back(curr);
			for (p++; p != links_curr.end(); p++)
			{
				to_visit.push_back(*p);
				to_visit_father.push_back(-1);
			}
		}
		else
		{//如果该点的links是空的
			inds_end.at(curr) = inds_start.at(curr);

			child ?
				(stat_i.at(inds_start.at(curr)).init(child->stats.at(curr).front(), base.get_eta())) :
				stat_i.at(inds_start.at(curr)).init(trainss.at(curr), base.get_eta());
			order_i.push_back(inds_start.at(curr));//表明了这个

			int f = father_i.at(inds_start.at(curr));
			while (f >= 0)//有大于0表示其子节点已经算完了
			{
				inds_end.at(f) = inds_end.at(curr);//那么该节点所包含的子节点就应该到这个位置结束了
				child ?
					(stat_i.at(inds_start.at(f)).init(child->stats.at(f).front(), base.get_eta())) :
					stat_i.at(inds_start.at(f)).init(trainss.at(f), base.get_eta());
				order_i.push_back(inds_start.at(f));
				for (p = links.at(f).begin(); p != links.at(f).end(); p++)
				{
					stat_i.at(inds_start.at(f)).update(stat_i.at(inds_start.at(*p)));//改变了stat中的inds的值
					stat_i.at(inds_start.at(*p)).clear_inds();

				}
				f = father_i.at(inds_start.at(f));//继续向上回溯
			}
		}
	}
	stat_i.front().clear_inds();
}

template<typename D> void Layer<D>::traverse_links()
{
	uni_tables_vec.clear();
	for (auto it_i = uni_tables.begin(); it_i != uni_tables.end(); it_i++)
	{
		for (auto it_j = it_i->begin(); it_j != it_i->end(); it_j++)
		{
			uni_tables_vec.push_back(*it_j);
		}
	}
	links.assign(trainss.size(), vector<int>());
	if (child)
	{///第1.2层使用
		for (auto it = child->uni_tables_vec.begin(); it != child->uni_tables_vec.end(); it++)//用的是child的数据更新本层数据
		{
			int cst = customers.at(*it);//这个customer.at(*it)是本层在*it位置的顾客与谁相连
			if (cst != *it)//如果不是自连接
			{
				links.at(cst).push_back(*it);//把该顾客（*it）放入顾客链接指向的节点（cst）的link列表中
			}
		}
	}
	else
	{//第0层使用
		int i = 0;
		for (auto it = customers.begin(); it != customers.end(); it++, i++)
		{

			if (*it != i)
			{
				links.at(*it).push_back(i);//*it是顾客i所指向的值，遍历所有顾客指向之后就得到了，所有指向顾客*it的顾客i组成的向量
			}

		}
	}

	old_tables = tables;
	trees.assign(trainss.size(), vector<int>());
	orders.assign(trainss.size(), vector<int>());
	stats.assign(trainss.size(), vector<STAT>());
	inds_start.assign(trainss.size(), -1);
	inds_end.assign(trainss.size(), -1);
#pragma omp parallel
	{
#pragma omp for //__gnu_parallel::for_each(uni_tables_vec.begin(), uni_tables_vec.end(), bind1st(mem_fun(&Layer<D>::traverse_single_table), this));
		for (int i = 0; i < uni_tables_vec.size(); i++)
		{
			this->traverse_single_table(uni_tables_vec.at(i));
		}
		//for_each(uni_tables_vec.begin(), uni_tables_vec.end(), bind1st(mem_fun(&Layer<D>::traverse_single_table), this));
		//cout << "stats.at(0).at(0).uni_qq.at(0) : " << stats.at(0).at(0).uni_qq.at(0) << endl;
	}

	//cout << "after traverse_link the link.size() is  : " << this->links.size() << endl;

}
#ifdef TRI_MULT_DIST

template<typename D> int Layer<D>::sample_ss(vector<int>& qq, vector<double>& eta, int w)
{//这里以减掉当前值的qq作为似然，然后进行采样得到新的采样值
	qq.at(w)--;//对应的统计量减掉一个
	qq.back()--;//对应的个数减一个
	vector<double> weights = eta;
	for (int i = 0; i != weights.size(); i++)
	{
		weights.at(i) += (double)qq.at(i);
	}
	int new_w = rand_mult_1(weights);
	qq.at(new_w)++;
	qq.back()++;
	return new_w;
}

template<typename D> void Layer<D>::sample_source_sink_c(int t, int c)
{
	if (child)
	{//如果不是底层
		for (auto it = trees.at(t).begin(); it != trees.at(t).end(); it++)
		{//对该餐桌t对应的树中的每一顾客,也即下一层的餐桌进行sample，这里的c代表最外层的类别标号
			child->sample_source_sink_c(*it, c);
		}
	}
	else
	{//追溯到底层的每个餐桌t,聚类标号c
		QQ& qq = *pos_classqq.at(c);
		for (auto it = trees.at(t).begin(); it != trees.at(t).end(); it++)
		{//树中的每一个节点ss,如果不是观测值就要进行重新采样
			SS& ss = trainss.at(*it);

			if (ss.gibbs_source)
			{
				ss.source = sample_ss(qq.qq_source, base.get_eta().eta_source, ss.source);
			}

			if (ss.gibbs_sink)
			{
				ss.sink = sample_ss(qq.qq_sink, base.get_eta().eta_sink, ss.sink);
			}
		}
	}
}

template<typename D> void Layer<D>::sample_source_sink()
{
	//__gnu_parallel::for_each(top->uni_tables_vec.begin(), top->uni_tables_vec.end(), [](int c){ top->sample_source_sink_c(c, c); });
	//#pragma omp parallel
	{
		//#pragma omp for
		//for ()
		//对每一个最顶层的餐桌进行采样，c代表餐桌标号
		for_each(top->uni_tables_vec.begin(), top->uni_tables_vec.end(), [](int c){ top->sample_source_sink_c(c, c); });
	}
}

template<typename D> void Layer<D>::update_ss_stats_c(int idx_table)
{
	vector<int>& tree_i = trees.at(idx_table);
	vector<int>& order_i = orders.at(idx_table);
	vector<STAT>& stat_i = stats.at(idx_table);
	vector<int>::iterator it_o;
	int curr;
	for (it_o = order_i.begin(); it_o != order_i.end(); it_o++)
	{
		curr = tree_i.at(*it_o);
		child ? stat_i.at(*it_o).copy_ss(child->stats.at(curr).front()) : stat_i.at(*it_o).init_ss(trainss.at(curr), base.get_eta());
		for (auto p = links.at(curr).begin(); p != links.at(curr).end(); p++)
		{
			stat_i.at(*it_o).update_ss(stat_i.at(inds_start.at(*p)));
			stat_i.at(inds_start.at(*p)).clear_inds();
		}
	}
}

template<typename D> void Layer<D>::update_ss_stats()
{
#pragma omp parallel
	{
#pragma omp for//	__gnu_parallel::for_each(uni_tables_vec.begin(), uni_tables_vec.end(), bind1st(mem_fun(&Layer<D>::update_ss_stats_c), this));
		for (int i = 0; i<uni_tables_vec.size(); i++)
		{
			//bind1st(mem_fun(&Layer<D>::update_ss_stats_c), &uni_tables_vec.at(i));
			this->update_ss_stats_c(uni_tables_vec.at(i));
		}
		//for_each(uni_tables_vec.begin(), uni_tables_vec.end(), bind1st(mem_fun(&Layer<D>::update_ss_stats_c), this));
	}
}

template<typename D> void Layer<D>::label_instances()
{
	for (int i = 0; i != bottom->num_groups; i++)
	{//对于底层的每个餐馆，也就是每一条轨迹
		vector<int> topics;
		vector<int> topic_stat;
		vector<int> pos(trainss.size(), -1);
		vector<int> &inds_items_i = bottom->inds_items.at(i);
		for (int j = 0; j != inds_items_i.size(); j++)
		{//对于每个节点，即轨迹片段
			int c = bottom->get_cluster(inds_items_i.at(j));//类别标签
			if (pos.at(c) < 0)
			{//如果这个类别标签是第一次出现，在该位置写进新的主题标号
				pos.at(c) = topics.size();
				topics.push_back(c);
				topic_stat.push_back(1);
			}
			else
			{
				topic_stat.at(pos.at(c))++;//记录主题的节点数目
			}
		}
		int max_stat = 0, idx = 0;
		for (int i = 0; i != topic_stat.size(); i++)
		{
			if (topic_stat.at(i) > max_stat)
			{
				max_stat = topic_stat.at(i);
				idx = i;
			}
		}
		labels.at(i) = topics.at(idx);//以节点数目最多的主题作为该餐馆的主题
	}
}

template<typename D> void Layer<D>::save_labels(string& save_dir, int iter)
{
	if (save_dir.back() != '/')
	{
		save_dir += '/';
	}
	ostringstream file_name_stream;
	file_name_stream << save_dir << "label" << iter;
	string label_file_name = file_name_stream.str() + ".txt";
	ofstream output_stream(label_file_name.c_str());
	cout << "saving trajectory labels at iteration " << iter << " ..." << endl;
	for (int i = 0; i != bottom->num_groups; i++)
	{
		output_stream << labels.at(i) << endl;
	}
	output_stream.close();
}

template<typename D> void Layer<D>::save_topic_labels(string& save_dir, int iter)
{
	if (save_dir.back() != '/')
	{
		save_dir += '/';
	}
	ostringstream file_name_stream;
	file_name_stream << save_dir << "topic_label" << iter;
	string label_file_name = file_name_stream.str() + ".txt";
	ofstream output_stream(label_file_name.c_str());
	cout << "saving topic labels for words at iteration " << iter << " ..." << endl;
	for (int i = 0; i != trainss.size(); i++)
	{
		output_stream << bottom->get_cluster(i) << endl;
	}
	output_stream.close();
}
#endif