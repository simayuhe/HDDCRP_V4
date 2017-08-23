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
	static D base;//�����˱���Ļ���ͳ����Ϣ��Ϊɶ��static
	static vector< list<QQ>::iterator > pos_classqq;//�ڳ���������λ�ô洢ָ����������classqq��ָ��
	static vector< SS > trainss;
	static vector<int> labels;//���label�ǹ켣�ı�ǩ

	static Layer<D> *top;
	static Layer<D> *bottom;//��������ָ����

	static double tot_lik;//�������������Ŵطֲ�����Ȼֵ�����Ƚϸ���iteration �Ĳ������ʱ�õ�

	static Layer<D> *cur_layer;//��runsampler��������ǵ�ǰ��
	static vector<bool> is_computed;//��������Ƿ�������ĳһ��Ϊ��ǵ��Ŵص���Ȼ����runsampler�л��ظ�����
	//������ڶ�link��������ر���
	static Layer<D> *cur_layer_sfl;//��runsampler��������ǵ�ǰ��
	static vector<bool> is_computed_sfl;//��������Ƿ�������ĳһ��Ϊ��ǵ��Ŵص���Ȼ����runsampler�л��ظ�����
	//
	static void initialize_base();//�����е�trianss����ʼ����һ��classqq��ȥ
	static double compute_tot_lik();//���㶥�����в�����Ȼֵ���ܺ�

	Layer<D> *parent;
	Layer<D> *child;

	int num_groups;
	double alpha_group;//����͹�֮�����������
	double alpha_item;
	double log_alpha_group;
	double log_alpha_item;

	vector< vector<int> > group_candidates;
	vector< vector<double> > log_group_priors;
	vector< vector<double> > group_priors;
	vector< vector< vector<int> > > item_candidates;//��ʲô�ط���ֵ�ģ�
	vector< vector< vector<double> > > log_item_priors;
	vector< vector< vector<double> > > item_priors;

	vector<int> customers; // customers to which each member links
	vector<int> tables; // tables each member belongs to
	vector<int> old_tables; // tables each member belongs to
	//vector<int> clusters;	
	vector< vector<int> > links; // customers from which each member is visited
	vector< list<int> > uni_tables; // unique tables in this restaurantΪʲô��listԪ����ɵ�������ÿһ���п����в�ͬ�Ĳ͹�
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

	vector< STAT >::iterator cur_stat;//����ÿһ����cur_item ����Ӧ��������ͳ�������ڷǸ��ڵ��ʱ����it_stat_cur��ͬ
	//�ڸ��ڵ�ʱ��it_stat_cur ָʾ�������ϲ��ͳ�ƣ����������е�link����ϣ��ʹ�ñ����cur_stat����¼�����иýڵ��ͳ��ֵ���������ϲ��link
	//��������Ŀ�����ڶԸ��ڵ�������²���ʱ���ܹ������ڵ㴦��ͳ��ֵ���ϲ���link����ͳ��ֵ�ֿ����벻ͬ���Ŵ���ȥ

	//������ڶ�link��������ر���
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

	vector< vector<int> > trees;//ÿ��ĳ���Ϊ�ڵ����������ÿ�������ڵ��ϼ�¼�������Ľڵ�ֲ�
	vector< vector<int> > orders;//����stats�����У��Ⱥ�˳�������Ҷ�ڵ㵽���ڵ�
	vector< vector<STAT> > stats;//��tree�еĽڵ����Ӧ��ͳ���˸ýڵ����е������Ĺ۲�ֵ
	vector< int > inds_start;//����Ϊ�ڵ���Ŀ����ʾ�˸ýڵ�������Ǵ�tree.at(i)���Ǹ�λ�ÿ�ʼ
	vector< int > inds_end;//Ҫ��tree���ʹ��
							
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
	void collect_connections();//�õ���Ӧ�������ڵ��ţ������ͳ����
	void check_link_status();
	void compute_marg_lik(int c);
	void compute_marg_liks();
	double compute_log_self_link_lik();
	void compute_log_probs_sampling();
	void sample_customer();
	void change_table(int _new_table);//�ı�connection_start end�����Ľڵ�Ĳ������Ϊ_new_table
	void update_link();//�ı�cur_item ��customer����table,��change_table��new_table_cure��
	void delete_table(int _old_table);//��idx_group_cur��ǰ�͹��а�ԭ�������old_talble erase����
	void change_customer(int _new_customer);//���� ������������������ڵ���ʧʱ����ԭ��ָ��cur_item�Ľڵ㶼ָ�����µĲ����ı�Žڵ㣬����һ�ֽ��ƴ���
	void merge_customers();
	void add_table();//����uni_table������list���е������ԣ��ڸò͹�idx_group_cur��ĳ���̶�λ�ò���һ���������
	void sample_new_customer();//���ϲ�Ϊ�˿Ͳ����µ�����
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
	//������ڶ�link��������غ��� sample_for_link  :sfl
	void get_candidates_sfl();
	int get_cluster_sfl(int _cur_link);
	void collect_clusters_sfl();
	void collect_connections_sfl();//�õ���Ӧ�������ڵ��ţ������ͳ����
	void check_link_status_sfl();
	void compute_marg_lik_sfl(int c);
	void compute_marg_liks_sfl();
	double compute_log_self_link_lik_sfl();
	void compute_log_probs_sampling_sfl();
	void sample_customer_sfl();
	void change_table_sfl(int _new_table);//�ı�connection_start end�����Ľڵ�Ĳ������Ϊ_new_table
	void update_link_sfl();//�ı�cur_item ��customer����table,��change_table��new_table_cure��
	void delete_table_sfl(int _old_table);//��idx_group_cur��ǰ�͹��а�ԭ�������old_talble erase����
	void change_customer_sfl(int _new_customer);//���� ������������������ڵ���ʧʱ����ԭ��ָ��cur_item�Ľڵ㶼ָ�����µĲ����ı�Žڵ㣬����һ�ֽ��ƴ���
	void merge_customers_sfl();
	void add_table_sfl();//����uni_table������list���е������ԣ��ڸò͹�idx_group_cur��ĳ���̶�λ�ò���һ���������
	void sample_new_customer_sfl();//���ϲ�Ϊ�˿Ͳ����µ�����
	void sample_for_single_sfl();
	void run_sampler_sfl();
	//
	void traverse_single_table(int idx_table);
	void traverse_links();
#ifdef TRI_MULT_DIST
	static int sample_ss(vector<int>& qq, vector<double>& eta, int w);
	void sample_source_sink_c(int t, int c);
	static void sample_source_sink();
	void update_ss_stats_c(int idx_table);//ͨ��order��ָ����Ҷ�ӽڵ㿪ʼ��stat���и���
	void update_ss_stats();

	static void label_instances();
	static void save_labels(string& save_dir, int iter);
	static void save_topic_labels(string& save_dir, int iter);
#endif
};

template<typename D> D Layer<D>::base;
template<typename D> vector< list<QQ>::iterator > Layer<D>::pos_classqq;//ָ��class����qq��ָ��
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
//��������
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
	pos_classqq.at(0) = base.add_class();//����һ��ָ��classqqĩβ��ָ��
	for (vector<SS>::iterator it_w = trainss.begin(); it_w != trainss.end(); it_w++)
	{
		//	base.add_data(base.get_classqq().front(), trainss.at(i));
		base.add_data(base.get_classqq().front(), *it_w);//������Multinomial.cpp ���� Tri_Mult.cpp�У�ȡ����base�������������ڳ���Ϊ1000�ʵ�qq�ж�Ӧ��λ��ͳ��ÿ�����ʳ��ֵĴ���
	}
	is_computed.assign(tot_num_words, false);
	log_pred_liks.assign(tot_num_words, 0.0);
	pred_liks.assign(tot_num_words, 0.0);
	//Ϊlink������׼��
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
		num_groups = 1;//����layer��0������ʼ��Ϊ1���Ŵ�
		//num_groups = 100;//����layer��0������ʼ��Ϊ100���Ŵ�
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
				trainss.push_back((int)(pd[j] - 1));//ΪʲôҪ��һ�أ�ԭʼ��77��������76������ʲô�ô��أ�
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
		//����ʡ���˺ܶ�reverse�Ļ��ڣ����ⷴ����д�ĵ�
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
			while (fscanf(fileptr, "%d", &word) != EOF)//������������ĵ�����Ԥ��֪������
			{
				trainss.push_back(word-1);//word �Ǵ�1��25������ͳ�ƴ�Ƶ��ʱ��Ҫ��0ͳ��
				inds_groups.push_back(i);
				inds_items_i.push_back(num_words++);
				//printf("num_words ��%d", word - 1);
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
				group_candidates_i.push_back((int)(pd_cands_i[j]) - 1);//�����б�Ŷ�����һ��
				log_group_priors_i.push_back(pd_log_priors_i[j]);
				group_priors_i.push_back(exp(pd_log_priors_i[j]));
			}

			group_candidates_i.push_back(i);//���������ӵ�ֵ��alpha 1
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
			//����ÿƪ�ĵ�
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
				group_candidates_i.push_back(word - 1);//�����б�Ŷ�����һ��
				fscanf(fileptr2, "%d", &logprior);
				log_group_priors_i.push_back(logprior);
				group_priors_i.push_back(exp(logprior));
				//cout << word  << ";";
			}
			//while (fscanf(fileptr, "%d", &word) != EOF)//������������ĵ�����Ԥ��֪������
			//{
			//	trainss.push_back(word - 1);//word �Ǵ�1��25������ͳ�ƴ�Ƶ��ʱ��Ҫ��0ͳ��
			//	inds_groups.push_back(i);
			//	inds_items_i.push_back(num_words++);
			//	//printf("num_words ��%d", word - 1);
			//}
			fclose(fileptr1);
			fclose(fileptr2);
			
			group_candidates_i.push_back(i);//���������ӵ�ֵ��alpha 1
			log_group_priors_i.push_back(log_alpha_group);
			group_priors_i.push_back(alpha_group);
			///cout << endl;
		}
	

	}
}

template<typename D> void Layer<D>::initialize_link(int num_groups, int num_init_cls)
{//��ʼ���Ŵ�
	int i;
	int tot_num_words = trainss.size();
	customers.assign(tot_num_words, 0);//ÿһ���customer��table
	tables.assign(tot_num_words, 0);
	if (num_groups == 1)
	{
		inds_groups.assign(tot_num_words, 0);//ÿ���켣���������Ĳ͹�
		inds_items.push_back(vector<int>());//word �����ݼ��е�����
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
	//����������ʲô��������Ϊɶ�м��Ҫ�õĺ�layer0��backҪ�õ�������
	if (num_init_cls == tot_num_words)
	{
		//ÿ���ڵ��Գ�һ��
		for (i = 0; i != inds_items.size(); i++)
		{
			list<int> &uni_tables_i = uni_tables.at(i);
			vector<int> &inds_items_i = inds_items.at(i);
			vector<int>::iterator it_i;
			int idx_j, idx_0 = inds_items_i.front();
			for (it_i = inds_items_i.begin(); it_i != inds_items_i.end(); it_i++)
			{
				idx_j = *it_i;
				customers.at(idx_j) = idx_j;//ÿ���˿͵�����ָ���Լ�����ÿ���˿��Գ�һ��
				tables.at(idx_j) = idx_j;//d�������͹˿���Ȼ�Ƕ�Ӧ��ϵ
				uni_tables_i.push_back(idx_j);//ÿ���͹��ڵĲ������
				pos_uni_tables.at(idx_j) = --uni_tables_i.end();
			}
		}
	}
	else
	{
		//ÿ���͹��еĹ˿�ȫ����ʼ����һ����
		for (i = 0; i != inds_items.size(); i++)
		{

			list<int> &uni_tables_i = uni_tables.at(i);
			vector<int> &inds_items_i = inds_items.at(i);
			vector<int>::iterator it_i;
			int idx_j, idx_0 = inds_items_i.front();
			for (it_i = inds_items_i.begin(); it_i != inds_items_i.end(); it_i++)
			{//Ϊÿһ����������ǩ
				idx_j = *it_i;
				customers.at(idx_j) = idx_0;//ÿ���͹��еĹ˿ͳ�ʼ��Ϊ1��
				tables.at(idx_j) = idx_0;//������ı�ž��ǵ�һ������ò͹ݵĹ˿�
			}
			uni_tables_i.push_back(idx_0);//�Ѹò͹��е����в����Ƶ�һ��list��ȥ����ʼ����ÿ���͹�ֻ��һ��������
			pos_uni_tables.at(idx_0) = --uni_tables_i.end();//����uni_tables �ĵ�����
		}
	}
}

template<typename D> void Layer<D>::collect_customers()
{//�� child-> uni_table ��ͳ�Ʊ�����ÿ���͹ݵĹ˿ͱ�ţ�Ҳ����child�еĲ�����ţ�
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
				int idx_table = *it_tab_j;//childlayer�в����ı��
				//cout << "idx_table " << idx_table << endl;
				int idx_group = inds_groups.at(idx_table);//���������Ǹ������ϵ�group���
				//cout << "idx_group " << idx_group << endl;
				inds_items.at(idx_group).push_back(idx_table);//�����inds_items �Ѳ��������Ϊ���ʷ��ڶ�Ӧ��group�У����group���ܴ���͹���Ϣ
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
		int i = inds_groups.at(cur_item);//�ҵ����word��Ӧ��group
		vector<int> &inds_items_cur = inds_items.at(i);//�ҵ����group ��ǰ����word������ֵ
		int j = cur_item - inds_items_cur.front();//��ǰ��word������һ���ǩλ�õĶ�Զ
		if (item_candidates.empty())//��item_candidates ���жϣ�ȷû�ж������и�ֵ������
		{//���item_candidates�ǿյģ��Ͷ�item_cands_cur��inds_items_cur�е�ֵ��Ҳ��������һ������Щword ������
			vector<int>::iterator start = inds_items_cur.begin();
			vector<int>::iterator end = inds_items_cur.begin() + j + 1;
			item_cands_cur.assign(start, end);//item_priors_cur.assign(item_cands_cur.size(), 1.0);
			log_item_priors_cur.assign(item_cands_cur.size(), 0.0);//log(1)=0;//item_priors_cur.back() = alpha_item;
			log_item_priors_cur.back() = log_alpha_item;//������������ά�ȿ��ܲ�һ��
		}
		else
		{//�����ֵ��ʹ��item_candidate�е�ֵ�Ե�ǰ��item_cands���и�ֵ
			item_cands_cur = item_candidates.at(i).at(j);//item_priors_cur = item_priors.at(i).at(j);
			log_item_priors_cur = log_item_priors.at(i).at(j);
		}
	}
	else
	{//layer��1����layer��2��
		item_cands_cur.clear();
		item_priors_cur.clear();
		log_item_priors_cur.clear();
		int idx_table_child = cur_item;//��ǰword
		int idx_group_child = child->inds_groups.at(idx_table_child);//��Ӧ��child�е�group����
		vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//��������child ��group candidate
		vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
		vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
		vector<int>::iterator it_c = group_candidates_i.begin();
		vector<double>::iterator it_p = group_priors_i.begin();
		vector<double>::iterator it_log_p = log_group_priors_i.begin();
		for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//�����ڵĲ͹������ӵĲ͹�
		{
			list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer�� ��ò͹����ӵĲ͹��е����в���
			list<int>::iterator it_t = uni_tables_i.begin();
			for (; it_t != uni_tables_i.end(); it_t++)
			{
				item_cands_cur.push_back(*it_t);
			}
			item_priors_cur.insert(item_priors_cur.end(), uni_tables_i.size(), *it_p);
			log_item_priors_cur.insert(log_item_priors_cur.end(), uni_tables_i.size(), *it_log_p);
		}
		list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//��������ǰ����鶼�����ѡ
		list<int>::iterator it_t = uni_tables_i.begin();
		while (it_t != uni_tables_i.end() && cur_item > *it_t)
		{//cout << "uni_tables_i for item_cands_cur" << *it_t << endl;
			item_cands_cur.push_back(*it_t);
			item_priors_cur.push_back(1.0);
			log_item_priors_cur.push_back(0.0);
			it_t++;
		}
		item_cands_cur.push_back(idx_table_child);//���Լ�����һ��λ�ú���Ӧ�ĸ���ֵ��
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
		cluster_i = get_cluster(*it);//�õ����Ĳ�����
		cls_cands_cur.push_back(cluster_i);
		if (!flag.at(cluster_i))//��֤����uni_cls_cands_cur �еı����û���ظ���
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
		double temp = base.marg_likelihood(*pos_classqq.at(c), *it_stat_cur);//�����ƺ�ҲӦ�ø����Ƿ�Ϊ���ڵ�ʹ��it_stat_cur ������ cur_stat
		//double temp = base.marg_likelihood(*pos_classqq.at(c), *cur_stat);//1201�����ƺ�ҲӦ�ø����Ƿ�Ϊ���ڵ�ʹ��it_stat_cur ������ cur_stat
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
	collect_clusters();//uni_cls_cands_cur,�洢����Щ��ѡ���ӵĶ����Ŵر��
	compute_marg_liks();

	if (parent)
	{
		log_self_link_lik = parent->compute_log_self_link_lik();//��һ���ݹ飬ֱ�����log_self_link_lik��ֵ��������qq_temp
	}
	else
	{
		base.reset_class(qq_temp);//����Ϊ��
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
	vector<double>::iterator it_log_p = log_item_priors_cur.begin();//��Ѱ�Һ�ѡ��ʱ����Ѿ������
	double log_prob_i;
	for (; it_c != cls_cands_cur.end(); it_c++, it_log_p++)//�����õ��Ĳ���uni_cls_cands_cur 
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

	int idx_ci = rand_mult_1(log_probs_sampling);//����һ�������λ��
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
{//�Ȳ�����ɾ������
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
	change_table(new_table_cur);//�ò��ñ���������*/
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
	sample_customer();//�������õ�log_ֵҪȡ�Ա���ĺ�����㣬�ڼ���log_self_lin_lik��ʱ���������������Բ�ͬ��ļ��㣬���Բ��ᱻ����
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
		new_table_cur = cur_item;//customer�����Ѿ���ֵ��
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
		int i = inds_groups.at(_cur_point);//�ҵ����word��Ӧ��group
		vector<int> &inds_items_cur = inds_items.at(i);//�ҵ����group ��ǰ����word������ֵ
		int j = _cur_point - inds_items_cur.front();//��ǰ��word������һ���ǩλ�õĶ�Զ
		if (item_candidates.empty())//��item_candidates ���жϣ�ȷû�ж������и�ֵ������
		{//���item_candidates�ǿյģ��Ͷ�item_cands_cur��inds_items_cur�е�ֵ��Ҳ��������һ������Щword ������
			vector<int>::iterator start = inds_items_cur.begin();
			vector<int>::iterator end = inds_items_cur.begin() + j + 1;
			item_cands_cur.assign(start, end);//item_priors_cur.assign(item_cands_cur.size(), 1.0);
			log_item_priors_cur.assign(item_cands_cur.size(), 0.0);//log(1)=0;//item_priors_cur.back() = alpha_item;
			log_item_priors_cur.back() = log_alpha_item;
		}
		else
		{//�����ֵ��ʹ��item_candidate�е�ֵ�Ե�ǰ��item_cands���и�ֵ
			item_cands_cur = item_candidates.at(i).at(j);//item_priors_cur = item_priors.at(i).at(j);
			log_item_priors_cur = log_item_priors.at(i).at(j);
		}
	}
	else
	{//layer��1����layer��2��
		item_cands_cur.clear();
		item_priors_cur.clear();
		log_item_priors_cur.clear();
		int idx_table_child = _cur_point;//��ǰword
		int idx_group_child = child->inds_groups.at(idx_table_child);//��Ӧ��child�е�group����
		vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//��������child ��group candidate
		vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
		vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
		vector<int>::iterator it_c = group_candidates_i.begin();
		vector<double>::iterator it_p = group_priors_i.begin();
		vector<double>::iterator it_log_p = log_group_priors_i.begin();
		for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//�����ڵĲ͹������ӵĲ͹�
		{
			list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer�� ��ò͹����ӵĲ͹��е����в���
			list<int>::iterator it_t = uni_tables_i.begin();
			for (; it_t != uni_tables_i.end(); it_t++)
			{
				item_cands_cur.push_back(*it_t);
			}
			item_priors_cur.insert(item_priors_cur.end(), uni_tables_i.size(), *it_p);
			log_item_priors_cur.insert(log_item_priors_cur.end(), uni_tables_i.size(), *it_log_p);
		}
		list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//��������ǰ����鶼�����ѡ
		list<int>::iterator it_t = uni_tables_i.begin();
		while (it_t != uni_tables_i.end() && _cur_point > *it_t)
		{//cout << "uni_tables_i for item_cands_cur" << *it_t << endl;
			item_cands_cur.push_back(*it_t);
			item_priors_cur.push_back(1.0);
			log_item_priors_cur.push_back(0.0);
			it_t++;
		}
		item_cands_cur.push_back(idx_table_child);//���Լ�����һ��λ�ú���Ӧ�ĸ���ֵ��
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
	change_table(new_table_cur);//���û�� 1201
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
		cur_link = *it;//������item �Ĳ�����link�Ĳ�������һ�������������ܶ�item�Ĳ���
		sample_for_single_sfl();//1201֮���û�иĶ���
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
	//��traverse��������



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
		return;//���ֻ���Լ����˳�ѭ����
	}
	collect_clusters();
	if (parent)
	{
		parent->check_link_status_point(_cur_point);//���������㣨parent����null��,������׷�ݣ�ֱ�����Ȼ��collect_connections.�õ���һ���connection_start �� connection_end
	}
	else
	{
		collect_connections_points(_cur_point);//�õ�����connection_start �� connection_end
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
		log_self_link_lik = base.marg_likelihood(qq_temp, *cur_stat);//���ڶ�����˵������cur_stat Ӧ�ú� it_stat_cur���
	}
	compute_log_probs_sampling();
	sample_customer();
	if (new_customer_cur != old_customer_cur)
	{//���ڸ��ڵ�������У�old_customer_cur==_cur_point,��new_customer_cur != old_customer_cur�������ڵ���ʧ�����
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
	sample_customer();//�������õ�log_ֵҪȡ�Ա���ĺ�����㣬�ڼ���log_self_lin_lik��ʱ���������������Բ�ͬ��ļ��㣬���Բ��ᱻ����
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
		new_table_cur = _cur_point;//customer�����Ѿ���ֵ��
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
	collect_connections_points(_cur_point);//�õ�����connection_start �� connection_end
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
		log_self_link_lik = base.marg_likelihood(qq_temp, *it_stat_cur);//���ڶ�����˵������cur_stat Ӧ�ú� it_stat_cur���
	}
	compute_log_probs_sampling();
	sample_customer();
	if (new_customer_cur != old_customer_cur)
	{//����Ҷ�ӽڵ��������,���ܴ��������ӣ����ǲ������ϲ�ڵ���ʧ�����
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
	idx_group_cur = inds_groups.at(cur_item);//��ǰ�ڵ����ڵ��Ŵر��
	get_candidates();//item_cand_cur �ռ��ڱ��͹���������cur_item֮ǰ���ֵĹ˿ͣ��Լ����Ӳ�������ͨ������D����������ӵĲ͹��еĲ������ڱ����н����˿ͣ�
	if (item_cands_cur.size() == 1)
	{
		return;//���ֻ���Լ����˳�ѭ����
	}
	collect_clusters();//uni_cls_cands_cur,�洢����Щ��ѡ���ӵĶ�����
	old_customer_cur = customers.at(cur_item);
	old_table_cur = tables.at(old_customer_cur);
	is_self_linked = (old_customer_cur == cur_item) ? true : false;
	if (is_self_linked)
	{//�������һ�����Լ�����
		if (parent)
		{
			parent->check_link_status();//���������㣨parent����null��,������׷�ݣ�ֱ�����Ȼ��collect_connections.�õ���һ���connection_start �� connection_end
		}
		else
		{
			collect_connections();//�õ�����connection_start �� connection_end
		}
	}
	else
	{
		collect_connections();
	}
	old_cls_cur = get_cluster(cur_item);//ֻ�ǵõ��Լ����Ŵر��
	base.del_data(*pos_classqq.at(old_cls_cur), *it_stat_cur);//�������Ŵ�����cur_item������word ��ͳ��ֵɾ��
	is_computed.assign(trainss.size(), false);
	compute_marg_liks();//�����Ŵ�uni_cls_cands_cur����ȻֵȻ��洢��pred_links.at(c)��,(��Ϊ����һЩ֮����иı�)
	/*cout << "we are here" << endl;*/
	//�����������ӵ������ĵط�����Ȼ���������������ӵ���Ȼ

	if (parent)
	{
		if (is_self_linked)
		{
			
			log_self_link_lik = base.marg_likelihood(*pos_classqq.at(old_cls_cur), *it_stat_cur);//��������������ӵ�old_cls_cur�ϵ���Ȼ
			/*cout << "we are here " << endl;*/
		}
		else
		{//����������
			log_self_link_lik = parent->compute_log_self_link_lik();//���칫ʽ5.8
		}
	}
	else
	{//���ڶ���
		base.reset_class(qq_temp);
		log_self_link_lik = base.marg_likelihood(qq_temp, *it_stat_cur);
	}
	compute_log_probs_sampling();//����Ѱ�����Ĵ���max_log_prob
	sample_customer();
	//�Ѿ��õ��µĲ������ӣ���ʼ���㵱ǰ���ӻ�Բ������ò���ʲôӰ��
	if (new_customer_cur != old_customer_cur)
	{
		//cout << "got a new customer :" << endl;
		clock_t t_start = clock();
		if (cur_item != new_customer_cur)
		{
			new_table_cur = tables.at(new_customer_cur);
			update_link();
			new_cls_cur = get_cluster(new_customer_cur);
			//base.add_data(*pos_classqq.at(new_cls_cur), *it_stat_cur);//1130,������������ϲ�Ĳ����Ļ����Ͳ��ܰ����е�link������������ͳ�������ӽ�ȥ
			
			//���԰���һ���ŵ��ж�����
			if (is_self_linked)
			{//�ȵ��ϲ�ȥ��������ɾ������
				base.add_data(*pos_classqq.at(new_cls_cur), *cur_stat);//1201
				delete_table(cur_item);
				if (parent)
				{
					//parent->new_customer_cur = new_table_cur;//1201
					parent->merge_customers();//ע�����������ϲ������Ĵ���
				}
				else
				{
					base.del_class(pos_classqq.at(cur_item));
				}
				//1128
				//if (parent)
				//{
				//	parent->new_customer_cur = new_table_cur;
				//	parent->merge_customers();//ע�����������ϲ������Ĵ���
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
		//����ȼ������
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
	int idx_table_child = cur_link;//��ǰword
	int idx_group_child = child->inds_groups.at(idx_table_child);//��Ӧ��child�е�group����
	vector<int> &group_candidates_i = child->group_candidates.at(idx_group_child);//��������child ��group candidate
	vector<double> &group_priors_i = child->group_priors.at(idx_group_child);
	vector<double> &log_group_priors_i = child->log_group_priors.at(idx_group_child);
	vector<int>::iterator it_c = group_candidates_i.begin();
	vector<double>::iterator it_p = group_priors_i.begin();
	vector<double>::iterator it_log_p = log_group_priors_i.begin();
	for (; it_c != group_candidates_i.end() - 1; it_c++, it_p++, it_log_p++)//�����ڵĲ͹������ӵĲ͹�
	{
		list<int> &uni_tables_i = child->uni_tables.at(*it_c);//child layer�� ��ò͹����ӵĲ͹��е����в���
		list<int>::iterator it_t = uni_tables_i.begin();
		for (; it_t != uni_tables_i.end(); it_t++)
		{
			item_cands_cur_sfl.push_back(*it_t);
		}
		item_priors_cur_sfl.insert(item_priors_cur_sfl.end(), uni_tables_i.size(), *it_p);
		log_item_priors_cur_sfl.insert(log_item_priors_cur_sfl.end(), uni_tables_i.size(), *it_log_p);
	}
	list<int> &uni_tables_i = child->uni_tables.at(idx_group_child);//��������ǰ����鶼�����ѡ
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
	item_cands_cur_sfl.push_back(idx_table_child);//���Լ�����һ��λ�ú���Ӧ�ĸ���ֵ��
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
	vector<int>::iterator it, end = item_cands_cur_sfl.end() - 1;//����û����������
	int cluster_i;
	vector<bool> flag(trainss.size(), false);
	for (it = item_cands_cur_sfl.begin(); it != end; it++)
	{
		cluster_i = get_cluster_sfl(*it);//�õ����Ĳ�����
		cls_cands_cur_sfl.push_back(cluster_i);
		if (!flag.at(cluster_i))//��֤����uni_cls_cands_cur �еı����û���ظ���
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
		log_self_link_lik_sfl = parent->compute_log_self_link_lik_sfl();//��һ���ݹ飬ֱ�����log_self_link_lik��ֵ��������qq_temp
	}
	else
	{
		base.reset_class(qq_temp);//����Ϊ��
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
	{//�����õ��Ĳ���uni_cls_cands_cur
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

	int idx_ci = rand_mult_1(log_probs_sampling_sfl);//����һ�������λ��
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
	sample_customer_sfl();//�������õ�log_ֵҪȡ�Ա���ĺ�����㣬�ڼ���log_self_lin_lik��ʱ���������������Բ�ͬ��ļ��㣬���Բ��ᱻ����
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
		new_table_cur_sfl = cur_link;//customer�����Ѿ���ֵ��
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
	idx_group_cur_sfl= inds_groups.at(cur_link);//��ǰ�ڵ����ڵ��Ŵر��
	get_candidates_sfl();
	//if (item_cands_cur_sfl.size() == 1)
	//{//���ֻ���Լ����˳�ѭ����,���������ڶ�item�Ĳ��������ֻ���Լ�Ҳ��ֻ���������ӵ������,��������������п���
	//	//������������п��ǣ����Բ��ص����г���
	//	cout << "zi ji " << endl;
	//	int iil;
	//	cin >> iil;
	//	collect_connections_sfl();//�ռ����е�����

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
	collect_clusters_sfl();//uni_cls_cands_cur,�洢����Щ��ѡ���ӵĶ�����
	collect_connections_sfl();//�ռ����е�����
	//old_cls_cur_sfl = get_cluster_sfl(cur_item);//ֻ�ǵõ��Լ����Ŵر��
	////�Ѿ�����һ���������ټ���
	//base.del_data(*pos_classqq.at(old_cls_cur_sfl), *it_stat_cur_sfl);//�������Ŵ�����cur_item������word ��ͳ��ֵɾ��
	is_computed_sfl.clear();
	is_computed_sfl.assign(trainss.size(), false);
	compute_marg_liks_sfl();//�����Ŵ�uni_cls_cands_cur����ȻֵȻ��洢��pred_links.at(c)��,(��Ϊ����һЩ֮����иı�)
	if (parent)
	{
		log_self_link_lik_sfl = parent->compute_log_self_link_lik_sfl();
	}
	else
	{
		base.reset_class(qq_temp);
		log_self_link_lik_sfl = base.marg_likelihood(qq_temp, *it_stat_cur_sfl);
	}
	compute_log_probs_sampling_sfl();//����Ѱ�����Ĵ���max_log_prob
	sample_customer_sfl();
	if (new_customer_cur_sfl == cur_link)//����²����������Գ�һ��
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
		cur_link = *it;//������item �Ĳ�����link�Ĳ�������һ�������������ܶ�item�Ĳ���
		sample_for_single_sfl();
		//customers.at(*it) = _new_customer;
	}
	//delete_table
}

template<typename D> void Layer<D>::traverse_single_table(int idx_table)
{//�����ǲ����ı�ţ��Բ�����������������������洢��tree.at(idx_table)����ͳ����۲�ֵ������Ӧ��stats�У�
	vector<int>& tree_i = trees.at(idx_table);//
	vector<int>& order_i = orders.at(idx_table);
	vector<STAT>& stat_i = stats.at(idx_table);
	vector<int> to_visit, to_visit_father, father_i;
	int cnt = 0;
	tree_i.push_back(idx_table);//��ʼԪ������������ı�ŵ�
	father_i.push_back(-1);//���ڵ㴦���Ϊ-1 Ӧ����û�и��ڵ����˼
	stat_i.push_back(STAT());
	inds_start.at(idx_table) = cnt++;//Ϊ���ڵ㴦��״̬��1
	vector<int>& links_t = links.at(idx_table);//��һ��������Ŵ���linkֵ
	vector<int>::iterator p = links_t.begin();
	if (p != links_t.end())//���ﲢ��ѭ��ֻ���ж��Ƿ�Ϊ��
	{
		to_visit.push_back(*p);//to_visit װ�������һ��Ҫ���ʵ�λ��
		to_visit_father.push_back(idx_table);//װ����Ǳ������ڵĽڵ�
		p++;
	}
	else
	{//������������linkֵ�ǿյ� 
		inds_end.at(idx_table) = 0;
		child ?//child Ϊtrue����expr1��Ϊfalse����expr2
			(stat_i.front() = child->stats.at(idx_table).front()) : //������ӽڵ�
			(stat_i.front().init(trainss.at(idx_table), base.get_eta()));//�������ײ�,��ʼ��idx_table��һ���uni_ss��uni_qq;��ֵ

		order_i.push_back(0);
		return;
	}

	for (; p != links_t.end(); p++)
	{
		to_visit.push_back(*p);//ָ����һ��ֵ
		to_visit_father.push_back(-1);//���ڵ��Ѿ�������ˣ����ﶼ��ǳ�-1
	}
	int curr;
	while (!to_visit.empty())//����������������֮��to_visit���ǿյģ��򣿣�
	{
		curr = to_visit.back();//�����һ����ʼ
		inds_start.at(curr) = cnt++;//���cnt�Ǹ������������
		tree_i.push_back(curr);
		stat_i.push_back(STAT());
		father_i.push_back(to_visit_father.back());
		to_visit.pop_back();
		to_visit_father.pop_back();//��tovisit��tovisitfater��ֵ��tree_i ,father_i֮���ͷ�
		vector<int> & links_curr = links.at(curr);
		if (!links_curr.empty())
		{//�����curr���link���ǿյ�
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
		{//����õ��links�ǿյ�
			inds_end.at(curr) = inds_start.at(curr);

			child ?
				(stat_i.at(inds_start.at(curr)).init(child->stats.at(curr).front(), base.get_eta())) :
				stat_i.at(inds_start.at(curr)).init(trainss.at(curr), base.get_eta());
			order_i.push_back(inds_start.at(curr));//���������

			int f = father_i.at(inds_start.at(curr));
			while (f >= 0)//�д���0��ʾ���ӽڵ��Ѿ�������
			{
				inds_end.at(f) = inds_end.at(curr);//��ô�ýڵ����������ӽڵ��Ӧ�õ����λ�ý�����
				child ?
					(stat_i.at(inds_start.at(f)).init(child->stats.at(f).front(), base.get_eta())) :
					stat_i.at(inds_start.at(f)).init(trainss.at(f), base.get_eta());
				order_i.push_back(inds_start.at(f));
				for (p = links.at(f).begin(); p != links.at(f).end(); p++)
				{
					stat_i.at(inds_start.at(f)).update(stat_i.at(inds_start.at(*p)));//�ı���stat�е�inds��ֵ
					stat_i.at(inds_start.at(*p)).clear_inds();

				}
				f = father_i.at(inds_start.at(f));//�������ϻ���
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
	{///��1.2��ʹ��
		for (auto it = child->uni_tables_vec.begin(); it != child->uni_tables_vec.end(); it++)//�õ���child�����ݸ��±�������
		{
			int cst = customers.at(*it);//���customer.at(*it)�Ǳ�����*itλ�õĹ˿���˭����
			if (cst != *it)//�������������
			{
				links.at(cst).push_back(*it);//�Ѹù˿ͣ�*it������˿�����ָ��Ľڵ㣨cst����link�б���
			}
		}
	}
	else
	{//��0��ʹ��
		int i = 0;
		for (auto it = customers.begin(); it != customers.end(); it++, i++)
		{

			if (*it != i)
			{
				links.at(*it).push_back(i);//*it�ǹ˿�i��ָ���ֵ���������й˿�ָ��֮��͵õ��ˣ�����ָ��˿�*it�Ĺ˿�i��ɵ�����
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
{//�����Լ�����ǰֵ��qq��Ϊ��Ȼ��Ȼ����в����õ��µĲ���ֵ
	qq.at(w)--;//��Ӧ��ͳ��������һ��
	qq.back()--;//��Ӧ�ĸ�����һ��
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
	{//������ǵײ�
		for (auto it = trees.at(t).begin(); it != trees.at(t).end(); it++)
		{//�Ըò���t��Ӧ�����е�ÿһ�˿�,Ҳ����һ��Ĳ�������sample�������c���������������
			child->sample_source_sink_c(*it, c);
		}
	}
	else
	{//׷�ݵ��ײ��ÿ������t,������c
		QQ& qq = *pos_classqq.at(c);
		for (auto it = trees.at(t).begin(); it != trees.at(t).end(); it++)
		{//���е�ÿһ���ڵ�ss,������ǹ۲�ֵ��Ҫ�������²���
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
		//��ÿһ�����Ĳ������в�����c����������
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
	{//���ڵײ��ÿ���͹ݣ�Ҳ����ÿһ���켣
		vector<int> topics;
		vector<int> topic_stat;
		vector<int> pos(trainss.size(), -1);
		vector<int> &inds_items_i = bottom->inds_items.at(i);
		for (int j = 0; j != inds_items_i.size(); j++)
		{//����ÿ���ڵ㣬���켣Ƭ��
			int c = bottom->get_cluster(inds_items_i.at(j));//����ǩ
			if (pos.at(c) < 0)
			{//����������ǩ�ǵ�һ�γ��֣��ڸ�λ��д���µ�������
				pos.at(c) = topics.size();
				topics.push_back(c);
				topic_stat.push_back(1);
			}
			else
			{
				topic_stat.at(pos.at(c))++;//��¼����Ľڵ���Ŀ
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
		labels.at(i) = topics.at(idx);//�Խڵ���Ŀ����������Ϊ�ò͹ݵ�����
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