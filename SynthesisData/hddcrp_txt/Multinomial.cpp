#include "StdAfx.h"
#include "Multinomial.h"
#include <math.h>

Stat_Mult& Stat_Mult::init(int ss)
{
	uni_ss.push_back(ss);
	uni_qq.push_back(1);
	tot = 1;
	return *this;
}

Stat_Mult& Stat_Mult::init(int ss, vector<double>& _eta)
{
	
	int voc_size = _eta.size() - 1;
	//cout << "*" << voc_size << "*";
	inds.assign(voc_size, -1);
	inds.at(ss) = 0;//ss�����˴ʵ䵱�е�һ��λ�ã������inds�������ĵط�����ֵΪ-1��ֻ�����λ��Ϊ0
	uni_ss.assign(1, ss);//Ϊuni_ss��ֵss,����ͳ����uni_qq�ϼ���1
	uni_qq.assign(1, 1);
	tot = 1;
	return *this;
}

Stat_Mult& Stat_Mult::update(int ss)
{
	if (inds.at(ss) < 0)
	{
		inds.at(ss) = uni_ss.size();
		uni_ss.push_back(ss);
		uni_qq.push_back(1);
	}
	else
	{
		uni_qq.at(inds.at(ss))++;
	}
	tot++;
	return *this;
}

void Stat_Mult::part_update(int ss, int cnt)
{
	//cout<< inds.at(ss)<<"|";//�տ�ʼ��ʱ��ÿ��stat�е�inds��һ����Ϊ1000 ��vectorֻ���ڱ�stat�ĵ�һ��ss����456����Ϊ0
	if (inds.at(ss) < 0)//ÿ�����µ���������ʱ��ind.at(ss)����-1
	{
		inds.at(ss) = uni_ss.size();
		uni_ss.push_back(ss);
		uni_qq.push_back(cnt);
	}
	else
	{
		uni_qq.at(inds.at(ss)) += cnt;//inds ָʾ�˸õ�����uni_ss�е�λ�ã�Ҳ����uni_qq�е�λ��
	}
	//cout << inds.at(ss)<<":"<<uni_ss.back()<<":"<<uni_qq.at(inds.at(ss))<<" ";//��������֮��inds��ʾ��stat�еĵڼ��ࣻ
	//qq��ʾ��һ���еĵڼ�����ss��ʾ�����ֵ�������Ͱ���ͬ�ĵ��ʿ���һ���˿���
}

Stat_Mult& Stat_Mult::init(Stat_Mult& stat, vector<double>& _eta)
{
	int voc_size = _eta.size() - 1;
	inds.assign(voc_size, -1);
	uni_ss = stat.uni_ss;
	uni_qq = stat.uni_qq;
	tot = stat.tot;
	int cnt = 0;
	for_each(uni_ss.begin(), uni_ss.end(), [&](int ss){inds.at(ss) = cnt++;});
	return *this;
}

void Stat_Mult::part_update(Stat_Mult& stat)
{
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		part_update(*it_s, *it_q);
	}
}

Stat_Mult& Stat_Mult::update(Stat_Mult& stat)
{
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		part_update(*it_s, *it_q);
	}
	tot += stat.tot;
	return *this;
}

//Stat_Mult& Stat_Mult::operator=(const Stat_Mult& stat)
//{
//	uni_ss = stat.uni_ss;
//	uni_qq = stat.uni_qq;
//	tot = stat.tot;
//	inds = stat.inds;
//	return *this;
//}

void Stat_Mult::clear_inds()
{
	inds.clear();
	vector<int>().swap(inds);
}

bool Stat_Mult::check_stat(int _tot)
{
	int sum = 0;
	for (int i = 0; i != uni_qq.size(); i++)
	{
		sum += uni_qq.at(i);
	}
	if (_tot != sum)
	{
		cout << "error" << endl;
		return false;
	}
	else
	{
		//cout << "OK" << endl;
		return true;
	}
}







Multinomial::Multinomial(vector<double>& _eta): eta(_eta) 
{
	eta.push_back(0.0);
	for (vector<double>::const_iterator it = _eta.begin(); it != _eta.end(); it++) eta.back() += *it;
}

void Multinomial::initialize_eta(vector<double>& _eta)
{
	//δ������
	eta = _eta;
	eta.push_back(0.0);
	for (vector<double>::const_iterator it = _eta.begin(); it != _eta.end(); it++) eta.back() += *it;
}

void Multinomial::initialize_eta(vector<double>& _eta, vector<double>& eta)
{
	eta = _eta;
	eta.push_back(0.0);//����ͳ�����е�ֵ�ĺ�
	for (vector<double>::const_iterator it = _eta.begin(); it != _eta.end(); it++) eta.back() += *it;//����ֻ��eta �����һ��ֵ����_eta��ͬ��
}

double Multinomial::marg_likelihood(vector<int>& qq, int ss)
{
	//return log( (eta.at(ss)+qq.at(ss))/(eta.back()+qq.back()) );
	//return log_pos_vals.at(0).at(qq.at(ss)) - log_pos_vals.at(1).at(qq.back());
	return log_pos_vals[0][qq[ss]] - log_pos_vals[1][qq.back()];
}

//double Multinomial::marg_likelihood(vector<int>& qq, Stat_Mult& stat)
//{
//	double lik = 0.0;
//	vector<double> temp(stat.uni_ss.size(), 0.0);
//	transform(stat.uni_ss.begin(), stat.uni_ss.end(), stat.uni_qq.begin(), temp.begin(), 
//			[&](int idx, int cnt){return log_pos_vals[0][qq[idx]+cnt] - log_pos_vals[0][qq[idx]];});
//	lik = accumulate(temp.begin(), temp.end(), 0.0);
//	lik -= log_pos_vals[1][qq.back()+stat.tot];
//	lik += log_pos_vals[1][qq.back()];
//	return lik;
//}

double Multinomial::marg_likelihood(vector<int>& qq, Stat_Mult& stat)
{
	double lik = 0.0;
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		
		lik += log_pos_vals[0][qq[*it_s]+(*it_q)];
		lik -= log_pos_vals[0][qq[*it_s]];
	}
	
	lik -= log_pos_vals[1][qq.back()+stat.tot];
	lik += log_pos_vals[1][qq.back()];
	
	return lik;
}

//double Multinomial::marg_likelihood(vector<int>& qq, Stat_Mult& stat, vector< vector<double> >& log_pos_vals, int tot)
//{
//	double lik = 0.0;
//	vector<double> temp(stat.uni_ss.size(), 0.0);
//	transform(stat.uni_ss.begin(), stat.uni_ss.end(), stat.uni_qq.begin(), temp.begin(), 
//			[&](int idx, int cnt){return log_pos_vals[0][qq[idx]+cnt] - log_pos_vals[0][qq[idx]];});
//	lik = accumulate(temp.begin(), temp.end(), 0.0);
//	lik -= log_pos_vals[1][qq.back()+tot];
//	lik += log_pos_vals[1][qq.back()];
//	return lik;
//}

double Multinomial::marg_likelihood(vector<int>& qq, Stat_Mult& stat, vector< vector<double> >& log_pos_vals, int tot)
{
	double lik = 0.0;
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	////������ʾ����ϸ��
	//cout << "when computing the marg_likelihood ��" << lik << endl;
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		/**/
		/*cout << "stat.uni_ss is" << *it_s << "|";
		cout << "stat.uni_qq" << *it_q << "|";*/
		/**/
		/*cout << endl; cout << qq[*it_s] << endl;*/
		lik += log_pos_vals[0][qq[*it_s]+(*it_q)];
		lik -= log_pos_vals[0][qq[*it_s]];
		/*cout << "we finish the log computation" << endl;*/
	}
	/*cout << "stat.back is" << qq.back() << "|";
	cout << "stat.tot" << stat.tot << "|";*/
	lik -= log_pos_vals[1][qq.back()+tot];
	lik += log_pos_vals[1][qq.back()];
	//cout << "after calculate ,the lik is " << lik << endl;
	///*int wwww;
	//cin >> wwww;*/
	return lik;
}

void Multinomial::marg_likelihoods(vector<double>& clik,  int ss) 
{
	double etas, eta0;
	eta0 = eta.back();
	etas = eta.at(ss);
	vector<double>::iterator it_clik = clik.begin();
	for ( list< vector<int> >::const_iterator it = classqq.begin(); it != classqq.end(); it++, it_clik++ ) 
		*it_clik = (etas + it->at(ss)) / (eta0 + it->back());
}


void Multinomial::add_data(vector<int>& qq, int ss) {
	//cout << "here" << endl;
	qq.at(ss)++;//ss��words�����ݣ�Ҳ����trainss�е����ڴʵ��е�λ�ã�����ͨ������ͳ��������
	qq.back()++;//���һ��������ͳ���ܹ��ĵ�����Ŀ
	//cout << qq.back()<<endl;
	
}


void Multinomial::add_data(vector<int>& qq, Stat_Mult& stat) {
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		qq.at(*it_s) += (*it_q);
	}
	qq.back() += stat.tot;
}

void Multinomial::add_data(vector<int>& qq, Stat_Mult& stat, int tot) {
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		qq.at(*it_s) += (*it_q);
	}
	qq.back() += tot;
}


double Multinomial::add_data_lik(vector<int>& qq, int ss) {
	//return log( (eta.at(ss) + (qq.at(ss)++)) / (eta.back() + (qq.back()++)) );
	//return log_pos_vals.at(0).at(qq.at(ss)++) - log_pos_vals.at(1).at(qq.back()++);
	return log_pos_vals[0][qq[ss]++] - log_pos_vals[1][qq.back()++];
}


void Multinomial::del_data(vector<int>& qq, int ss) {
	qq.at(ss)--;
	qq.back()--;
}


void Multinomial::del_data(vector<int>& qq, Stat_Mult& stat) {
	vector<int>::const_iterator it_s = stat.uni_ss.begin();
	vector<int>::const_iterator it_q = stat.uni_qq.begin();
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		qq.at(*it_s) -= (*it_q);
	}
	qq.back() -= stat.tot;
}

void Multinomial::del_data(vector<int>& qq, Stat_Mult& stat, int tot) {
	vector<int>::const_iterator it_s = stat.uni_ss.begin();//stat .uni_ss ��Ӧ��word �ڴʵ��м�ȥ��Ӧ������
	vector<int>::const_iterator it_q = stat.uni_qq.begin();//stat �д洢�˵��ʣ�uni_ss���Լ����ʵĸ�����uni_qq��,tot������
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		qq.at(*it_s) -= (*it_q);
	}
	qq.back() -= tot;
}


list< vector<int> >::iterator Multinomial::add_class(){
	classqq.push_back(vector<int>(eta.size(), 0));
	cout << "classqq.back().size() : "<<classqq.back().size() << endl;
	return --classqq.end();//ָ��classqqĩβ��һ��ָ��
}

void Multinomial::del_class(list< vector<int> >::iterator it){
	classqq.erase(it);
}

void Multinomial::output(string& save_dir, int iter)
{
	if (save_dir.back() != '/')
	{
		save_dir += '/';
	}
	ostringstream file_name_stream;
	file_name_stream << save_dir << "classqq" << iter;
	string topic_file_name = file_name_stream.str() + ".txt";
	ofstream output_stream(topic_file_name.c_str());
	cout << "saving topic data at iteration " << iter << " ..." << endl;
	list< vector<int> >::iterator it_q;
	for (it_q = classqq.begin(); it_q != classqq.end(); it_q++)
	{
		vector<int>::iterator it;
		vector<int>::iterator end = it_q->end() - 1;
		for (it = it_q->begin(); it != end; it++)
		{
			output_stream << *it << " ";
		}
		output_stream << endl;
	}
	output_stream.close();
}

void Multinomial::reset_class(vector<int>& qq)
{
	qq.assign(eta.size(), 0);
}

void Multinomial::init_log_pos_vals(int tot_num_words)
{
	log_pos_vals.assign(2, vector<double>());
	log_pos_vals.at(0).reserve(tot_num_words+1);
	log_pos_vals.at(1).reserve(tot_num_words+1);
	log_pos_vals.at(0).push_back( 0.0 );
	log_pos_vals.at(1).push_back( 0.0 );

	for (int i = 0; i < tot_num_words; i++)
	{	
		
		log_pos_vals.at(0).push_back( log( eta.front() + (double)i ) + log_pos_vals.at(0).back() );
		log_pos_vals.at(1).push_back( log( eta.back() + (double)i )  + log_pos_vals.at(1).back() );
	}

}

void Multinomial::init_log_pos_vals(int tot_num_words, vector< vector<double> >& log_pos_vals, vector<double>& eta)
{
	log_pos_vals.assign(2, vector<double>());
	log_pos_vals.at(0).reserve(tot_num_words+1);
	log_pos_vals.at(1).reserve(tot_num_words+1);
	log_pos_vals.at(0).push_back( 0.0 );
	log_pos_vals.at(1).push_back( 0.0 );
	for (int i = 0; i < tot_num_words; i++)
	{
		//eta(0.001,..., 0.001, 1 )
		log_pos_vals.at(0).push_back( log( eta.front() + (double)i ) + log_pos_vals.at(0).back() );
		log_pos_vals.at(1).push_back( log( eta.back() + (double)i )  + log_pos_vals.at(1).back() );
	}
	//��Դ��D M BLEI �Ĵ������Ƚ���ͬȡֵ�ĺ����������ɱ���Ա���ȡ
	//log_pos_vals[0][n]��ʾ sum(log(j + eta0)) ����j=1,...,n ,Ҳ�� log[ gamma(n + eta0)/gamma( eta0 ) ]
	//log_pos_vals[1][n]��ʾ sum(log(j + eta)) ����j=1,...,n ,Ҳ�� log[ gamma(n + eta)/gamma( eta ) ]
	//Ŀ�����Ȱ���Ȼ�е������ܰ��������������0 ~ tot_num���ֱ���������ɱ���Ա���ѯ


}

bool Multinomial::check_qq(vector<int>& qq)
{
	int sum = 0;
	for (int i = 0; i != qq.size()-1; i++)
	{
		if (qq.at(i) < 0)
		{
			cout << "qq[" << i << "] = " << qq.at(i) << endl;
			return false;
		}
		else
		{
			sum += qq.at(i);
		}
	}
	if (sum != qq.back())
	{
		cout << "qq.back() != sum" << endl;
		return false;
	}
	else
	{
		//cout << "OK" << endl;
		return true;
	}
}
