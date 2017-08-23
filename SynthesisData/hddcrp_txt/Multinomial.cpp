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
	inds.at(ss) = 0;//ss代表了词典当中的一个位置，这里的inds把其他的地方都赋值为-1，只有这个位置为0
	uni_ss.assign(1, ss);//为uni_ss赋值ss,并在统计量uni_qq上计数1
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
	//cout<< inds.at(ss)<<"|";//刚开始的时候每个stat中的inds是一个长为1000 的vector只有在本stat的第一个ss（如456）处为0
	if (inds.at(ss) < 0)//每次有新的类别产生的时候ind.at(ss)会是-1
	{
		inds.at(ss) = uni_ss.size();
		uni_ss.push_back(ss);
		uni_qq.push_back(cnt);
	}
	else
	{
		uni_qq.at(inds.at(ss)) += cnt;//inds 指示了该单词在uni_ss中的位置，也即在uni_qq中的位置
	}
	//cout << inds.at(ss)<<":"<<uni_ss.back()<<":"<<uni_qq.at(inds.at(ss))<<" ";//更新万事之后inds表示在stat中的第几类；
	//qq表示这一类中的第几个，ss表示具体的值，这样就把相同的单词看成一个顾客了
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
	//未被引用
	eta = _eta;
	eta.push_back(0.0);
	for (vector<double>::const_iterator it = _eta.begin(); it != _eta.end(); it++) eta.back() += *it;
}

void Multinomial::initialize_eta(vector<double>& _eta, vector<double>& eta)
{
	eta = _eta;
	eta.push_back(0.0);//用来统计所有的值的和
	for (vector<double>::const_iterator it = _eta.begin(); it != _eta.end(); it++) eta.back() += *it;//这里只有eta 的最后一个值是与_eta不同的
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
	////用于显示程序细节
	//cout << "when computing the marg_likelihood ：" << lik << endl;
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
	qq.at(ss)++;//ss是words的内容，也就是trainss中单词在词典中的位置，这里通过加来统计其数量
	qq.back()++;//最后一个数用来统计总共的单词数目
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
	vector<int>::const_iterator it_s = stat.uni_ss.begin();//stat .uni_ss 对应的word 在词典中减去相应的数量
	vector<int>::const_iterator it_q = stat.uni_qq.begin();//stat 中存储了单词（uni_ss）以及单词的个数（uni_qq）,tot是总数
	for (; it_s != stat.uni_ss.end(); it_s++, it_q++)
	{
		qq.at(*it_s) -= (*it_q);
	}
	qq.back() -= tot;
}


list< vector<int> >::iterator Multinomial::add_class(){
	classqq.push_back(vector<int>(eta.size(), 0));
	cout << "classqq.back().size() : "<<classqq.back().size() << endl;
	return --classqq.end();//指向classqq末尾的一个指针
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
	//来源于D M BLEI 的代码中先将不同取值的后验算好制造成表格以备查取
	//log_pos_vals[0][n]表示 sum(log(j + eta0)) 其中j=1,...,n ,也即 log[ gamma(n + eta0)/gamma( eta0 ) ]
	//log_pos_vals[1][n]表示 sum(log(j + eta)) 其中j=1,...,n ,也即 log[ gamma(n + eta)/gamma( eta ) ]
	//目的是先把似然中的两项能包含的所有情况（0 ~ tot_num）分别算出来做成表格以备查询


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
