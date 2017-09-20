%求取holdout data 的likelihood
%源于D:\code\MotionPatternLearning2_win\compute_likelihood_CASIA.m
%%
%加载模型和数据
close all;
clear all;
clc;
%加载数据，要与训练集采用同一字典编码，包括字典长度，字典内容
load('holdout_data.mat');
codes=holdout_data;
%对于每一条轨迹（也即每一篇文档），构造与训练数据相同的数据结构
J = length(codes);
data_struct = struct('obs',cell(1,J),'blockSize',cell(1,J));
for jj = 1:J
    data_struct(jj).obs = codes{jj};
    data_struct(jj).blockSize = ones(1,length(data_struct(jj).obs));
    data_struct(jj).blockEnd = cumsum(data_struct(jj).blockSize);
end

myiter=[0.001 0.01 0.1 1 10 100 1000];

for myi=1:1:length(myiter)
    for myj=1:3
        for n=2500:100:2500
            start_exp = myiter(myi)+myj;
            exp_num=start_exp;
            result_dir = './results/CASIA/';
            save_dir = [result_dir 'HMMdata/exp' num2str(exp_num) '/'];
            filename = strcat(save_dir,'/HDPHMMDPstats','iter',num2str(n),'trial1.mat');
            load(filename);
            theta=S.theta(1);
            pi_init = S.dist_struct.pi_init;
            pi_z = S.dist_struct.pi_z;
            pi_s = S.dist_struct.pi_s;
            Kc = size(pi_z,3);
            Kz = size(pi_z,1);
            Ks = size(pi_s,2);
            clear S
            obsModelType = 'Multinomial';
            liks_out = zeros(J, Kc);
            %max_liks_out = zeros(1,J);
            %is_out_vec = zeros(1,J);
            for jj = 1:J
                blockEnd = data_struct(jj).blockEnd;
                [likelihood log_normalizer] = compute_likelihood(data_struct(jj),theta,obsModelType,Kz,Ks);
                for cc = 1:Kc
                    [~, neglog_c_t] = forward_message_vec(likelihood,log_normalizer,blockEnd,pi_z(:,:,cc),pi_s,pi_init(:,cc)');
                    liks_out(jj,cc) = sum(neglog_c_t);
                end
                %     [max_liks_out(jj), max_c] = max(liks_out(jj,:));
                %     if max_liks_out(jj) < thr_liks(max_c)
                %         is_out_vec(jj) = 1;
                %     end
            end           
        end
        likelihood_holdoutdata(myi,myj)=sum(sum(liks_out));
        clear liks_out;
    end
end
%%
ave_liks=mean(likelihood_holdoutdata');
A=mean(ave_liks);
var_liks=std(likelihood_holdoutdata');
log_myiter=log10(myiter);
errorbar(log_myiter,ave_liks,var_liks,'rs');
axis([-5 5 -3.5e5 -2.6e5 ]);
hold on
plot([-5 5],[A A],'--')
hold off

