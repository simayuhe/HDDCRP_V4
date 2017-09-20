% function function [stateSeq INDS stateCounts] = sample_zs(data_struct,dist_struct,theta,obsModelType)
% Sample the mode and sub-mode sequence given the observations, transition
% distributions, and emission parameters. If SLDS model, the "observations"
% are the sampled state sequence.
% Modified by Guodong Tian

function [stateSeq, INDS, stateCounts] = sample_zs(cluster,data_struct,dist_struct,theta,stateCounts,stateSeq,INDS,obsModelType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define and initialize parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define parameters:
pi_z = dist_struct.pi_z;  % transition distributions with pi_z(i,j) the probability of going from i->j
pi_s = dist_struct.pi_s;  % mixture weights with pi_s(i,j) the probability of s_t=j when z_t=i
pi_init = dist_struct.pi_init;  % initial distribution on z_1
pi_c = dist_struct.pi_c;

Kc = size(pi_c,2);
Kz = size(pi_z,2);  % truncation level for transition distributions
Ks = size(pi_s,2);  % truncation level for MoG emissions

% Initialize state count matrices:
N = zeros(Kz+1,Kz,Kc);
Ns = zeros(Kz,Ks);

if ~isfield(data_struct(1),'test_cases')
    data_struct(1).test_cases = 1:length(data_struct);
end

% Preallocate INDS
% INDS = struct('obsIndzs',cell(1,length(data_struct)));
% stateSeq = struct('z',cell(1,length(data_struct)),'s',cell(1,length(data_struct)));
% for ii = 1:length(data_struct)
%   T = length(data_struct(ii).blockSize);
%   INDS(ii).obsIndzs(1:Kz,1:Ks) = struct('inds',sparse(1,data_struct(ii).blockEnd(end)),'tot',0);
%   % Initialize state sequence structure:
%   stateSeq(ii) = struct('z',zeros(1,T),'s',zeros(1,data_struct(ii).blockEnd(end)));
% end

% cluster = zeros(1,length(data_struct));
for ii=data_struct(1).test_cases  % those sequences for which z_{1:T} is unknown
%     clearvars -except dist_struct theta obsModelType data_struct ii pi_z pi_s pi_init pi_c Kc Kz Ks L N Ns INDS stateSeq cluster
    T = length(data_struct(ii).blockSize);
    blockSize = data_struct(ii).blockSize;
    blockEnd = data_struct(ii).blockEnd;

    % Initialize state and sub-state sequences:
    z = zeros(1,T);
    s = zeros(1,blockEnd(end));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute likelihoods and messages %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute likelihood(kz,ks,u_i) of each observation u_i under each
    % parameter theta(kz,ks):
    [likelihood, ~] = compute_likelihood(data_struct(ii),theta,obsModelType,Kz,Ks);
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample the state and sub-state sequences %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Sample (z(1),{s(1,1)...s(1,N1)}).  We first sample z(1) given the
    % observations u(1,1)...u(1,N1) having marginalized over the associated s's
    % and then sample s(1,1)...s(1,N1) given z(1) and the observations.

%     totSeq = zeros(Kz,Ks);
%     indSeq = zeros(data_struct(ii).blockEnd(end),Kz,Ks);
    % Compute backwards messages:
%        bwds_msg = zeros(Kz,T);
%        partial_marg = zeros(Kz,T);
    c = cluster(ii);
    [bwds_msg, partial_marg] = backwards_message_vec(likelihood, blockEnd, pi_z(:,:,c), pi_s);
    for t=1:T
        % Sample z(t):
        if (t == 1)
            Pz = pi_init(:,c) .* partial_marg(:,1);
            obsInd = 1:blockEnd(1);
        else
            Pz = pi_z(z(t-1),:,c)' .* partial_marg(:,t);
            obsInd = blockEnd(t-1)+1:blockEnd(t);
        end
        Pz   = cumsum(Pz);
        z(t) = 1 + sum(Pz(end)*rand(1) > Pz);

        % Add state to counts matrix:
        if (t > 1)
            N(z(t-1),z(t),c) = N(z(t-1),z(t),c) + 1;
        else
            N(Kz+1,z(t),c) = N(Kz+1,z(t),c) + 1;  % Store initial point in "root" restaurant Kz+1
        end

        % Sample s(t,1)...s(t,Nt) and store sufficient stats:
        for k=1:blockSize(t)
            % Sample s(t,k):
            if Ks > 1
                Ps = pi_s(z(t),:) .* likelihood(z(t),:,obsInd(k));
                Ps = cumsum(Ps);
                s(obsInd(k)) = 1 + sum(Ps(end)*rand(1) > Ps);
            else
                s(obsInd(k)) = 1;
            end

            % Add s(t,k) to count matrix and observation statistics:
            Ns(z(t),s(obsInd(k))) = Ns(z(t),s(obsInd(k))) + 1;
%             totSeq(z(t),s(obsInd(k))) = totSeq(z(t),s(obsInd(k))) + 1;
%             indSeq(totSeq(z(t),s(obsInd(k))),z(t),s(obsInd(k))) = obsInsd(k);
            INDS(ii).obsIndzs(z(t),s(obsInd(k))).tot  = INDS(ii).obsIndzs(z(t),s(obsInd(k))).tot+1;
            INDS(ii).obsIndzs(z(t),s(obsInd(k))).inds(INDS(ii).obsIndzs(z(t),s(obsInd(k))).tot)  = obsInd(k);

        end
    end

    stateSeq(ii).z = z;
    stateSeq(ii).s = s;

end

for ii=setdiff(1:length(data_struct),data_struct(1).test_cases) % for sequences ii with fixed z_{1:T}
    T = length(data_struct(ii).blockSize);
    blockSize = data_struct(ii).blockSize;
    blockEnd = data_struct(ii).blockEnd;
    %INDS(ii).obsIndzs(1:Kz,1:Ks) = struct('inds',sparse(1,T),'tot',0);

    % Initialize state and sub-state sequences:
    z = data_struct(ii).true_labels;
    s = ones(1,sum(blockSize));

    % Add s(1,1)...s(1,N1) counts and store sufficient stats:
    for i=1:blockSize(1)
        % Add s(t,i) to counts matrix:
        Ns(z(1),s(i)) = Ns(z(1),s(i)) + 1;
        INDS(ii).obsIndzs(z(1),s(i)).tot = INDS(ii).obsIndzs(z(1),s(i)).tot + 1;
        INDS(ii).obsIndzs(z(1),s(i)).inds(INDS(ii).obsIndzs(z(1),s(i)).tot) = i;
    end
    % Add z(1) count:
    N(Kz+1,z(1)) = N(Kz+1,z(1)) + 1;

    % Sample (z(t),{s(t,1)...s(t,Nt)}).  We first sample z(t) given the
    % observations u(t,1)...u(t,Nt) having marginalized over the associated s's
    % and then sample s(t,1)...s(t,Nt) given z(t) and the observations.
    for t=2:T

        % Add state to counts matrix:
        N(z(t-1),z(t)) = N(z(t-1),z(t))+1;

        % Sample s(t,1)...s(t,Nt) and store sufficient stats:
        for i=1:blockSize(t)
            obsInd = blockEnd(t-1) + i;

            % Add s(t,i) to counts matrix:
            Ns(z(t),s(obsInd)) = Ns(z(t),s(obsInd)) + 1;

            INDS(ii).obsIndzs(z(t),s(obsInd)).tot = INDS(ii).obsIndzs(z(t),s(obsInd)).tot + 1;
            INDS(ii).obsIndzs(z(t),s(obsInd)).inds(INDS(ii).obsIndzs(z(t),s(obsInd)).tot) = obsInd;
        end

    end
    
    stateSeq(ii).z = z;
    stateSeq(ii).s = s;

end

binNs = zeros(size(Ns));
binNs(Ns>0) = 1;
uniqueS = sum(binNs,2);

stateCounts.uniqueS = uniqueS;
stateCounts.N = N;
stateCounts.Ns = Ns;
return;
