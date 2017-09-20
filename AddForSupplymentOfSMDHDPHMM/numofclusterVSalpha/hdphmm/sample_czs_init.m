function [cluster, stateSeq, INDS, stateCounts] = sample_czs_init(data_struct,dist_struct,obsModelType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define and initialize parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define parameters:
pi_z = dist_struct.pi_z;
pi_s = dist_struct.pi_s;
pi_init = dist_struct.pi_init;
pi_c = dist_struct.pi_c;

Kc = size(pi_c,2);
Kz = size(pi_z,2);
Ks = size(pi_s,2);

% Initialize state count matrices:
L = zeros(1,Kc);
N = zeros(Kz+1,Kz,Kc);
Ns = zeros(Kz,Ks);

if ~isfield(data_struct(1),'test_cases')
    data_struct(1).test_cases = [1:length(data_struct)];
end

% Preallocate INDS
INDS = struct('obsIndzs',cell(1,length(data_struct)));
stateSeq = struct('z',cell(1,length(data_struct)),'s',cell(1,length(data_struct)));
for ii = 1:length(data_struct)
  T = length(data_struct(ii).blockSize);
  INDS(ii).obsIndzs(1:Kz,1:Ks) = struct('inds',sparse(1,data_struct(ii).blockEnd(end)),'tot',0);
  % Initialize state sequence structure:
  stateSeq(ii) = struct('z',zeros(1,T),'s',zeros(1,data_struct(ii).blockEnd(end)));
end

cluster = zeros(1,length(data_struct));
for ii=data_struct(1).test_cases
    if isfield(data_struct(ii),'z_init')
        [cluster(ii) stateSeq(ii).z stateSeq(ii).s INDS(ii) L N Ns] = setZtoFixedSeq(data_struct(ii),dist_struct,INDS(ii),L,N,Ns,data_struct(ii).c_init,data_struct(ii).z_init,0);
    else
        [cluster(ii) stateSeq(ii).z stateSeq(ii).s INDS(ii) L N Ns] = sampleZfromPrior(data_struct(ii),dist_struct,INDS(ii),L,N,Ns);
    end   
end

binNs = zeros(size(Ns));
binNs(find(Ns)) = 1;
uniqueS = sum(binNs,2);
uniqueL = sum(L>0);

stateCounts.uniqueS = uniqueS;
stateCounts.N = N;
stateCounts.Ns = Ns;
stateCounts.L = L;
stateCounts.uniqueL = uniqueL;
return;

function [c z s INDS L N Ns] = sampleZfromPrior(data_struct,dist_struct,INDS,L,N,Ns)

% Define parameters:
pi_c = dist_struct.pi_c;
pi_z = dist_struct.pi_z;
pi_s = dist_struct.pi_s;
pi_init = dist_struct.pi_init;

Kc = size(pi_c,2);
Kz = size(pi_z,2);
Ks = size(pi_s,2);

T = length(data_struct.blockSize);
blockSize = data_struct.blockSize;
blockEnd = data_struct.blockEnd;

% Initialize state and sub-state sequences:
z = zeros(1,T);
s = zeros(1,blockEnd(end));

Pc = pi_c;
Pc = cumsum(Pc);
c = 1 + sum(rand*Pc(end)>Pc);
L(c) = L(c) + 1;

for t=1:T
    % Sample z(t):
    if (t == 1)
        Pz = pi_init(:,c)';
        obsInd = 1:blockEnd(1);
    else
        Pz = pi_z(z(t-1),:,c)';
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
            Ps = pi_s(z(t),:);
            Ps = cumsum(Ps);
            s(obsInd(k)) = 1 + sum(Ps(end)*rand(1) > Ps);
        else
            s(obsInd(k)) = 1;
        end
        
        % Add s(t,k) to count matrix and observation statistics:
        Ns(z(t),s(obsInd(k))) = Ns(z(t),s(obsInd(k))) + 1;
        INDS.obsIndzs(z(t),s(obsInd(k))).tot  = INDS.obsIndzs(z(t),s(obsInd(k))).tot+1;
        INDS.obsIndzs(z(t),s(obsInd(k))).inds(INDS.obsIndzs(z(t),s(obsInd(k))).tot)  = obsInd(k);
    end
end

return;

function [c z s INDS L N Ns] = setZtoFixedSeq(data_struct,dist_struct,INDS,L,N,Ns,c_fixed,z_fixed,sampleS)

% Define parameters:
pi_z = dist_struct.pi_z;
pi_s = dist_struct.pi_s;
pi_init = dist_struct.pi_init;
pi_c = dist_struct.pi_c;

Kc = size(pi_c,2);
Kz = size(pi_z,2);
Ks = size(pi_s,2);

T = length(data_struct.blockSize);
blockSize = data_struct.blockSize;
blockEnd = data_struct.blockEnd;

% Initialize state and sub-state sequences:
c = c_fixed;
L(c) = L(c) + 1;
z = z_fixed;
if sampleS
    for t=1:T
        % Sample z(t):
        if (t == 1)
            obsInd = [1:blockEnd(1)];
        else
            obsInd = [blockEnd(t-1)+1:blockEnd(t)];
        end

        % Sample s(t,1)...s(t,Nt) and store sufficient stats:
        for k=1:blockSize(t)
            % Sample s(t,k):
            if Ks > 1
                Ps = pi_s(z(t),:);
                Ps = cumsum(Ps);
                s(obsInd(k)) = 1 + sum(Ps(end)*rand(1) > Ps);
            else
                s(obsInd(k)) = 1;
            end
        end
    end
else
    s = ones(1,sum(blockSize));
end


for t=1:T
    % Sample z(t):
    if (t == 1)
        obsInd = [1:blockEnd(1)];
    else
        obsInd = [blockEnd(t-1)+1:blockEnd(t)];
    end

    % Add state to counts matrix:
    if (t > 1)
        N(z(t-1),z(t),c) = N(z(t-1),z(t),c) + 1;
    else
        N(Kz+1,z(t),c) = N(Kz+1,z(t),c) + 1;  % Store initial point in "root" restaurant Kz+1
    end

    % Sample s(t,1)...s(t,Nt) and store sufficient stats:
    for k=1:blockSize(t)

        % Add s(t,k) to count matrix and observation statistics:
        Ns(z(t),s(obsInd(k))) = Ns(z(t),s(obsInd(k))) + 1;
        INDS.obsIndzs(z(t),s(obsInd(k))).tot  = INDS.obsIndzs(z(t),s(obsInd(k))).tot+1;
        INDS.obsIndzs(z(t),s(obsInd(k))).inds(INDS.obsIndzs(z(t),s(obsInd(k))).tot)  = obsInd(k);
    end
end

return;