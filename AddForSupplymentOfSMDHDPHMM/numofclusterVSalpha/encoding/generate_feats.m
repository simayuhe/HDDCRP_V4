function features = generate_feats(save_dir,Y,n_dim,varargin)
if ~exist(save_dir,'file')
	mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
T = size(Y,2)/2;
w = 50;
Y = [Y w*(Y(:,2:T)-Y(:,1:T-1)) w*(Y(:,T+2:end)-Y(:,T+1:end-1))];
if isempty(varargin)
    [COEFF,SCORE] = princomp(Y);
    features = SCORE(:,1:n_dim);
else
    COEFF = varargin{1};
    M = varargin{2};
    features = (Y-repmat(M,size(Y,1),1))*COEFF;
end
save([save_dir 'features.mat'],'features');
