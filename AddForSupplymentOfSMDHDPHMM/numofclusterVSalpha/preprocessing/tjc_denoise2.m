function x_den = tjc_denoise2(x)
% load(settings.file_original_tjcs,'TestTrajDatabase');

level = 8;
wname = 'sym4';
tptr  = 'sqtwolog';
sorh  = 's';
npc_app = 2;
npc_fin = 2;

% comp_rate = 10;
% smoothedTrajectories = cell(length(TestTrajDatabase),1);
% for ii = 1:length(TestTrajDatabase)
%     input(num2str(ii));
%     x = TestTrajDatabase{ii};
    [x_den, npc, nestco] = wmulden(x, level, wname, npc_app, npc_fin, tptr, sorh);
    
    % compensate the offset
%     x_center = mean(x,1);
%     x_den_center = mean(x_den,1);
%     x_offset = x_center - x_den_center;
%     x_den = x_den + x_offset(ones(1,size(x,1)),:);
        
%     smoothedTrajectories{ii} = x_den;
%     figure(1);
%     subplot(121);
%     plot(x(:,1),x(:,2));
%     subplot(122);
%     plot(x_den(:,1),x_den(:,2));
%     figure(1);
% end
% TestTrajDatabase = smoothedTrajectories;
% save(settings.file_smoothed_tjcs,'TestTrajDatabase');
% save('.\data\SyntheticTrajectoriesSmoothed.mat','TestTrajDatabase');