function [datass, thetas] = genbars2(imsize,noiselevel,numbarpermix,numgroup,numdata,numcluster)
numdim   = imsize^2;
numbars  = imsize*2;
mixtheta = zeros(numdim,numbars);
figure;
colormap(gray);
for ii=1:imsize
  im = ones(imsize,imsize)*noiselevel/imsize^2;
  im(:,ii) = im(:,ii) + 1/imsize;
  mixtheta(:,ii) = im(:)/sum(im(:));
end
for ii = 1:imsize
  im = ones(imsize,imsize)*noiselevel/imsize^2;
  im(ii,:) = im(ii,:) + 1/imsize;
  mixtheta(:,imsize+ii) = im(:)/sum(im(:));
end
%两种主题共10个

datass = cell(numgroup,1);
thetas = cell(numgroup,1);
for ii = 1:numcluster
    for jj = (ii-1)*numgroup/numcluster+1 : ii*numgroup/numcluster
        nb = randmult(numbarpermix);
        kk = randperm(numbars*.5);
        kk = kk(1:nb) + (ii-1)*numbars*.5;
        theta = mean(mixtheta(:,kk),2);
        subplot(5,10,jj);
        datass{jj} = randmult(repmat(theta,1,numdata),1);
        thetas{jj} = reshape(theta,5,5);
        imagesc(thetas{jj},[0, 0.2]);
    end
end