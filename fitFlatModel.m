function fitFlatModel()
% Obtain model parameters through a grid search.
% Copyright (C) 2017-2019 Hormet Yiltiz
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% The model response matrix will be HUGE:
% 180 neurons x 2 (mask and target) x 227850 parameter combinations x 6 conditions x 38 stimulus levels
% 3937248000 Bytes x 38 = 140GB (in memory)
% later files will store larger files, since initially most model parameters are too small, resulting in
% very sparse model responses R, which will affect save IO speed, and file size.

warning('off', 'MATLAB:hg:WillBeRemovedReplaceWith');
warning('off', 'MATLAB:dispatcher:UnresolvedFunctionHandle');
%addpath('../lib/NIfTImatlab/matlab/');

adapterNames = {'contingent-15';      % plaid
                'noncontingent-15';   % grating
                'blank';              % blank
                'contingent-45';
                'noncontingent-45';
                'noncontingent-15b';
                'noncontingent-45b'};
if 0
f=dir('bootstrapSim_CrossOrientation_*.mat');
nObs = numel(f);
adapterIdx = [1 4 2 5 3 3];
else
% bootstrapSim_CrossOrientationALLbutSimExpIgnoreFirstHalf_20180424160645.mat
f = load('bootstrapSim_CrossOrientationALLbutSimExp_20180425103330.mat');
nObs = numel(f.T);
adapterIdx = [1 4 6 7 3 3];
end

for i=1:nObs
  try
  temp=load(f(i).name);
  catch
    emulateFile{i}.matFilesPath = sprintf('./Data/HS/ContrastDetection/%s/%s*.mat', f.observers{i}, f.observers{i}(1:2));
    emulateFile{i}.mergedFits = f.T{i}; % {15, 45 deg} x {plaid, grating, blank and no adaptation}
    temp = emulateFile{i};
  end
  c=arrayfun(@(x) x.noiseModel.x, temp.mergedFits, 'UniformOutput', false);
  A{i}=unique(cell2mat(c(:)'));
end
% stimulus strength
xFull = unique(cell2mat(A)); % the stimulus strengths ever used for any observers
xFull(1) = 1e-8; % first value can't be zero or fitting will explode into Inf and NaN

% noiseModel is 2x3, {plaid, grating, blank and no adaptation} x {15, 45 deg} (cite: analyzeContrastDetection.m)
vectorizedNChooseK = @(n,k) factorial(round(n))./(factorial(round(n-k)).*factorial(round(k)));
orientations = repmat([15 45], [1 numel(adapterIdx)/2]);


for iFile=1:nObs
  try
  [ss,dd] = computeD(f(iFile).name, xFull);
  catch
      [ss,dd] = computeD(emulateFile{i}, xFull);
  end
  s{iFile} = ss;
  dMeasured{iFile} = dd;
end


% now specify the model with some parameters
if 0
  % [baseRateSet, sigmaSet, temp_leakSet; ...]
  params = combvec([0 0.01 0.2 0.4 0.6], [0.01 0.16 0.3], [ 0.01 0.1])';
  bandWidths = [15 30 60];
else
  % params = combvec(0:0.02:0.4, 0.01:0.01:0.3, logspace(-2, -1, 10))';
  params = combvec(0:0.01:0.1, 0.01:0.01:0.2, logspace(-2, -1, 5), logspace(-2,2,11))';
  bandWidths = [15 30 60];
  % bandWidths = [1 3 5 6 9 10 14 15 16 17 18 19 20 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 45 59 60 80 89 91];
end

try parpool(maxNumCompThreads, 'IdleTimeout', 120);end
nParams = size(params,1);
nBandwidths = numel(bandWidths);
fprintf('%s: Gridding model space %8d simulations in parallel with %2d kernels...\n-----\n', datestr(now, 'YYYYmmDDHHMMSS'), nParams*nBandwidths, maxNumCompThreads)
% T = zeros(size(params,1)*numel(bandWidths), size(params,2)+1+1);
T = [];
dModelS = [];
for iBandwidth=bandWidths
  tic;
  TT = zeros(size(params,1), size(params,2)+1+1+3*numel(dMeasured));
  fprintf('Processing bandwidth: %d deg during %s', iBandwidth, datestr(now, 'YYYYmmDDHHMMSS'));
  S = struct();
  % adapter0 = flatFeedforward(num2cell([0,0.16,30,0.05]));
  % smallS = struct('adapter', adapter0, 'logL', [0,0.16,30,0.05,-100]);
  % stupid Mathworlks cannot share file identifiers across their stupid "workers"
  fname = sprintf('parfor_fitFlatModel_%s.log', datestr(now, 'YYYYmmDDHHMMSS'));
  for iSim=1:size(params,1)
    opts = params(iSim,:);
    opts = [opts([1 2]) iBandwidth opts([3 4])];

    % ========================================
    adapter = flatFeedforward(num2cell(opts));
    % ========================================

    % adapter = adapter(whichAdapter);
    sigma = opts(2);
    baseRate = opts(1);

    % compute model d'
    dModel = zeros(size(dMeasured{1}));
    R = zeros(180, numel(xFull), 2, numel(adapterIdx));
    for i=1:numel(adapterIdx)
      Rmask = computeR([-1]*orientations(i) +90, [repmat(.6, size(xFull(:)))], adapter(adapterIdx(i)).T, adapter(adapterIdx(i)).s, adapter(adapterIdx(i)).W, sigma, baseRate); % to mask only at 15 deg relative to vertical
      Rtarget = computeR([-1 1]*orientations(i) +90, [repmat(.6, size(xFull(:))) xFull(:) ], adapter(adapterIdx(i)).T, adapter(adapterIdx(i)).s, adapter(adapterIdx(i)).W, sigma, baseRate); % to mask + target
      R(:,:,1,i) = [Rmask];
      R(:,:,2,i) = [Rtarget];
      % imRange=[0.4 0.7];figure;subplot(1,3,1);imagesc(Rmask, imRange);colorbar;subplot(1,3,2);imagesc(Rtarget, imRange);colorbar;subplot(1,3,3); imagesc(Rtarget-Rmask);colorbar;
      % for i=1:6;
      % Rmask=S(500).R(:,:,1,i);
      % Rtarget=S(500).R(:,:,2,i);
      dModel(:,i) = sqrt(sum((Rtarget-Rmask).^2))';
    end


    % output additive noise scales our estimate of d' (dModel), thus we
    % find a scale parameter k that minimizes sum(dMeasured - k dModel).^2
    % Calculus shows the following:
    outputSigmaEst = cellfun(@(x) sum(x(:).*dModel(:))/sum(dModel(:).^2), dMeasured);
    eachlogL = cellfun(@(k,ss) logLGivenD(k*dModel,ss,xFull, adapterIdx), num2cell(outputSigmaEst), s); % logL from our best guess

    % But ultimately we need to minimize logL given data, rather than L2
    % distance in d' space, thus minimization
    % Simple 1D optimization in continuous space, so let's use the fastest
    outputSigma = zeros(size(outputSigmaEst));
    nlogL = zeros(size(outputSigma));
    for ik=1:numel(outputSigmaEst)
      [outputSigma(ik), nlogL(ik)] = fminsearch(@(k) -logLGivenD(k*dModel, s{ik},xFull, adapterIdx), outputSigmaEst(ik));
    end
    if all(outputSigma>0)  % if optimization is valid, ignore our best guess
      eachlogL = -nlogL;
    end
    logL = sum(eachlogL);


    if 0

      figure

      for i=1:numel(adapterIdx)
        subplot(2,3,i);
        data = s.mergedFits(i).binomModel.data;
        hold on

        [~,accuracyModel]=logLGivenD(outputSigmaEst(1)*dModel, s{1});
        plot(xFull,accuracyModel(:,i), 'r')

        scatter(data(:,1), data(:,3)./data(:,2), data(:,2), 'ok');
        axis([0 0.4 0 1]);
        if i==1
          xlabel('Contrast');
          ylabel('Percent correct')
        end
      end
      hold off
    end


    S(iSim).dModel = dModel;
    S(iSim).R = R;
    S(iSim).logL = logL;
    S(iSim).opts = opts;
    S(iSim).outputSigma = outputSigma;
    S(iSim).outputSigmaEst = outputSigmaEst;
    TT(iSim, :) = [opts logL outputSigma outputSigmaEst eachlogL];
    % smallSS(iSim).adapter = adapter;
    % smallSS(iSim).logL = [opts logL];


    % stupid Mathworlks cannot share file identifiers across their stupid "workers"
    [fid, message] = fopen(fname, 'a');
    if fid < 0;
      fprintf(2, 'failed to open "%s" because "%s"\n', fname, message);
      %and here, get out gracefully
      warning('Failed to open log file');
      continue
    end
    fprintf(fid, '.');
    fclose(fid);
%     break;
  end

  T = [T; TT];
  dModelS = [dModelS rmfield(S, 'R')];
  % smallS = [smallS; smallSS];
  fprintf(' %s, duration %5.2f s\n', datestr(now, 'YYYYmmDDHHMMSS'), toc);
  tic;
  save(sprintf('ModelFits_%s_BW%d_%s', 'allObs',iBandwidth,datestr(now, 'YYYYmmDDHHMMSS')), '-nocompression', '-v7.3', '-regexp', '^(?!(T|dModelS)$).');
  toc;
end

% save T, perform basic grid analysis based on T, and export NIFTI file
outputTimestamp = datestr(now, 'YYYYmmDDHHMMSS');
save(sprintf('ModelFits_%s_Grid_%s', 'allObs',outputTimestamp), 'T');
save(sprintf('ModelFits_%s_dModel_%s', 'allObs',outputTimestamp), 'dModelS');

nCol=[];for i=1:size(params,2)+1;nCol(i)=numel(unique(T(:,i)));end
TT=sortrows(T(:,1:numel(nCol)+1), numel(nCol):-1:1); % important that the axes are ordered so the volumes are not jumbled
% cbiWriteNifti(sprintf('ModelFits_%s_Grid_%s', 'allObs',outputTimestamp), reshape(TT(:,5), nCol));

idx=[5 2 3 1 6];
TT=T(abs(T(:,4)-0.03) < 0.01, idx);% pick a moderate leak temp and grid the rest of the space
nCol=[];for i=1:size(params,2)+1;nCol(i)=numel(unique(TT(:,i)));end
TT=sortrows(TT(:,1:numel(nCol)), numel(nCol)-1:-1:1); % important that the axes are ordered so the volumes are not jumbled
%        scaleC 39.811         sigma 0.1          bw 15        baseline 0.04      -543.09
%cbiWriteNifti(sprintf('ModelFits_%s_Grid_%s', 'allObs',outputTimestamp), reshape(TT(:,5), nCol(1:end-1)));
end

function [logL, accuracyModel] = logLGivenD(dModel,s,xFull, adapterIdx)
vectorizedNChooseK = @(n,k) factorial(round(n))./(factorial(round(n-k)).*factorial(round(k)));


% We fit the model to the data; since our task is 2AFC we have:
% d' := z(PC1) + z(PC2) = (ignoring interval bias) 2z(PC)
% normcdf(9)==1 in Matlab, while norminv(1)==inf, bleh!
accuracyModel = normcdf(dModel/2); % assuming readout/late gaussian noise
% the lapse rate is needed for the probabilities not d'
accuracyModel(accuracyModel==0)=1e-4; % some lapse rate
accuracyModel(accuracyModel==1)=1-1e-4; % some lapse rate

logL = 0;
for i=1:numel(adapterIdx)
  data = s.mergedFits(i).binomModel.data;
  % idxX = ismember(data(:,1) , xFull); % doesn't work for floats
  idxX = arrayfun(@(x) find(abs(x-xFull)<10*xFull(1)), data(:,1))'; % use xFull(1) as floating error tolerance
  % logLi = log(vectorizedNChooseK(data(:,2),data(:,3))) + log(data(:,3).*accuracyModel(idxX,i)) + log((data(:,2)-data(:,3)).*(1-accuracyModel(idxX,i))); % we can pre-calculate some of it
  Li = [ones([size(data,1) 1]) data(:,3) (data(:,2)-data(:,3))].*log([vectorizedNChooseK(data(:,2),data(:,3)) [accuracyModel(idxX,i) (1-accuracyModel(idxX,i))]]);


  % the first term in first column is independent of model parameters
  % ignore 0 terms as they correspond to stimulus levels that were not tested
  logL = logL + sum(Li(:));
  % model fit score
  % score = sum(sum((dMeasured-dModel).^2));
end
end

function [s, dMeasured] = computeD(matFile, xFull)
if ischar(matFile)
s = load(matFile);
else
  s = matFile;
end

guessRate = 1/2;
lnrm.f = @(p,x) logncdf(x,p(1),p(2))*(1-guessRate-p(3))+guessRate;


% Use the likelihood function for a binomial model
dMeasured = zeros(numel(xFull), numel(s.mergedFits));

for i=1:numel(s.mergedFits)
  % Fill in the data for the likelihood function
  % psychometric function d'
  dMeasured(:,i) = norminv(lnrm.f(s.mergedFits(i).noiseModel.pfit, xFull))';
  dMeasured(isinf(dMeasured))=8.2;

  [m,freq] = grpstats(s.mergedFits(i).noiseModel.resp, s.mergedFits(i).noiseModel.fitX, {@numel, @mean});
  xUsed = grpstats(s.mergedFits(i).noiseModel.fitX, s.mergedFits(i).noiseModel.fitX, {@mean});
  n = m.*freq;

  s.mergedFits(i).binomModel.data = [xUsed m n];
  % s.mergedFits(i).binomModel.f = @(p) log(vectorizedNChooseK(data(:,2),data(:,3))) + log(data(:,3).*p) + log((data(:,2)-data(:,3)).*(1-p));
end
end

function t = fast_rmfield(s,field)
  % get fieldnames of struct
  f = fieldnames(s);
  [f,ia] = setdiff(f,field,'R2012a');
 % convert struct to cell array
  c = squeeze(struct2cell(s));
% rebuild struct
  t = cell2struct(c(ia,:),f)';
end
