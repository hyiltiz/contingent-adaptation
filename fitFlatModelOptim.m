function pFit = fitFlatModelOptim()
% Obtain model parameters through optimization (CMAES or fmincon)
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

%     0.0043129 0.057421 88.726 0.5
f=dir('bootstrapSim_CrossOrientation_*.mat');
for i=1:numel(f);s=load(f(i).name);
  c=arrayfun(@(x) x.noiseModel.x, s.mergedFits, 'UniformOutput', false);
  A{i}=unique(cell2mat(c(:)'));
end
% stimulus strength
xFull = unique(cell2mat(A)); % the stimulus strengths ever used for any observers
xFull(1) = 1e-8; % first value can't be zero or fitting will explode into Inf and NaN

s = load('bootstrapSim_CrossOrientation_HTY_20170922223832.mat');
guessRate = 1/2;
lnrm.f = @(p,x) logncdf(x,p(1),p(2))*(1-guessRate-p(3))+guessRate;


% Use the likelihood function for a binomial model
dMeasured = zeros(numel(xFull), numel(s.mergedFits));

for i=1:numel(s.mergedFits)
  % Fill in the data for the likelihood function
  % psychometric function d'
  dMeasured(:,i) = norminv(lnrm.f(s.mergedFits(i).noiseModel.pfit, xFull))';

  [m,freq] = grpstats(s.mergedFits(i).noiseModel.resp, s.mergedFits(i).noiseModel.fitX, {@numel, @mean});
  xUsed = grpstats(s.mergedFits(i).noiseModel.fitX, s.mergedFits(i).noiseModel.fitX, {@mean});
  n = m.*freq;

  s.mergedFits(i).binomModel.data = [xUsed m n];
  % s.mergedFits(i).binomModel.f = @(p) log(vectorizedNChooseK(data(:,2),data(:,3))) + log(data(:,3).*p) + log((data(:,2)-data(:,3)).*(1-p));

end

if 0
  params = combvec([0 0.01 0.2], [0 0.16 0.3], [0 0.01 0.1])';
  bandWidths = [15 30 60];
else
  params = combvec(0:0.01:0.2, 0:0.01:0.3, logspace(-2, -1, 10))';
  bandWidths = [1 3 5 6 9 10 14 15 16 17 18 19 20 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 45 59 60 80 89 91];
end

p0 = [0 0.05 60 0.1];
opts.LBounds = [0 0 3 0]'; opts.UBounds = [0.5 0.5 90 0.5]';
opts.MaxFunEvals = 5e4;
opts.TolFun = 1e-3;
opts.LogTime = 100;
% Can also use fmincon
pFit = cmaes(@(p) -flatFeedforwardLogL(p,s,xFull), p0', [0.01 0.05 5 0.01]', opts);
%   opts.baseRate = baseRateSet(iSim); % baseline firing rate
%   opts.sigma = sigmaSet(iSim);
%   opts.halfWidthHalfHeight = bandwidthSet(iSim); % half-width at half-height in deg (pi/6); full-width should be 30 deg
end

function logL = flatFeedforwardLogL(opts, s,xFull)
try
  opts(3) = round(opts(3));
  adapter = flatFeedforward(num2cell(opts));
catch
  % most likely when we are requesting tuning curves impossible to tile
  logL = NaN;
  return;
end

% adapter = adapter(whichAdapter);
sigma = opts(2);
baseRate = opts(1);

adapterIdx = [1 2 3 4 5 3];
orientations = [15 15 15 45 45 45];
noiseModelIdx = [1 3 5 2 4 6];
vectorizedNChooseK = @(n,k) factorial(round(n))./(factorial(round(n-k)).*factorial(round(k)));

dMeasured = zeros([numel(xFull), numel(s.mergedFits)]);
R = zeros(180, numel(xFull), 2, numel(adapterIdx));
logL = 0;
for i=1:numel(adapterIdx)
  Rmask = computeR([-1]*orientations(i) +90, [repmat(.6, size(xFull(:)))], adapter(adapterIdx(i)).T, adapter(adapterIdx(i)).s, adapter(adapterIdx(i)).W, sigma, baseRate); % to mask only at 15 deg relative to vertical
  Rtarget = computeR([-1 1]*orientations(i) +90, [repmat(.6, size(xFull(:))) xFull(:) ], adapter(adapterIdx(i)).T, adapter(adapterIdx(i)).s, adapter(adapterIdx(i)).W, sigma, baseRate); % to mask + target
  dModel(:,i) = sqrt(sum((Rtarget-Rmask).^2))';
end

% We fit the model to the data; since our task is 2AFC we have:
% d' := z(PC1) + z(PC2) = (ignoring interval bias) 2z(PC)
% normcdf(9)==1 in Matlab, while norminv(1)==inf, bleh!
accuracyModel = normcdf(dModel/2); % assuming readout/late gaussian noise
% the lapse rate is needed for the probabilities not d'
accuracyModel(accuracyModel==0)=1e-4; % some lapse rate
accuracyModel(accuracyModel==1)=1-1e-4; % some lapse rate

for i=1:numel(adapterIdx)
  data = s.mergedFits(noiseModelIdx(i)).binomModel.data;
  % idxX = ismember(data(:,1) , xFull); % doesn't work for floats
  idxX = arrayfun(@(x) find(abs(x-xFull)<10*xFull(1)), data(:,1))'; % use xFull(1) as floating error tolerance
  % logLi = log(vectorizedNChooseK(data(:,2),data(:,3))) + log(data(:,3).*accuracyModel(idxX,i)) + log((data(:,2)-data(:,3)).*(1-accuracyModel(idxX,i))); % we can pre-calculate some of it
  Li = [ones([size(data,1) 1]) data(:,3) (data(:,2)-data(:,3))].*log([vectorizedNChooseK(data(:,2),data(:,3)) [accuracyModel(idxX,i) (1-accuracyModel(idxX,i))]]);

  % the first term in first column is independent of model parameters
  % ignore 0 terms as they correspond to stimulus levels that were not tested
  logL = logL + sum(Li(:));
end

end
