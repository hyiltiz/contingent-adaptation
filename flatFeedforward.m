function adapter = flatFeedforward(opts)
% Full model implementation the Hebbian model of conventional and contingent adaptation.
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

contrast = 1;
sigma = contrast/2/3;

halfWidthHalfHeight = 30; % half-width at half-height in deg (pi/6); full-width should be 30 deg
% neurons whose preferred orientations are equally spaced within [0,pi)

% We have: sampleFreq:2*bw == 180 breaks : 180 deg
sampleFreq = 120;
% we want 360 neurons
overCompletenessRatio = 60;

% specifying n and N overrides sampleFreq and overCompletenessRatio
n = 180; % 180 discrete points in the [0, pi)=[0, 180) domain;
N = 180; % N neurons
% N=720;


% overlap := dSpacing/(2*width);
% overlap = (2*halfWidthHalfHeight*180/pi/2)/overCompletenessRatio; % FIXME: wrong
noiseSD = 1; % deg, noise of the each neurons, i.i.d.
whichAdapters = [3 4 5 6 7 8 9];
scaleC = 1;
% whichAdapters = [11];
% whichAdapters = [8 9];
createVideo = false;
% createVideo = true;
trackEigHistory = false;
epoch = 5e1;
nMemory = 1; % we store `nMemory` copies of recent (plus current) weights
baseRate = 0; % baseline firing rate
temp_leak = 5e-2;
temp = 100;

if nargin == 1
%   baseRate = opts.baseRate; % baseline firing rate
%   contrast = opts.contrast;
%   sigma = opts.sigma;
%   halfWidthHalfHeight = opts.halfWidthHalfHeight; % half-width at half-height in deg (pi/6); full-width should be 30 deg
[baseRate, sigma, halfWidthHalfHeight, temp_leak, scaleC] = deal(opts{:});
end
%% Initialize the network
% [T s preferredS] = tuningCurves(60, 1, 4); % to debug
% this set of parameters produces 360 neurons within [0, pi), on a grid of
% size 180.
minTuningSize = size(tuningCurves(2*halfWidthHalfHeight, 1, 2*halfWidthHalfHeight)); % minimal neurons and discretization
[T0, s, preferredS] = tuningCurves(2*halfWidthHalfHeight, N/minTuningSize(1), 2*halfWidthHalfHeight*n/minTuningSize(2)); % a set of tuning curves

T = T0 + baseRate;
T = T./mean(sqrt(sum(T.^2,1))); % this ripples due to the cross term of the form: 2b(cos+sin)
W = ones(numel(preferredS)); % weight
R = T.^2./(sigma.^2 + W/size(W,1)*T.^2); % response; NOTE: divided by network size hoping that the range of C should scale with the network size

p = ones(size(s)); % probabilities of the stimulus
p = normpdf(s, 0, sqrt(-0.5*10^2/log(0.5))) * mean(diff(s));


% Response products for this network where orientations follow a uniform distribution
C = scaleC*R*(eye(numel(s))/numel(s))*R';

% Interacts with temperature, needs to be closer to C for cooler temperatures
% temp = 10^(abs(diff(log10([mean(W(:)) mean(C(:))])))); % learning temperature
% temp_leak = 1e-2;
% temp = 10;
% temp = 1;


%% Learn weights given adapters
% see computeR.m for how values defined under `adapter` is used
adapter = loadAdapters(s);
% adapter = adapter(whichAdapters);
adapter = adapter([1 end]); % temporary: testing a masking paradigm
[adapter(:).T] = deal(T);
[adapter(:).s] = deal(s);
[adapter(:).preferredS] = deal(preferredS);

% add in a dynamic adapter with sinosoids

for whichAdapter = 1:numel(adapter)
  % whichAdapter = 4;
  V = adapter(whichAdapter);



  W = ones([numel(preferredS) numel(preferredS) nMemory]); % weight
  W0 = W;
  wMemory = zeros([1 1 nMemory]); % if updating weight using recent weight history
  wMemory(:) = exp(-(0:nMemory-1)/1);
  if trackEigHistory; eigHistory = zeros([size(W,1) epoch]);end

  idxW = mod(0:epoch-1, nMemory)+1;
  edge = zeros(epoch,2);
  if createVideo
    h=figure('Position', [500 500 1600 800]);
    set(gcf,'color','w');
    F(epoch) = struct('cdata',[],'colormap',[]);
    myVideo = VideoWriter(sprintf('myfile_%s_%s', V.name, datestr(now, 'YYYYmmDDHHMM')), 'MPEG-4');
    myVideo.Quality = 75;    % Default 75
    myVideo.FrameRate = 10;
    open(myVideo);
  end
  for i=1:epoch
    R = computeR(V.Vdeg(i), V.contrast, T, s, W(:,:,idxW(i)), sigma, baseRate);% over-represented stimuli


    if createVideo
      subplot(2,3,1);imagesc(C);colorbar;title('C');axis xy image;
      subplot(2,3,2);imagesc(R*V.p*R');colorbar;title('RR^T');axis xy image;
      subplot(2,3,3);imagesc(R*V.p*R'-C);colorbar;title('\DeltaW:=RR^T-C');axis xy image;
      subplot(2,3,4);imagesc(W);colorbar;title('W');axis xy image;xlabel(sprintf('Iter: %d,Temp: %.2f', i,  ((temp-1)*exp(-i/temp)+1)));
      Tpost = T.^2 ./(sigma.^2 + W/size(W,1)*T.^2);
      edge(i,:) = [min(Tpost(:)) max(Tpost(:))];
      subplot(2,3,5);imagesc(Tpost, [-5 5]);colorbar;title('T_{post}');axis xy image;colormap(gca, 'gray');xlabel(sprintf('Range:[%.0e,%.0e]', edge(i,1), edge(i,2)));
%       subplot(2,3,6);imagesc(T);colorbar;title('T_{pre}');axis xy image;xlabel(sprintf('Base rate: %.0e', baseRate));
      subplot(2,3,6);imagesc(W-W0);colorbar;title('W-W_0');axis xy image;xlabel(sprintf('Base rate: %.0e', baseRate));
      drawnow;
      F(i) = getframe(gcf);
      writeVideo(myVideo, F(i));

      Tpost = T.^2 ./(sigma.^2 + W(:,:,idxW(i))/size(W,1)*T.^2);
      edge(i,:) = [min(Tpost(:)) max(Tpost(:))];
      pause(0.1);
    end


    if trackEigHistory; eigHistory(:,i) = eig(W); end
    % Calculate response products averaged over the stimulus ensemble
    % Response products for this network where orientations follow a distribution p
    iidx = i:-1:i-nMemory+1; % reuse the memory pool88ttgv6tgg
    idx = idxW(iidx(iidx>0));

    % ---------- BEGIN: Weight update equation ----------
    %     W(:,:,idxW(i)) = sum(bsxfun(@times, W(:,:,idx), wMemory(:,:,idx)),3) + temp*exp(-i/temp)*(R*V.p*R' - C); % + 1e-2*randn(size(W));
    % attractor W => W' := W - a(W-T)
    % W(:,:,idxW(i)) = max(0, W(:,:,idx) + ((temp-1)*exp(-i/temp)+1)*(R*V.p*R' - C) - temp_leak*(W-C)); % beta is independent of alpha
    alpha = ((temp-1)*exp(-i/temp)+1);
    W(:,:,idxW(i)) = max(0, W(:,:,idx) + alpha*(R*V.p*R' - C));% - alpha*temp_leak*(W-C)); % beta is a ratio of alpha
%     W(:,:,idxW(i)) = max(0, W(:,:,idx) + ((temp-1)*exp(-i/temp)+1)*(R*V.p*R' - C) - temp_leak*W0);
    % ----------   END: Weight update equation ----------

    %{
    a = temp*exp(-i/temp)*(R*V.p*R' - C);
    b = -temp_leak*(W-W0);
    c = W(:,:,idx) + a + b;
    d = max(0, c);
    %}

%       W(:,:,idxW(i)) = W(:,:,idx) + abs(temp*exp(-i/temp)*(R*V.p*R' - C) - temp_leak*W0);
%       W(:,:,idxW(i)) = max(0,W(:,:,idx).*(1 + temp*max(0,R*V.p*R' - C))); % temp := [-1, 1]




  end
  if createVideo;close(myVideo);end

  if trackEigHistory
  figure;
  for i=1:size(eigHistory,2);
    [~,idx]=sort(angle(eigHistory(:,i)));
    polarplot(eigHistory(idx,i));
    rlim([-50 max(max(abs(eigHistory)))]);
    title(sprintf('%s adapter weight eigenvalue history: %d/%d', adapter(whichAdapter).name, i, size(eigHistory,2)));
    pause(0.01);
  end
  end

  % Effective tuning curves post-normalization before learning
  Tpost = T.^2 ./(sigma.^2 + W(:,:,idxW(end))/size(W,1)*T.^2);

  if 0 % plot the state
    preferredSdeg = preferredS;
    figure('Position', [500 500 500 500]);
    subplot(2,2,1);
    imagesc(preferredSdeg, preferredSdeg, C);
    axis xy image
    title('C');
    ylabel('Neuron');

    subplot(2,2,2);
    imagesc(preferredSdeg, preferredSdeg, W(:,:,idxW(end)));
    axis xy image
    title('W_{post}');

    subplot(2,2,3);
    imagesc(preferredSdeg, s, T');
    title('T_{pre}');
    axis xy image square
    ylabel('\theta');
    xlabel('Neuron')

    subplot(2,2,4);
    imagesc(preferredSdeg, s, Tpost');
    axis xy image square
    title('T_{post}');
    xlabel('Neuron');
    colormap gray;
  end

  adapter(whichAdapter).W = W(:,:,idxW(end));
  adapter(whichAdapter).Tpost = Tpost;
end % adapters

if 1
  cleanGray = flipud(colormap('gray'));
  preferredSdeg = preferredS;
%   figure('Position', [0 0 (1+numel(adapter))*800 700]);
  figure('Position', [0 0 1900 700]);

  TimageRange = 1*[0 1];
%   TimageRange = [0.7 0.75];
  subplot(3,1+numel(adapter),1);
  imagesc(preferredSdeg, preferredSdeg, C);
%   showIm(C);
  axis xy image
  title('C');
  ylabel('Neuron');
  xlabel('Neuron');

  subplot(3,numel(adapter)+1,numel(adapter)+2);
  imagesc(preferredSdeg, s, T, TimageRange);
%   showIm(T');
  title('T_{pre}');
  axis xy image square
%   ylabel('\theta');
  ylabel('Neuron')


    subplot(3,1+numel(adapter),2*(1+numel(adapter))+1);
    plot(s, adapter(1).T(1:60:end,:)','k-');
    axis([0 180 0 max(TimageRange)]);
    xlabel('\theta');
    ylabel('Firing rate');
    axis square


  for whichAdapter=1:numel(adapter)
    subplot(3,1+numel(adapter),whichAdapter+1);
    imagesc(preferredSdeg, preferredSdeg, adapter(whichAdapter).W, [0 30]);
%     showIm(adapter(whichAdapter).W);
    axis xy image
    title(sprintf('%s', adapter(whichAdapter).name));
    if whichAdapter>1;set(gca, 'XTick',[],'YTick',[]);end
    colormap(cleanGray);

    subplot(3,1+numel(adapter),1+numel(adapter)+whichAdapter+1);
    imagesc(preferredSdeg, s, adapter(whichAdapter).Tpost, TimageRange);
%   showIm(adapter(whichAdapter).Tpost');
    axis xy image square
    if whichAdapter>1;set(gca, 'XTick',[],'YTick',[]);
    else
    title('T_{post}');
%     xlabel('Neuron');
    end
    colormap(cleanGray);

    subplot(3,1+numel(adapter),2*(1+numel(adapter))+whichAdapter+1);
    plot(s, adapter(whichAdapter).Tpost(1:15:end,:)','k-');
    axis([0 180 0 max(TimageRange)]);
    set(gca, 'XTick',[],'YTick',[]);
    if whichAdapter==1
%     title('T_{post}');
    end
    axis square

  end
end

%% target and mask
% mask

Rmask = computeR([-15]+90, [.3], T, s, W(:,:,idxW(end)), sigma, baseRate); % to mask only at 15 deg relative to vertical

% target
Rtarget = computeR([-15 15]+90, [.3 .6], T, s, W(:,:,idxW(end)), sigma, baseRate); % to mask + target


% Assuming Gaussian noise
dPrime = norm(Rmask-Rtarget);

%% readouts
% argmax
% pre-adaptation matching orientation
% pre-adaptation orientation weight profile
% maximizing information

R = R + noiseSD*randn(size(R)); % Addive gaussian noise, noisy readout

% A. Pick response of the neuron based on pre-learning tuning sensitivities
idx = preferredS==(15 + 90);
dA = Rtarget(idx) - Rmask(idx);


% B. Weigh the output using pre-learning tuning sensitivities
idx = s==(15+90);
dB = T(:,idx)'*Rtarget - T(:,idx)'*Rmask;

if dA < dB
  % disp('Deciding by a single neuron performs worse than weighting in all neurons');
end


end

function [R, sR, allCenters] = tuningCurves(bandwidth, overCompleteRatio, sampleFreq)
% Creates `overCompleteRatio` sets of raised cosine tuning curves with the
% the (half) `bandwidth` (from peak to 0 in radians) equally spaced over [0,pi).
% Each tuning curve is discretized at `sampleFreq`.
%
% First create a set of minimal sufficient basis raised cosine functions
% where sum of squared basis function is a constant such that their
% energy (response) is constant across orientations. For this basis set,
% the width constrain the spacing. We will tile [0,pi/2) with N raised
% cosines such that they intersect at their quarter periods. We then
% shift then by pi/2 so it corresponds to N raised sines. This 2N raised
% cosines are minimal basis set given a bandwidth by the constrain:
%     N = pi/bandwidth, where N is minimal basis set size in [0,pi/2)
%
% We can duplicate this 2N basis set by shifting it by any orientation
% to get over-sufficient basis set. Therefore, the number of raised
% cosine tuning curves will be 2M(pi/bandwidth) where M is any positive
% whole number.
%
% As the implementation goes, consider the whole range [-pi, pi).
% Construct the N basis within [0,pi/2). Only the first raised
% cosine will bleed out of [0,pi/2) into nagative. This is still true
% for the minimal 2N basis over [0,pi). We simply wrap around this
% half period of the first raised cosine to the end of [0,pi).
%
% width -> N centers in [0, pi/2)
%       -> shift by pi/2 to get 2N centers in [0, pi)
%       -> duplicate by M to get over-complete basis set
%       -> evaluate a raised cosine at those centers with the bandwidth
%       -> normalize by the energy such that, for any orientation the
%          response is 1.
%
% NOTE: St. DH's enlightening proverb:
%       All you need to know on discrete finite domain signal processing is:
%             2*pi*k*x/N, where x <- [0,N-1]
%
% Licensed under GNU GPL version 3 or later.
% Hörmet Yiltiz, <hyiltiz@gmail.com>
% 2017/11/11

% DONE: Use degs, as they are whole numbers
if nargin < 2
  overCompleteRatio = 1; % not overcomplete by default
end

if nargin < 3
  sampleFreq = 1e2;
end

powerK = 1; % just stick to simple old sin() and cos()
% assert(powerK==1, 'No idea how to tile cosines raised to arbitrary powers yet.');

assert(mod(sampleFreq, 2)==0, 'Need even samples to properly handle wrap-arounds in circular axis.');

N = round(180/bandwidth);
cosCenters = linspace(0, 90, N+1); % [0,pi/2]
cosCenters(end) = []; % [0, pi/2)
sinCenters = cosCenters + 90; % [pi/2, pi)
basisCenters = [cosCenters sinCenters]; % [0, pi)
shifts = linspace(0, bandwidth/2, overCompleteRatio+1); % [0, bandwidth]
shifts(end) = []; % [0, bandwidth)
allCenters = bsxfun(@plus, basisCenters, shifts'); % each row is a complete set
allCenters = sort(allCenters(:)); % ordered in [0, pi), a column

% same as below but in has a rounding error at 0
% s = linspace(-pi, pi, N*sampleFreq+1);
% s(end)=[];

% Explicitly avoid the floating point error at 0
sMinus = linspace(-180, 0, N*sampleFreq/2+1);
sCenter = linspace(0, 180, N*sampleFreq/2+1);
sPlus = linspace(180, 2*180, N*sampleFreq/2+1);
s = [sMinus(1:end-1) sCenter(1:end-1) sPlus(1:end-1)]; % [-pi:sampleFreq:pi], a row, centered at 0

% centered at 0, reaches to 0 at +-bw, only contains the positive bump at 0
% raisedCosine = @(x,bw,k) cos(min(abs((pi/bw)*x), pi/2)).^k;

% centered at 0, asymptotes to 0 at +-bw
% NOTE: cos(2x) = 2cos(x)^2-1
raisedCosine = @(x,bw, powerK) (cosd(min(abs((180/bw)*x), 180))+1)/2;


% each row is a curve using centers in `allCenters` in order, in [0-pi,pi+pi)
r = bsxfun(@(x,center) raisedCosine(x-center, bandwidth, powerK), s, allCenters);

% to wrap around, we need an even sampleFreq such that
% wrap [-pi,0) to (pi,0]
wrappablesL = s <= 0-2*eps; % to avoid any floating point errors
wrappablesR =  s >= 180-2*eps;
assert(numel(s)/3==sum(wrappablesL), 'You really expect commenting out the previous assertion would solve your problem!? Try to oblige next time :D');


% now go into our beloved [0,pi) orientation axis
sR = s(:,  ~(wrappablesL|wrappablesR));
R = r(:,wrappablesL) + r(:, ~(wrappablesL|wrappablesR)) + r(:,wrappablesR);


% assert(sum(R.^2, 1)==(3*overCompleteRatio/2));

% This is also worthwhile to note that, this network's response to any
% power sums up to a constant.
% normalize such that, given a stimulus, the network's energy response sum to 1
try
  R = R./sqrt(sum(R.^2,1));
catch
  R = bsxfun(@rdivide, R, sqrt(sum(R.^2,1))); % old Matlab complains about the above
end

assert(all(abs(sum(R.^2,1)-1) < 10*eps), 'Energy responses of the tuning curves is not normalized.');

end

function T = validBandwidths4Network(n,N)
% n = 180; N = 180;
T = [];

bandWidthRange = 1:180;
for i=bandWidthRange
  halfWidthHalfHeight=i;
  try
    minTuningSize = size(tuningCurves(2*halfWidthHalfHeight, 1, 2*halfWidthHalfHeight)); % minimal neurons and discretization
    flag=all(size(tuningCurves(2*halfWidthHalfHeight, N/minTuningSize(1), 2*halfWidthHalfHeight*n/minTuningSize(2)))==[N n]);
    if flag;T=[T i];end
  end
end

end


function R = computeR(Vdeg, contrast, T, s, W, sigma, baseRate)
% Vdeg = [-15 15];
% contras = [.3 0.6];

% Vidx = sum(bsxfun(@eq, s, Vdeg'))==1; % indices into T given Vdeg
% Veach = T(:,Vidx); % tuning responses of all neurons to each orientations in Vdeg
% V = Veach*contrast'; % sum using contrast as weights

% Might have to worry about floating point rounding here
%   V = T(:,sum(s==Vdeg',1)==1)*contrast';
% V = T(:,sum(bsxfun(@(x,y) abs(x-y)<2*eps, s, Vdeg'),1)==1)*contrast';
% V = T(:,sum(bsxfun(@eq, s, Vdeg'),1)==1)*contrast';
% V = T(:,any(bsxfun(@eq, s, Vdeg'),1))*contrast';


% NOTE: Too small base rates can clip network weight
% The effect is similar to applying a window function to W


if isnan(Vdeg)
  V = baseRate + zeros(size(T,1));
else
  [idx2s,tmp]=find(bsxfun(@eq, s', Vdeg));
  V = T(:,idx2s)*contrast';
end


% normalized output with input V and weight W
R = V.^2 ./(sigma^2 + W/size(W,1)*V.^2);

end

function adapter = loadAdapters(s)
% `contrast` is a weight for the sum when computing the response
% of a single neuron to a set of inputs presented simultaneously.
adapter = [];

overRepresented = linspace(0,180, 10+1);
overRepresented(end)=[];
p=ones(size(overRepresented));
p(floor(numel(overRepresented)/2)+1) = 5;

V.name = 'Benucci-unbiased';
V.Vdeg = @(i) overRepresented;
V.p = diag(numel(overRepresented));
V.p = diag(V.p/sum(V.p));
V.contrast = [1];
adapter = [adapter V];

V.name = 'Benucci-biased';
V.Vdeg = @(i) overRepresented;
V.p = diag(p);
V.contrast = [1];
adapter = [adapter V];

V.name = 'contingent-15';
V.Vdeg = @(i) [(-1)^i*[-15 15]+90];
V.p = [0.5]; % either this or nothing (blank)
V.contrast = [.5 .5]; % each component is at half a contrast
adapter = [adapter V];

V.name = 'noncontingent-15';
% actual presentation sequentially
% V.Vdeg = @(i) [(-1)^i*[15]+90];
% V.p = [1];
% V.contrast = [0.5];
V.Vdeg = @(i) [[-15 15]+90]; % actual presentation sequentially
V.p = diag([0.5 0.5]);
V.contrast = [1]; % full contrast grating adapter
adapter = [adapter V];

V.name = 'blank';
V.Vdeg = @(i) NaN; % FIXME: this necessitates a base rate
V.p = 1;
V.contrast = [1];
 adapter = [adapter V];

 V.name = 'contingent-45';
 V.Vdeg = @(i) [(-1)^i*[-45 45]+90];
 V.p = [0.5];
 V.contrast = [.5 .5];
 adapter = [adapter V];

 V.name = 'noncontingent-45';
 V.Vdeg = @(i) [[-45 45]+90];
 V.p = diag([0.5 0.5]);
 V.contrast = [1];  % full contrast grating adapter
 adapter = [adapter V];

 V.name = 'noncontingent-15b';
V.Vdeg = @(i) [[-15 15]+90];
V.p = diag([0.5 0.5]);
V.contrast = [0.5]; % half contrast grating adapter
adapter = [adapter V];

 V.name = 'noncontingent-45b';
 V.Vdeg = @(i) [[-45 45]+90];
 V.p = diag([0.5 0.5]);
 V.contrast = [0.5]; % half contrast grating adapter
 adapter = [adapter V];

V.name = 'gaussian';
V.Vdeg = @(i) s;
V.p = diag(normpdf(s,90,sqrt(-0.5*30^2 / log(0.5)))); % halfWidthHalfHeight=30 deg; values below 27.18 degs will produce extreme weights give baserate=1e-2
V.contrast = [1];
% adapter = [adapter V];

V.name = 'contingent^*';
V.Vdeg = @(i) [(-1)^i*[-15 15]+90 overRepresented];
V.p = [5e0 ones(size(overRepresented))];
V.p = diag(V.p/sum(V.p));
V.contrast = eye(numel(V.Vdeg(1)));
V.contrast(1,:) = [];
V.contrast(1, [1 2]) = [0.5 0.5];
adapter = [adapter V];

V.name = 'noncontingent^*';
V.Vdeg = @(i) [(-1)^i*[15]+90 overRepresented];
V.p = [5e0 ones(size(overRepresented))];
V.p = diag(V.p/sum(V.p));
V.contrast = eye(numel(V.Vdeg(1)));
adapter = [adapter V];

V.name = 'bandpass-contingent';
V.Vdeg = @(i) s;
V.p = sum(normpdf(s,90+[-15;15],sqrt(-0.5*30^2 / log(0.5))),1);
V.p = diag(V.p/sum(V.p));
V.contrast = [1];
% adapter = [adapter V];

V.name = 'bandpass-noncontingent';
V.Vdeg = @(i) s;
V.p = sum(normpdf(s,90+[-15;15],sqrt(-0.5*30^2 / log(0.5))),1);
V.p = diag(V.p/sum(V.p));
V.contrast = [1];
% adapter = [adapter V];

V.name = 'Benucci-noiseMasked';
V.Vdeg = @(i) [(1+(-1).^i)/2*overRepresented + (1-(-1).^i)/2*randi(90, size(overRepresented))];
V.p = diag(numel(overRepresented));
V.p = diag(V.p/sum(V.p));
V.contrast = [1];
adapter = [adapter V];

end
