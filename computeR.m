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
