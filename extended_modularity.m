
function [modularity] = extended_modularity(B,Z0,m)
	Z0n = sqrt(sum(Z0.*Z0,2));
	Z0n(Z0n < 1e-10)=1;
	Z0n = repmat(Z0n,1,size(Z0,2));
	Z1 = Z0./(Z0n);
	modularity = 1/(2*m)*(sum(sum(B.*(Z1*Z1'))));
end
