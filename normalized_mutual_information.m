function nmi = normalized_mutual_information(Z1, Z2, way)
% Compute NMI as described in:
% Lancichinetti, Fortunato, and Kertesz. Detecting the overlapping and 
% hierarchical community structure of complex networks. 2009.
% Input: two sparse binary matrices with size n x k1 and n x k2.
% Output: NMI score of two covers. 

if nnz(Z1) ~= nnz(Z1==1)
  error('Z1 should be a binary matrix!');
end

if nnz(Z2) ~= nnz(Z2==1)
  error('Z2 should be a binary matrix!');
end

H_x = marginal_entropy(Z1);
H_y = marginal_entropy(Z2);
	if  way~=3
		H_x_y = conditional_entropy(Z1,Z2,H_x,way);
		H_y_x = conditional_entropy(Z2,Z1,H_y,way);
		nmi = 1 - 0.5 * (H_x_y + H_y_x);
	else
		H_x_s = sum(H_x);
		H_y_s = sum(H_y);
		H_x_y = conditional_entropy(Z1,Z2,H_x,way);
		H_y_x = conditional_entropy(Z2,Z1,H_y,way);
		I_x_y = 0.5*(H_x_s-H_x_y+H_y_s-H_y_x);
		nmi = (I_x_y)/max(H_x_s,H_y_s);
		
	end

end


function H = marginal_entropy(Z)
% Compute marginal entropy H(X_k) for all k. 
% Return value H is a column vector of length k.
n = size(Z,1);
prob = sum(Z,1)' / n; % nnz of each column divided by n
prob = max(prob, 1e-9);
H = - prob .* log2(prob) - (1 - prob) .* log2(1 - prob);

end
% function H_joint = joint_entropy(Z1, Z2)

% end
function H_cond = conditional_entropy(Z1,Z2,H_x,way)

% Compute joint entropy H(X_k, Y_l) for all k and l
% Return value H is a k1 x k2 matrix
	n = size(Z1,1);
	k1 = size(Z1,2);
	k2 = size(Z2,2);
	H_cond_pre = zeros(k1,k2);
	for c1 = 1:k1
	  col1 = Z1(:,c1);
	  for c2 = 1:k2
	    col2 = Z2(:,c2);
	    intersect_size = nnz(col1 .* col2);
	    union_size = nnz(col1 + col2);
	    p00 = 1 - union_size / n;
	    p01 = (nnz(col2) - intersect_size) / n;
	    p10 = (nnz(col1) - intersect_size) / n;
	    p11 = intersect_size / n;
	    prob = [p00; p01; p10; p11];
	    assert(sum(prob) - 1 < 1e-9);
	    prob = max(prob, 1e-9); % clip to avoid 0 * log(0)
	    H_prob = -prob.*log2(prob); 
	    if(way ~= 1 & (H_prob(1)+H_prob(4))<(H_prob(2)+H_prob(3)))
	    	prob2 =[p00+p01;p10+p11];
	      	H_cond_pre(c1,c2) =  - prob2' * log2(prob2);
	    else   
	    	prob2 =[p00+p10;p11+p01];	
	       	H_cond_pre(c1,c2) = - prob' * log2(prob) + prob2' * log2(prob2);
	   	end  
	  end
	end	
	% Compute conditional entropy H(X|Y), a scalar.
	% H_x and H_y are column vectors of length k1 and k2.
	% H_cond_pre is a k1 x k2 matrix.
	% Subtract H_y' from every row of H_xy, get a k1 x k2 matrix
	% Find min for every row, get a column vector of length k1
	H_cond = min(H_cond_pre, [], 2); 

	% Normalize, average over k1
	if way ~= 3
		H_cond = mean(H_cond ./ H_x);
	else
		H_cond = sum(H_cond);
	end

end