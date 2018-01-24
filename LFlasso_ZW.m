%% solving the following problem:
%%
%%     min \frac{1}{2}\|R-M\|^2_F  + lambda*\|M\|_S
%% 
%% where \|.\|_S is atomic norm based on the atomic set:
%%
%%     S = { zz' | z_i\in {0,1} }.
%%
%% Our solver reterns c and Z=[z_1,...,z_k] s.t.  M=\sum_j c_j* z_jz_j'.

%% R_true and R_test is dense here 
function  [nmi_b,em_b,nmi_s,em_s,Z,W,snap_tcom,obj_suc] = LFlasso_ZW(R_true,R_test,R_train,R0_sp,R1_sp,snap_cell,syn_parameter,parameter,Z0,W0,data_cell,out,snap_load,outname,outresult)

	obj_suc = 1.0;
	datatype = data_cell{5};
	%setting parameter
	ground_truth_existence = parameter(1,1);
	noisy = parameter(1,2);
	T=parameter(1,5);
	lambda =parameter(1,6);
	TOL = parameter(1,7);
	tol_rate = parameter(1,8);
	T2 =parameter(1,9);
	SDP_rank = parameter(1,10);
	SDP_iter = parameter(1,11);
	mu = parameter(1,12)
	stepsize = parameter(1,13);
	mod_num = parameter(1,14);
	snap_run = parameter(1,16);
	threshold = parameter(1,17):parameter(1,18):parameter(1,19);
	nmi = -1e30;
	em  = -1e30;
	nmi_b = zeros(1,4)-10;
	nmi_s = zeros(1,6)-10;
	% nmi_g4 : sanp nmi
	em_b = zeros(1,5)-10;
	em_s = zeros(1,6)-10;
	snap_tcom = -10;
	% register a clean up fuction
	finishup = onCleanup(@() myCleanupFun(syn_parameter,parameter,nmi_s,em_s,nmi_b,em_b,snap_cell,snap_tcom,data_cell,outname,outresult,obj_suc));
	% em_g4 :snap modularity em_g5 :ground_truth modularity
	f = @(Z1,z0,thd) sum( Z1~=(z0*ones(1,size(Z1,2))) ) <= thd;
	n = size(R0_sp,1);

	Z = [];
	c = [];
	W =[];

	best_auc= 0;
	prev_obj = 1e300;




	%first calculate modularity matrix B
	B = full(R1_sp); 
	m = sum(sum(B))/2;
	degree = sum(B,2);
	BB = (degree*degree')/(2*m);
	B = B-BB;

	if(ground_truth_existence > 0.5)
		em_b(5) = extended_modularity(B,Z0,m); 	
		g_loss = snap_obj(R_true,Z0);
	end

	%can actually calculate groundtruth guess
	Rt_guess=0;
	g_auc=0;
	g_acc=0;
	g_rec=0;
	g_prec=0;
	g_F1_1=0;
	g_F1_2=0;
	if(strcmp(datatype,'Synthetic'))
		Rt_guess = 1./(1+exp(-Z0*W0));
		[g_acc,g_rec,g_prec,g_F1,g_F1_2,g_auc] = all_statistic(R_true,R_test,Rt_guess,threshold);
		%beta is actually dummy variable in all_statistics
	end	
	Z6 = 0;
	snap_loss = 1e300;
	Rs_guess = 0;
	s_auc=0;
	s_acc=0;
	s_rec=0;
	s_prec=0;
	s_F1_1=0;
	s_F1_2=0;
	if (parameter(1,16) > 0.5)
		Z6 = load(snap_load);
		Z6 = Z6';
		snap_tcom = size(Z6,2);
		Rs_guess = 1-exp(-Z6*Z6');
		[s_acc,s_rec,s_prec,s_F1_1,s_F1_2,s_auc] = all_statistic(R_true,R_test,Rs_guess,threshold);
		snap_loss = snap_obj(R_true,Z6);
		em_b(4) = extended_modularity(B,Z6,m);
		if(ground_truth_existence > 0.5)
			nmi_b(4) = normalized_mutual_information(Z0,Z6,1);
		end
	end

	for t = 1:T


		t
		% compute A and use it to solve maxcut for Z empty and not empty
		if length(c)==0
			A = setA_dense(zeros(n,1),zeros(1,n),R0_sp,R1_sp);
			z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
			% Z = [Z z];
			% c = [c;0];
			% Zc = Z * diag(sqrt(c));
			% [W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
			% 'maxcut done'
		else
			Zc = Z * diag(sqrt(c));
			[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
			A = setA_dense(Zc,W_o,R0_sp,R1_sp);
			z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
			% Z = [Z z];
			% c = [c;0];
			% Zc = Z * diag(sqrt(c));
			% [W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
			% 'maxcut done'
		end

		Z = [Z z];
		c = [c;0];
		% 'maxcut done'





		
		%fully corrective by prox-GD


		k = length(c);
		h = diag_hessian(Z);

		eta = stepsize/(max(h)*k); %step size

		for t2 = 1:T2

			grad_c = gradient_c_fast(R0_sp,R1_sp,Z,c,mu);

			c = prox( c - eta*grad_c, eta*mu);
			

		end
		'prox done'


		tic;
		%shrink c and Z for j:cj=0
		Z = Z(:,c'>TOL);
		c = c(c>TOL);
		[obj,~,~] = LFLassoObj(R0_sp,R1_sp,lambda,Z,c);
		obj
		%dump info
		if obj > prev_obj
						obj_suc = 0;
						return;
		end
		prev_obj = obj;

		if mod(t,mod_num)==0
				%M = compute_M(c,Z);
				%obj = boolLassoObj(R,lambda,M,c, beta);
				if(ground_truth_existence > 0.5)
					match = zeros(1,length(c));
					for k = 1:size(Z0,2)
							match_k = f(Z,Z0(:,k),n*tol_rate);
							match(match_k>0)=k;
					end
					P = [match;c'];
					P
				end
				[~,ind] = sort(c,'descend');
				xx = -1
				if(noisy < 0.5)
					xx = length(c);
				end
					for K0 = max(min(2,length(c)),xx):1:length(c)
						Z2 = Z(:,ind(1:min(end,K0)));
						c2 = c(ind(1:min(end,K0))); 
						[obj,R_guess,W] = LFLassoObj(R0_sp,R1_sp,lambda,Z2,c2);
						[acc,rec,prec,F1,F1_2,auc] = all_statistic(R_true,R_test,R_guess,threshold);
						if(ground_truth_existence > 0.5 )
							% compute mutual information
							nmi = normalized_mutual_information(Z0,Z2,1);
							% compute extended modularity 
							em = extended_modularity(B,Z2,m);
			        	    if(parameter(1,16)>0.5)
				             	fprintf(2,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_b(4),em,em_b(4),em_b(5));
								fprintf(out,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_b(4),em,em_b(4),em_b(5));
							else
								fprintf(2,'t=%f nmi=%f em=%f groundtruth_em=%f\n',t,nmi,em,em_b(5));
								fprintf(out,'t=%f nmi=%f em=%f groundtruth_em=%f\n',t,nmi,em,em_b(5));
							end
								
							if(nmi>nmi_b(1))
								nmi_b(1) = nmi;
								nmi_b(2) = t;
								nmi_b(3) = K0;
							end
							if(em > em_b(1))
								em_b(1) = em;
								em_b(2) = t;
								em_b(3) = K0;
							end
							[nmi_s,em_s]=update_track(snap_tcom,snap_cell,nmi_s,em_s,nmi,em,t,K0,ground_truth_existence);
						else
							% compute extended modularity 
							em = extended_modularity(B,Z2,m);
							if(parameter(1,16)>0.5)
				              	fprintf('t=%f em=%f snap_em=%f\n',t,em,em_b(4));
								fprintf(out,'t=%f em=%f\n snap_em=%f\n',t,em,em_b(4));
							else
								fprintf('t=%f em=%f\n',t,em);
								fprintf(out,'t=%f em=%f',t,em);
							end							
							if(em > em_b(1))
								em_b(1) = em;
								em_b(2) = t;
								em_b(3) = K0;
							end
							[nmi_s,em_s]=update_track(snap_tcom,snap_cell,nmi_s,em_s,nmi,em,t,K0,ground_truth_existence);
						end		
						%print other statistical information
						lfsnaploss = snap_obj(R_true,Z2);
						if(strcmp(datatype,'Synthetic'))
							if(parameter(1,16)>0.5)
								fprintf(2,'t=%f auc=%f s_auc=%f g_auc=%f obj=%f\n',t,auc,s_auc,g_auc,obj);
								fprintf(out,'t=%f auc=%f s_auc=%f g_auc=%f obj=%f\n',t,auc,s_auc,g_auc,obj);
								fprintf(2,'t=%f lf_snap_loss=%f snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,snap_loss,g_loss);
								fprintf(out,'t=%f lf_snap_loss=%f snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,snap_loss,g_loss);
							else  
								fprintf(2,'t=%f auc=%f g_auc=%f obj=%f\n',t,auc,g_auc,obj);
								fprintf(out,'t=%f auc=%f g_auc=%f obj=%f\n',t,auc,g_auc,obj);
								fprintf(2,'t=%f lf_snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,g_loss);
								fprintf(out,'t=%f lf_snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,g_loss);
							end
						else
							if(parameter(1,16)>0.5)
								if(ground_truth_existence > 0.5)
									fprintf(2,'t=%f auc=%f s_auc=%f obj=%f\n',t,auc,s_auc,obj);
									fprintf(out,'t=%f auc=%f s_auc=%f obj=%f\n',t,auc,s_auc,obj);
									fprintf(2,'t=%f lf_snap_loss=%f snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,snap_loss,g_loss);
									fprintf(out,'t=%f lf_snap_loss=%f snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,snap_loss,g_loss);
								else 
									fprintf(2,'t=%f auc=%f s_auc=%f obj=%f\n',t,auc,s_auc,obj);
									fprintf(out,'t=%f auc=%f s_auc=%f obj=%f\n',t,auc,s_auc,obj);
									fprintf(2,'t=%f lf_snap_loss=%f snap_loss=%f\n',t,lfsnaploss,snap_loss);
									fprintf(out,'t=%f lf_snap_loss=%f snap_loss=%f\n',t,lfsnaploss,snap_loss);
								end
							else
								if(ground_truth_existence > 0.5)
									fprintf(2,'t=%f auc=%f obj=%f\n',t,auc,obj);
									fprintf(out,'t=%f auc=%f obj=%f\n',t,auc,obj);
									fprintf(2,'t=%f lf_snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,g_loss);
									fprintf(out,'t=%f lf_snap_loss=%f groundtruth_loss=%f\n',t,lfsnaploss,g_loss);
								else 
									fprintf(2,'t=%f auc=%f\n',t,auc,s_auc,g_auc,obj);
									fprintf(out,'t=%f auc=%f\n',t,auc,s_auc,g_auc,obj);
									fprintf(2,'t=%f lf_snap_loss=%f\n',t,lfsnaploss);
									fprintf(out,'t=%f lf_snap_loss=%f\n',t,lfsnaploss);
								end
							end
						end 							
						for u=1:length(threshold)
							fprintf(2,'threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
							fprintf(out,'threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
						end	
						if(parameter(1,16)>0.5)
							for u=1:length(threshold)
								fprintf(2,'threshold=%f s_acc=%f s_prec=%f s_rec=%f s_F1_1=%f s_F1_2=%f\n',threshold(u),s_acc(u),s_prec(u),s_rec(u),s_F1_1(u),s_F1_2(u));
								fprintf(out,'threshold=%f s_acc=%f s_prec=%f s_rec=%f s_F1_1=%f s_F1_2=%f\n',threshold(u),s_acc(u),s_prec(u),s_rec(u),s_F1_1(u),s_F1_2(u));
							end	
						end	
						if(parameter(1,16)>0.5)
							fprintf(2,'Best em at K=snap_force=%f occurs at t=%f:%f\n',em_s(6),em_s(5),em_s(4));
							fprintf(2,'Best em at K=snap_true=%f occurs at t=%f:%f\n',em_s(3),em_s(2),em_s(1));
							fprintf(out,'Best em at K=snap_force=%f occurs at t=%f:%f\n',em_s(6),em_s(5),em_s(4));
							fprintf(out,'Best em at K=snap_true=%f occurs at t=%f:%f\n',em_s(3),em_s(2),em_s(1));
							if(ground_truth_existence > 0.5)
								fprintf(2,'Best nmi at K=snap_force=%f occurs at t=%f:%f\n',nmi_s(6),nmi_s(5),nmi_s(4));
								fprintf(2,'Best nmi at K=snap_true=%f occurs at t=%f:%f\n',nmi_s(3),nmi_s(2),nmi_s(1));
								fprintf(out,'Best nmi at K=snap_force=%f occurs at t=%f:%f\n',nmi_s(6),nmi_s(5),nmi_s(4));
								fprintf(out,'Best nmi at K=snap_true=%f occurs at t=%f:%f\n',nmi_s(3),nmi_s(2),nmi_s(1));
							else
							end							
						end						
					end
				
		end

	end
	function myCleanupFun(syn_parameter,parameter,nmi_s,em_s,nmi_b,em_b,snap_cell,snap_tcom,data_cell,outname,outresult,obj_suc)

		fprintf('Executing cleanup function\n');
		fid = outresult;
		if(obj_suc < 0.5)
			fprintf(2,'Objective was increasing. The process had been terminated\n');
			return;
		else
			fprintf(fid,'Process terminated by Ctrl-c, log file:%s\n',outname);
			if(strcmp(data_cell{6},'ZZ^t'))			
				fprintf(fid,'beta = %f\n',z_beta);
				if(strcmp(data_cell{5},'Synthetic'))				
					fprintf(fid,'n = %f\n',syn_parameter(1,1));
					fprintf(fid,'k = %f\n',syn_parameter(1,2));
					fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,3));
				end
			else
				if(strcmp(data_cell{5},'Synthetic'))
					fprintf(fid,'mean = %f\n',syn_parameter(1,1));
					fprintf(fid,'var = %f\n',syn_parameter(1,2));
					fprintf(fid,'tunningI = %f\n',syn_parameter(1,3));
					fprintf(fid,'n = %f\n',syn_parameter(1,4));
					fprintf(fid,'k = %f\n',syn_parameter(1,5));
					fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,6));
				end
			end
			if(parameter(1,16)>0.5)
				fprintf(fid,'snap set communitiy = %f\n',snap_cell{3});
				fprintf(fid,'snap true communitiy = %f\n',snap_tcom);
			end
			fprintf(fid,'ground_truth_existence = %f\n',parameter(1,1));
			fprintf(fid,'noisy = %f\n',parameter(1,2));
			fprintf(fid,'positive_sample_rate = %f\n',parameter(1,3));
			fprintf(fid,'negative_sample_rate = %f\n',parameter(1,4));
			fprintf(fid,'iteration_num = %f\n',parameter(1,5));
			if(strcmp(data_cell{6},'ZZ^t'))
				fprintf(fid,'linear_regularization_lambda = %f\n',parameter(1,6));
			else
				fprintf(fid,'w_solver_lambda = %f\n',parameter(1,6));
			end	
			fprintf(fid,'c_TOL= %f\n',parameter(1,7));
			fprintf(fid,'match_tol_rate = %f\n',parameter(1,8));
			fprintf(fid,'inner_iteration_num = %f\n',parameter(1,9));
			fprintf(fid,'SDP_rank = %f\n',parameter(1,10));
			fprintf(fid,'SDP_iter = %f\n',parameter(1,11));
			fprintf(fid,'mu_multiple_edge = %f\n',parameter(1,12));
			fprintf(fid,'stepsize = %f\n',parameter(1,13));
			fprintf(fid,'mod = %f\n',parameter(1,14));	
			fprintf(fid,'train = %f\n',parameter(1,15));
			fprintf(fid,'snap_run = %f\n',parameter(1,16));	
			fprintf(fid,'threshold_prob_for_link_start = %f\n',parameter(1,17));	
			fprintf(fid,'threshold_prob_for_link_space = %f\n',parameter(1,18));
			fprintf(fid,'threshold_prob_for_link_end = %f\n',parameter(1,19));
			if(parameter(1,16)>0.5)
				fprintf(fid,'Best em occurs at t=%f k=%f value=%f\n',em_b(2),em_b(3),em_b(1));
				fprintf(fid,'Best em at K=snap_force=%f occurs at t=%f:%f\n',em_s(6),em_s(5),em_s(4));
				fprintf(fid,'Best em at K=snap_true=%f occurs at t=%f:%f\n',em_s(3),em_s(2),em_s(1));
				fprintf(fid,'SNAP_em:%f\n',em_b(4));
				if(parameter(1,1)>0.5)
					fprintf(fid,'ground_truth_em:%f\n',em_b(5));
					fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f\n',nmi_b(2),nmi_b(3),nmi_b(1));
					fprintf(fid,'Best nmi at K=snap_force=%f occurs at t=%f:%f\n',nmi_s(6),nmi_s(5),nmi_s(4));
					fprintf(fid,'Best nmi at K=snap_true=%f occurs at t=%f:%f\n',nmi_s(3),nmi_s(2),nmi_s(1));
				end	
			else
				fprintf(fid,'Best em occurs at t=%f k=%f value=%f\n',em_b(2),em_b(3),em_b(1));
				if(parameter(1,1)>0.5)
					fprintf(fid,'ground_truth_em:%f\n',em_b(5));
					fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f\n',nmi_b(2),nmi_b(3),nmi_b(1));
				end				
			end	
		end	
	end
end

function M = compute_M(c,Z)
	
	[n,k] = size(Z);
	M = zeros(n,n);
	for j = 1:k
		M = M + c(j)*Z(:,j)*Z(:,j)';
	end
end


function grad_M = gradient_M(R,M, beta)
	
	grad_M = -R .* beta .* exp(-beta*M) ./ (1-exp(-beta*M)+1e-2) + (1-R)*beta;
end


function grad_c = gradient_c(R,Z,c, beta)
	
	k = length(c);
	grad_c = zeros(k,1);
	%for j=1:k
	%	ZTzj = Z'*Z(:,j); %k by 1
	%	grad_c(j) = -Z(:,j)'*R*Z(:,j) + 0.5*c'*(ZTzj.^2) + 0.5*c(j)*ZTzj(j).^2;
	%end
	grad_M = gradient_M(R,compute_M(c,Z), beta);
	for j=1:k
		grad_c(j) = Z(:,j)'*grad_M*Z(:,j);
	end
end

function grad_c = gradient_c_fast(R0_sp,R1_sp,Z,c,lambda)

	k = length(c);
	grad_c = zeros(k,1);
	Zc = Z * diag(sqrt(c));
	[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),size(R0_sp,1)),lambda);
	A = setA_dense(Zc,W_o,R0_sp,R1_sp);

	for j=1:k
		grad_c(j) = -Z(:,j)'*A*A'*Z(:,j);
	end
end

function h = diag_hessian(Z)
	k = size(Z,2);
	h = zeros(k,1);
	for i = 1:k
		h(i) = (Z(:,i)'*Z(:,i)).^2;
	end
end

function c2 = prox( c, lambda )
	
	c2 = c;
	c2(c<=lambda)=0;
	c2(c>lambda) = c(c>lambda)-lambda;
end
function A = setA_dense(Z,W,R0_sp,R1_sp)
	% -grad_L(ZW)

	[I1,J1,V1] = find(R0_sp);
	ind1 = find(R0_sp);
	E_1 = sum(Z(I1,:).*(W(:,J1)'),2);
	temp1 = 1./(1+exp(-E_1));
	[I2,J2,V2] = find(R1_sp);
	ind2 = find(R1_sp);
	E_2 = sum(Z(I2,:).*(W(:,J2)'),2);
	temp2 = -exp(-E_2)./(1+exp(-E_2));
	n = size(Z,1);
	S1 = sparse(I1,J1,temp1,n,n);
	S2 = sparse(I2,J2,temp2,n,n);
	A = -S1-S2;
	% E = Z*W;
	% R = full(R1_sp);
	% A = -(-R.*exp(-E)+1-R)./(1+exp(-E));
end
function [obj,R_guess,W_o] = LFLassoObj(R0_sp,R1_sp,lambda,Z,c)
	Zc = Z * diag(sqrt(c));
	[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Z,2),size(R0_sp,1)),lambda);
	E = Zc*W_o;
	R_guess = 1 ./ (1+exp(-Zc*W_o));
	R = full(R1_sp);
	obj = sum(sum(-R.*log(1./(1+exp(-E)))-(1-R).*log(1-1./(1+exp(-E)))))+0.5*lambda*norm(W_o,'fro')^2;
end

function [sobj] = snap_obj(R_true,Z)
	sobj=0;
	[I1,J1,V1] = find(R_true);
	E_1 = sum(Z(I1,:).*Z(J1,:),2);
	E_1 = 1-exp(-E_1);
	E_1 = max(E_1, 1e-10);
	E_1 = sum(log(E_1));
	[I1,J1,V1] = find(1-R_true);
	E_2 = sum(sum(Z(I1,:).*Z(J1,:),2));
	sobj = E_2-E_1;
end
function [nmi_s,em_s]=update_track(snap_tcom,snap_cell,nmi_s,em_s,nmi,em,t,K0,ground_truth_existence)
	if(snap_tcom < 0)
		return;
	end
	if K0 == snap_tcom 
		if (ground_truth_existence > 0.5)
			if(nmi>nmi_s(1))
					nmi_s(1) = nmi;
					nmi_s(2) = t;
					nmi_s(3) = K0;
			end
			if(em > em_s(1))
				em_s(1) = em;
				em_s(2) = t;
				em_s(3) = K0;
			end
		else
			if(em > em_s(1))
				em_s(1) = em;
				em_s(2) = t;
				em_s(3) = K0;
			end
		end
	end
	if K0 == snap_cell{3}
		if (ground_truth_existence > 0.5)
			if(nmi>nmi_s(4))
					nmi_s(4) = nmi;
					nmi_s(5) = t;
					nmi_s(6) = K0;
			end
			if(em > em_s(4))
				em_s(4) = em;
				em_s(5) = t;
				em_s(6) = K0;
			end
		else
			if(em > em_s(4))
				em_s(4) = em;
				em_s(5) = t;
				em_s(6) = K0;
			end
		end
	end
end
