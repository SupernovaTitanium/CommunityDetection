%  use it as a general interface to run communitiy detection test data
%  usage  general_script path_to_setting_file
function [] = general_script(set_file_path)
	fprintf(2,'Reading setting file from %s\n',set_file_path);	
	fid = fopen(set_file_path,'r');
	assert(fid > 0,'Open file error\n');
	parameter = zeros(1,15);
	syn_parameter = zeros(1,6);
	tline = fgetl(fid);
	syn = 0;
	data_path ='/'; 
	data_type ='Synthetic';
	cluster_path = 'none';
	model_type = 'none';
	snap_path = 'none';
	snap_out = 'none';
	snap_community = 0;
	z_beta = 0;
	if(strcmp(tline,'Synthetic'));
		syn= 1;
	elseif(strcmp(tline,'Real'));
		syn= 0;
	else
		assert(1 < 0,'Unknown data type\n');
	end
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	model_type = C{3};
	if(strcmp(model_type,'ZZ^t') & (syn > 0))
		syn = 2;
	end
	if(strcmp(model_type,'ZZ^t'))
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		z_beta = str2num(C{3});
	end
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_path = C{3};
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_out = C{3};
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_community = str2num(C{3});
	if(syn > 0.5 & syn < 1.5)
		for i = 1:6
			tline = fgetl(fid);
			fprintf(2,'Try to set %s\n',tline);
			C = strsplit(tline,' ');
			syn_parameter(1,i) = str2num(C{3});
		end
	end
	if(syn > 1.5)
		for i = 1:3
			tline = fgetl(fid);
			fprintf(2,'Try to set %s\n',tline);
			C = strsplit(tline,' ');
			syn_parameter(1,i) = str2num(C{3});
		end
	end
	if(syn < 0.5 )
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		cluster_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_type = C{3};	
	end
	tline = fgetl(fid);
	i=1;
	while ischar(tline)
		fprintf(2,'Try to set %s\n',tline);
		C = strsplit(tline,' ');
		parameter(1,i) = str2num(C{3}); 
		i=i+1;
		tline = fgetl(fid);	
	end
	% fix random number
	rand('seed',1);
	randn('seed',1);
	% perform unit_test
	% unittest(syn,model_type,z_beta,snap_path,snap_out,snap_community,data_type,data_path,cluster_path,syn_parameter,parameter);
	fprintf(2,'Converting data to adjancy matrix')
	if(syn < 1.5 & syn > 0.5)
		n = syn_parameter(1,4)
		k = syn_parameter(1,5)
		Z_true = zeros(n,k);
		% use Z_sparsity to set binary matrix Z_true
		nnz_row = ceil(syn_parameter(1,6)*k);
		for i=1:n
			rand1 = randperm(k);
			Z_true(i,rand1(1:nnz_row)) = 1;
		end
		I = eye(k);
		M = normrnd(syn_parameter(1,1),syn_parameter(1,2),k,k);
		V = syn_parameter(1,3)*I + (M+M')/2;
		W_true = V*Z_true';
		S = Z_true*W_true;
		R = 1 ./ (1+exp(-S));
		%mean(mean(R))
		M2 = rand(n,n);
		Rb = double( (M2+M2')/2 < R);
		fprintf(2,'write the data to Syn_data for snap usage');
		C = strsplit(set_file_path,'/');
		if(7 ~= exist('Syn_data','dir'))
			mkdir('Syn_data');
		end
		snap_data = strcat('Syn_data/',C{2},'_snap.txt');
		fid5 = fopen(snap_data,'w+');
		assert(fid5>0,'Open file error\n');
		for i=1:n
			for j=1:n
				if(Rb(i,j) > 0.9 )
					fprintf(fid5,'%d\t%d\n',i,j);
				end
			end
		end
		fclose(fid5);		
		%nnz(Rb)
		positive_edge = sum(sum(Rb))
		parameter(1,13) = positive_edge*parameter(1,13); 
		negative_edge = n*n-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4)
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = min(sample_negative_edge/negative_edge,1);
		Sp = double(rand(n,n) > 1-positive_ratio);
		Sn = double(rand(n,n) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(n,n);
		%save('tmp1');
		%exit(0);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd(data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',model_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Running bigclam first\n');
		snapload = run_snap(snap_data,snap_path,snap_out,snap_community,n);
		fprintf(2,'Start running algorithm LFlasso_ZW\n');
		[nmi_g,em_g,Z_guess,W_guess,snap_tcom] = LFlasso_ZW(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out,snapload);
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,model_type,parameter,syn_parameter,syn,nmi_g,em_g,snap_tcom,z_beta,snap_community);
		fclose(out);
		fclose(outresult);
	end
	if(syn > 1.5)
		n = syn_parameter(1,1)
		k = syn_parameter(1,2)
		Z_true = zeros(n,k);
		% use Z_sparsity to set binary matrix Z_true
		nnz_row = ceil(syn_parameter(1,3)*k);
		for i=1:n
			rand1 = randperm(k);
			Z_true(i,rand1(1:nnz_row)) = 1;
		end
		M_true = Z_true*Z_true';
		L = 1-exp(-z_beta*M_true);
       	t1 = triu(rand(n),1);
       	t2 = triu(rand(n),0);
       	rand_mat = t1+t1'; 
       	Rb = double(rand_mat < L);	
		%nnz(Rb) It actually suggest self links
		fprintf(2,'write the data to Syn_data for snap usage');
		C = strsplit(set_file_path,'/');
		if(7 ~= exist('Syn_data','dir'))
			mkdir('Syn_data');
		end
		snap_data = strcat('Syn_data/',C{2},'_snap.txt');
		fid5 = fopen(snap_data,'w+');
		assert(fid5>0,'Open file error\n');
		for i=1:n
			for j=1:n
				if(Rb(i,j) > 0.9)
					fprintf(fid5,'%d\t%d\n',i,j);
				end
			end
		end
		fclose(fid5);	
		positive_edge = sum(sum(Rb))
		parameter(1,13) = positive_edge*parameter(1,13); 
		negative_edge = n*n-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4)
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = min(sample_negative_edge/negative_edge,1);
		Sp = double(rand(n,n) > 1-positive_ratio);
		Sn = double(rand(n,n) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(n,n);
		%save('tmp1');
		%exit(0);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd(data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',model_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Running bigclam first\n');
		snapload = run_snap(snap_data,snap_path,snap_out,snap_community,n);
		fprintf(2,'Start running algorithm LFlasso_ZZt\n');
		[nmi_g,em_g,Z,snap_tcom] = LFlasso_ZZt(Rb,R1,Sr,parameter,z_beta,Z_true,data_type,1,out,snapload);
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,model_type,parameter,syn_parameter,syn,nmi_g,em_g,snap_tcom,z_beta,snap_community);
		fclose(out);
		fclose(outresult);
	end
	if(strcmp(data_type,'email_EU'))
		%notice: index start from zero
		node = 1005
		edge_num = 25571
		com_num = 42;
		data_tmp = load(data_path);
		data_tmp = data_tmp + 1;
		v1 = ones(edge_num,1);
		Rb = sparse(data_tmp(:,1),data_tmp(:,2),v1,node,node);	
		Rb1 = full(Rb);	
		C = strsplit(data_path,'.');
		snap_data = strcat(C{3},'_snap.txt');
		if exist(snap_data, 'file') ~= 2
			fprintf(2,'write the data to %s for snap usage\n',snap_data);
			fid5 = fopen(snap_data,'w+');
			assert(fid5>0,'Open file error\n');
			for i=1:node
				for j=1:node
					if(Rb1(i,j) > 0.9)
						fprintf(fid5,'%d\t%d\n',i,j);
					end
				end
			end
			fclose(fid5);
		end	
		Z_true = zeros(node,42);
		cluster_tmp = load(cluster_path);
		cluster_tmp = cluster_tmp + 1;
		for i=1:node
			Z_true(cluster_tmp(i,1),cluster_tmp(i,2)) = 1;
		end
		positive_edge = sum(sum(Rb))
		parameter(1,13) = positive_edge*parameter(1,13); 
		negative_edge = node*node-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4) 
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = sample_negative_edge/negative_edge;
		Sp = double(rand(node,node) > 1-positive_ratio);
		Sn = double(rand(node,node) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(node,node);

		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd(data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',model_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Running bigclam first\n');
		snapload = run_snap(snap_data,snap_path,snap_out,snap_community,node);
		if(strcmp(model_type,'ZW'))
			W_true = normrnd(0,1,42,node);
			fprintf(2,'Start running algorithm LFlasso_ZW\n');
			[nmi_g,em_g,Z_guess,W_guess,snap_tcom] = LFlasso_ZW(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out,snapload);
		else
			fprintf(2,'Start running algorithm LFlasso_ZZt\n');
		    [nmi_g,em_g,Z,snap_tcom] = LFlasso_ZZt(Rb,R1,Sr,parameter,z_beta,Z_true,data_type,1,out,snapload);
		end			
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,model_type,parameter,syn_parameter,syn,nmi_g,em_g,snap_tcom,z_beta,snap_community);
		fclose(out);
		fclose(outresult);
	end
	if(strcmp(data_type,'email'))
		%notice: index start from 1 undirected (1133,1133)?
		node = 1133
		edge_num = 10903
		data_tmp = load(data_path);
		Rb = sparse(data_tmp(:,1),data_tmp(:,2),data_tmp(:,3),node,node);
		Rb1 = full(Rb);	
		C = strsplit(data_path,'.');
		snap_data = strcat(C{3},'_snap.txt');
		if exist(snap_data, 'file') ~= 2
			fprintf(2,'write the data to %s for snap usage\n',snap_data);
			fid5 = fopen(snap_data,'w+');
			assert(fid5>0,'Open file error\n');
			for i=1:node
				for j=1:node
					if(Rb1(i,j) > 0.9)
						fprintf(fid5,'%d\t%d\n',i,j);
					end
				end
			end
			fclose(fid5);
		end	
		positive_edge = sum(sum(Rb))		
		negative_edge = node*node-sum(sum(Rb))
		parameter(1,13) = positive_edge*parameter(1,13); 
		sample_positive_edge = positive_edge*parameter(1,3) 
		sample_negative_edge = positive_edge*parameter(1,4) 
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = sample_negative_edge/negative_edge;
		Sp = double(rand(node,node) > 1-positive_ratio);
		Sn = double(rand(node,node) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(node,node);

		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd (data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',model_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Running bigclam first\n');
		snapload = run_snap(snap_data,snap_path,snap_out,snap_community,node);
		Z_true = double(rand(node,1)>0.8);
		if(strcmp(model_type,'ZW'))
			W_true = normrnd(0,1,42,node);
			fprintf(2,'Start running algorithm LFlasso_ZW\n');
			[nmi_g,em_g,Z_guess,W_guess,snap_tcom] = LFlasso_ZW(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out,snapload);
		else
			fprintf(2,'Start running algorithm LFlasso_ZZt\n');
		    [nmi_g,em_g,Z,snap_tcom] = LFlasso_ZZt(Rb,R1,Sr,parameter,z_beta,Z_true,data_type,1,out,snapload);
		end			
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,model_type,parameter,syn_parameter,syn,nmi_g,em_g,snap_tcom,z_beta,snap_community);
		fclose(out);
		fclose(outresult);
	end
end

function [] = write_to_result(fid,model_type,parameter,syn_parameter,syn,nmi_g,em_g,snap_tcom,z_beta,snap_community)
	if(syn < 1.5 & syn > 0.5)
		fprintf(fid,'mean = %f\n',syn_parameter(1,1));
		fprintf(fid,'var = %f\n',syn_parameter(1,2));
		fprintf(fid,'tunningI = %f\n',syn_parameter(1,3));
		fprintf(fid,'n = %f\n',syn_parameter(1,4));
		fprintf(fid,'k = %f\n',syn_parameter(1,5));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,6));
	end
	if(syn > 1.5)
		fprintf(fid,'beta = %f\n',z_beta);
		fprintf(fid,'n = %f\n',syn_parameter(1,1));
		fprintf(fid,'k = %f\n',syn_parameter(1,2));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,3));
	end
	fprintf(fid,'snap set communitiy = %f\n',snap_community);
	fprintf(fid,'ground_truth_existence = %f\n',parameter(1,1));
	fprintf(fid,'noisy = %f\n',parameter(1,2));
	fprintf(fid,'positive_sample_rate = %f\n',parameter(1,3));
	fprintf(fid,'negative_sample_rate = %f\n',parameter(1,4));
	fprintf(fid,'threshold_prob_for_link = %f\n',parameter(1,5));
	fprintf(fid,'iteration_num = %f\n',parameter(1,6));
	if(strcmp(model_type,'ZZ^t'))
		fprintf(fid,'linear_regularization_lambda = %f\n',parameter(1,7));
	else
		fprintf(fid,'w_solver_lambda = %f\n',parameter(1,7));
	end	
	fprintf(fid,'c_TOL= %f\n',parameter(1,8));
	fprintf(fid,'match_tol_rate = %f\n',parameter(1,9));
	fprintf(fid,'inner_iteration_num = %f\n',parameter(1,10));
	fprintf(fid,'SDP_rank = %f\n',parameter(1,11));
	fprintf(fid,'SDP_iter = %f\n',parameter(1,12));
	fprintf(fid,'mu = %f\n',parameter(1,13));
	fprintf(fid,'stepsize = %f\n',parameter(1,14));
	fprintf(fid,'mod = %f\n',parameter(1,15));	
	if(parameter(1,1) > 0)
		fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f snap_k=%f snap_nmi=%f\n',nmi_g(2),nmi_g(3),nmi_g(1),snap_tcom,nmi_g(4));
		fprintf(fid,'Best em occurs at t=%f k=%f value=%f and snap_k=%f snap_em=%f ground_truth_em=%f\n',em_g(2),em_g(3),em_g(1),snap_tcom,em_g(4),em_g(5));
	else
		fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f snap_k=%f snap_nmi=%f\n',nmi_g(2),nmi_g(3),nmi_g(1),snap_tcom,em_g(5));
		fprintf(fid,'Best em occurs at t=%f k=%f value=%f  snap_k=%f snap_nmi=%f\n',em_g(2),em_g(3),em_g(1),snap_tcom,em_g(4));
	end		
end

function []= unittest(syn,model_type,z_beta,snap_path,snap_out,snap_community,data_type,data_path,cluster_path,syn_parameter,parameter)
	fid=2;
	fprintf(fid,'data_type = %s\n',data_type);
	fprintf(fid,'model_type = %s\n',model_type);
	if(strcmp(model_type,'ZZ^t'))
		fprintf(fid,'beta = %f\n',z_beta);
	end
	fprintf(fid,'snap_path = %s\n',snap_path);
	fprintf(fid,'snap_out = %s\n',snap_out);
	fprintf(fid,'snap_community = %f\n',snap_community);
	if(syn < 1.5 & syn > 0.5)
		fprintf(fid,'mean = %f\n',syn_parameter(1,1));
		fprintf(fid,'var = %f\n',syn_parameter(1,2));
		fprintf(fid,'tunningI = %f\n',syn_parameter(1,3));
		fprintf(fid,'n = %f\n',syn_parameter(1,4));
		fprintf(fid,'k = %f\n',syn_parameter(1,5));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,6));
	end
	if(syn > 1.5)
		fprintf(fid,'n = %f\n',syn_parameter(1,1));
		fprintf(fid,'k = %f\n',syn_parameter(1,2));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,3));
	end
	if(syn < 0.5)
		fprintf(fid,'data_path = %s\n',data_path);
		fprintf(fid,'cluster_path = %s\n',cluster_path);
	end	
	fprintf(fid,'ground_truth_existence = %f\n',parameter(1,1));
	fprintf(fid,'noisy = %f\n',parameter(1,2));
	fprintf(fid,'positive_sample_rate = %f\n',parameter(1,3));
	fprintf(fid,'negative_sample_rate = %f\n',parameter(1,4));
	fprintf(fid,'threshold_prob_for_link = %f\n',parameter(1,5));
	fprintf(fid,'iteration_num = %f\n',parameter(1,6));
	if(strcmp(model_type,'ZZ^t'))
		fprintf(fid,'linear_regularization_lambda = %f\n',parameter(1,7));
	else
		fprintf(fid,'w_solver_lambda = %f\n',parameter(1,7));
	end	
	fprintf(fid,'c_TOL= %f\n',parameter(1,8));
	fprintf(fid,'match_tol_rate = %f\n',parameter(1,9));
	fprintf(fid,'inner_iteration_num = %f\n',parameter(1,10));
	fprintf(fid,'SDP_rank = %f\n',parameter(1,11));
	fprintf(fid,'SDP_iter = %f\n',parameter(1,12));
	fprintf(fid,'mu_multiple  = %f\n',parameter(1,13));
	fprintf(fid,'stepsize = %f\n',parameter(1,14));
	fprintf(fid,'mod = %f\n',parameter(1,15));	
end
function [snapload] = run_snap(snap_data,snap_path,snap_out,snap_community,node)
	cur_work_path = pwd;
	snap_data = strcat(cur_work_path,'/',snap_data);
	fprintf(2,'Move to snap directory %s\n',snap_path);
	cd(snap_path);
	if (snap_community > 0.5 )
		operation = strcat('./bigclam -o:',snap_out,' -i:',snap_data,' -c:',num2str(snap_community));
		fprintf(2,'Running operation %s\n',operation);
		system(operation);
	else
		operation = strcat('./bigclam -o:',snap_out,' -i:',snap_data);
		fprintf(2,'Running operation %s\n',operation);
		system(operation);
	end
	fprintf('Convert snap result\n');
	% convert snapresult to directly readible data by python
	snap_out = strcat(snap_out,'cmtyvv.txt');
	operation = strcat('python cmt_to_Z.py',{' '},snap_out,{' '},num2str(node));
	operation = operation{1};
	fprintf(2,'Running operation %s\n',operation);
	system(operation);
	snapload = strcat(snap_path,'/',snap_out,'.Z');
	cd(cur_work_path);
end