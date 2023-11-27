clear;
load ERGM_data.mat;

L = length(Wn);     % number of Monte Carlo repetition
N = size(Wn{1},1);   % number of nodes in network
T = 10000; % length of MCMC
R = 2;     % number of loops in simulating network
M = 20;    % size of block to update latent variable
K = ceil(N/M);

for s = 1:L % Monte Carlo repetitions
    W = Wn{s};
    X = Xn{s};
    Z = Zn{s};
    
    % jumping rate in proposal distributions
    c0 = 1e-2;
    c1 = 1e-6;
    acc0 = 0;
    acc1 = 0;
    acc_rate0 = zeros(T,1);
    acc_rate1 = zeros(T,1);
    
    beta = zeros(T,4);
    sig2 = zeros(T,1);
    nk = size(beta,2);
    
    % initial values to start MCMC 
    beta(1,:) = [-4.5 0.9 0.9 -0.9];
    sig2(1) = 1;
    z1 = ones(N,1); % individual random effects
    
    % hyper parameters
    kappa = 10;
    alpha = 2;
        
    for t = 2:T % start MCMC
        disp(t);
        tic;
        
        beta0 = beta(t-1,:);
        H = beta0(1)+beta0(2)*(X-X');
        
        if ltnt == 1 % DMH algorithm to update latent variables
            acc_0v = 0;
            for k = 1:K
                z0 = z1;
                if k == K
                    z1((k-1)*M+1:N) = z0((k-1)*M+1:N)+sqrt(c0)*randn(N-(k-1)*M,1);
                else
                    z1((k-1)*M+1:k*M) = z0((k-1)*M+1:k*M)+sqrt(c0)*randn(M,1);
                end
                
                W1 = W;
                for r = 1:R  % start to simulate auxiliary networks
                    for i = 1:N
                        for j = 1:N
                            if i~=j
                                pw = H(i,j)+z1(i)+z1(j)+beta0(3)*W1(j,i)+beta0(4)*W1(j,:)*W1(:,i);
                                pw = ((-1)^W1(i,j))*pw;
                                if log(rand(1)) <= pw
                                    W1(i,j) = 1-W1(i,j);
                                end
                            end
                        end
                    end
                end
                
                if abs(sum(sum(W1))-sum(sum(W))) <= 60
                    dz = z1-z0;
                    pp = sum(sum(W.*(dz+dz')))-sum(sum(W1.*(dz+dz')));
                    if (k==K)
                        pp = pp+log(mvnpdf(z1((k-1)*M+1:N),zeros(N-(k-1)*M,1),sig2(t-1)*eye(N-(k-1)*M)))-log(mvnpdf(z0((k-1)*M+1:N),zeros(N-(k-1)*M,1),sig2(t-1)*eye(N-(k-1)*M)));
                    else
                        pp = pp+log(mvnpdf(z1((k-1)*M+1:k*M),zeros(M,1),sig2(t-1)*eye(M)))-log(mvnpdf(z0((k-1)*M+1:k*M),zeros(M,1),sig2(t-1)*eye(M)));
                    end
                    
                    if log(rand(1)) <= pp
                        z0 = z1;
                        acc_0v = acc_0v+1;
                    end
                end
            end
            if acc_0v >= 2
                acc0 = acc0+1;
            end
            acc_rate0(t) = acc0/t;
            
            if acc_rate0(t)<0.2 && c0>=1e-10
                c0 = c0/1.01;
            elseif acc_rate0(t)>0.3 && c0<=1.0
                c0 = c0*1.01;
            end
            % update hyper parameter
            sig2(t) = (1/chi2rnd(alpha+N))*((z1-mean(z1))'*(z1-mean(z1))+kappa);
        end
        
        % propose beta by Adaptive M-H (Haario, H., Saksman, E., Tamminen, J.: An adaptive Metropolis algorithm. Bernoulli 7(2), 223-242 (2001))
        if t < 500
            beta1 = mvnrnd(beta0,c1*eye(nk));
        else
            beta1 = mvnrnd(beta0,cov(beta(1:t-1,:))*2.38^2/nk)*0.6+mvnrnd(beta0,c1*eye(nk))*0.4;
        end
        
        H = beta1(1)+beta1(2)*(X-X');
        if ltnt == 1
            H = H+z1+z1';
        end
        W1 = W;
        for r = 1:R  % start to simulate auxiliary networks
            for i = 1:N
                for j = 1:N
                    if i~=j
                        pw = H(i,j)+beta0(3)*W1(j,i)+beta0(4)*W1(j,:)*W1(:,i);
                        pw = ((-1)^W1(i,j))*pw;
                        if log(rand(1)) <= pw
                            W1(i,j) = 1-W1(i,j);
                        end
                    end
                end
            end
        end
        
        if abs(sum(sum(W1))-sum(sum(W))) > 60  % condition to reject the generated network
            beta(t,:) = beta0;
        else
            ZZ1 = [sum(sum(W1)),sum(sum(W1.*(X-X'))),sum(sum(W1.*W1'))/2,sum(sum(W1.*(W1*W1)'))/3];
            ZZ0 = [sum(sum(W)), sum(sum(W.*(X-X'))), sum(sum(W.*W'))/2,  sum(sum(W.*(W*W)'))/3];
            pp  = (ZZ0-ZZ1)*(beta1-beta0)'+log(mvnpdf(beta1,zeros(1,nk),100*eye(nk)))-log(mvnpdf(beta0,zeros(1,nk),100*eye(nk)));
            
            if log(rand(1)) <= pp
                beta(t,:) = beta1;
                acc1 = acc1+1;
            else
                beta(t,:) = beta0;
            end
        end
        
        acc_rate1(t) = acc1/t;
        time = toc;
        
        fprintf('time = %5.3f seconds\n',time);
        fprintf('beta = %5.3f %5.3f %5.3f %5.3f\n',beta(t,:));
        fprintf('sig2 = %5.3f\n',sig2(t));
        fprintf('c0 = %10.7f\n',c0);
        fprintf('acc_rate0 = %5.3f\n',acc_rate0(t));
        fprintf('acc_rate1 = %5.3f\n',acc_rate1(t));
        fprintf('\n');
    end
end