clear;
L = 1; % number of Monte Carlo repetitions
N = 100; % number of nodes in the network
R = 1000; % number of loops for simulating networks
%% DGP parameters 
beta = [-3,1,1.0,-1.0];
sig2 = 0.5; % individual random effects
ltnt = 0; % 1 if including latent individual effects
%% 
Wn = cell(L,1); % network
Xn = cell(L,1);
Zn = cell(L,1);

for s = 1:L
    disp(s);
    X = randn(N,1);
    %X = exp(X); % X is lognormal
    Z = sig2*randn(N,1);
    
    H = beta(1)+beta(2)*(X-X');
    if ltnt == 1
        H = H+Z+Z';
    end
    W = double(H>0);
    
    for r = 1:R  % simulating network using an MH algorithm
        for i = 1:N
            for j = 1:N
                if i ~= j
                    pw = H(i,j)+beta(3)*W(j,i)+beta(4)*W(j,:)*W(:,i);
                    pw = ((-1)^W(i,j))*pw;
                    if log(rand(1)) <= pw
                        W(i,j) = 1-W(i,j);
                    end
                end
            end
        end
    end
    disp([sum(sum(W)) max(sum(W,1)) max(sum(W,2))]);
    Wn{s} = W;
    Xn{s} = X;
    Zn{s} = Z;
end
save ERGM_data.mat Wn Xn Zn ltnt