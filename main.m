clear all; close all;

param.M = 16;
param.N = 32;
param.K=3;
param.d = 0.5;
dic_range = [-45:1:45].';
param.cont_ang = [-45:0.01:45].';

param_range = [0:10:50].';
rmse_SDP = zeros(length(param_range),1);
rmse_proposed = zeros(length(param_range),1);

SNR_dB = 30;
iter_num=1e2;
MC_num = 100;

G = randn(param.N,param.M)+1j*randn(param.N,param.M);
psi = 10+rand(1,1);
theta = [-30, -10, 25].'+rand(param.K,1);
param.vec = @(MAT) MAT(:);
param.vecH = @(MAT) MAT(:).';
param.get_steer = @(theta, L) exp(1j*2*pi*[0:1:L-1].'*param.d*param.vecH(sind(theta)));

target_sig = exp(1j*rand(param.K,1)*2*pi);
target_q = 10*exp(1j*rand(1,1)*2*pi);
c = G*param.get_steer(psi, param.M);
noise = randn(param.N,1)+1j*randn(param.N,1);
r1 = G*param.get_steer(theta, param.M)*target_sig;
r2 = c*target_q;
alpha = sqrt(norm(r1)^2/db2pow(SNR_dB)/norm(noise)^2);
noise = alpha*noise;

r = r1+r2+noise;
rho = 10*sqrt(norm(noise)^2/length(noise))*sqrt(param.M*log(param.M));
tau = 0.5*rho; 

est_x=zeros(param.M, 1); 
cvx_begin sdp quiet
    variable est_x(param.M, 1) complex
    variable est_u(param.M, 1) complex
    variable est_q complex
    variable est_Z(param.M,param.M) hermitian toeplitz
    variable est_t
    minimize(quad_form(r-G*est_x-c*est_q,eye(length(r)))+real(rho/2*(est_t+est_u(1))))
    subject to
        [est_Z, est_x; est_x', est_t]>=0
        est_Z(:, 1) == est_u
cvx_end

est_sp = MUSIConesnapshot(est_x, param);
est_sp = est_sp/max(est_sp);
est_sp = pow2db(est_sp);

L = @(x,q,u,t,z,Y,P,w) norm(r-z)^2+rho/2*(u(1)+t)+real(trace((Y-[toeplitz(conj(u)), x; x', t])'*P))...
    +real(w'*(G*x+c*q-z))+tau*norm(Y-[toeplitz(conj(u)), x; x', t],'fro')^2 ...
    +tau*norm(G*x+c*q-z)^2;

q = 0;
x = zeros(param.M,1);
u = zeros(param.M,1);
u(1) = real(u(1));
t = 0;
z = G*x+c*q;

P = zeros(param.M+1,param.M+1);
P1 = P(1:param.M, 1:param.M);
p2 = P(1:param.M, param.M+1);
p3 = P(param.M+1,param.M+1);

Y = [toeplitz(conj(u)), x; x', t];
Y1 = Y(1:param.M, 1:param.M);
y2 = Y(1:param.M, param.M+1);
y3 = Y(param.M+1,param.M+1);

w = zeros(param.N,1);

L_value = zeros(iter_num,1);
obj = zeros(iter_num,1);

inv_mat = inv(tau*G'*G+2*tau*eye(param.M));

ZZ = [toeplitz(conj(u)), x; x', t];
for idx = 1:iter_num

    q = 1/(tau*norm(c)^2)*c'*(tau*z-0.5*w-tau*G*x);
    x = inv_mat*(2*tau*y2+p2-G'*(0.5*w+tau*q*c-tau*z));
    
    u=zeros(param.M,1);
    for idx_tmp = 0:param.M-1
        if idx_tmp==0
            em = 1;
        else
            em = 0;
        end
        u(idx_tmp+1) = 1/(2*tau*(param.M-idx_tmp))*(sum(diag(P1,-idx_tmp))+2*tau*sum(diag(Y1,-idx_tmp))-rho*em-2*tau*sum(diag(Y1,-idx_tmp))*em);
    end
    u(1) = real(u(1)); 
   
    t = real(y3+p3/(2*tau)-rho/(4*tau));
    
    ZZ = [toeplitz(conj(u)), x; x', t];
    
    z = 1/(1+tau)*(r+0.5*w+tau*(G*x+c*q));
    
    [U,Lambda] = eig(ZZ-1/(2*tau)*P); 
    lambda = real(diag(Lambda));
    lambda(lambda<0) = 0;
    Lambda = diag(lambda); 
    Y = U*Lambda*U';  
    Y1 = Y(1:param.M, 1:param.M);
    y2 = Y(1:param.M, param.M+1);
    y3 = real(Y(param.M+1,param.M+1));
    Y(param.M+1,param.M+1) = y3;
    
    
     
    P = P+rho*(Y-ZZ);
    P1 = P(1:param.M, 1:param.M);
    p2 = P(1:param.M, param.M+1);
    p3 = real(P(param.M+1,param.M+1));
    P(param.M+1,param.M+1) = p3;
    
    w = w+0.5*rho*(G*x+c*q-z);
    
    L_value(idx) = real(L(x,q,u,t,z,Y,P,w));
    obj(idx) = real(norm(r-z)^2+rho/2*(u(1)+t))+tau*norm(Y-ZZ,'fro')^2 +tau*norm(G*x+c*q-z)^2;
end

figure; plot(param.cont_ang, est_sp, 'LineWidth', 2);
hold on; stem(theta, zeros(param.K,1), 'BaseValue', min(est_sp)-10);
hold on; stem(psi, zeros(1,1),'x', 'BaseValue', min(est_sp)-10);

est_sp = MUSIConesnapshot(x, param);
est_sp = est_sp/max(est_sp);
est_sp = pow2db(est_sp);
hold on; plot(param.cont_ang, est_sp, 'LineWidth', 2);
legend('SDP method','Ground-truth DOA','AP interference', 'Proposed method')
grid on;
xlabel('Spatial angle (deg)');
ylabel('Spatial spectrum (dB)')
