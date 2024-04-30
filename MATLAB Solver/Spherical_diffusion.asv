clc;
clear;
% close all;
tic
t_all = 600;

delta_t = 1;

step = t_all/delta_t;

% Zeng's data
Ns = 10001;
Rp = 5e-6;
j = -5.35e-5;
C0 = 2e4*ones(Ns,1);

cs_all = zeros(Ns, length(t_all)+1);
cs_all(:,1) = C0;

for i = 1:step
    disp(i);
    [A_cs, b_cs] = cs_CVM(Rp, j, cs_all(:,i), Ns);
    cs_all(:,i+1) = A_cs\b_cs;
end

figure;
rad = 0:Rp/(Ns-1)*1e6:Rp*1e6;
plot(rad, cs_all(:,1));
hold on;
plot(rad, cs_all(:,101));
plot(rad, cs_all(:,201));
plot(rad, cs_all(:,301));
plot(rad, cs_all(:,401));
plot(rad, cs_all(:,501));
plot(rad, cs_all(:,601));

toc