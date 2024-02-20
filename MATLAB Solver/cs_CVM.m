function [A, b] = cs_CVM(Rp, j, C_prev, Ns)

delta_t = 1;

num = 2:Ns-1;

delta_r = Rp/(Ns-1);

% compute diffusivity between points
C_mid = (C_prev(1:end-1)+C_prev(2:end))/2;
Ds_mid = Ds_SOC(C_mid)';

ri = arrayfun(@(num) (num-1), num)*delta_r;

% compute coefficients of stencils
ci_coeff = -Ds_mid(2:end).*(ri+delta_r/2).^2/delta_r ...
    -Ds_mid(1:end-1).*(ri-delta_r/2).^2/delta_r;

cip1_coeff = Ds_mid(2:end).*(ri+delta_r/2).^2/delta_r;

cim1_coeff = Ds_mid(1:end-1).*(ri-delta_r/2).^2/delta_r;

% mass matrix for the time derivative
At = diag([delta_r^3/24, 1/3*((ri+delta_r/2).^3-(ri-delta_r/2).^3), ...
    1/3*(Rp^3-(Rp-delta_r/2)^3)]);

% mass matrix for the space derivative
As = diag([-Ds_mid(1)*(delta_r/2)^2/delta_r, ci_coeff, ...
    -Ds_mid(end)*(Rp-delta_r/2)^2/delta_r]) + ...
    diag([Ds_mid(1)*(delta_r/2)^2/delta_r, cip1_coeff],1) + ...
    diag([cim1_coeff, Ds_mid(end)*(Rp-delta_r/2)^2/delta_r],-1);

% ionic flux from boundary condition
b_flux = [zeros(Ns-1,1); -Rp^2*j];

% implement crank-nicolson scheme
A = 2*At-delta_t*As;
b = (delta_t*As+2*At)*C_prev+2*delta_t*b_flux;

end