function Ds = Ds_SOC(c)
% This function computes Ds based on the SOC [Zeng et al., JCS, 2013]

Ds_ref = 2e-16;
c_max = 4.665e4;
c_theory = 277.84;
c_prac = 160;

SOC = (c_max-c)/c_max*c_theory/c_prac;

Ds = Ds_ref*(1+100*SOC.^1.5);

end
