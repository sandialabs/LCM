function [element] = Constitutive_Linear_Elastic(element)

E = element.modulus;
A = element.area;
l = element.stretch;
strain = l - 1;
q = element.internal;
ds = element.damage_saturation;
md = element.maximum_damage;
update = element.update;

% Update energy, stress and Hessian.
W = 0.5 * E * strain^2;
P = E * strain;
C = E;

element.energy_density = W;
element.stress = P;
element.hessian = C;
element.internal = q;