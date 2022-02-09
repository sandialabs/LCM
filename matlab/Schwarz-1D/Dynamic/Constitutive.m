function [element] = Constitutive(element)

E = element.modulus;
A = element.area;
l = element.stretch;
q = element.internal;
ds = element.damage_saturation;
md = element.maximum_damage;
update = element.update;

W0 = 0.5 * E * (1.0 / l / l + l * l - 2.0);

% Update internal variables only.
if update == 1
  q = max([q, W0]);
end

% Update energy, stress and Hessian.
dmg = md * (1.0 - exp(-q / ds));
W = (1.0 - dmg) * W0;
P0 = E * (l - 1 / l / l / l);
P = (1.0 - dmg) * P0;
C0 = E * (3 / l / l / l / l + 1);
C = (1.0 - dmg) * C0;

element.energy_density = W;
element.stress = P;
element.hessian = C;
element.internal = q;