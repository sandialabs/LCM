function [K, M, f, q, P, e, V] = ...
    Assembly(X, u, E, A, rho, q, ds, md, diagonal, lumped, update, constitutive)

number_nodes = length(X);
number_elements = number_nodes - 1;

K = zeros(number_nodes, number_nodes);

if diagonal == 1
  M = zeros(number_nodes, 1);
else
  M = zeros(number_nodes, number_nodes);
end

f = zeros(number_nodes, 1);
P = zeros(number_elements, 1);
e = zeros(number_elements, 1);

V = 0;

for i = 1 : number_elements
  element.positions = X(i : i + 1);
  element.displacements = u(i : i + 1);
  element.internal = q(i);
  element.modulus = E(i);
  element.area = A(i);
  element.density = rho(i);
  element.damage_saturation = ds(i);
  element.maximum_damage = md(i);
  element.diagonal= diagonal;
  element.lumped = lumped;
  element.update = update;
  element.constitutive_law = constitutive;

  [element] = StiffForce(element);

  Ki = element.stiffness;
  Mi = element.mass;
  fi = element.internal_force;
  q(i) = element.internal;
  P(i) = element.stress;
  e(i) = element.stretch;

  K(i : i + 1, i : i + 1) = K(i : i + 1, i : i + 1) + Ki;

  if (diagonal == 1)
    M(i : i + 1) = M(i : i + 1) + Mi;
  else
    M(i : i + 1, i : i + 1) = M(i : i + 1, i : i + 1) + Mi;
  end

  f(i : i + 1) = f(i : i + 1) + fi;

  V = V + element.stored_energy;
end

K = sparse(K);

if diagonal == 0
    M = sparse(M);
end