function [element] = StiffForce(element)

X = element.positions;
u = element.displacements;
A = element.area;
rho = element.density;
lumped = element.lumped;
diagonal = element.diagonal;

x = X + u;
L = X(2) - X(1);
l = x(2) - x(1);

lambda = l / L;

element.stretch = lambda;

%For some reason, this doesn't feel right passing a struct to a function
%inside of it, but
[element] = element.constitutive_law(element);

P = element.stress;
C = element.hessian;
A = element.area;

% Use midpoint rule to integrate.
[Na, DNa] = Lagrange1D2(0.0);
GradNa = 2.0 / L .* DNa;
B = GradNa;
f = L * B * P * A;
K = L * B * C * A * B';

if diagonal == 1
  M = [0.5; 0.5] * L * A * rho;
else
  if lumped == 1
    M = L * A * rho * [0.5, 0; 0, 0.5];
  else
    M = L * A * rho * [2.0, 1.0; 1.0, 2.0] / 6.0;
  end
end

element.mass = M;
element.stiffness = K;
element.internal_force = f;
element.stored_energy = L * A * element.energy_density;
