function [ q, qm ] = ...
    Update(X, u, E, A, q, domain_size, update, constitutive)

number_nodes = length(X);
number_elements = number_nodes - 1;
element_size = X(2) - X(1);

if (update == 1)

  for i = 1 : number_elements

    Xi = X(i : i + 1);
    ui = u(i : i + 1);
    xi = Xi + ui;

    Li = Xi(2) - Xi(1);
    li = xi(2) - xi(1);
    lambda = li / Li;

    element.modulus = E(i);
    element.area = A(i);
    element.strain = lambda;
    element.internal = q(i);
    element.constitutive_law = constitutive;

    % Dummies
    element.damage_saturation = 1.0;
    element.maximum_damage = 0.0;

    element.update_state = update;

    [element] = element.constitutive_law(element);

    q(i) = element.internal;

  end

end

size_ratio = round(domain_size / element_size);
number_nonlocal_domains = number_elements / size_ratio;
number_nonlocal_nodes = number_nonlocal_domains + 1;

%Averaging of the internal variables.
qm = 0.0 * q;

for i = 1 : number_nonlocal_domains
  qmi = 0;
  for j = 1 : size_ratio
    qmi = qmi + q(size_ratio*(i - 1) + j);
  end
  qmi = qmi / size_ratio;
  for j = 1 : size_ratio
    qm(size_ratio*(i - 1) + j) = qmi;
  end
end
