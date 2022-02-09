function [ y_element ] = A_function(x_nodal, y_element)

N = length(x_nodal);

% factor to scale the function
alpha = 1.0;

x_element = zeros(1, N - 1);

for i = 1 : N - 1
  x_element(i) = 0.5 * (x_nodal(i) + x_nodal(i+1));
end

factor = sqrt(x_element .* alpha);
y_element = factor .* y_element;