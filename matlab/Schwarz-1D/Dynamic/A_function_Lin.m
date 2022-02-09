function [ y ] = A_function(x, y)

N = length(x);

xi = x(1);
xf = x(N);

a = 0.1;
b = 1.0;

m = (b - a) / (xf - xi);

xe = zeros(1,N-1);

for i = 1:N-1
  xe(i) = 0.5*(x(i) + x(i+1));
end

f = m .* (xe - xi) + a;
y = f .* y;