function [ u ] = GaussSeidel(K, u, r)

n = size(u);

for i = 1 : n

    L = 1 : i - 1;
    U = i + 1 : n;

    u(i) = (r(i) - K(i, L) * u(L) - K(i, U) * u(U)) / K(i, i);

end