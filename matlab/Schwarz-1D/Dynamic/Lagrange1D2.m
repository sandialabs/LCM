function [ Na, DNa ] = Lagrange1D2(xi)

Na  = zeros(2, 1);
Na(1) = 0.5 * (1.0 - xi);
Na(2) = 0.5 * (1.0 + xi);

DNa = zeros(2, 1);
DNa(1) = -0.5;
DNa(2) =  0.5;
