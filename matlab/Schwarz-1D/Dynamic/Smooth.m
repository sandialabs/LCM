function [ u ] = Smooth(K, u, r)

type = 6;
damping = 1.0;

switch type
  case 0
    u = 0 * r;
  case 1
    u = damping .* r ./ diag(K);
  case 2
    u = r ./ norm(K);
  case 3
    u = GaussSeidel(K, u, r);
  case 4
    n = round(log2(size(u)) * 2);
    for i = 1 : n
        u = GaussSeidel(K, u, r);
    end
  case 5
    error = 1.0;
    tol = 1.0e-10;
    while error > tol,
        v = u;
        u = GaussSeidel(K, u, r);
        n = norm(v);
        if n > 0.0
            error = norm(u - v) / n;
        else
            error = norm(u);
        end
    end
  case 6
    u = K \ r;
end