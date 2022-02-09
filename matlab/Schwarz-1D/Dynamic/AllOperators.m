function [ P, R ] = AllOperators(X)

[number_levels, number_nodes_bottom] = size(X);

P = cell(number_levels, 1);
R = cell(number_levels, 1);

number_elements_bottom = number_nodes_bottom - 1;

number_elements_top = ...
    number_elements_bottom / 2^(number_levels - 1);

number_nodes_top = number_elements_top + 1;

number_nodes_level = number_nodes_top;

for level = 1 : number_levels - 1

    X_level = X(level, 1 : number_nodes_level)';
    [P_level, R_level] = Operators(X_level);

    P{level} = P_level;
    R{level} = R_level;

    number_nodes_level = 2 * (number_nodes_level - 1) + 1;

end
