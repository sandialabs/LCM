function [element_properties, nodal_fields] = SetupSchwarzDD(top_level_params)

number_steps = top_level_params.number_steps;
number_domains = top_level_params.number_domains;
number_elements_domain = top_level_params.number_elements_domain;

maximum_number_elements = max(number_elements_domain);
maximum_number_nodes = maximum_number_elements + 1;

positions = cell(number_domains);
displacements = cell(number_domains);
residuals = cell(number_domains);

limits_domains = top_level_params.limits_domains;
number_elements_domain = top_level_params.number_elements_domain;
overlap_domains = top_level_params.overlap_domains;
initial_displacements = top_level_params.initial_displacements;
initial_velocities = top_level_params.initial_velocities;

for domain = 1 : number_domains
  left = limits_domains(domain, 1);
  right = limits_domains(domain, 2);
  number_elements = number_elements_domain(domain);
  number_nodes = number_elements + 1;
  length = right - left;
  delta = length / number_elements;
  positions{domain} = [left : delta : right];
  displacements{domain} = zeros(1, number_nodes);
  velocities{domain} = zeros(1, number_nodes);
  accelerations{domain} = zeros(1, number_nodes);
  residuals{domain} = zeros(1, number_nodes);
  disp_histories{domain} = zeros(number_steps + 1, number_nodes);
  velo_histories{domain} = zeros(number_steps + 1, number_nodes);
  acce_histories{domain} = zeros(number_steps + 1, number_nodes);
  domain_left = overlap_domains(domain, 1);
  domain_right = overlap_domains(domain, 2);
  for node = 1 : number_nodes
      node_position = positions{domain}(node);
      displacements{domain}(node) = initial_displacements(node_position);
      velocities{domain}(node) = initial_velocities(node_position);
      disp_histories{domain}(1, node) = initial_displacements(node_position);
      velo_histories{domain}(1, node) = initial_velocities(node_position);
  end
%   if domain_left == -1
%     displacements{domain}(1) = initial_displacements(1);
%     velocities{domain}(1) = initial_velocities(1);
%     disp_histories{domain}(1, 1) = initial_displacements(1);
%     velo_histories{domain}(1, 1) = initial_velocities(1);
%   end
%   if domain_right == -1
%     displacements{domain}(number_nodes) = initial_displacements(2);
%     velocities{domain}(number_nodes) = initial_velocities(2);
%     disp_histories{domain}(1, number_nodes) = initial_displacements(2);
%     velo_histories{domain}(1, number_nodes) = initial_velocities(2);
%   end
%   if domain_left == -2 || domain_right == -2
%     for node = 1 : number_nodes
%       displacements{domain}(node) = initial_displacements(1);
%       velocities{domain}(node) = initial_velocities(1);
%       disp_histories{domain}(1, node) = initial_displacements(1);
%       velo_histories{domain}(1, node) = initial_velocities(1);
%     end
%   end
end

nodal_fields.positions = positions;
nodal_fields.displacements = displacements;
nodal_fields.velocities = velocities;
nodal_fields.accelerations = accelerations;
nodal_fields.residuals = residuals;

nodal_fields.disp_histories = disp_histories;
nodal_fields.velo_histories = velo_histories;
nodal_fields.acce_histories = acce_histories;

element_properties.moduli = cell(number_domains);
element_properties.areas = cell(number_domains);
element_properties.densities = cell(number_domains);
element_properties.internals = cell(number_domains);
element_properties.damage_saturation = cell(number_domains);
element_properties.maximum_damage = cell(number_domains);

element_properties.forces = cell(number_domains);
element_properties.strains = cell(number_domains);

for domain = 1 : number_domains
  number_elements = number_elements_domain(domain);

  E = top_level_params.youngs_modulus * ones(1, number_elements);
  A = top_level_params.bar_area * ones(1, number_elements);
  rho = top_level_params.material_density * ones(1, number_elements);
  q = zeros(1, number_elements);
  ds = 1.0e11 * ones(1, number_elements);
  md = 0.0 * ones(1, number_elements);

  % Area and Young's modulus may be non-constant.
  % Use these functions to compute at each element.
  X = positions{domain};

  %E = E_function(X, E);
  %A = A_function(X, A);

  element_properties.constitutive_law = top_level_params.constitutive_law;
  element_properties.moduli{domain} = E;
  element_properties.areas{domain} = A;
  element_properties.densities{domain} = rho;
  element_properties.internals{domain} = q;
  element_properties.damage_saturation{domain} = ds;
  element_properties.maximum_damage{domain} = md;

  element_properties.forces{domain} = zeros(1, number_elements);
  element_properties.strains{domain} = zeros(1, number_elements);
end
