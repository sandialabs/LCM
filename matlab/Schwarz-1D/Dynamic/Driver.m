initial_time = 0;
final_time = 1e-3;
top_level_params.time_interval = [0, final_time];
top_level_params.gamma = 0.5;
top_level_params.beta = 0.25;
top_level_params.nonlocal_domain_size = 0.1;

%initial conditions
a = 0.01;
b = 0.5;
s = 0.02;
top_level_params.initial_displacements = @(x) a * exp(-(x-b)*(x-b)/2/s/s);
top_level_params.initial_velocities = @(x) 0;

%bar properties
top_level_params.bar_area = 1.0e-06;

%material properties
top_level_params.constitutive_law = @(element)Constitutive_Linear_Elastic(element);
top_level_params.youngs_modulus = 1.0e09;
top_level_params.material_density = 1000;

% These depend on the number of domains
number_domains = 2;
switch number_domains
case 1
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [0];
  top_level_params.time_steps = [1e-7];
  top_level_params.number_elements_domain = [1000]';
  top_level_params.limits_domains = [0, 1.0];
  top_level_params.overlap_domains = [0, 0];
  top_level_params.regularization_toggles = [0];
case 2
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [0, 0];
  top_level_params.time_steps = [1e-7, 1e-7];
  top_level_params.number_elements_domain = [750, 750]';
  top_level_params.limits_domains = [0, 0.75; 0.25, 1.0];
  top_level_params.overlap_domains = [0, 2; 1, 0];
  top_level_params.regularization_toggles = [0, 0];
case 3
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [0, 0, 0];
  top_level_params.time_steps = [0.001, 0.001, 0.001];
  top_level_params.number_elements_domain = [64, 16, 4]';
  top_level_params.limits_domains = [0, 0.5; 0.25, 0.75; 0.5, 1.0];
  top_level_params.overlap_domains = [0, 2; 1, 3; 2, 0];
  top_level_params.regularization_toggles = [1, 0, 0];
otherwise
end
time_interval = top_level_params.time_interval;
time_steps = top_level_params.time_steps;
time_step = 1.0e-7;
time_difference = final_time - initial_time;
number_steps = round(time_difference ./ time_step);
top_level_params.number_steps = number_steps;
top_level_params.time_step = time_step;
times = initial_time + (0 : number_steps) * time_difference / number_steps;

fprintf('Initial number of steps: %d\n', number_steps);
top_level_params.maximum_applied_displacement = [0.0, 0.0; 0.0, 0.0];

top_level_params.rel_tol_domain = 1e-8;
top_level_params.rel_tol_schwarz = 1e-8;
top_level_params.abs_tol_domain = 1e-11;
top_level_params.abs_tol_schwarz = 1e-11;

top_level_params.max_iter_schwarz = 1000;
top_level_params.max_iter_domain = 100000;

top_level_params.step_interval = 100;
top_level_params.schwarz_interval = 1;
top_level_params.domain_interval = 1;

[element_properties, nodal_fields] =  SetupSchwarzDD(top_level_params);

[element_properties, nodal_fields] = ...
TopLevelSchwarzDD(top_level_params, element_properties, nodal_fields);

% Plotting of results
number_plots = 15;
number_domains = top_level_params.number_domains;
number_elements_domain = top_level_params.number_elements_domain;
number_steps = top_level_params.number_steps;
plot_str = ['r-*';'g-*';'b-*';'k-*'];
schemes = ['IMPLICIT'; 'EXPLICIT'];
integration_schemes = top_level_params.integration_schemes;
plot_interval = number_steps / number_plots;
title_str = schemes(integration_schemes(1) + 1, :);
for domain = 2 : number_domains
  integration_scheme = integration_schemes(domain);
  title_str = strcat(title_str, '-', schemes(integration_scheme + 1, :));
end

ustr = ['U0.txt'; 'U1.txt'];
vstr = ['V0.txt'; 'V1.txt'];
astr = ['A0.txt'; 'A1.txt'];

figure(1);
subplot(3,1,1);
hold on;
title(title_str);
for domain = 1 : number_domains
  X = nodal_fields.positions{domain};
  UU = zeros(number_steps + 2, length(X));
  UU(1, :) = X;
  for step = 1 : number_steps + 1
    if mod(step - 1, plot_interval) == 0
      U = nodal_fields.disp_histories{domain}(step, :);
      plot(X, U, plot_str(domain,:));
      UU(step + 1, :) = U;
    end
  end
  save(ustr(domain,:), 'UU', '-ascii');
end
xlabel('POSITION');
ylabel('DISPLACEMENT');
%axis tight;
axis([0 1 -0.01 0.01]);
hold off;
subplot(3,1,2);
hold on;
title(title_str);
for domain = 1 : number_domains
  X = nodal_fields.positions{domain};
  VV = zeros(number_steps + 2, length(X));
  VV(1, :) = X;
  for step = 1 : number_steps + 1
    if mod(step - 1, plot_interval) == 0
      V = nodal_fields.velo_histories{domain}(step, :);
      plot(X, V, plot_str(domain,:));
      VV(step + 1, :) = V;
    end
  end
  save(vstr(domain,:), 'VV', '-ascii');
end
xlabel('POSITION');
ylabel('VELOCITY');
%axis tight;
axis([0 1 -200 200]);
hold off;
subplot(3,1,3);
hold on;
title(title_str);
for domain = 1 : number_domains
  X = nodal_fields.positions{domain};
  AA = zeros(number_steps + 2, length(X));
  AA(1, :) = X;
  for step = 1 : number_steps + 1
    if mod(step - 1, plot_interval) == 0
      A = nodal_fields.acce_histories{domain}(step, :);
      plot(X, A, plot_str(domain,:));
      AA(step + 1, :) = A;
    end
  end
  save(astr(domain,:), 'AA', '-ascii');
end
xlabel('POSITION');
ylabel('ACCELERATION');
%axis tight;
axis([0 1 -2e7 2e7]);
hold off;
%{
subplot(3,1,4);
hold on;
title(title_str);
for domain = 1 : number_domains
  if mod(step - 1, plot_interval) == 0
    PE = nodal_fields.potential_energies(domain, :);
    KE = nodal_fields.kinetic_energies(domain, :);
    TE = PE + KE;
    T = times(2:number_steps + 1);
    plot(T, PE, 'r-*');
    plot(T, KE, 'g-*');
    plot(T, TE, 'b-*');
  end
end
xlabel('TIME');
ylabel('ENERGY');
axis tight;
hold off;
%}
