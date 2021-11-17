%Simulation looking at the performance of Schwarz applied to the 1D linear
%wave equation (linear elastic bar) using different timesteps
clc
clear all
close all

initial_time = 0;
final_time = 1;
top_level_params.time_interval = [0, final_time];

%newmark-beta timestepper parameters
top_level_params.gamma = 0.5;
top_level_params.beta = 0.25;

top_level_params.nonlocal_domain_size = 0.0;

%bar properties
top_level_params.bar_area = 1.0;

%material properties
top_level_params.constitutive_law = @(element) ...
    Constitutive_Linear_Elastic(element);
top_level_params.youngs_modulus = 1;
top_level_params.material_density = 1;

%initial condition
amplitude = 1;
center = 0.5;
sigma = 0.02;
initial_displacements = @(x) ...
    amplitude * exp( - (x - center)^2 / (2 * sigma^2) );
% initial_displacements = @(x) ...
%     amplitude * ( heaviside(x - 1/3) - heaviside(x - 2/3) );
top_level_params.initial_displacements = ...
    initial_displacements;
top_level_params.initial_velocities = @(x) 0;

%true solution
wave_speed = sqrt(top_level_params.youngs_modulus / ...
    top_level_params.material_density);
period = 1 / wave_speed;
b = center;

%there must be an easier way to do this, but I'm writing the true solution
%symbolically, then taking derivatives. Then I'm turning those derivatives
%into function handles so I can evaluate them easily. To do this, I convert
%the symbolic expressions to chars, then evaluate them.
syms x;
syms t;
symbolic_solution = ...
    1/2 * initial_displacements(x - wave_speed*t) ...
    + 1/2 * initial_displacements(x + wave_speed*t) ...
    - 1/2 * initial_displacements(x - wave_speed*(period-t)) ...
    - 1/2 * initial_displacements(x + wave_speed*(period-t));
char_displacements = char(symbolic_solution);
char_velocities = char( diff( symbolic_solution, t) );
char_accelerations = char( diff( symbolic_solution, t, 2) );
%replace all the ^'s and *'s with .^ and .* so the functions work for
%vector inputs
char_displacements = strrep(strrep(char_displacements,'^','.^'),'*','.*');
char_velocities = strrep(strrep(char_velocities,'^','.^'),'*','.*');
char_accelerations = strrep(strrep(char_accelerations,'^','.^'),'*','.*');
true_solution = eval( ['@(x,t)' char_displacements] );
true_velocity = eval(['@(x,t)' char_velocities] );
true_acceleration = eval(['@(x,t)' char_accelerations] );

% These depend on the number of domains
number_domains = 1;
switch number_domains
case 1
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [1];
  top_level_params.time_steps =  [1e-2];
  top_level_params.number_elements_domain = [100]';
  top_level_params.limits_domains = [0, 1.0];
  top_level_params.overlap_domains = [0, 0];
  top_level_params.regularization_toggles = [0];
case 2
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [1, 1];
  top_level_params.time_steps = [1e-2, 1e-3];
  top_level_params.number_elements_domain = [75, 75]';
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
time_step = max(time_steps);
time_difference = final_time - initial_time;
number_steps = round(time_difference ./ time_step);
top_level_params.number_steps = number_steps;
top_level_params.time_step = time_step;
times = initial_time + (0 : number_steps) * time_difference / number_steps;

fprintf('Initial number of steps: %d\n', number_steps);
top_level_params.maximum_applied_displacement = 0.0*[0.0, 0.0; 0.0, 0*final_time];

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
plot_str = {'r-*','g--s','b-*','k-*'};
schemes = ['IMPLICIT'; 'EXPLICIT'];
integration_schemes = top_level_params.integration_schemes;
plot_interval = number_steps / number_plots;
title_str = schemes(integration_schemes(1) + 1, :);
for domain = 2 : number_domains
  integration_scheme = integration_schemes(domain);
  title_str = strcat(title_str, '-', schemes(integration_scheme + 1, :));
end

%write files that are descriptive of what was run:
%nd - number of domains
%d - Domain index
%dt - timestep
%expl - explicit (1) or implicit(0)
file_ending_str_format = '_nd%u_d%u_dt%1.3e_exp%u.txt';
for domain = 1 : number_domains
    file_ending = sprintf(file_ending_str_format, number_domains, ...
        domain, time_steps(domain), ...
        top_level_params.integration_schemes(domain));
end

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
      plot(X, U, plot_str{domain});
      UU(step + 1, :) = U;
    end
  end
  save(['U', file_ending], 'UU', '-ascii');
end
xlabel('POSITION');
ylabel('DISPLACEMENT');
axis tight;
%axis([0 1 -amplitude amplitude]);
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
      plot(X, V, plot_str{domain});
      VV(step + 1, :) = V;
    end
  end
  save(['V', file_ending], 'VV', '-ascii');
end
xlabel('POSITION');
ylabel('VELOCITY');
axis tight;
%axis([0 1 -50 150]);
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
      plot(X, A, plot_str{domain});
      AA(step + 1, :) = A;
    end
  end
  save(['A', file_ending], 'AA', '-ascii');
end
xlabel('POSITION');
ylabel('ACCELERATION');
axis tight;
%axis([0 1 -2e7 2e7]);
hold off;


frame_number=1;
number_frames = min(120, number_steps);
frame_interval = round(number_steps / number_frames);
frames(number_frames) = struct('cdata',[],'colormap',[]);
figure
displacement_errors = zeros(number_domains, number_steps);
velocity_errors = zeros(number_domains, number_steps);
acceleration_errors = zeros(number_domains, number_steps);
total_energy = zeros(number_domains, number_steps);
for step = 1 : number_steps + 1
    % error calculation
    time = initial_time + (step-1) * time_step;
    for domain = 1: number_domains
        X = nodal_fields.positions{domain};
        U = nodal_fields.disp_histories{domain}(step, :);
        V = nodal_fields.velo_histories{domain}(step, :);
        A = nodal_fields.acce_histories{domain}(step, :);
        time = initial_time + (step-1) * time_step;

        %store errors normalized by the maximum norm of the displacement
        %(velocity or accelleration)
        displacement_errors(domain, step) = ...
            norm( true_solution(X, time) - U ) / norm(true_solution(X,0));
        velocity_errors(domain, step) = ...
            norm( true_velocity(X, time) - V ) / norm(true_velocity(X,period/2));
        acceleration_errors(domain, step) = ...
            norm( true_acceleration(X, time) - A ) / norm(true_acceleration(X,0));

        %energies
        PE = nodal_fields.potential_energies(domain, step);
        KE = nodal_fields.kinetic_energies(domain, step);
        total_energy(domain, step) = PE + KE;
    end

    %making the video of the displacements
    if mod(step - 1, frame_interval) == 0
        for domain = 1 : number_domains
            X = nodal_fields.positions{domain};
            UU = zeros(number_steps + 2, length(X));
            UU(1, :) = X;
            U = nodal_fields.disp_histories{domain}(step, :);
            plot(X, U, plot_str{domain});
            axis([0 1 -amplitude amplitude]);
            hold on;
            UU(step + 1, :) = U;
        end
        plot(X, true_solution(X, time), '-k');
        drawnow;
        frames(frame_number) = getframe(gcf);
        frame_number = frame_number + 1;
        hold off;
    end
end

figure
plot(0:time_step:final_time, total_energy);
xlabel('time');
ylabel('total energy');

figure
for domain = 1:number_domains
    subplot(3,1,1);
    plot(0:time_step:final_time, displacement_errors(domain,:));
    hold all;
    ylabel('U');

    subplot(3,1,2);
    plot(0:time_step:final_time, velocity_errors(domain,:));
    hold all;
    ylabel('V');

    subplot(3,1,3);
    plot(0:time_step:final_time, acceleration_errors(domain,:))
    hold all;
    ylabel('A');
end
hold off
xlabel('time')

% fig = figure;
% movie(fig, frames, 1);

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