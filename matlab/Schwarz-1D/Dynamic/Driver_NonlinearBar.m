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
  Constitutive(element);
top_level_params.youngs_modulus = 1;
top_level_params.material_density = 1;

%initial condition
amplitude = 0.02;
center = 0.5;
sigma = 0.02;
initial_displacements = @(x) ...
    amplitude * exp( - (x - center)^2 / (2 * sigma^2) );
% initial_displacements = @(x) ...
%     amplitude * ( heaviside(x - 1/3) - heaviside(x - 2/3) );
top_level_params.initial_displacements = ...
    initial_displacements;
top_level_params.initial_velocities = @(x) 0.1;

% These depend on the number of domains
number_domains = 2;
switch number_domains
case 1
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [1];
  top_level_params.time_steps =  [1e-2];
  top_level_params.number_elements_domain = [100]';
  top_level_params.limits_domains = [0, 1.0];
  top_level_params.overlap_domains = [-1, -1];
  top_level_params.regularization_toggles = [0];
case 2
  top_level_params.number_domains = number_domains;
  top_level_params.integration_schemes = [0, 0];
  top_level_params.time_steps = [2e-3 2e-3];
  top_level_params.number_elements_domain = [75, 75]';
  top_level_params.limits_domains = [0, 0.75; 0.25, 1.0];
  top_level_params.overlap_domains = [-1, 2; 1, -1];
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

plot_str = {'r-*','g--s','b-*','k-*'};
frame_number=1;
number_frames = min(120, number_steps);
frame_interval = round(number_steps / number_frames);
frames(number_frames) = struct('cdata',[],'colormap',[]);
figure
for step = 1 : number_steps + 1
    % error calculation
    time = initial_time + (step-1) * time_step;

    %making the video of the displacements
    if mod(step - 1, frame_interval) == 0
        for domain = 1 : number_domains
            X = nodal_fields.positions{domain};
            UU = zeros(number_steps + 2, length(X));
            UU(1, :) = X;
            U = nodal_fields.disp_histories{domain}(step, :);
            plot(X, U, plot_str{domain});
            axis([0 1 -5*amplitude 5*amplitude]);
            hold on;
            UU(step + 1, :) = U;
        end
        drawnow;
        frames(frame_number) = getframe(gcf);
        frame_number = frame_number + 1;
        hold off;
    end
end