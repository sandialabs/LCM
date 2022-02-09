%IKT main file
function [element_properties, nodal_fields] = ...
        TopLevelSchwarzDD(top_level_params, element_properties, nodal_fields)


number_domains = top_level_params.number_domains;
number_steps = top_level_params.number_steps;
maximum_applied_displacement = top_level_params.maximum_applied_displacement;
overlap_domains = top_level_params.overlap_domains;
regularization_toggles = top_level_params.regularization_toggles;
number_elements_domain = top_level_params.number_elements_domain;
nonlocal_domain_size = top_level_params.nonlocal_domain_size;
time_interval = top_level_params.time_interval;
rel_tol_domain = top_level_params.rel_tol_domain;
rel_tol_schwarz = top_level_params.rel_tol_schwarz;
abs_tol_domain = top_level_params.abs_tol_domain;
abs_tol_schwarz = top_level_params.abs_tol_schwarz;
max_iter_domain = top_level_params.max_iter_domain;
max_iter_schwarz = top_level_params.max_iter_schwarz;
domain_interval = top_level_params.domain_interval;
schwarz_interval = top_level_params.schwarz_interval;
step_interval = top_level_params.step_interval;
time_steps = top_level_params.time_steps;
time_step = top_level_params.time_step;
integration_schemes = top_level_params.integration_schemes;

positions = nodal_fields.positions;
displacements = nodal_fields.displacements;
velocities = nodal_fields.velocities;
accelerations = nodal_fields.accelerations;
residuals = nodal_fields.residuals;
disp_histories = nodal_fields.disp_histories;
velo_histories = nodal_fields.velo_histories;
acce_histories = nodal_fields.acce_histories;

moduli = element_properties.moduli;
areas = element_properties.areas;
densities = element_properties.densities;
internals = element_properties.internals;
damage_saturation = element_properties.damage_saturation;
maximum_damage = element_properties.maximum_damage;
forces = element_properties.forces;
strains = element_properties.strains;

initial_time = time_interval(1);
final_time = time_interval(2);

beta = top_level_params.beta;
gamma = top_level_params.gamma;

% Estimate stable time step for each domain in case of explicit time
% integration
fmt_str = 'Warning domain %d, reduce time step from %0.8e to at most %0.8e.\n';
for domain = 1 : number_domains
  if integration_schemes(domain) == 1
    number_elements = number_elements_domain(domain);
    element_sizes = zeros(1, number_elements);
    X = positions{domain};
    for element = 1 : number_elements
      element_sizes(element) = X(element + 1) - X(element);
    end
    wave_speeds = sqrt(moduli{domain} ./ densities{domain});
    stable_steps = element_sizes ./ wave_speeds;
    stable_step = 1.0 * min(stable_steps);
    tolerance = 1e-15;
    if time_steps(domain) - stable_step > tolerance
      fprintf(fmt_str, domain, time_steps(domain), stable_step);
      time_steps(domain) = stable_step;
      error('Program terminated.');
    end
  end
end
top_level_params.time_steps = time_steps;
proposed_time_step = max(time_steps);
%if proposed_time_step < time_step
%  time_step = proposed_time_step;
%  number_steps = round((final_time - initial_time) / time_step);
%  top_level_params.time_step = time_step;
%  top_level_params.number_steps = number_steps;
%  fprintf('Adjusted number of steps: %d\n', number_steps);
%end

potential_energies = zeros(number_domains, number_steps);
kinetic_energies = zeros(number_domains, number_steps);

%before our first time step, compute the kinetic and potential energy as
%well as the initial acceleration in each domain
for domain = 1 : number_domains
  X = positions{domain}';
  U = displacements{domain}';
  V = velocities{domain}';
  E = moduli{domain}';
  a = areas{domain}';
  rho = densities{domain}';
  Q = internals{domain}';
  ds = damage_saturation{domain}';
  md = maximum_damage{domain}';
  regularize = regularization_toggles(domain);

  [~, M, f, ~, ~, ~, PE] = AssembleUpdate(X, U, E, a, ...
    rho, Q, ds, md, nonlocal_domain_size, regularize, 0, 1, 1, ...
    element_properties.constitutive_law);

  acce_histories{domain}(1,:) = - M\f;
  accelerations{domain} = acce_histories{domain}(1,:);
  KE = 0.5 * V' * M * V;

  potential_energies(domain, 1) = PE;
  kinetic_energies(domain, 1) = KE;
end

%
% Main loop
%
for step = 1 : number_steps

  time = initial_time + (step - 1) * time_step;

  iter_schwarz = 1;
  display_step = mod(step - 1, step_interval) == 0 || step == number_steps;

  if display_step == 1
    fprintf('step: %d, time: %0.8e\n', step, time);
  end

  % Before the Schwarz algorithm, set an initial guess for the
  % displacements in each Schwarz domain. Here, lets use the displacements
  % from an explicit dynamics update.
  for domain = 1 : number_domains

    number_domain_per_global = round(time_step / time_steps(domain));
    if number_domain_per_global < 1
      number_domain_per_global = 1;
    end
    domain_time_step = time_step / number_domain_per_global;

    displacements{domain} = displacements{domain} + ...
      domain_time_step * ( velocities{domain} + ...
      0.5 * domain_time_step * accelerations{domain});
    disp_histories{domain}(step + 1, :) = 1.0 * displacements{domain};

    velocities{domain} = velocities{domain} + ...
      domain_time_step * accelerations{domain};

  end
  prev_internals = internals;

  % Schwarz iteration
  while true,

    prev_schwarz_displacements = displacements;
    prev_schwarz_velocities = velocities;
    internals = prev_internals;

    display_schwarz = ...
        mod(iter_schwarz, schwarz_interval) == 0 && display_step;

    baseline_norms = zeros(number_domains, 1);
    difference_norms = zeros(number_domains, 1);

    % Go one domain at a time.
    for domain = 1 : number_domains

      prev_global_loadstep_displacements = disp_histories{domain}(step, :);
      prev_global_loadstep_velocities = velo_histories{domain}(step, :);
      prev_global_loadstep_accelerations = acce_histories{domain}(step, :);

      % Adjust the domain time step to be an integer multiple
      % of the global time step.
      number_domain_per_global = round(time_step / time_steps(domain));
      if number_domain_per_global < 1
        number_domain_per_global = 1;
      end
      domain_time_step = time_step / number_domain_per_global;

      number_elements = number_elements_domain(domain);
      number_nodes = number_elements + 1;

      domain_left = overlap_domains(domain, 1);
      domain_right = overlap_domains(domain, 2);

      position_left = positions{domain}(1);
      position_right = positions{domain}(number_nodes);

      prev_timestep_displacements = prev_global_loadstep_displacements;
      prev_timestep_velocities = prev_global_loadstep_velocities;
      prev_timestep_accelerations = prev_global_loadstep_accelerations;

      for domain_step = 1 : number_domain_per_global

        domain_time = time + domain_time_step * domain_step;
        free_dof = (2 : number_nodes - 1)';

        % Determine boundary conditions.
        % Check whether there is enough history to interpolate in time.
        switch domain_left
        case {-2, -1}
          free_dof = [1; free_dof];
        case 0,
          T = time_interval';
          U = maximum_applied_displacement(:, 1);
          disp_left = interp1(T, U, domain_time, 'linear', 'extrap');
          velo_left = (U(2) - U(1)) / (T(2) - T(1));
          acce_left = 0.0;
          displacements{domain}(1) = disp_left;
          velocities{domain}(1) = velo_left;
          accelerations{domain}(1) = acce_left;
        otherwise,
          % This results in interpolation with previous history data or
          % pure extrapolation in case there is enough data to do so.
          if step == 1
            limit = step + 1;
          else
            limit = step + 1;
          end
          %times = (0 : limit - 1) * time_step;
          times = (step - 1 : step) * time_step;
          [X, T] = meshgrid(positions{domain_left}, times);
          x = position_left;
          t = domain_time;
          %U = disp_histories{domain_left}(1 : limit, :);
          %V = velo_histories{domain_left}(1 : limit, :);
          %A = acce_histories{domain_left}(1 : limit, :);
          U = disp_histories{domain_left}(step : step + 1, :);
          V = velo_histories{domain_left}(step : step + 1, :);
          A = acce_histories{domain_left}(step : step + 1, :);
          disp_left = interp2(X, T, U, x, t, 'spline');
          velo_left = interp2(X, T, V, x, t, 'spline');
          acce_left = interp2(X, T, A, x, t, 'spline');
          displacements{domain}(1) = disp_left;
          velocities{domain}(1) = velo_left;
          accelerations{domain}(1) = acce_left;
        end

        switch domain_right
        case {-2, -1}
          free_dof = [free_dof; number_nodes];
        case 0,
          T = time_interval';
          U = maximum_applied_displacement(:, 2);
          disp_right = interp1(T, U, domain_time, 'linear', 'extrap');
          velo_right = (U(2) - U(1)) / (T(2) - T(1));
          acce_right = 0.0;
          displacements{domain}(number_nodes) = disp_right;
          velocities{domain}(number_nodes) = velo_right;
          accelerations{domain}(number_nodes) = acce_right;
        otherwise,
          % This results in interpolation with previous history data or
          % pure extrapolation in case there is enough data to do so.
          if step == 1
            limit = step + 1;
          else
            limit = step + 1;
          end
          %times = (0 : limit - 1) * time_step;
          times = (step - 1 : step) * time_step;
          [X, T] = meshgrid(positions{domain_right}, times);
          x = position_right;
          t = domain_time;
          %U = disp_histories{domain_right}(1 : limit, :);
          %V = velo_histories{domain_right}(1 : limit, :);
          %A = acce_histories{domain_right}(1 : limit, :);
          U = disp_histories{domain_right}(step : step + 1, :);
          V = velo_histories{domain_right}(step : step + 1, :);
          A = acce_histories{domain_right}(step : step + 1, :);
          disp_right = interp2(X, T, U, x, t, 'spline');
          velo_right = interp2(X, T, V, x, t, 'spline');
          acce_right = interp2(X, T, A, x, t, 'spline');
          displacements{domain}(number_nodes) = disp_right;
          velocities{domain}(number_nodes) = velo_right;
          accelerations{domain}(number_nodes) = acce_right;
        end

        E = moduli{domain}';
        a = areas{domain}';
        rho = densities{domain}';
        Q = internals{domain}';
        ds = damage_saturation{domain}';
        md = maximum_damage{domain}';

        regularize = regularization_toggles(domain);

        X = positions{domain}';
        U = displacements{domain}';
        V = velocities{domain}';
        A = accelerations{domain}';

        display_domain = mod(domain_step, domain_interval) == 0 ...
            && display_schwarz;

        fmt_str = 'domain: %d, domain step: %d\n';

        if display_domain == 1
          fprintf(fmt_str, domain, domain_step);
        end

        % Integrate implicitly or explicitly
        if integration_schemes(domain) == 0
          % Implicit integration

          % Predictor
          U_pre = U;
          U_pre(free_dof) = prev_timestep_displacements(free_dof) + ...
            domain_time_step * (prev_timestep_velocities(free_dof) + ...
            (0.5 - beta) * domain_time_step * prev_timestep_accelerations(free_dof));

          V_pre = V;
          V_pre(free_dof) = prev_timestep_velocities(free_dof) + ...
            (1 - gamma) * domain_time_step * prev_timestep_accelerations(free_dof);

          iter_domain = 1;

          %U(free_dof) = U_pre(free_dof);
          % Nonlinear solver iteration
          while true,

            q = Q;

            [K, M, f, q, P, e, PE] = AssembleUpdate(X, U, E, a, ...
              rho, q, ds, md, nonlocal_domain_size, regularize, 0, 1, 1, ...
              element_properties.constitutive_law);

            T = (M / beta / domain_time_step / domain_time_step + K);

            A(free_dof) = (U(free_dof) - U_pre(free_dof)) / ...
              beta / domain_time_step / domain_time_step;

            r = - M * A - f;

            delta = T(free_dof, free_dof) \ r(free_dof);

            U(free_dof) = U(free_dof) + delta;

            V(free_dof) = ...
                V_pre(free_dof) + gamma * domain_time_step * A(free_dof);

            KE = 0.5 * V' * M * V;

            norm_residual = norm(r(free_dof));
            abs_error = norm(delta);
            norm_disp = norm(U);

            if norm_disp > 0
              rel_error = abs_error / norm_disp;
            else
              if abs_error > 0
                rel_error = 1.0;
              else
                rel_error = 0.0;
              end
            end

            converged_rel = rel_error < rel_tol_domain;
            converged_abs = abs_error < abs_tol_domain;
            converged = (converged_rel || converged_abs);

            fmt_str = 'nl iter: %d, rel error: %0.8e, abs error: %0.8e\n';

            if display_domain == 1
              fprintf(fmt_str, iter_domain, rel_error, abs_error);
            end

            if converged == 1 || iter_domain == max_iter_domain
              if display_domain == 0 && display_schwarz == 1
                fprintf(fmt_str, iter_domain, rel_error, abs_error);
              end
              break;
            end

            iter_domain = iter_domain + 1;

          end % iterate domain

        else
          % Explicit integration

          % Predictor
          U(free_dof) = prev_timestep_displacements(free_dof) + ...
            domain_time_step * (prev_timestep_velocities(free_dof) + ...
            0.5 * domain_time_step * prev_timestep_accelerations(free_dof));

          V(free_dof) = prev_timestep_velocities(free_dof) + ...
            (1 - gamma) * domain_time_step * prev_timestep_accelerations(free_dof);

          [~, M, f, q, P, e, PE] = AssembleUpdate(X, U, E, a, ...
            rho, Q, ds, md, nonlocal_domain_size, regularize, 1, 1, 1, ...
            element_properties.constitutive_law);

          r = - M .* A - f;

          delta = r(free_dof) ./ M(free_dof);

          A(free_dof) = A(free_dof) + delta;

          % Corrector
          V(free_dof) = V(free_dof) + gamma * domain_time_step * A(free_dof);

          KE = 0.5 * V' * diag(M) * V;

        end

        %update previous state
        prev_timestep_displacements = U;
        prev_timestep_velocities = V;
        prev_timestep_accelerations = A;

      end % domain steps

      displacements{domain} = U';
      velocities{domain} = V';
      accelerations{domain} = A';
      residuals{domain} = r';

      internals{domain} = q';
      forces{domain} = P';
      strains{domain} = e';

      % Advanced one global step, record histories
      disp_histories{domain}(step + 1, :) = displacements{domain};
      velo_histories{domain}(step + 1, :) = velocities{domain};
      acce_histories{domain}(step + 1, :) = accelerations{domain};

      potential_energies(domain, step+1) = PE;
      kinetic_energies(domain, step+1) = KE;

      baseline_norms(domain) = norm(U) + time_step * norm(V);
      DU = U' - prev_schwarz_displacements{domain};
      DV = V' - prev_schwarz_velocities{domain};
      difference_norms(domain) = norm(DU) + time_step * norm(DV);

    end % for each domain

    baseline_norm = norm(baseline_norms);
    abs_schwarz_error = norm(difference_norms);

    if baseline_norm > 0.0
      rel_schwarz_error = abs_schwarz_error / baseline_norm;
    else
      if abs_schwarz_error > 0
        rel_schwarz_error = 1.0;
      else
        rel_schwarz_error = 0.0;
      end
    end

    if number_domains == 1
      break;
    end

    converged_rel = rel_schwarz_error < rel_tol_schwarz;
    converged_abs = abs_schwarz_error < abs_tol_schwarz;
    converged = converged_rel || converged_abs;

    fmt_str = 'schwarz iter: %d, rel error: %0.8e, abs error: %0.8e\n';
    if display_schwarz == 1
      fprintf(fmt_str, iter_schwarz, rel_schwarz_error, abs_schwarz_error);
    end

    if converged == 1 || iter_schwarz == max_iter_schwarz
      if display_schwarz == 0 && display_step == 1
        fprintf(fmt_str, iter_schwarz, rel_schwarz_error, abs_schwarz_error);
      end
      break;
    end

    iter_schwarz = iter_schwarz + 1;

  end % iterate domains (Schwarz iteration)

end % load step

nodal_fields.displacements = displacements;
nodal_fields.velocities = velocities;
nodal_fields.accelerations = accelerations;
nodal_fields.residuals = residuals;

nodal_fields.disp_histories = disp_histories;
nodal_fields.velo_histories = velo_histories;
nodal_fields.acce_histories = acce_histories;

nodal_fields.potential_energies = potential_energies;
nodal_fields.kinetic_energies = kinetic_energies;

element_properties.internals = internals;
element_properties.forces = forces;
element_properties.strains = strains;
