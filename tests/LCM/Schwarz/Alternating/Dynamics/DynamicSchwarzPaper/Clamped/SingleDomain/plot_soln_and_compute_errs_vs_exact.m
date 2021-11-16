
close all;
clear all;

[x,y,z,dispz] = extract_fields_from_exo('clamped.e', 3);
[Nx, Nt] = size(dispz);
ind = find((x == 0.0005) & (y == 0.0005));


c = sqrt(1e9/1e3);
a = 0.01;
b = 0.5;
s = 0.02;
T = 1e-3;

Nt = 101;

fig1 = figure(1);
winsize = get(fig1,'Position');
Movie=moviein(Nt,fig1);
set(fig1,'NextPlot','replacechildren')
j = 1;
for i=1:Nt
  time = 10*(i-1)*(1e-6);
  times(i) = time;
  clearvars dispzex zSort;
  dz = dispz(ind, i);
  [zSort, I] = sort(z(ind));
  dispzex = 1/2*a*(exp(-(zSort-c*time-b).^2/2/s^2) + exp(-(zSort+c*time-b).^2/2/s^2))...
      - 1/2*a*(exp(-(zSort-c*(T-time)-b).^2/2/s^2) + exp(-(zSort+c*(T-time)-b).^2/2/s^2));
  plot(zSort, dispzex,'b');
  hold on;
  plot(zSort, dz(I), '--r');
  norm_sol(i) = norm(dispzex);
  abs_err(i) = norm(dispzex-dz(I));
  if (norm_sol(i)  < 4.0e-3)
    rel_err(i) = abs_err(i);
  else
    rel_err(i) = abs_err(i)/norm_sol(i);
  end
  set(fig1,'NextPlot','replacechildren')
  xlabel('z');
  ylabel('z-disp');
  title(['Time = ', num2str(time)]);
  axis([0 1.0 -0.01, 0.01]);
  legend('Exact', 'Sierra', 'Location','Best');
  pause(0.1)
  Movie(:,j)=getframe(fig1);
  mov(j) = getframe(gcf);
  j = j+1;
end
figure();
plot(times, rel_err) ;
xlabel('time');
ylabel('Relative error (displacement)');
figure();
plot(times,norm_sol);
xlabel('time');
ylabel('Norm displacement');
fprintf('Mean relative error in displacement = %f\n', mean(rel_err));
fprintf('Max relative error in displacement = %f\n', max(rel_err));
%movie2avi(Movie,'clamped_zdisp_1000_E1e9_zpts.avi','fps',7,'quality',10,'Compression','None');
