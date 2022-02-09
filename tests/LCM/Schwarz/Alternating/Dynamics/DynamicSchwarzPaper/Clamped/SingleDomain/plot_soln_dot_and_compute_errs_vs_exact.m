
close all;
clear all;


[x,y,z,dispz] = extract_fields_from_exo('clamped.e', 12);
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
  clearvars velz zsort;
  dz = dispz(ind, i);
  [zsort,I] = sort(z(ind));
  velz = c/2*a/s^2*((zsort-c*time-b).*exp(-(zsort-c*time-b).^2/2/s^2) - (zsort+c*time-b).*exp(-(zsort+c*time-b).^2/2/s^2))...
      + c/2*a/s^2*((zsort-c*(T-time)-b).*exp(-(zsort-c*(T-time)-b).^2/2/s^2) - ...
      (zsort+c*(T-time)-b).*exp(-(zsort+c*(T-time)-b).^2/2/s^2));
  plot(zsort, velz,'b');
  hold on;
  plot(zsort, dz(I), '--r');
  norm_sol(i) = norm(velz);
  abs_err(i) = norm(velz-dz(I));
  if (norm_sol(i)  < 100)
    rel_err(i) = abs_err(i);
  else
    rel_err(i) = abs_err(i)/norm_sol(i);
  end
  set(fig1,'NextPlot','replacechildren')
  xlabel('z');
  ylabel('z-vel');
  time = 10*(i-1)*(1e-6);
  title(['Time = ', num2str(time)]);
  axis([0 1.0 -500, 500]);
  legend('Exact', 'Sierra', 'Location','Best');
  pause(0.1)
  Movie(:,j)=getframe(fig1);
  mov(j) = getframe(gcf);
  j = j+1;
end
figure();
plot(times, rel_err) ;
xlabel('time');
ylabel('Relative error (velocity)');
figure();
plot(times,norm_sol);
xlabel('time');
ylabel('Norm velocity');
fprintf('Mean relative error in velocity = %f\n', mean(rel_err(1:end-1)));
fprintf('Max relative error in velocity = %f\n', max(rel_err(1:end-1)));
%movie2avi(Movie,'clamped_zvel_1000_E1e9_zpts.avi','fps',7,'quality',10,'Compression','None');
