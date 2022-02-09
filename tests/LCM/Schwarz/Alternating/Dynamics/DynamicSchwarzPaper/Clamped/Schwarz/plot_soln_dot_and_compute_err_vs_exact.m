
close all;
clear all;


[x0,y0,z0,dispz0] = extract_fields_from_exo('clamped_left.e', 12);
[Nx, Nt] = size(dispz0);
ind0 = find((x0 == 0.0005) & (y0 == 0.0005));

[x1,y1,z1,dispz1] = extract_fields_from_exo('clamped_right.e', 12);
[Nx, Nt] = size(dispz1);
ind1 = find((x1 == 0.0005) & (y1 == 0.0005));

z = [0:0.001:1];
c = sqrt(1e9/1e3);
a = 0.01;
b = 0.5;
s = 0.02;
T = 1e-3;

fig1 = figure(1);
winsize = get(fig1,'Position');
Movie=moviein(Nt,fig1);
set(fig1,'NextPlot','replacechildren')
j = 1;
for i=1:Nt
  time = 10*(i-1)*(1e-6);
  times(i) = time;
  clearvars velz;
  velz = c/2*a/s^2*((z-c*time-b).*exp(-(z-c*time-b).^2/2/s^2) - (z+c*time-b).*exp(-(z+c*time-b).^2/2/s^2))...
      + c/2*a/s^2*((z-c*(T-time)-b).*exp(-(z-c*(T-time)-b).^2/2/s^2) - ...
      (z+c*(T-time)-b).*exp(-(z+c*(T-time)-b).^2/2/s^2));
  plot(z, velz,'b');
  hold on;
  dz0 = dispz0(ind0, i);
  [zsort0,I] = sort(z0(ind0));
  velzEx0 = c/2*a/s^2*((zsort0-c*time-b).*exp(-(zsort0-c*time-b).^2/2/s^2) - (zsort0+c*time-b).*exp(-(zsort0+c*time-b).^2/2/s^2))...
      + c/2*a/s^2*((zsort0-c*(T-time)-b).*exp(-(zsort0-c*(T-time)-b).^2/2/s^2) - ...
      (zsort0+c*(T-time)-b).*exp(-(zsort0+c*(T-time)-b).^2/2/s^2));
  norm_sol0(i) = norm(velzEx0);
  abs_err0(i) = norm(velzEx0-dz0(I));
  if (norm_sol0(i)  < 80)
    rel_err0(i) = abs_err0(i);
  else
    rel_err0(i) = abs_err0(i)/norm_sol0(i);
  end
  plot(zsort0, dz0(I), '--r');
  hold on;
  dz1 = dispz1(ind1, i);
  [zsort1,I] = sort(z1(ind1));
  velzEx1 = c/2*a/s^2*((zsort1-c*time-b).*exp(-(zsort1-c*time-b).^2/2/s^2) - (zsort1+c*time-b).*exp(-(zsort1+c*time-b).^2/2/s^2))...
      + c/2*a/s^2*((zsort1-c*(T-time)-b).*exp(-(zsort1-c*(T-time)-b).^2/2/s^2) - ...
      (zsort1+c*(T-time)-b).*exp(-(zsort1+c*(T-time)-b).^2/2/s^2));
  norm_sol1(i) = norm(velzEx1);
  abs_err1(i) = norm(velzEx1-dz1(I));
  if (norm_sol1(i)  < 80)
    rel_err1(i) = abs_err1(i);
  else
    rel_err1(i) = abs_err1(i)/norm_sol1(i);
  end
  plot(zsort1, dz1(I), '-.g');
  set(fig1,'NextPlot','replacechildren')
  xlabel('z');
  ylabel('z-vel');
  time = 10*(i-1)*(1e-6);
  title(['Time = ', num2str(time)]);
  axis([0 1.0 -500, 500]);
  legend('Exact', '\Omega_0', '\Omega_1');
  pause(0.1)
  Movie(:,j)=getframe(fig1);
  mov(j) = getframe(gcf);
  j = j+1;
end
figure();
plot(times, rel_err0) ;
hold on;
plot(times, rel_err1);
xlabel('time');
ylabel('Relative error (velocity)');
legend('\Omega_0','\Omega_1', 'Location','Best');
figure();
plot(times,norm_sol0);
hold on;
plot(times, norm_sol1);
xlabel('time');
ylabel('Norm velocity');
legend('\Omega_0','\Omega_1', 'Location','Best');
fprintf('Mean relative error in velocity in left domain = %f\n', mean(rel_err0(1:end-1)));
fprintf('Max relative error in velocity in left domain = %f\n', max(rel_err0(1:end-1)));
fprintf('Mean relative error in velocity in right domain  = %f\n', mean(rel_err1(1:end-1)));
fprintf('Max relative error in velocity in right domain = %f\n', max(rel_err1(1:end-1)));
fprintf('Mean relative error in velocity = %f\n', (mean(rel_err0(1:end-1))+mean(rel_err1(1:end-1)))/2);
fprintf('Max relative error in velocity = %f\n', max(max(rel_err0(1:end-1)), max(rel_err1(1:end-1))));

%movie2avi(Movie,'clamped_zvel_1000_E1e9_zpts.avi','fps',7,'quality',10,'Compression','None');
