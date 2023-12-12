
close all;
clear all;

%[x,y,z,dispz] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_svk.e', 3);
[x,y,z,dispz] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_dt1em7.e', 3);
[Nx, Nt] = size(dispz);
ind = find((x == 0.0) & (y == 0.0));
%[x,y,z,veloz] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_svk.e', 19);
%[x,y,z,accez] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_svk.e', 22);
[x,y,z,veloz] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_dt1em7.e', 19);
[x,y,z,accez] = extract_fields_from_exo('clamped_nonlinear_implicit_stab_dt1em7.e', 22);


load('1sd_nonlin_neohookean_dt1em7.mat'); 
times = times(1:100:end); 
z2 = nodal_fields.positions{1}; 
dispz2 = nodal_fields.disp_histories{1}(1:100:end,:);
veloz2 = nodal_fields.velo_histories{1}(1:100:end,:);
accez2 = nodal_fields.acce_histories{1}(1:100:end,:);

c = sqrt(1e9/1e3);
a = 0.001;
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
  subplot(3,1,1); 
  dz = dispz(ind, i);
  [zSort, I] = sort(z(ind));
  plot(zSort, dz(I), 'b');
  set(fig1,'NextPlot','replacechildren')
  hold on; 
  plot(z2, dispz2(i,:) , '--r');
  set(fig1,'NextPlot','replacechildren')
  hold off; 
  xlabel('z');
  ylabel('z-disp');
  title(['Time = ', num2str(time)]);
  axis([0 1.0 -a, a]);
  %legend('Albany', 'Location','Best');
  subplot(3,1,2); 
  vz = veloz(ind, i);
  plot(zSort, vz(I), 'b');
  set(fig1,'NextPlot','replacechildren')
  hold on; 
  plot(z2, veloz2(i,:), '--r');
  set(fig1,'NextPlot','replacechildren')
  hold off; 
  xlabel('z');
  ylabel('z-velo');
  axis([0 1.0 -50, 50]);
  subplot(3,1,3); 
  az = accez(ind, i);
  plot(zSort, az(I), 'b');
  set(fig1,'NextPlot','replacechildren')
  hold on; 
  plot(z2, accez2(i,:), '--r');
  hold off;
  set(fig1,'NextPlot','replacechildren')
  xlabel('z');
  ylabel('z-acce');
  axis([0 1.0 -4e6, 8e6]);
  if (j < 11)
    figname = strcat('soln_00', num2str(j-1), '.png'); 
  elseif ( j < 101)
    figname = strcat('soln_0', num2str(j-1), '.png'); 
  else
    figname = strcat('soln_', num2str(j-1), '.png'); 
  end
  legend('Albany','MATLAB','Location','NorthEast'); 
  exportgraphics(fig1, figname); 
  pause(0.1)
  Movie(:,j)=getframe(fig1);
  mov(j) = getframe(gcf);
  j = j+1;
end
%movie2avi(Movie,'clamped_zdisp_1000_E1e9_zpts.avi','fps',7,'quality',10,'Compression','None');
