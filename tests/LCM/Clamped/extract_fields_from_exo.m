
function [x,y,z,dispz] = extract_fields_from_exo(exoname, varnum);

varname = strcat('vals_nod_var', num2str(varnum));
dispz = ncread(exoname, varname);
x = ncread(exoname, 'coordx');
y = ncread(exoname, 'coordy');
z = ncread(exoname, 'coordz');
