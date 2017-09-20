function theta = init_theta(theta, dist_struct)
% pi_c = dist_struct.pi_c;
% pi_z = dist_struct.pi_z;
Kz = size(dist_struct.pi_z,2);
for kz = 1:Kz
    theta.p(kz,1,kz) = 1;
end