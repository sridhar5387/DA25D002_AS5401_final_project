clc; clear; close all;
data_files = dir('*dat.h5');
data_files = data_files(1:4:end);
nx = 20;
nt = size(data_files, 1);
delta_t = 1;
X = zeros(3*nx^2, nt);
for i = 1:nt
    filename = data_files(i).name;
    u = h5read(filename, '/results/1/phase-1/cells/SV_U/1');
    v = h5read(filename, '/results/1/phase-1/cells/SV_V/1');
    T = h5read(filename, '/results/1/phase-1/cells/SV_T/1');
    X(:, i) = [u; v; T];
end

%Normalization
u_mean = mean(X(1:nx^2, :), 'all');
u_std = std(X(1:nx^2, :), 0, 'all');
v_mean = mean(X(nx^2+1:2*nx^2, :), 'all');
v_std = std(X(nx^2+1:2*nx^2, :), 0, 'all');
T_mean = mean(X(2*nx^2+1:3*nx^2, :), 'all');
T_std = std(X(2*nx^2+1:3*nx^2, :), 0, 'all');

X_norm = X;
X_norm(1:nx^2, :) = (X(1:nx^2, :) - u_mean) / u_std;
X_norm(nx^2+1:2*nx^2, :) = (X(nx^2+1:2*nx^2, :) - v_mean) / v_std;
X_norm(2*nx^2+1:3*nx^2, :) = (X(2*nx^2+1:3*nx^2, :) - T_mean) / T_std;

[phi, S, ~] = svd(X_norm, 'econ');
singular_values = diag(S);

cummulative_singular_value_ratio = cumsum(singular_values)/sum(singular_values);
disp(cummulative_singular_value_ratio);
figure;
plot(cummulative_singular_value_ratio, '-o');
xlabel('Number of modes (r)');
ylabel('Cummulative Singular Value Ratio');
title('Cummulative Singular Value Ratio vs Number of modes');
%99% threshold
r_99 = find(cummulative_singular_value_ratio >= 0.99, 1);
disp("Number of modes for 99% energy capture (r_99):");
disp(r_99)

%Function for symmetric kronecker product
function y = skron(x)
    n = length(x);
    A = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if i == j
                A(i, j) = x(i)*x(j);
            else
                A(i, j) = 0.5*(x(i)*x(j) + x(j)*x(i));
            end
        end
    end
    y = A(tril(true(n)));
end

r_values = [20, 25, 30, 35, 40, 45, 50, 51];
errors = zeros(length(r_values), 1);
lambda = 1e-5;

function X_reconstructed = get_rom(r, phi, X_norm, delta_t, nt, lambda)
    phi_r = phi(:, 1:r);
    X_r = phi_r' * X_norm;

    %Obtaining X dot
    X_t = zeros(size(X_r));
    for i = 1:nt-1
    X_t(:, i) = (X_r(:, i+1) - X_r(:, i))/delta_t;
    end



    D = zeros(r + r*(r+1)/2, nt);
    for i = 1:nt
    D(:, i) = [X_r(:, i); skron(X_r(:, i))];
    end

    O = (X_t * D') / (D * D' + lambda*eye(r + r*(r+1)/2));
    A = O(:, 1:r);
    H = O(:, r+1:end);

    %ROM simulation
    X_rom = zeros(r, nt);
    X_rom(:, 1) = X_r(:, 1);
    for i = 1:nt-1
    kron_X = skron(X_rom(:, i));
    dX_dt = A*X_rom(:, i) + H*kron_X;
    X_rom(:, i+1) = X_rom(:, i) + delta_t*dX_dt;
    end
    X_reconstructed = phi_r*X_rom;
end

for idx = 1:length(r_values)
    r = r_values(idx);
    X_reconstructed = get_rom(r, phi, X_norm, delta_t, nt, lambda);
    errors(idx) = norm(X_norm - X_reconstructed, 'fro') / norm(X_norm, 'fro');
end

figure;
semilogy(r_values, errors, '-o');
xlabel('Reduced Order (r)');
ylabel('Relative Error');
title('Log plot of error vs reduced order');



X_reconstructed = get_rom(r_99, phi, X_norm, delta_t, nt, lambda);
disp('Relative Error for r = r_99:');
rel_error_r99 = norm(X_norm - X_reconstructed, 'fro') / norm(X_norm, 'fro');
disp(rel_error_r99);

%Plotting
X_reconstructed(1:nx^2, :) = X_reconstructed(1:nx^2, :) * u_std + u_mean;
X_reconstructed(nx^2+1:2*nx^2, :) = X_reconstructed(nx^2+1:2*nx^2, :) * v_std + v_mean;
X_reconstructed(2*nx^2+1:3*nx^2, :) = X_reconstructed(2*nx^2+1:3*nx^2, :) * T_std + T_mean;

L = 0.2;
x_node = linspace(0, L, nx + 1);
y_node = linspace(0, L, nx + 1);
x_centroid = x_node(1:end-1) + diff(x_node)/2;
y_centroid = y_node(1:end-1) + diff(y_node)/2;

[x_grid, y_grid] = meshgrid(x_centroid, y_centroid);
figure;
pause(10);
for i = 1:nt
    velocity_mag = sqrt(reshape(X(1:nx^2, i), nx, nx)'.^2 + reshape(X(nx^2+1:2*nx^2, i), nx, nx)'.^2);
    velocity_mag_ROM = sqrt(reshape(X_reconstructed(1:nx^2, i), nx, nx)'.^2 + reshape(X_reconstructed(nx^2+1:2*nx^2, i), nx, nx)'.^2);

    tl = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    title(tl, "t = " + num2str(i-1));
    
    nexttile;
    contourf(x_grid, y_grid, reshape(X(2*nx^2+1:3*nx^2, i), nx, nx)', 20, 'LineColor', 'none');
    colormap(jet) 
    axis equal tight;
    xlabel('X (m)');
    ylabel('Y (m)');
    title('Temperature (K) (FOM)');

    nexttile;
    contourf(x_grid, y_grid, reshape(X_reconstructed(2*nx^2+1:3*nx^2, i), nx, nx)', 20, 'LineColor', 'none');
    colormap(jet)
    axis equal tight;
    xlabel('X (m)');
    ylabel('Y (m)');
    title('Temperature (K) (ROM)');
    colorbar;

    nexttile;
    contourf(x_grid, y_grid, reshape(X(2*nx^2+1:3*nx^2, i)-X_reconstructed(2*nx^2+1:3*nx^2, i), nx, nx)'.^2, 20, 'LineColor', 'none');
    colormap(jet)
    axis equal tight;
    xlabel('X (m)');
    ylabel('Y (m)');
    title('Temperature Error (K^2)');
    colorbar;

    nexttile;
    contourf(x_grid, y_grid, velocity_mag, 20, 'LineColor', 'none');
    colormap(jet)
    axis equal tight;
    title('Magnitude of velocity (m/s) (FOM)');
    xlabel('X (m)');
    ylabel('Y (m)');
    nexttile;

    contourf(x_grid, y_grid, velocity_mag_ROM, 20, 'LineColor', 'none');
    colormap(jet)
    axis equal tight;
    title('Magnitude of velocity (m/s) (ROM)');
    xlabel('X (m)');
    ylabel('Y (m)');
    colorbar;

    nexttile;
    contourf(x_grid, y_grid, (velocity_mag - velocity_mag_ROM).^2, 20, 'LineColor', 'none');
    colormap(jet)
    axis equal tight;
    title('Velocity Error (m/s)^2');
    xlabel('X (m)');
    ylabel('Y (m)');
    colorbar;

    pause(0.1);
end
