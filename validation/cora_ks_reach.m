% cora_ks_reach.m
% Run CORA reach for the kinematic single-track (KS) bicycle model and save
% per-step bounds to CSV for comparison against pycora.
%
% Setup (one-time):
%   Install CORA via MATLAB Add-Ons:
%     Home tab → Add-Ons → search "CORA" → Install
%   (Auto-adds to MATLAB path; no manual addpath needed.)
%
% Then run this script:
%     run('cora_ks_reach.m')
%
% This script does NOT use steeringConstraints / accelerationConstraints
% (we want a constraint-free comparison against pycora's KS).

clear; clc;

% --------- Output directory (must be writable from both Windows and WSL) -----
% Adjust this if needed. Using the project's validation/ folder:
output_dir = 'C:\Users\larisa\Projects\commonroad-frenetix-project\validation\cora_outputs';
% Equivalent WSL path: /home/larisa/Projects/commonroad-frenetix-project/validation/cora_outputs
% Alternative if WSL path doesn't resolve in MATLAB:
% output_dir = 'C:\temp\pycora_validation';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% --------- KS bicycle vehicle parameters (BMW 320i, matches pycora test) ----
a_param = 1.1562;   % distance c.o.g. to front axle [m]
b_param = 1.4227;   % distance c.o.g. to rear axle [m]
l_wb = a_param + b_param;

% --------- KS dynamics (constraint-free, matches pycora) --------------------
% State: [x, y, delta, v, psi]
% Input: [delta_dot, a]
% Slip angle uses smooth tanh sign (matches pycora make_ks_dynamics)
KS_f = @(x,u) [
    x(4) * cos(atan(tan(x(3)) * b_param / l_wb) * tanh(100*x(4)) + x(5));
    x(4) * sin(atan(tan(x(3)) * b_param / l_wb) * tanh(100*x(4)) + x(5));
    u(1);
    u(2);
    x(4) * cos(atan(tan(x(3)) * b_param / l_wb) * tanh(100*x(4))) ...
         * tan(x(3)) / l_wb;
];

% Wrap as nonlinearSys (5 states, 2 inputs)
sys = nonlinearSys('KS', KS_f, 5, 2);

% --------- Reach configuration (matches pycora test_ks_curve_orientation_grows)
delta0 = 0.1;
v0 = 8.0;
x0 = [0.0; 0.0; delta0; v0; 0.0];
init_radii = [0.01; 0.01; 1e-4; 0.01; 1e-4];

R0 = zonotope(x0, diag(init_radii));

% Input set centered on (0, 0); essentially no input over the horizon
u_center = [0.0; 0.0];
u_radii = [1e-6; 1e-6];
U_uncertain = zonotope(zeros(2,1), diag(u_radii));

dt = 0.1;
n_steps = 10;

params.tStart = 0.0;
params.tFinal = dt * n_steps;
params.R0 = R0;
params.U = U_uncertain;
params.uTrans = u_center;

options.timeStep = dt;
options.taylorTerms = 6;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% --------- Run reach --------------------------------------------------------
fprintf('Running CORA reach: %d steps at dt=%.3f...\n', n_steps, dt);
R = reach(sys, params, options);

% --------- Extract bounds per step ------------------------------------------
% Time-point sets are in R.timePoint.set; time-interval sets are in
% R.timeInterval.set.
n_tp = numel(R.timePoint.set);
fprintf('Got %d time-point sets.\n', n_tp);

bounds_tp = zeros(n_tp, 1 + 2*5);  % [t, lb1, ub1, ..., lb5, ub5]
for k = 1:n_tp
    Z = R.timePoint.set{k};
    I = interval(Z);
    bounds_tp(k, 1) = R.timePoint.time{k};
    for d = 1:5
        bounds_tp(k, 2*d) = infimum(I(d));
        bounds_tp(k, 2*d + 1) = supremum(I(d));
    end
end

% Save to CSV
csv_path = fullfile(output_dir, 'ks_curve_reach_bounds.csv');
header = 't,lb_x,ub_x,lb_y,ub_y,lb_delta,ub_delta,lb_v,ub_v,lb_psi,ub_psi';
fid = fopen(csv_path, 'w');
fprintf(fid, '%s\n', header);
for k = 1:n_tp
    fprintf(fid, '%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
        bounds_tp(k, :));
end
fclose(fid);
fprintf('Wrote: %s\n', csv_path);

% --------- Print summary ----------------------------------------------------
fprintf('\nFinal reach set bounds (t=%.2f):\n', bounds_tp(end, 1));
fprintf('  x:     [%.4f, %.4f]\n', bounds_tp(end, 2), bounds_tp(end, 3));
fprintf('  y:     [%.4f, %.4f]\n', bounds_tp(end, 4), bounds_tp(end, 5));
fprintf('  delta: [%.4f, %.4f]\n', bounds_tp(end, 6), bounds_tp(end, 7));
fprintf('  v:     [%.4f, %.4f]\n', bounds_tp(end, 8), bounds_tp(end, 9));
fprintf('  psi:   [%.4f, %.4f]\n', bounds_tp(end, 10), bounds_tp(end, 11));
