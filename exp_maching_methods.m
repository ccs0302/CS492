clc; clear all; close all;

%% Prepare data for doing experiment.
data_file = "bun180.ply";
data_file_split = split(data_file, '.');

data_name = data_file_split(1);
file_format = data_file_split(2);

if file_format == 'ply'
    Source_pcd = pcread(data_file).Location;
    Source_pcd = double(transpose(Source_pcd)); 
    N = size(Source_pcd,2);
elseif file_format == 'pcd'
    Source_pcd = load(data_file);
    Source_pcd = double(transpose(Source_pcd));
    N = size(Source_pcd,2);
end

% Translation values (a.u.):
Tx = 0.5;
Ty = -0.3;
Tz = 0.2;
T = [Tx; Ty; Tz];

% Rotation values (rad.):
rx = 0.3;
ry = -0.2;
rz = 0.05;
Rx = [1 0 0;
      0 cos(rx) -sin(rx);
      0 sin(rx) cos(rx)];
Ry = [cos(ry) 0 sin(ry);
      0 1 0;
      -sin(ry) 0 cos(ry)];
Rz = [cos(rz) -sin(rz) 0;
      sin(rz) cos(rz) 0;
      0 0 1];
R = Rx*Ry*Rz;

% Transform data 
Target_pcd = R * Source_pcd + repmat(T, 1, N);

% % plus noise
% rng(2912673);
% Target_pcd = Target_pcd + 0.01*randn(3,n);
% Source_pcd = Source_pcd + 0.01*randn(3,n);

%% Run ICP: Source --(Transform_[R,T])--> Target

opt_iter = 50;

total_result_fg = figure;

matching_method = ["bruteForce", "kDtree", "Delaunay"];
for mm = matching_method
    [ER, Ticp, Ricp, t] = f_icp(data_name, Source_pcd, Target_pcd, opt_iter, mm);
    
    E_trans = zeros(1, opt_iter);
    E_rot = zeros(1, opt_iter);
    for k= 1: opt_iter+1
        err_rot = acos((trace(R*inv(Ricp(:,:,k))-1)/2));
        if err_rot > pi
            err_rot = err_rot - pi;
        end
        E_rot(k) = err_rot;
        E_trans(k) = norm(Ticp(:,:,k) - T);
    end

    figure(total_result_fg);
    % Plot elapsed_time curve
    subplot(3,1,1); 
    plot(0:opt_iter, t,'--x', 'DisplayName', convertStringsToChars(mm));
    hold on;

    % Plot Err_trans curve
    subplot(3,1,2);
    plot(0:opt_iter,E_trans,'--x', 'DisplayName', convertStringsToChars(mm));
    hold on;

    % Plot Err_rot curve
    subplot(3,1,3); 
    plot(0:opt_iter,E_rot,'--x', 'DisplayName', convertStringsToChars(mm));
    hold on;
end

figure(total_result_fg);
subplot(3,1,1);
xlabel('iteration #');
ylabel('time_{elapsed}');
title('Optimization progress');
legend;

subplot(3,1,2);
xlabel('iteration #');
ylabel('d_{trans}');
legend;

subplot(3,1,3);
xlabel('iteration #');
ylabel('d_{rot}');
legend;






