% vPCA: Voting Principle Components Analysis for PSI
% Authors:
% Shouyu Wang @ OptiX+ Laboratory, Wuxi University
% Javier Vargas @ CSIC
% Chen Li @ Wuxi University
% Wei Yu @ OptiX+ Laboratory, Wuxi University
% Aihui Sun @ Computational Optics Laboratory, Jiangnan University
% Version 1, August, 2026

%% CCC
clear;
close;
clc;

%% PSI
% ----------------------Constants------------------------------------------
Height_Full = 300;
Width_Full = 300;
Height = 256;
Width = 256;
M = Height*Width;
N = 50;
A = 0.5;
B = 0.5;
Vibration_Ratio = 0.2;
Turbulence_Ratio = 0.1;
% rng(42);
% ----------------------Phase-Shifting Angles------------------------------
Delta_NoNoise = (0:N-1)*pi/3;
Delta_Noise = rand(1,N)*0.1;
Delta_Truth = Delta_NoNoise+Delta_Noise;
Delta_Truth(1) = 0;
Delta_Truth = mod(Delta_Truth,2*pi);
% ----------------------Phase-Shifting Interferograms----------------------
[X, Y] = meshgrid(linspace(-1,1,Height_Full),linspace(-1,1,Width_Full));
Phi_Truth_Full = 20*pi*(0.3*(X.^2+Y.^2)+0.2*X);%5*peaks(Height_Full);%
Phi_Truth = Phi_Truth_Full(Height_Full/2-Height/2+1:Height_Full/2+Height/2,Width_Full/2-Width/2+1:Width_Full/2+Width/2);
PSI_NoNoise_Full = zeros(Height_Full,Width_Full,N);
for n = 1:N
    PSI_NoNoise_Full(:,:,n) = A+B*cos(Phi_Truth_Full+Delta_Truth(n));
end
PSI_NoNoise = PSI_NoNoise_Full(Height_Full/2-Height/2+1:Height_Full/2+Height/2,Width_Full/2-Width/2+1:Width_Full/2+Width/2,:);
% ----------------------Vibrations-----------------------------------------
Vibration_Frames = randperm(N, floor(Vibration_Ratio*N));
Vibration_Number = length(Vibration_Frames);
for j = 1:Vibration_Number
    Height_Shift = floor((rand(1,1)-0.5)*40);
    Width_Shift = floor((rand(1,1)-0.5)*40);
    PSI_NoNoise(:,:,Vibration_Frames(j)) = PSI_NoNoise_Full(Height_Full/2-Height/2-Height_Shift+1:Height_Full/2+Height/2-Height_Shift,Width_Full/2-Width/2-Width_Shift+1:Width_Full/2+Width/2-Width_Shift,Vibration_Frames(j));
end
% ----------------------Turbulence-----------------------------------------
Turbulence_Frames = randperm(N, floor(Turbulence_Ratio*N));
Turbulence_Number = length(Turbulence_Frames);
Turbulence = 2.5*pi*(0.1*(X.^2+Y.^2));
for j = 1:Turbulence_Number
    PSI_Turbulence = A+B*cos(Phi_Truth_Full+Delta_Truth(Turbulence_Frames(j))+Turbulence);
    PSI_NoNoise(:,:,Turbulence_Frames(j)) = PSI_Turbulence(Height_Full/2-Height/2+1:Height_Full/2+Height/2,Width_Full/2-Width/2+1:Width_Full/2+Width/2);
end
% ----------------------Noise----------------------------------------------
Noise = (rand(Height,Width,N)-0.5)*0.2;
PSI_Noise = PSI_NoNoise+Noise;
PSI_Noise(PSI_Noise<0) = 0;
PSI_Noise(PSI_Noise>1) = 1;
% ----------------------Disturbance-Corrupted Frames-----------------------
Bad_Frames_Union_Set = union(Vibration_Frames,Turbulence_Frames);

%% PCA
% ----------------------Matrix Rearrangement and Centering-----------------
PSI_Noise_Squeeze = reshape(PSI_Noise,M,N);
PSI_Noise_Squeeze_Mean = mean(PSI_Noise_Squeeze,2);
PSI_Tilde = PSI_Noise_Squeeze-PSI_Noise_Squeeze_Mean;
% ----------------------Covariance Matrix----------------------------------
C = (PSI_Tilde'*PSI_Tilde);
% ----------------------Eigen Decomposition--------------------------------
[U,Lambda] = eig(C);
[~,idx] = sort(diag(Lambda),'descend');
U = U(:,idx);
Sigma = diag(Lambda(idx,idx));
% ----------------------Phase-Shifting Angle Estimation--------------------
U1 = U(:,1);
U2 = U(:,2);
Delta_PCA = atan2(U2,U1);
Delta_PCA = Delta_PCA-Delta_PCA(1);
% ----------------------Phase-Shifting Angle Correction--------------------
if Delta_PCA(2) < 0
    Delta_PCA = -Delta_PCA;
end
Delta_PCA = mod(Delta_PCA,2*pi);
% ----------------------Phase Under Detection Estimation-------------------
Y1 = PSI_Tilde*U1;
Y2 = PSI_Tilde*U2;
Phi_PCA_Squeeze = atan2(-Y2,Y1);
Phi_PCA = reshape(Phi_PCA_Squeeze,Height,Width);

%% HT
% ----------------------Constants------------------------------------------
PSI_Centered = reshape(PSI_Tilde,Height,Width,N);
Pixel_Positions = [128,128;64,64;192,192;64,192;192,64];
Bad_Frames_Union = [];
HV_Number = 5;
for k = 1:HV_Number
% ----------------------Data-----------------------------------------------
    PSI_SinglePixel = squeeze(PSI_Centered(Pixel_Positions(k,1),Pixel_Positions(k,2),:));
% ----------------------Hough Voting---------------------------------------
    [Xi,Eta,HV,PSI_SinglePixel_HT] = hough_voting(Delta_PCA,PSI_SinglePixel);
% ----------------------Disturbance-Corrupted Frame Identification---------
    Error_Ratio = abs(PSI_SinglePixel-PSI_SinglePixel_HT)./abs(PSI_SinglePixel);
    Error_Ratio_Threshold = 1/2;
    Bad_Frames = find(Error_Ratio>Error_Ratio_Threshold);
    Bad_Frames_Cell{k} = Bad_Frames;
end
All_Frames = vertcat(Bad_Frames_Cell{:});
[unique_frames, ~, idx] = unique(All_Frames);
Counts = accumarray(idx, 1);
Threshold = HV_Number / 2;
Selected_frames = unique_frames(Counts >= Threshold);
Bad_Frames_Union = Selected_frames';

%% LSM
% ----------------------Disturbance-Free Data------------------------------
Good_Frames = setdiff(1:N,Bad_Frames_Union);
N_LSM = length(Good_Frames);
if N_LSM < 3
    error('vPCA: No Enough Disturbance-Free Interferograms');
end
PSI_LSM = PSI_Noise(:,:,Good_Frames);
PSI_LSM_Tilde = reshape(PSI_LSM,M,N_LSM);
Delta_LSM = Delta_PCA(Good_Frames(:));
% ----------------------LSM Iterations-------------------------------------
Iteration_Number = 5;
for i = 1:Iteration_Number
% ----------------------Phase Under Detection Update-----------------------
    X = [ones(N_LSM,1),cos(Delta_LSM),sin(Delta_LSM)];
    P = (X'*X)\(X'*PSI_LSM_Tilde');
    Phi_LSM_Tilde = atan2(-P(3,:)',P(2,:)');
    Phi_LSM = reshape(Phi_LSM_Tilde,Height,Width);
% ----------------------Phase-Shifting Angle Update------------------------
    Q = [P(2,:)',P(3,:)'];
    Delta_LSM = zeros(N_LSM,1);
    for n = 1:N_LSM
        Z = (Q'*Q)\(Q'*(PSI_LSM_Tilde(:,n)-P(1,:)'));
        Delta_LSM_SingleFrame = atan2(Z(2),Z(1));
        Delta_LSM(n) = Delta_LSM_SingleFrame;
    end
    Delta_LSM = mod(Delta_LSM,2*pi);
end

%% Results
% ----------------------vPCA Results---------------------------------------
Phi_vPCA = Phi_LSM;
Delta_vPCA = Delta_LSM;
% ----------------------Disturbance-Corrupted Frame Results----------------
disp('Set Disturbance-Corrupted Frames:');
disp(Bad_Frames_Union_Set);
disp('vPCA Obtained Disturbance-Corrupted Frames:');
disp(Bad_Frames_Union);
% ----------------------Phase RMSE Results---------------------------------
RMSE_Phi_PCA = sqrt(mean(mean((Phi_PCA-angle(exp(1i*Phi_Truth))).^2)));
RMSE_Phi_vPCA = sqrt(mean(mean((Phi_vPCA-angle(exp(1i*Phi_Truth))).^2)));
disp('PCA Phase RMSE:');
disp(RMSE_Phi_PCA);
disp('vPCA Phase RMSE:');
disp(RMSE_Phi_vPCA);
% ----------------------Phase-Shifting Angle RMSE Results------------------
RMSE_Delta_PCA = sqrt(mean((Delta_PCA'-Delta_Truth).^2));
RMSE_Delta_vPCA = sqrt(mean((Delta_vPCA'-Delta_Truth(Good_Frames)).^2));
disp('PCA Phase-Shifting Angle RMSE:');
disp(RMSE_Delta_PCA);
disp('vPCA Phase-Shifting Angle RMSE:');
disp(RMSE_Delta_vPCA);

%% Plotting
subplot(3,5,1); imagesc(PSI_Noise(:,:,1)); title('PSI No.1'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,2); imagesc(PSI_Noise(:,:,2)); title('PSI No.2'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,3); imagesc(PSI_Noise(:,:,N-1)); title('PSI No.N-1'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,4); imagesc(PSI_Noise(:,:,N)); title('PSI No.N'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,5); imagesc(Phi_Truth); title('Phase Under Detection'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,6); imagesc(angle(exp(1i*Phi_Truth))); title('Wrapped Phase Under Detection'); axis square; colorbar; colormap(gca, 'gray'); axis off;
subplot(3,5,7); imagesc(Phi_PCA); title('PCA: Wrapped Phase Under Detection'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,8); plot(0:N-1, Delta_Truth, 'b-o', 'LineWidth', 1.5); hold on; plot(0:N-1, Delta_PCA, 'r-*', 'LineWidth', 1.5); title('PCA: Phase-Shifting Angle'); xlabel('Frame'); ylabel('Phase-Shifting Angle (rad)'); legend('GT', 'PCA'); grid on;
subplot(3,5,9); imagesc(HV); title('Hough Voting'); colormap(gray); axis square; hold on; hold off;  axis off;
subplot(3,5,10); plot(1:N, PSI_SinglePixel, 'b-o', 'LineWidth', 1.5); title('Hough Fitting'); hold on; plot(1:N,PSI_SinglePixel_HT, 'r-*', 'LineWidth', 1.5); xlabel('Frame'); ylabel('Intensity'); legend('GT', 'HV', 'Location', 'best');
subplot(3,5,11); imagesc(Phi_vPCA); title('vPCA: Wrapped Phase Under Detection'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,12); plot(0:N_LSM-1, Delta_Truth(Good_Frames), 'b-o', 'LineWidth', 1.5); hold on; plot(0:N_LSM-1, Delta_vPCA, 'r-*', 'LineWidth', 1.5); title('vPCA: Phase-Shifting Angle'); xlabel('Frame'); ylabel('Phase-Shifting Angle (rad)'); legend('GT', 'vPCA'); grid on;
subplot(3,5,13); imagesc(Phi_PCA-angle(exp(1i*Phi_Truth))); title('PCA-GT'); axis square; colorbar; colormap(gray); axis off;
subplot(3,5,14); imagesc(Phi_vPCA-angle(exp(1i*Phi_Truth))); title('vPCA-GT'); axis square; colorbar; colormap(gray); axis off;

%% HT Function
function [Xi,Eta,Hough_Voting_Smooth,Intensity_HT] = hough_voting(Delta,Intensity,varargin)
% Hough Voting: Intensity = Xi*cos(Delta)-Eta*sin(Delta)
%
% Input:
%   Delta                - Phase-Shifting Angles
%   Intensity            - Single Pixel Intensity of Phase-Shifting Interferograms
%   Optional
%       'Range'          - Hough Voting Range, Default 1.5
%       'HoughSize'      - Hough Voting Size [v_num, u_num], Default [200, 200]
%       'SmoothSize'     - Size of Gaussian Smoothing Kernel (Odd Number), Default 7
%       'SmoothSigma'    - S.D. of Gaussian Smoothing Kernel (Odd Number), Default 3
%
% Output:
%   Xi, Eta              - Fitted Coefficients
%   Hough_Voting_Smooth  - Hough Voting Results
%   Intensity_HT         - HV Fitted Results
%
% Examples:
%   [Xi,Eta] = hough_voting(Delta_PCA, PSI_SinglePixel);
%   [Xi,Eta,HV,I_HT] = hough_voting(Delta_PCA, PSI_SinglePixel, 'Range', 2.0, 'HoughSize', [300, 300]);

% ----------------------Input----------------------------------------------
    p = inputParser;
    addRequired(p,'Delta',@(x) isvector(x) && isnumeric(x));
    addRequired(p,'Intensity',@(x) isvector(x) && isnumeric(x));
    addParameter(p,'Range',1.5, @(x) isscalar(x) && x>0);
    addParameter(p,'HoughSize',[200, 200], @(x) isvector(x) && length(x)==2 && all(x>0));
    addParameter(p,'SmoothSize',7, @(x) isscalar(x) && mod(x,2)==1);
    addParameter(p,'SmoothSigma',3, @(x) isscalar(x) && x>0);
    parse(p,Delta,Intensity,varargin{:});
    Hough_Range = p.Results.Range;
    Hough_Size = p.Results.HoughSize;
    Smooth_Size = p.Results.SmoothSize;
    Smooth_Sigma = p.Results.SmoothSigma;
    Delta = Delta(:);
    Intensity = Intensity(:);
    N = length(Delta);
% ----------------------Pair Construction----------------------------------
    pairs = [];
    for k = 2:N
        for l = 1:k-1
            pairs = [pairs;k,l];
        end
    end
    nPairs = size(pairs,1);
% ----------------------Difference Computation-----------------------------
    Cosine = cos(Delta);
    Sine = sin(Delta);
    dX = Cosine(pairs(:,1))-Cosine(pairs(:,2));
    dY = Sine(pairs(:,1))-Sine(pairs(:,2));
    dIntensity = Intensity(pairs(:,1))-Intensity(pairs(:,2));
% ----------------------Hough Space----------------------------------------
    u_range = [-Hough_Range,Hough_Range];
    v_range = [-Hough_Range,Hough_Range];
    v_edges = linspace(v_range(1),v_range(2),Hough_Size(1)+1);
    u_edges = linspace(u_range(1),u_range(2),Hough_Size(2)+1);
    v_centers = (v_edges(1:end-1)+v_edges(2:end))/2;
    u_centers = (u_edges(1:end-1)+u_edges(2:end))/2;
    dv = v_centers(2)-v_centers(1);
    du = u_centers(2)-u_centers(1);
% ----------------------Hough Voting---------------------------------------
    Hough_Voting = zeros(Hough_Size);
    for p = 1:nPairs
        Xp = dX(p);
        Yp = dY(p);
        dIp = dIntensity(p);
        if abs(Xp) > 1e-10
            for ui = 1:Hough_Size(2)
                u_candidate = u_centers(ui);
                if abs(Yp) > 1e-10
                    v_candidate = (dIp-u_candidate*Xp)/Yp;
                else
                    continue;
                end
                if v_candidate >= v_range(1) && v_candidate <= v_range(2)
                    vi = round((v_candidate-v_range(1))/dv)+1;
                    if vi >= 1 && vi <= Hough_Size(1)
                        Hough_Voting(vi,ui) = Hough_Voting(vi,ui) + 1;
                    end
                end
            end
        else
            if abs(Yp) > 1e-10
                v0 = dIp/Yp;
                vi = round((v0-v_range(1))/dv)+1;
                if vi >= 1 && vi <= Hough_Size(1)
                    Hough_Voting(vi,:) = Hough_Voting(vi,:)+1;
                end
            end
        end
    end
% ----------------------Gaussian Smoothing Kernel--------------------------
    h = fspecial('gaussian',[Smooth_Size, Smooth_Size],Smooth_Sigma);
    Hough_Voting_Smooth = imfilter(Hough_Voting,h,'same');
% ----------------------Hough Voting Peak----------------------------------
    [~,idx] = max(Hough_Voting_Smooth(:));
    [vi_peak,ui_peak] = ind2sub(Hough_Size,idx);
    u_est = u_centers(ui_peak);
    v_est = v_centers(vi_peak);
% ----------------------Output---------------------------------------------
    Xi = u_est;
    Eta = -v_est;
    Intensity_HT = Xi*cos(Delta)-Eta*sin(Delta);
end
