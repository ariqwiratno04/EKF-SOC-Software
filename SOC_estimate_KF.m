% --------- INITIALIZATION ---------
clc; clear;

% Parameters (Example values - replace with your identified model)
Qn = 2.37 * 3600;   % Nominal capacity in Coulombs
dt = 1;             % Sampling time in seconds
N = 1000;           % Simulation steps

% Battery model parameters (replace with lookup or piecewise functions)
Rt = 0.0441;        % Terminal resistance [Ohm]
Rp = 0.0186;        % Polarization resistance [Ohm]
Cp = 69176;         % Polarization capacitance [F]

% State: [SoC; Vp] where Vp is the voltage across Rp-Cp
x = [0.6; 0];       % Initial guess of states

% Covariances
P = eye(2) * 0.01;
Q = diag([1e-7, 1e-5]);   % Process noise covariance
R = 1e-3;                % Measurement noise variance

% Placeholder for results
x_est = zeros(2, N); x_est(:,1) = x;
V_est = zeros(1, N);
SOC_true = linspace(0.6, 0.2, N);  % Simulated true SOC profile (optional)

% Input current profile (negative = discharge)
I = -0.5 * ones(1, N);  % Constant 0.5A discharge

% OCV lookup (use your experimental data)
OCV = @(soc) 3.5 + 0.5 * soc;  % Simplified linear model, replace with piecewise

% --------- MAIN EKF LOOP ---------
for k = 2:N
    % Previous state
    SoC_prev = x(1);
    Vp_prev = x(2);
    
    % --- Prediction ---
    SoC_pred = SoC_prev - (dt * I(k-1)) / Qn;
    Vp_pred = exp(-dt/(Rp*Cp)) * Vp_prev + Rp * (1 - exp(-dt/(Rp*Cp))) * I(k-1);
    x_pred = [SoC_pred; Vp_pred];

    % Jacobians
    A = [1, 0;
         0, exp(-dt/(Rp*Cp))];
    B = [-dt/Qn;
         Rp * (1 - exp(-dt/(Rp*Cp)))];
    
    P_pred = A * P * A' + Q;

    % --- Measurement update ---
    Vt_meas = OCV(SoC_pred) - Vp_pred - I(k-1) * Rt;
    C = [OCV_derivative(SoC_pred), -1];  % H matrix (Jacobian of output)
    K = P_pred * C' / (C * P_pred * C' + R);  % Kalman gain

    % Assume voltage measured from experiment:
    V_meas = OCV(SOC_true(k)) - Vp_prev - I(k-1) * Rt + sqrt(R)*randn();  % Simulated measurement

    y = V_meas;
    y_hat = OCV(SoC_pred) - Vp_pred - I(k-1) * Rt;

    x = x_pred + K * (y - y_hat);
    P = (eye(2) - K * C) * P_pred;

    % Save estimates
    x_est(:,k) = x;
    V_est(k) = y_hat;
end

% --------- PLOTTING ---------
time = (0:N-1)*dt;
figure;
subplot(2,1,1);
plot(time, x_est(1,:) * 100, 'b', 'DisplayName','Estimated SoC');
hold on; plot(time, SOC_true * 100, 'k--', 'DisplayName','True SoC');
xlabel('Time (s)'); ylabel('SoC (%)'); legend; grid on;

subplot(2,1,2);
plot(time, V_est, 'r'); xlabel('Time (s)'); ylabel('Estimated Voltage (V)'); grid on;

% --- Derivative of OCV (you should replace this with actual slope values)
function dVoc = OCV_derivative(soc)
    dVoc = 0.5; % constant for linear function: OCV = 3.5 + 0.5*soc
end
