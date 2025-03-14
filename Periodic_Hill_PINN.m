%% Add the npy-matlab library to the path

addpath(fullfile(pwd, 'npy-matlab'));
disp(['Current directory: ', pwd]);

%cheching if the files are there in the directory 
fullFilePath = fullfile(pwd, 'data', 'komegasst_PHLL_case_1p0_Cx.npy');
if exist(fullFilePath, 'file') ~= 2
    error('File does not exist: %s', fullFilePath);
end

% Loading the datasets
Cx = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_Cx.npy'));
Cy = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_Cy.npy'));
Ux = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_Ux.npy'));
Uy = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_Uy.npy'));
Pressure = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_p.npy'));
gradU = readNPY(fullfile('data', 'komegasst_PHLL_case_1p0_gradU.npy'));



gradUxx = gradU(:,1);
gradUxy = gradU(:,2);
gradUyx = gradU(:,4);
gradUyy = gradU(:,5);

% Step1.1: Combine the Data into a Single Matrix
data = [Cx, Cy, Ux, Uy,gradUxx,gradUxy,gradUyx,gradUyy,Pressure];


% Step 1.2: Handle Missing or Invalid Data
% Check for missing (NaN) or infinity values
if any(isnan(data(:))) || any(isinf(data(:)))
    disp('Missing or invalid values detected. Cleaning the data...');
    
    % Fill missing values with linear interpolation
    data = fillmissing(data, 'linear', 1); % Among columns
    %data = fillmissing(data, 'linear', 2); % Along rows
    
    % Replace infinite values with the columnn mean
    data(isinf(data)) = mean(data(~isinf(data)));
else
    disp('No missing or invalid values detected.');
end



% Step 1.3: Normalize the Data (Min-Max Normalization to [0, 1])
dataMin = min(data, [], 1); % Minimum of each column
dataMax = max(data, [], 1); % Maximum of each column

% Avoid division by zero in case dataMax == dataMin
range = dataMax - dataMin;
range(range == 0) = 1;

normalizedData = (data - dataMin) ./ range;



% Step 1.4: Split Features and Target
% Features: Cx,Cy,Ux,Uy,  gradU | Target: Pressure
inputs = normalizedData(:, 1:8);   % Input features
targets = normalizedData(:, 9);    % Target output (Pressure)

%% Step 2: Defining Neural Network Architecture

inputSize = 8;
outputSize = 1;
hiddenLayerSize = [50, 50, 50];

layers = [
    featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(hiddenLayerSize(1), 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(hiddenLayerSize(2), 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(hiddenLayerSize(3), 'Name', 'fc3')
    tanhLayer('Name', 'tanh3')
    fullyConnectedLayer(outputSize, 'Name', 'output')
];

net = dlnetwork(layerGraph(layers));


%% Step 3: Training and Validation

numEpochs = 1000;
learningRate = 1e-4;
lambda = 0.001;
miniBatchSize = 8192;
numObservations = size(inputs, 1);

% Split data into training and validation(80% train, 20% validation)
cv = cvpartition(numObservations, 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

trainInputs = inputs(idxTrain, :);
trainTargets = targets(idxTrain, :);
valInputs = inputs(idxVal, :);
valTargets = targets(idxVal, :);

%% Step 3.1: Training Loop with Timing

avgGrad = [];
avgSqGrad = [];
trainLossHistory = zeros(numEpochs, 1);
valLossHistory = zeros(numEpochs, 1);

tic; % Start timing
for epoch = 1:numEpochs
    % Mini-batch selection
    idx = randperm(size(trainInputs, 1), miniBatchSize);
    X = trainInputs(idx, :);
    T = trainTargets(idx, :);
    
    % Convert to dlarray(deep learning array)
    dlX = dlarray(X', 'CB');  % 8 x miniBatchSize
    dlT = dlarray(T', 'CB');  % 1 x miniBatchSize
    
    % Compute gradients and loss
    [loss, gradients] = dlfeval(@modelGradients, net, dlX, dlT, lambda);
    
    % Update parameters
    [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad, avgSqGrad, epoch, learningRate);
    
    % Store training loss
    trainLossHistory(epoch) = extractdata(loss);
    
    % Validation loss every 500 epochs
    if mod(epoch, 500) == 0
        dlValX = dlarray(valInputs', 'CB');
        dlValT = dlarray(valTargets', 'CB');
        valP_pred = forward(net, dlValX);
        valLoss = mean((valP_pred - dlValT).^2, 'all');
        valLossHistory(epoch) = extractdata(valLoss);
        
        fprintf('Epoch %d, Training Loss = %.4f, Validation Loss = %.4f\n', ...
            epoch, trainLossHistory(epoch), valLossHistory(epoch));
    end
end
trainingTime = toc; % End timing
fprintf('PINN Training Time: %.2f seconds\n', trainingTime);

%% Step 3.2: Prediction and Visualization

dlInputs = dlarray(inputs', 'CB');
P_pred = forward(net, dlInputs);
P_pred = extractdata(P_pred)';  % Extract predictions and transpose to match targets
Cx_plot = inputs(:, 1);
Cy_plot = inputs(:, 2);

% Create a figure for predicted vs true pressure
figure('Position', [100, 100, 600, 800]);
% Subplot 1: Predicted Pressure
subplot(2,1,1);
triPred = delaunay(Cx_plot, Cy_plot);
trisurf(triPred, Cx_plot, Cy_plot, P_pred, 'EdgeColor', 'none');
view(2); shading interp; axis equal tight; colormap jet; colorbar;
title('Predicted Pressure');
xlabel('Cx (X-Coordinates)');
ylabel('Cy (Y-Coordinates)');

% Subplot 2: True Pressure
subplot(2,1,2);
triTruth = delaunay(Cx_plot, Cy_plot);
trisurf(triTruth, Cx_plot, Cy_plot, targets, 'EdgeColor', 'none');
view(2); shading interp; axis equal tight; colormap jet; colorbar;
title('True Pressure');
xlabel('Cx (X-Coordinates)');
ylabel('Cy (Y-Coordinates)');

%% Step 3.3: Validation Metrics and Error Visualization
% Compute error metrics on validation set

valP_pred = extractdata(forward(net, dlValX))';
MAE_val = mean(abs(valP_pred - valTargets));
RMSE_val = sqrt(mean((valP_pred - valTargets).^2));


% Error visualization on full dataset
figure('Position', [100, 100, 600, 400]);
triError = delaunay(Cx_plot, Cy_plot);
trisurf(triError, Cx_plot, Cy_plot, abs(P_pred - targets), 'EdgeColor', 'none');
view(2); shading interp; axis equal tight; colormap jet; colorbar;
title('Absolute Error in Pressure Prediction');
xlabel('Cx (X-Coordinates)');
ylabel('Cy (Y-Coordinates)');

% Plot training and validation loss history
figure('Position', [100, 100, 600, 400]);
plot(1:numEpochs, trainLossHistory, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
hold on;
plot(500:500:numEpochs, valLossHistory(500:500:end), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss Over Epochs');
legend('Location', 'northeast');
grid on;

%% Step 3.4: Comparison Table for 10 Validation Samples

sample_idx = 1:min(10, length(valTargets));
comparison_table = table(valTargets(sample_idx), valP_pred(sample_idx), ...
    'VariableNames', {'True Pressure', 'Predicted Pressure'}, ...
    'RowNames', arrayfun(@(x) sprintf('Sample %d', x), sample_idx, 'UniformOutput', false));
disp('Comparison Table (10 Validation Samples):');
disp(comparison_table);


%% Step 4: Analysis and Interpretation
% Error Metrics on Full Dataset and validation datasett

MAE_full = mean(abs(P_pred - targets));
RMSE_full = sqrt(mean((P_pred - targets).^2));

fprintf('Full Dataset Mean Absolute Error: %.4f\n', MAE_full);
fprintf('Validation Mean Absolute Error: %.4f\n', MAE_val);

fprintf('Full Dataset Root Mean Squared Error: %.4f\n', RMSE_full);
fprintf('Validation Mean Absolute Error: %.4f\n', MAE_val);



% Visualize Flow Fields
figure('Position', [100, 100, 800, 1200]);
subplot(3,1,1);
trisurf(delaunay(Cx, Cy), Cx, Cy, Ux, 'EdgeColor', 'none');
view(2); shading interp; colorbar;
title(' Velocity Field in X-Direction (Ux)');
xlabel('Cx'); ylabel('Cy'); axis equal;

subplot(3,1,2);
trisurf(delaunay(Cx, Cy), Cx, Cy, Uy, 'EdgeColor', 'none');
view(2); shading interp; colorbar;
title(' Velocity Field in Y-Direction (Uy)');
xlabel('Cx'); ylabel('Cy'); axis equal;

subplot(3,1,3);
trisurf(triPred, Cx_plot, Cy_plot, P_pred, 'EdgeColor', 'none');
view(2); shading interp; colorbar;
title('PINN Predicted Pressure Field');
xlabel('Cx'); ylabel('Cy'); axis equal;

% Computational Efficiency
fprintf('PINN Training Time: %.2f seconds\n', trainingTime);


% Discussion
disp('Analysis Summary:');
disp(['- Accuracy: PINN predictions show MAE = ', num2str(MAE_full, '%.4f'), ...
    ' and RMSE = ', num2str(RMSE_full, '%.4f')]);
disp(['- Efficiency: PINN training took ', num2str(trainingTime, '%.2f'), ...
    ' seconds']);


%% Loss Function(Navier-stokess equation)

function [loss, gradients] = modelGradients(net, dlX, dlT, lambda)
    % Forward pass
    P_pred = forward(net, dlX);
    
    % Data loss
    dataLoss = mean((P_pred - dlT).^2, 'all');
    
    % Physics-aware gradients
    spatial = dlX(1:2,:);  % Extract spatial coordinates
    gradP = dlgradient(sum(P_pred,'all'), spatial, 'EnableHigherDerivatives', true);
    
    dP_dx = gradP(1,:);
    dP_dy = gradP(2,:);
    
    % Extract physics parameters
    Ux = dlX(3,:);
    Uy = dlX(4,:);
    gradUxx = dlX(5,:);
    gradUxy = dlX(6,:);
    gradUyx = dlX(7,:);
    gradUyy = dlX(8,:);
    
    % Physics residual
    resCont = gradUxx + gradUyy;
    resX = dP_dx + Ux.*gradUxx + Uy.*gradUxy;
    resY = dP_dy + Ux.*gradUyx + Uy.*gradUyy;
    
    physicsLoss = mean(resCont.^2 + resX.^2 + resY.^2, 'all');
    
    % Total loss
    loss = dataLoss + lambda * physicsLoss;
    
    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end