% Reset
clear; clc;

% Load the image data
imds = imageDatastore('data', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the data into training and validation sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);
numClasses = numel(categories(imdsTrain.Labels));

tfmImdsTrain = transform(imdsTrain, @lbp_filter);
tfmImdsTest = transform(imdsTest, @lbp_filter);

trainStore = combine(tfmImdsTrain, arrayDatastore(imdsTrain.Labels));
testStore = combine(tfmImdsTest, arrayDatastore(imdsTest.Labels));

% CNN layers
inputSize = [48 48 1];
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer([5 5], 64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5], 64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5], 16, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Set the training options for the network
options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', testStore, ...
    'ValidationFrequency', 30);

% Train the network using the training data
netmdl = trainNetwork(trainStore, layers, options);

% Evaluate the accuracy of the trained network using the validation data
gt = imdsTest.Labels;

[YPred, scores] = classify(netmdl, testStore);
accuracy = mean(YPred == gt);

% Calculate the confusion matrix
[m, order] = confusionmat(gt, YPred);

