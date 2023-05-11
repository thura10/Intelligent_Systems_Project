% Reset
clear; clc;

% Load the image data
imds = imageDatastore('data', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the data into training and validation sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);
numClasses = numel(categories(imdsTrain.Labels));

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

pixelRange = [-5 5];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Set the training options for the network
options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30);

% Train the network using the training data
netmdl = trainNetwork(augImdsTrain, layers, options);

% Evaluate the accuracy of the trained network using the validation data
gt = imdsTest.Labels;

[YPred, scores] = classify(netmdl, imdsTest);
accuracy = mean(YPred == gt);

% Calculate the confusion matrix
[m, order] = confusionmat(gt, YPred);

