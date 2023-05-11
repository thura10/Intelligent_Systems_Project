% Reset
clear; clc;

% Load the image data
imds = imageDatastore('data', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the data into training and validation sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7);
trainLabels = imdsTrain.Labels;
testLabels = imdsTest.Labels;

pixelRange = [-5 5];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% Hog variables
cellSize = [4 4];
hogFeatureSize = 4356; % output feature size for a cell size of 4x4

% Transform image into HOG features
numTrain = numel(imdsTrain.Files);
trainFeatures = zeros(numTrain, hogFeatureSize, 'single');

for i = 1:numTrain
    img = readimage(imdsTrain, i);
    img = im2gray(img);
    img = imageAugmenter.augment(img);

    trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Transform test
numValidation = numel(imdsTest.Files);
testFeatures = zeros(numValidation, hogFeatureSize, 'single');

for i = 1:numValidation
    img = readimage(imdsTest, i);
    img = im2gray(img);

    testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Train a SVM classifier
svm_mdl = fitcecoc(trainFeatures, trainLabels);

% Predict the validation set using the classifier
svm_pred = predict(svm_mdl, testFeatures);
% Confusion matrix
svm_conf = confusionmat(testLabels, svm_pred);
% Calculate accuracy
svm_accu = mean(svm_pred == testLabels);

% Train a deep learning CNN
numClasses = numel(categories(trainLabels));
layers = [
    featureInputLayer(hogFeatureSize, Normalization="zscore")
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Set the training options for the network
options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', { testFeatures, testLabels }, ...
    'ValidationFrequency', 30 ...
    );

% Train the network using the training data
cnn_mdl = trainNetwork(trainFeatures, trainLabels, layers, options);

cnn_pred = classify(cnn_mdl, testFeatures);
cnn_accu = mean(cnn_pred == testLabels);

% Calculate the confusion matrix
cnn_conf = confusionmat(testLabels, cnn_pred);

