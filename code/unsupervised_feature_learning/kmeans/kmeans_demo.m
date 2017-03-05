
% CIFAR_DIR='/Users/EmileMathieu/Projets/kmeans_demo/cifar-10-batches-mat/';

% assert(~strcmp(CIFAR_DIR, '/path/to/cifar/cifar-10-batches-mat/'), ...
%        ['You need to modify kmeans_demo.m so that CIFAR_DIR points to ' ...
%         'your cifar-10-batches-mat directory.  You can download this ' ...
%         'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);
clear;clc;
%% Configuration
addpath minFunc;
rfSize = 6;
%numCentroids=1600; % ORIGINAL PAPER VALUE
numCentroids=1600/5/10*3; % TO BE TUNED
whitening=true;
%numPatches = 10000; % ORIGINAL PAPER VALUE
numPatches = 400000/5/10*3; % TO BE TUNED
CIFAR_DIM=[32 32 3];

%%
X = csvread('../../../data/Xtr.csv');
X = double(X(:,1:end-1));
% RESCALE DATA
X = X + abs(min(X(:)));
X = X * (255 / max(X(:)));
X = round(X);
%% COPY "Ytr.csv" FILE, DELETE HEADER AND RENAME IT "Ytr_wo_header.csv"
Y = csvread('../../../data/Ytr_wo_header.csv');
Y = double(Y(:,2) + 1); %% LABELS TO BE IN (1,10); NO IDEA WHY

%% Load CIFAR training data
fprintf('Loading training data...\n');
trainX = X(1:3000,:);
trainY = Y(1:3000);
%%
% % testY = Y(3001:end);
% % testX = X(3001:end,:);
% %f1=load([CIFAR_DIR '/data_batch_1.mat']);
% %f2=load([CIFAR_DIR '/data_batch_2.mat']);
% %f3=load([CIFAR_DIR '/data_batch_3.mat']);
% %f4=load([CIFAR_DIR '/data_batch_4.mat']);
% %f5=load([CIFAR_DIR '/data_batch_5.mat']);
% 
% %trainX = double(f1.data(1:5000,:));
% %trainY = double(f1.labels(1:5000)) + 1; % add 1 to labels!
% %trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
% %trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
% %clear f1 %f2 f3 f4 f5;
% 
% % extract random patches
% patches = zeros(numPatches, rfSize*rfSize*3);
% for i=1:numPatches
%   if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
%   
%   r = random('unid', CIFAR_DIM(1) - rfSize + 1);
%   c = random('unid', CIFAR_DIM(2) - rfSize + 1);
%   %patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
%   patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
%   patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
%   patches(i,:) = patch(:)';
% end
% % normalize for contrast
% csvwrite('../../../data/patches.csv',patches);
% patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
% 
% % whiten
% if (whitening)
%   C = cov(patches);
%   M = mean(patches);
%   [V,D] = eig(C);
%   P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
%   patches = bsxfun(@minus, patches, M) * P;
% end
% 
% % run K-means
% csvwrite('../../../data/MBpatches.csv',patches);
% centroids = run_kmeans(patches, numCentroids, 50);
% %show_centroids(centroids, rfSize); drawnow;
% 
% csvwrite('../../../data/centroids_learn.csv',centroids);
% csvwrite('../../../data/M.csv',M);
% csvwrite('../../../data/P.csv',P);
% % extract training features
% if (whitening)
%   trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
% else
%   trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
% end
% 
% % standardize data
% csvwrite('../../../data/trainXC.csv',trainXC);
% trainXC_mean = mean(trainXC);
% trainXC_sd = sqrt(var(trainXC)+0.01);
% trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
% trainXCs = [trainXCs, ones(size(trainXCs,1),1)];
%csvwrite('../../../data/X_features_kmeans.csv',trainXCs);
%%
%train classifier using SVM
trainXCsp = csvread('../../../data/features_python.csv');
trainXCsp = trainXCsp(1:3000,:);
%% train
C = 100;
theta = train_svm(trainXCsp, trainY, C, 100);

[val,labels] = max(trainXCsp*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

%% Load CIFAR test data
fprintf('Loading test data...\n');
testX = X(3001:end,:);
testY = Y(3001:end);
centroids = csvread('../../../data/centroids.csv');
M = csvread('../../../data/M.csv');
P = csvread('../../../data/P.csv');
trainXC_mean = csvread('../../../data/XCmean.csv');
trainXC_sd = csvread('../../../data/XCvar.csv');
trainXC_mean = trainXC_mean';
trainXC_sd = trainXC_sd';
%%
%f1=load([CIFAR_DIR '/test_batch.mat']);
%testX = double(f1.data);
%testY = double(f1.labels) + 1;
%clear f1;

% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

