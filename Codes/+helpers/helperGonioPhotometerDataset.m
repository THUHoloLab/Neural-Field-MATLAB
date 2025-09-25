function [dsTrain, dsValid, params] = helperGonioPhotometerDataset
% This helper function uses a labeled data set which comes from NeRF synthetic
% dataset
% Copyright 2023 The MathWorks, Inc.

datasetPath = fullfile("data","goniophotometer","20240729spiral");

[~,name,~] = fileparts(datasetPath);
trTrainJson = fileread(fullfile(datasetPath,"transforms_train.json"));
trValJson = fileread(fullfile(datasetPath,"transforms_val.json"));
trTrain = jsondecode(trTrainJson);
trVal = jsondecode(trValJson);
filePathTrain = arrayfun(@(x)string(x.file_path),trTrain.frames);
filePathVal = arrayfun(@(x)string(x.file_path),trVal.frames);
imdsTrain = imageDatastore(fullfile(datasetPath,filePathTrain));
I = read(imdsTrain);

params.modelName = name;
params.height = size(I,1);
params.width = size(I,2);
ncols = params.width;
nrows = params.height;
params.numCoarse = 64;
params.numFine = 192;
params.noise = 0;
params.sampleFromSingle = false;
params.roi = [-1,1,-1,1,-1,1]/2;
params.useViewDirs = true;
params.valStepPixels = 2;
params.valNumFrames = 1;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
% params.bg = single([174; 232; 175]/255);
params.bg = single([0; 0; 0]);
% focal = 340;
fx_org = trTrain.fl_x;
fy_org = trTrain.fl_y;
cx_org = params.width / 2;
cy_org = params.height / 2;
k1 = 0;
k2 = 0;
p1 = 0;
p2 = 0;

fx = fx_org * params.width  / ncols;
fy = fy_org * params.height / nrows;
cx = cx_org * params.width  / ncols;
cy = cy_org * params.height / nrows;

intrinsicMatrix = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
distortionCoefficients = [k1, k2, p1, p2, 0];
imageSize = [params.height, params.width];
params.intrinsics = cameraIntrinsicsFromOpenCV(intrinsicMatrix,distortionCoefficients,imageSize);

imdsTrain = imageDatastore(fullfile(datasetPath,filePathTrain),...
    "ReadFcn",@(x) undistortImage(imresize(imread(x),[params.height, params.width]),params.intrinsics));

imdsValid = imageDatastore(fullfile(datasetPath,filePathVal),...
    "ReadFcn",@(x) undistortImage(imresize(imread(x),[params.height, params.width]),params.intrinsics));

scale = 1;

% R_combi = [0 -1 0; 1 0 0; 0 0 1] * [0 1 0; 1 0 0; 0 0 1];
% 
% transform_func = @(x) rigidtform3d([...
%     R_combi * [x(1:3,1:2), -x(1:3,3)], x(1:3,4)*scale; ...   % 旋转部分 + 缩放平移
%     0, 0, 0, 1 ...                                              % 齐次坐标最后一行
% ]);
% 
% camPosesTrain = cellfun(transform_func, {trTrain.frames.transform_matrix});
% camPosesValid = cellfun(transform_func, {trVal.frames.transform_matrix});


camPosesTrain = cellfun(@(x)rigidtform3d([eul2rotm(rotm2eul([x(1:3,[2,1]),-x(1:3,3)]))*[0,-1,0;1,0,0;0,0,1],x(1:3,4)*scale; 0,0,0,1]),...
    {trTrain.frames.transform_matrix});
camPosesValid = cellfun(@(x)rigidtform3d([eul2rotm(rotm2eul([x(1:3,[2,1]),-x(1:3,3)]))*[0,-1,0;1,0,0;0,0,1],x(1:3,4)*scale; 0,0,0,1]),...
    {trVal.frames.transform_matrix});

dsPosesTrain = arrayDatastore(camPosesTrain');
dsPosesValid = arrayDatastore(camPosesValid');

dsTrain = combine(imdsTrain,dsPosesTrain);
dsValid = combine(imdsValid,dsPosesValid);
params.t_n = 0.1;
params.t_f = 3;

% Define a camera trajectory for novel view synthesis after training
% rotCenter = [0,0,0];
% N = 30*2;
% numRot = 1;
% angels = 2*pi*((1:N)'-1)*numRot/N;
% zrange = 0.5*ones(N,1);
% yrange = -1.5*ones(N,1);
% xrange = 0*ones(N,1);
% tr = [xrange,yrange,zrange];
% pitch = pi - atan2(tr(:,3),tr(:,2));
% poses = num2cell([tr, pitch, angels],2);
% tforms = cellfun(@(x)rigidtform3d(trvec2tform(rotCenter)*...
%     eul2tform([x(5),0,0])*...
%     trvec2tform(-rotCenter)*...
%     trvec2tform(rotCenter+x(1:3))*...
%     eul2tform([0,0,-pi/2-x(4)])),poses);
% params.testCamPoses = tforms;
end