clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = imread("peppers.png");
img = foo(img)/255;
img = imresize(img,[512,256]);

target = dlarray(reshape(img,[],3),"BC");

x = linspace(0,1,size(img,2));
y = linspace(0,1,size(img,1));

[x,y] = meshgrid(x,y);
z = 0.3*ones(size(x));

pos_batch = foo([x(:),y(:),z(:)]);

pos_batch = dlarray(pos_batch,"BC");

addpath(genpath('support_train\'))
rootFolder = '';

layers = [
    featureInputLayer(3,"Name","xyz-input");
    Hash_EncodeingLayer( ...
        "base_res",     16,...
        "device",       gpuDevice(),...
        "bounding_box", [0,0,0;1,1,1],...
        "feature_len",  4,...
        "high_res",     2048,...
        "level",        8,...
        "log2_hashmap_size",21);
];

for con = 1:3
layers = [layers;
    FC_SimpleLayer(64,"Name","fc" + con);
    reluLayer();
];
end

layers = [layers;
    FC_SimpleLayer(3,"Name",'fin');
];

net = dlnetwork(layers);


optimizer_E = optimizers.Adam(0.9,0.99,1e-4);

learnRate = 0.01;

%begin training
for iteration = 1:10000
    tStart = tic;

    [loss,dldw] = dlfeval(@model_loss, net, pos_batch, target);
    
    net = optimizer_E.step(net,dldw,iteration,learnRate);
    % 
    % if mod(iteration,50) == 0 
    %     [img_test, score] = validation(net, pos_batch, img, size(img));
    %     figure(121);
    %     imshow(img_test,[])
    %     drawnow;
    %     fprintf("training takes: %.3f s, psnr = %.4f \n",toc(tStart),score);
    % end
    wait(gpuDevice());
    fprintf("training takes: %.3f s \n",toc(tStart));
    this_loss = extractdata(loss);
    
    if iteration > 300 && (mod(iteration,100) == 0)
        learnRate = learnRate * 0.7;
    end

    if learnRate < 1e-9
        break;
    end
end



function [loss,dldw] = model_loss(net, xyzs, target)

predict = net.predict(xyzs);

loss = l2loss(predict,target);
dldw = dlgradient(loss, net.Learnables);

end

function [img, score] = validation(net,xyzs,target,img_sz)
predict = net.forward(xyzs);
img = extractdata(predict)';
img = reshape(img,[img_sz(1),img_sz(2),3]);

score = psnr(img,target);
end