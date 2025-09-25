function [image_c, image_f, depth_img] = renderImage(o, d, ...
                                                     net_e, net_c, net_f,...
                                                     params,...
                                                     bound_box, occbox)
numSamples = size(o,1);
numIterations = ceil(numSamples/params.miniBatchSize);
numFractions = numSamples - params.miniBatchSize * (numIterations-1);
image_c = zeros(params.height * params.width,3,"single");
image_f = zeros(params.height * params.width,3,"single");
depth_img = zeros(params.height*params.width,1,"single");
for batchIndex = 1:numIterations
    clc
    disp(['remains: ',num2str(round(batchIndex/numIterations * 100)),'%'])
    % Read mini-batch of data.
    if batchIndex == numIterations && numFractions > 0
        index = (batchIndex-1)*params.miniBatchSize+1:numSamples;
    else
        index = (batchIndex-1)*params.miniBatchSize+1:batchIndex*params.miniBatchSize;
    end
    % Convert to dlarray
    dlo = o(index,:)';
    dld = d(index,:)';
    % if (params.executionEnvironment == "auto" && canUseGPU) || params.executionEnvironment == "gpu"
        dlo = gpuArray(dlo);
        dld = gpuArray(dld);
    % end
    [dlrgb_c,dlrgb_f, depth, ~, ~] = renderHier_occ(dlo, ...
                                                    dld, ...
                                                    net_e, ...
                                                    net_c, ...
                                                    net_f, ...
                                                    params, ...
                                                    bound_box, occbox);

    image_c(index,:) = gather(permute(extractdata(dlrgb_c),[2,1]));
    image_f(index,:) = gather(permute(extractdata(dlrgb_f),[2,1]));
    depth_img(index,:) = depth;
end
image_c = reshape(image_c,params.height,params.width,3);
image_f = reshape(image_f,params.height,params.width,3);
depth_img = reshape(depth_img,params.height,params.width);
end