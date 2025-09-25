function dataBatch = createMinibatch(rgbImgs, tforms, indSamplesAll, params)
rgbImgs = im2single(rgbImgs);
numImagePerBatch = size(indSamplesAll,1);
numRaysPerImage = size(indSamplesAll,2);

dataBatch = zeros(12,numImagePerBatch*numRaysPerImage,"single");
if (params.executionEnvironment == "auto" && canUseGPU) || params.executionEnvironment == "gpu"
    dataBatch = gpuArray(dataBatch);
end
for k = 1:numImagePerBatch
    rgbImg = rgbImgs(:,:,:,k);
    tform = tforms(:,:,k);
    indSamples = indSamplesAll(k,:);

    r = rgbImg(:,:,1);
    g = rgbImg(:,:,1);
    b = rgbImg(:,:,1);

    
    rgb = [r(indSamples); g(indSamples); b(indSamples)];


    [row,col] = ind2sub([params.height, params.width],indSamples);
    u = col(:) - 1;
    v = row(:) - 1;
    xy = ([u, v] - params.intrinsics.PrincipalPoint) ./ params.intrinsics.FocalLength;
    xyz1 = [xy, ones(size(xy,1),2)]';
    d = tform * xyz1;
    d = d(1:3,:);
    o = tform(1:3,4);
    d = d - o;
    d_norm = sqrt(sum(d.^2,1));
    t_n = params.t_n * ones(size(d_norm));
    t_f = params.t_f * ones(size(d_norm));
    dataBatch(:,((k-1)*numRaysPerImage+1):(k*numRaysPerImage)) = ...
        [rgb; repmat(single(o),[1,size(d,2)]); single(d); single(d_norm); single(t_n); single(t_f)];
end
end