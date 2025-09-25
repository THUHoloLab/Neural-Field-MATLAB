function [lossValid, imageDebug, psnr_f] = validation(net_e,net_c,net_f, ...
                                                      mbqValid, ...
                                                      iteration, ...
                                                      elapsedTime, ...
                                                      params,...
                                                      bound_box, ...
                                                      occbox)
% Evaluate the model on the first image
paramsInference = params;
stepPixels = paramsInference.valStepPixels;
numFrames = paramsInference.valNumFrames;
paramsInference.width = paramsInference.width / stepPixels;
paramsInference.height = paramsInference.height / stepPixels;
paramsInference.noise = 0;
paramsInference.perturb = false;
intrinsics = paramsInference.intrinsics;
paramsInference.intrinsics = cameraIntrinsics(intrinsics.FocalLength / stepPixels,...
    intrinsics.PrincipalPoint / stepPixels,...
    intrinsics.ImageSize / stepPixels);
paramsInference.train = false;
paramsInference.miniBatchSize = 1024 * 10;

% Split the data into three dlarrays
if ~iscell(mbqValid)
    reset(mbqValid);
end
lossValidN_f = zeros(numFrames,1);
lossValidN = zeros(numFrames,1);
for k = 1
    if iscell(mbqValid)
        rgbImg = mbqValid{k,1};
        tform = mbqValid{k,2};
        indSamples = 1:(size(rgbImg,1)*size(rgbImg,2));
        dlData = createMinibatch(rgbImg, tform, indSamples, params);
    else
        dlData = next(mbqValid);
    end
    dlData = reshape(dlData,12,params.height,params.width);
    dlrgb = dlData(1:3,1:stepPixels:end,1:stepPixels:end);
    dlo = reshape(dlData(4:6,1:stepPixels:end,1:stepPixels:end),3,[]);
    dld = reshape(dlData(7:9,1:stepPixels:end,1:stepPixels:end),3,[]);
    % dlrgb = dlarray(dlrgb);
    % dlo = dlarray(dlo);
    % dld = dlarray(dld);
    [image_c, image_f, depth] = renderImage(dlo', dld',net_e,net_c,net_f, ...
                                                        paramsInference,...
                                                        bound_box, ...
                                                        occbox);
    image_gt = gather(permute((dlrgb),[2,3,1]));
    lossValidN_f(k) = mean((image_gt(:)-image_f(:)).^2);
    lossValidN(k) = mean((image_gt(:)-image_c(:)).^2) + lossValidN_f(k);

    depth(depth < paramsInference.t_n) = paramsInference.t_n;
    depth(depth > paramsInference.t_f) = paramsInference.t_f;
    depth_norm = (depth - paramsInference.t_n) / (paramsInference.t_f - paramsInference.t_n);
    depth_norm(isinf(depth_norm)) = 1;
    depth_col = ind2rgb(im2uint8(depth_norm),jet);
    
    image_gt = insertText(image_gt,[1,1],"GT","BoxOpacity",0,"FontSize",26,"FontColor",[1,1,1]);
    image_gt = insertText(image_gt,[1,50],"Iter:"+string(iteration),"BoxOpacity",0.5,"FontSize",26,"FontColor",[1,1,1]);
    image_gt = insertText(image_gt,[1,110],"Time:"+string(elapsedTime),"BoxOpacity",0.5,"FontSize",26,"FontColor",[1,1,1]);
    image_c = insertText(image_c,[1,1],"Coarse","BoxOpacity",0,"FontSize",26,"FontColor",[1,1,1]);
    image_f = insertText(image_f,[1,1],"Fine","BoxOpacity",0,"FontSize",26,"FontColor",[1,1,1]);
    depth_col = insertText(depth_col,[1,1],"Depth","BoxOpacity",0,"FontSize",26,"FontColor",[1,1,1]);

    % figure(145);
    % subplot(221);imshow(image_gt,[])
    % subplot(222);imshow(image_c,[])
    % subplot(223);imshow(image_f,[])
    % subplot(224);imshow(depth_norm,[]);colormap jet;colorbar;

    ccc1 = mean(occbox.box,3);ccc1 = mat2gray(ccc1); ccc1 = imresize(ccc1,[size(image_f,1),size(image_f,2)],'nearest');
    ccc1 = ind2rgb(im2uint8(ccc1),turbo);

    ccc2 = mean(permute(occbox.box,[1,3,2]),3);
    ccc2 = mat2gray(ccc2); ccc2 = imresize(ccc2,[size(image_f,1),size(image_f,2)],'nearest');
    ccc2 = ind2rgb(im2uint8(ccc2),turbo);
    imageDebug = [image_gt, image_c, image_f;depth_col,ccc1,ccc2];
end
lossValid = mean(lossValidN);
lossValid_f = mean(lossValidN_f);
psnr_f = -10*log10(lossValid_f);
end