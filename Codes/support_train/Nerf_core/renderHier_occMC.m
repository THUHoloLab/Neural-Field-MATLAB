function [rgbImageC,rgbImageF, depthF, w, s] = renderHier_occMC(dlo, ...
                                                      dld, ...
                                                      net_e,...
                                                      net_c,...
                                                      net_f, ...
                                                      params, ...
                                                      bound_box, ...
                                                      occbox)

%% 粗采样
[dlx_c, dld_c, zVals] = samplePointsCoarse(dlo, dld, bound_box, params);
%{
    dld_c: 3 x B
    dlx_c: 3 x 64 x B (The batch size for this training points is 64 x B)
    zVals: 64 x B
%}

dlx_c = reshape(dlx_c,3,[]); % 3 x (64 x B)
dld_c = reshape(repmat(permute(dld_c,[1,3,2]),[1,params.numCoarse,1]),3,[]);

idx = max(min(round((dlx_c(1,:) - bound_box(1,1)) ...
                                          / occbox.dx(1,1)),occbox.res),1);
idy = max(min(round((dlx_c(2,:) - bound_box(1,2)) ...
                                          / occbox.dx(1,2)),occbox.res),1);
idz = max(min(round((dlx_c(3,:) - bound_box(1,3)) ...
                                          / occbox.dx(1,3)),occbox.res),1);
ind = sub2ind([occbox.res, occbox.res, occbox.res],idx,idy,idz);
if_occ = occbox.box(ind);
[~,col] = find(if_occ);

if sum(col) ~= 0
    idx = gpuArray.zeros(1,size(dlx_c,2),'single');
    idx(:,col) = 1;

    dlout = net_c.forward(...
        net_e.forward(dlarray(dlx_c,'CB')),...
        dlarray(dld_c,'CB'));
    
    dlout = reshape(dlout .* idx,...
                    4,params.numCoarse,[]);

    [rgbImageC,~,w] = renderVolume(dlout,zVals,dld,params);
else
    w = dlarray(gpuArray.zeros(size(zVals),'single'));
    depthF = dlarray(gpuArray.zeros(1,size(dld,2),'single'));
    rgbImageC = dlarray(gpuArray(zeros(3,size(w,2),'single')));
    rgbImageF = dlarray(gpuArray(zeros(3,size(dld,2),'single')));
    s = zeros(size(zVals));
    disp('returns')
    return;
end

% 细采样
idx = gpuArray.zeros(size(if_occ),'single');
idx(col) = 1;
w = w(:) .* idx';
w = reshape(w,params.numCoarse,[]);

zValsMid = (zVals(2:end,:) + zVals(1:end-1,:)) / 2;
deterministicFlag = params.perturb == false;
zSamples = samplePointsFine(zValsMid, w(2:end-1,:), deterministicFlag, params);
zVals = sort([zVals; extractdata(zSamples)],1);

if params.applyDistortionLoss
    s = convEuclidToNorm(zVals);
    s = (s - params.s_n) / (params.s_f - params.s_n);
else
    s = zeros(size(zVals));
end

viewdirs_f = dld ./ sqrt(sum(dld.^2,1));
pts_f = permute(dlo,[1,3,2]) + permute(dld,[1,3,2]) .* permute(zVals,[3,1,2]);
pts_f = dlarray(reshape(pts_f,3,[]),'CB');

viewdirs_ext_f = repmat(permute(viewdirs_f,[1,3,2]),[1,params.numCoarse + params.numFine,1]);
viewdirs_ext_f = dlarray(reshape(viewdirs_ext_f,3,[]),'CB');

% idx = max(min(round((pts_f(1,:) - bound_box(1,1)) ...
%                                           / occbox.dx(1,1)),occbox.res),1);
% idy = max(min(round((pts_f(2,:) - bound_box(1,2)) ...
%                                           / occbox.dx(1,2)),occbox.res),1);
% idz = max(min(round((pts_f(3,:) - bound_box(1,3)) ...
%                                           / occbox.dx(1,3)),occbox.res),1);
% ind = sub2ind([occbox.res, occbox.res, occbox.res],idx,idy,idz);
% if_occ = occbox.box(ind);
% [~,col] = find(if_occ);

if sum(col) ~= 0
    % idx = gpuArray.zeros(1,size(pts_f,2),'single');
    % idx(:,col) = 1;

    % [dlsigma_f,dlrgb_f] = net_f.forward(net_e.forward(pts_f), viewdirs_ext_f);
    dlout_f = net_f.forward(net_e.forward(pts_f), viewdirs_ext_f);
    dlout_f = reshape(dlout_f,...
                      4, params.numFine + params.numCoarse, []);

    [rgbImageF, depthF, ~] = renderVolume(dlout_f,zVals,dld,params);
else
    depthF = dlarray(gpuArray.ones(1,size(dld,2),'single'));
    rgbImageF = dlarray(gpuArray(zeros(3,size(dld,2),'single')));
end

end

 