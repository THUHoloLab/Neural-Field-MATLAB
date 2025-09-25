function [out_parser, depthF, w, s] = renderHier_occMC(dlo, ...
                                                      dld, ...
                                                      net_e,...
                                                      net_c,...
                                                      net_f, ...
                                                      params, ...
                                                      bound_box, ...
                                                      occbox)

% 粗采样
[dlx_c, dld_c, zVals] = samplePointsCoarse(dlo, dld, params);
%{
    dld_c: 3 x B
    dlx_c: 3 x 64 x B (The batch size for this training points is 64 x B)
    zVals: 64 x B
%}
dlx_c = reshape(dlx_c,3,[]); % 3 x (64 x B)
dld_c = reshape(repmat(permute(dld_c,[1,3,2]),[1,params.numCoarse,1]),3,[]);

idx = max(min(round((dlx_c(1,:) - bound_box(1,1)) / occbox.dx(1,1)),occbox.res),1);
idy = max(min(round((dlx_c(2,:) - bound_box(1,2)) / occbox.dx(1,2)),occbox.res),1);
idz = max(min(round((dlx_c(3,:) - bound_box(1,3)) / occbox.dx(1,3)),occbox.res),1);
ind = sub2ind([occbox.res,...
               occbox.res, ...
               occbox.res],idx,idy,idz);
if_occ = occbox.box(ind);
[~,col] = find(if_occ);
%
dlsigma = dlarray(gpuArray(zeros(1,size(dlx_c,2),'single')),'CB');
dlrgb   = dlarray(gpuArray(zeros(3,size(dlx_c,2),'single')),'CB');


dlx_c = dlarray(dlx_c,'CB');
dld_c = dlarray(dld_c,'CB');

if sum(col) ~= 0
x_emb = forward(net_e,dlx_c(:,col));
[dlsigma(:,col),dlrgb(:,col)]= forward(net_c,x_emb,dld_c(:,col));
end
% dlrgb = repmat(dlrgb,[3,1]);

dlout = [dlrgb; dlsigma];
dlout = reshape(dlout,4,params.numCoarse,[]);

[rgbImageC,~,w] = renderVolume(dlout,zVals,dld,params);

% 细采样


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
viewdirs_ext_f = repmat(permute(viewdirs_f,[1,3,2]),[1,params.numCoarse + params.numFine,1]);

viewdirs_ext_f = reshape(viewdirs_ext_f,3,[]);
pts_f = reshape(pts_f,3,[]);

pts_f = dlarray(pts_f,'CB');
viewdirs_ext_f = dlarray(viewdirs_ext_f,'CB');

idx = max(min(round((pts_f(1,:) - bound_box(1,1)) / occbox.dx(1,1)),occbox.res),1);
idy = max(min(round((pts_f(2,:) - bound_box(1,2)) / occbox.dx(1,2)),occbox.res),1);
idz = max(min(round((pts_f(3,:) - bound_box(1,3)) / occbox.dx(1,3)),occbox.res),1);
ind = sub2ind([occbox.res,...
               occbox.res, ...
               occbox.res],idx,idy,idz);
if_occ = occbox.box(ind);
[~,col] = find(if_occ);
%


dlsigma_f = dlarray(gpuArray(zeros(1,size(pts_f,2),'single')),'CB');
dlrgb_f   = dlarray(gpuArray(zeros(3,size(pts_f,2),'single')),'CB');


if sum(col) ~= 0
x_emb = forward(net_e,pts_f(:,col));
[dlsigma_f(:,col),dlrgb_f(:,col)]= forward(net_f,x_emb,viewdirs_ext_f(:,col));
end


% dlrgb_f = repmat(dlrgb_f,[3,1]);
dlout_f = [dlrgb_f; dlsigma_f];
dlout_f = reshape(dlout_f,4,params.numFine + params.numCoarse,[]);


[rgbImageF, depthF, ~] = renderVolume(dlout_f,zVals,dld,params);


out_parser = {rgbImageC, rgbImageF};

end

 