function [rgbImageC,rgbImageF, depthF, w, s] = renderHier_occ(dlo, ...
                                                      dld, ...
                                                      net_e,...
                                                      net_c,...
                                                      net_f, ...
                                                      params, ...
                                                      bound_box, occbox)

% global bound_box occbox
bboxMin = gpuArray(single(bound_box(1,:)));
%% 粗采样
[dlx_c, dld_c, zVals] = samplePointsCoarse(dlo, dld, bound_box, params);
%{
    dld_c: 3 x B
    dlx_c: 3 x 64 x B (The batch size for this training points is 64 x B)
    zVals: 64 x B
%}
dlx_c = reshape(dlx_c,3,[]); % 3 x (64 x B)
dld_c = reshape(repmat(permute(dld_c,[1,3,2]),[1,params.numCoarse,1]),3,[]);


col = checkOccbox(dlx_c,...
                  occbox.box,...
                  bboxMin,...
                  occbox.dx);

dlx_c = dlarray(dlx_c,'CB');
dld_c = dlarray(dld_c,'CB');

if sum(col) ~= 0
    dlout = net_c.forward(net_e.forward(dlx_c),dld_c);
    dlout = dlout .* col;
else
    dlout = dlarray(gpuArray.zeros(4,size(dlx_c,2),'single'));
end

dlout = reshape(dlout,4,params.numCoarse,[]);
[rgbImageC,~,w] = renderVolume(dlout,zVals,dld,params);

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

col = checkOccbox(pts_f,...
                  occbox.box,...
                  bboxMin,...
                  occbox.dx);

if sum(col) ~= 0
    dlout_f = net_f.forward(...
              net_e.forward(dlarray(pts_f,'CB')),...
                            dlarray(viewdirs_ext_f,'CB'));
    dlout_f = dlout_f .* col;
else
    dlout_f = dlarray(gpuArray.zeros(4,size(pts_f,2),'single'));
end

dlout_f = reshape(dlout_f,4,params.numFine + params.numCoarse,[]);
[rgbImageF, depthF, ~] = renderVolume(dlout_f,zVals,dld,params);

end

 