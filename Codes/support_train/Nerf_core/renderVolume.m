function [rgbImage,depthImage,weights] = renderVolume(raw,z_vals,rays_d,params)
dists = z_vals(2:end,:) - z_vals(1:end-1,:);
dists = [dists; 1e10*ones(1,size(dists,2))];
rays_d_norm = sqrt(sum(rays_d.^2,1));
dists = dists .* rays_d_norm;
dists = permute((dists),[3,1,2]);
rgb = sigmoid(raw(1:3,:,:));

if params.noise > 0
    sigma = raw(4,:,:) + params.noise*randn(size(raw(4,:,:)));
else
    sigma = raw(4,:,:);
end

alpha = 1 - exp(-relu(sigma).*dists);

% cumprod is not supported for dlarray
density = (1 - alpha(:,1:end-1,:) + 1e-10);
density = cat(2,ones(1,1,size(alpha,3),"single"),density);

L = tril(gpuArray.ones(size(density,2),'single'));
density_tril = L .* density + ~L;
T = permute(prod(density_tril,2),[2,1,3]);

weights = alpha .* T;
rgbImage = sum(weights.*rgb,2);

weights = permute(weights,[2,3,1]);
weights = squeeze(weights);
if params.train
    % "expectation" in training mode
    depthImage = sum(weights .* z_vals);
else
    % "median depth" in inference mode
    weights_denoised = weights;
    weights_denoised(weights_denoised < params.depthWeightThreshold) = 0;
    cumulative_contribution = cumsum(extractdata(weights_denoised));
    opaqueness = cumulative_contribution >= 0.3;
    padded_opaqueness = [false(1,size(opaqueness,2));
        opaqueness(1:end-1,:)];
    opaqueness_mask = xor(opaqueness,padded_opaqueness);
    depthImage = sum(opaqueness_mask .* z_vals);
end

acc = sum(weights,1);
rgbImage = permute(rgbImage,[1,3,2]) + params.bg.*(1 - acc);
end