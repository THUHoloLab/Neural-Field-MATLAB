function samples = samplePointsFine(bins, weights, det, params)
weights = weights + 1e-5;
pdf = weights ./ sum(weights,1);

% cdf = cumsum(extractdata(pdf),1); % replace cumsum as differentiable code
U = single(triu(ones(size(pdf,1))));
if (params.executionEnvironment == "auto" && canUseGPU) || params.executionEnvironment == "gpu"
    U = gpuArray(U);
end
pdf_tmp = U .* permute(pdf,[1,3,2]);
cdf2 = permute(sum(pdf_tmp,1),[2,3,1]);
cdf = [dlarray(zeros(1,size(pdf,2),'single')); cdf2];

if det
    r = linspace(single(0),single(1),params.numFine);
    r = repmat(r',[1,size(bins,2)]);
else
    r = rand(params.numFine,size(bins,2),"single");
end
if (params.executionEnvironment == "auto" && canUseGPU) || params.executionEnvironment == "gpu"
    r = gpuArray(r);
end

% Find falling bins for each random number
flags = permute(r,[1,3,2]) >= permute(cdf,[3,1,2]);
inds = sum(flags,2);
inds = permute(inds,[1,3,2]);

below = max(zeros(size(inds)), inds-1);
above = min((size(cdf,1)-1)*ones(size(inds)), inds);
inds_g = cat(3,below,above);
inds_g = permute(inds_g,[2,1,3]);

% Find the corresponding indeces for each bin
inds_g = permute(inds_g,[2,1,3]);

index_all = sub2ind(size(cdf),...
    reshape(inds_g+1,[],1),...
    repmat(repelem((1:size(r,2))',params.numFine,1),[2,1]));
cdf_g = permute(reshape(cdf(index_all),size(inds_g)),[2,1,3]);
bins_g = permute(reshape(bins(index_all),size(inds_g)),[2,1,3]);
denom = cdf_g(:,:,2) - cdf_g(:,:,1);
denom(denom < 1e-5) = 1;
denom = denom';
t = (r - cdf_g(:,:,1)') ./ denom;
samples = (bins_g(:,:,1)' + t.*(bins_g(:,:,2)'-bins_g(:,:,1)'));

end