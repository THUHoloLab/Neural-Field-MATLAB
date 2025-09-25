function [dlx, viewDirs, zVals] = samplePointsCoarse(rays_o, rays_d, bound_box, params)
viewDirs = rays_d;
viewDirs = viewDirs ./ sqrt(sum(viewDirs.^2,1));

rays_o = permute(rays_o,[1,3,2]);
rays_d = permute(rays_d,[1,3,2]);

if params.applyDistortionLoss
    s = linspace(0,1,params.numCoarse+1)';
    s = s(1:end-1,:);
    params.s_n = convEuclidToNorm(params.t_n);
    params.s_f = convEuclidToNorm(params.t_f);
    s = repmat(s,[1,size(rays_o,3)]);
    if params.perturb
        mids = 0.5 * (s(2:end,:) + s(1:end-1,:));
        upper = cat(1,mids, s(end,:));
        lower = cat(1,s(1,:),mids);
        t_rand = rand(size(s));
        s = lower + (upper - lower) .* t_rand;
    end
    zVals = convNormToEuclid(s*(params.s_f - params.s_n) + params.s_n);
else
    [t_min, t_max] = ray_aabb_intersection(rays_o, rays_d, bound_box);
    zVals = (t_max - t_min) .* linspace(0,1,params.numCoarse) + t_min;
    zVals = squeeze(zVals);
    
    if params.perturb
        mids = 0.5 * (zVals(2:end,:) + zVals(1:end-1,:));
        upper = cat(1,mids, zVals(end,:));
        lower = cat(1,zVals(1,:),mids);
        if (params.executionEnvironment == "auto" && canUseGPU) || params.executionEnvironment == "gpu"
            t_rand = rand(size(zVals),"single","gpuArray");
        else
            t_rand = rand(size(zVals));
        end
        zVals = lower + (upper - lower) .* t_rand;
    end
end

dlx = rays_o + rays_d .* permute(zVals,[3,1,2]);

end


function [t_min, t_max] = ray_aabb_intersection(o, d, bound_box)
% 计算多条光线与AABB的交点
% 输入:
%   o: 光线起点 [3, n] (n=1024)
%   d: 光线方向单位向量 [3, n] 
%   bound_box: 包围盒边界 [2, 3]，第一行是min，第二行是max
% 输出:
%   t_min: 进入深度 [1, n]，无交点时为-inf
%   t_max: 离开深度 [1, n]，无交点时为-inf
% 提取包围盒边界
bounds_min = bound_box(1,:)';
bounds_max = bound_box(2,:)';

t1 = (bounds_min - o) ./ (d + 1e-5);
t2 = (bounds_max - o) ./ (d + 1e-5);

swap_mask = t1 > t2;
temp = t1(swap_mask);
t1(swap_mask) = t2(swap_mask);
t2(swap_mask) = temp;

t_min = max(t1,[],1);
t_max = min(t2,[],1);
no_intersection = t_min > t_max;
t_min(no_intersection) = -1;
t_max(no_intersection) = -1;


% 检查交点是否在光线正方向上
behind_mask = t_max < 0;
t_min(behind_mask) = -1;
t_max(behind_mask) = -1;

% 处理起点在盒子内部的情况
inside_mask = t_min < 0 & t_max > 0;
t_min(inside_mask) = 0;
end