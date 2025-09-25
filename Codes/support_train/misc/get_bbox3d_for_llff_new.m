function [min_bound,max_bound] = get_bbox3d_for_llff_new(dsTrain,params,nn,ff)

min_bound = [ 100,  100,  100];
max_bound = [-100, -100, -100];
reset(dsTrain)
while dsTrain.hasdata
    cc = read(dsTrain);
    pos = cc{2};
    data = helpers.computeRay(params.intrinsics, pos, params);
    sz = params.intrinsics.ImageSize;

    rays_o = data(:,:,1:3);
    rays_d = data(:,:,4:6);
    d_norm = sqrt(sum(rays_d.^2,3));
    for xxx = [1,sz(2)]
        for yyy = [1,sz(1)]
            min_point = rays_o(yyy,xxx,:) + rays_d(yyy,xxx,:) .* nn;
            max_point = rays_o(yyy,xxx,:) + rays_d(yyy,xxx,:) .* ff;

            min_point = reshape(min_point,[1,3,1]);
            max_point = reshape(max_point,[1,3,1]);


            
            min_bound = min(min(min_bound,min_point),max_point);
            max_bound = max(max(max_bound,min_point),max_point);
        end
    end
end
    min_bound = min_bound - [0.1,0.1,0.1];
    max_bound = max_bound + [0.1,0.1,0.1];
end
