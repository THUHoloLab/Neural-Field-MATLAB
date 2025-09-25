function [min_bound,max_bound] = get_bbox3d_for_blender(dsTrain,params)


min_bound = [ 100,  100,  100];
max_bound = [-100, -100, -100];

while dsTrain.hasdata
    cc = read(dsTrain);

    pos = cc{2}.Translation;

    min_bound = min(min(min_bound,pos),pos);
    max_bound = max(max(max_bound,pos),pos);
end
end