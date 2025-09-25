classdef occupancy_box < handle
    properties
        res
        box
        dx
        next
    end

    properties (Access = private)
        ind
        cor
    end

    methods
        function self = occupancy_box(resolution,bound_box)
            self.res = resolution;
            self.box = gpuArray(ones(self.res,self.res,self.res,'single'));
            % self.box = logical(self.box);
            self.dx = (bound_box(2,:) - bound_box(1,:)) / self.res;
            self.dx = gpuArray(single(self.dx));

            xx = linspace(bound_box(1,1) + self.dx(1,1)/2,bound_box(2,1) + self.dx(1,1)/2,self.res);
            yy = linspace(bound_box(1,2) + self.dx(1,2)/2,bound_box(2,2) + self.dx(1,2)/2,self.res);
            zz = linspace(bound_box(1,3) + self.dx(1,3)/2,bound_box(2,3) + self.dx(1,3)/2,self.res);
            [xx,yy,zz] = meshgrid(xx,yy,zz);

            idx = max(min(round((xx - bound_box(1,1)) / self.dx(1,1)),self.res),1);
            idy = max(min(round((yy - bound_box(1,2)) / self.dx(1,2)),self.res),1);
            idz = max(min(round((zz - bound_box(1,3)) / self.dx(1,3)),self.res),1);

            ind = sub2ind([self.res,self.res,self.res],idy,idx,idz);

            self.cor = dlarray([xx(:),yy(:),zz(:)],'BC');
            self.ind = ind;
            self.next = 1;
        end

        function [self,bound_box] = update_box(self,net_E,net_C,bound_box)
            k = 5;

            x_emb = forward(net_E,self.cor);
            out = forward(net_C, x_emb, self.cor);
            dlsigma = out(end,:);
            % x_emb = forward(net_E,self.cor);
            % dlsigma = x_emb(1,:);
            % dlcolor = forward(net_c,viewdirs_ext_f(:,col),x_emb(2:end,:));

            dlsigma = extractdata(relu(dlsigma)); % relu(dlsigma).* sigmoid(max(dlcolor,[],1) + 6) + 1e-4 .* sigmoid(sigmoid(1*max(dlcolor,[],1)) - 3)
            temp = 0*self.box;
            temp(self.ind) = dlsigma;
            self.next = gpuArray(single(convn(temp,ones(k,k,k) / k^3,'same')));

            self.box = self.next / sum(self.next,'all');

            CC = bwconncomp(gather(self.box),26);
            for ii=1:CC.NumObjects
                currsum=sum(self.box(CC.PixelIdxList{ii}));
                if currsum < 0.1
                    self.box(CC.PixelIdxList{ii}) = 0;
                end
            end
            % toc
            idx = self.box > 0.025*max(self.box(:));
            self.box(idx) = 1;
            self.box(~idx) = 0;
            self.box = gpuArray(single(self.box));

                        % 
            % p = regionprops3(self.box,"Volume");
            % [~,maxIdx] = max([p.Volume]);
            % 
            % p = regionprops3(self.box,"VoxelIdxList");
            % cutoff = zeros(size(self.box),'single');
            % cutoff(p(maxIdx,1).VoxelIdxList{1}) = 1;
            % 
            % k = 5;
            % self.box = self.box .* cutoff;
            % self.box = convn(self.box,ones(k,k,k) / k^3,'same');
            % self.box(self.box>0) = 1;
        end

        % function self = step(self,ratio)
        %     self.box = ratio .* self.box + (1 - ratio) .* (self.next .* max(2*self.box - 1,0));
        %     self.box = min(self.box,2);
        % end
    end
end