classdef shEncodingLayer < nnet.layer.Layer...
                        & nnet.layer.Formattable

    properties
        C0 = 0.28209479177387814;
        
        C1 = 0.4886025119029199;

        C2 = [1.0925484305920792,-1.0925484305920792,0.31539156525252005,...
             -1.0925484305920792,0.5462742152960396];

        C3 = [-0.5900435899266435,2.890611442640554,-0.4570457994644658,...
               0.3731763325901154,-0.4570457994644658,1.445305721320277,-0.5900435899266435];

        C4 = [2.5033429417967046, -1.7701307697799304,  0.9461746957575601,...
             -0.6690465435572892,  0.10578554691520431,-0.6690465435572892,...
             0.47308734787878004,-1.7701307697799304,  0.6258357354491761];

        degree;

        outdim;
    end

    methods
        function layer = shEncodingLayer(degree,name)
            layer.degree = degree;
            layer.outdim = degree^2;
            layer.Name = name;
        end

        function Z = predict(layer,X)
            % X dlarray: CB, C = 3
            % start_tic = tic;
            Z = dlarray(gpuArray(single(zeros(layer.outdim,size(X,2)))),'CB');
            Z(1,:) = layer.C0;

            x = X(1,:);
            y = X(2,:);
            z = X(3,:);

            if layer.degree > 1
                Z(2,:) = -layer.C1 * y;
                Z(3,:) =  layer.C1 * z;
                Z(4,:) = -layer.C1 * x;
                if layer.degree > 2
                    xx = x.^2;      yy = y.^2;     zz = z.^2;
                    xy = x .* y;    yz = y .* y;   xz = x .* z;
                    Z(5,:) = layer.C2(1) .* xy;
                    Z(6,:) = layer.C2(2) .* yz;
                    Z(7,:) = layer.C2(3) .* (2 .* zz - xx - yy);
                    Z(8,:) = layer.C2(4) .* xz;
                    Z(9,:) = layer.C2(5) .* (xx - yy);
                    if layer.degree > 3
                        Z(10,:) = layer.C3(1) .* y .* (3 .* xx - yy);
                        Z(11,:) = layer.C3(2) .* xy .* z;
                        Z(12,:) = layer.C3(3) .* y .* (4 .* zz - xx - yy);
                        Z(13,:) = layer.C3(4) .* z .* (2 .* zz - 3 .* xx - 3 .* yy);
                        Z(14,:) = layer.C3(5) .* x .* (4 .* zz - xx - yy);
                        Z(15,:) = layer.C3(6) .* z .* (xx - yy);
                        Z(16,:) = layer.C3(7) .* x .* (xx - 3 .* yy);
                        if layer.degree > 4
                            Z(17,:) = layer.C4(1) .* xy .* (xx - yy);
                            Z(18,:) = layer.C4(2) .* yz .* (3 .* xx - yy);
                            Z(19,:) = layer.C4(3) .* xy .* (7 .* zz - 1);
                            Z(20,:) = layer.C4(4) .* yz .* (7 .* zz - 3);
                            Z(21,:) = layer.C4(5) .* (zz .* (35 .* zz - 30) + 3);
                            Z(22,:) = layer.C4(6) .* xz .* (7 .* zz - 3);
                            Z(23,:) = layer.C4(7) .* (xx - yy) .* (7 * zz - 1);
                            Z(24,:) = layer.C4(8) .* xz .* (xx - 3 .* yy);
                            Z(25,:) = layer.C4(9) .* (xx .* (xx - 3 .* yy) - yy .* (3 .* xx - yy));
                        end
                    end
                end
            end
            wait(gpuDevice());
            % fprintf("SH encoding takes: %f \n",toc(start_tic));
        end
    end

end