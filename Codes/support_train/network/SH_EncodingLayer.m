classdef SH_EncodingLayer < nnet.layer.Layer &...
                            nnet.layer.Formattable

    properties
        degree;
        outdim;
    end
    
    properties(Access = private)
        func;
    end

    methods
        function layer = SH_EncodingLayer(degree,name)
            layer.degree = degree;
            layer.outdim = degree^2;
            layer.Name = name;

            layer.func = fused_SHEncoding();
        end

        function Z = predict(layer,X)
            % X dlarray: CB, C = 3
            % start_tic = tic;
            
            Z = layer.func( ...
                X,...
                layer.degree...
            );
            wait(gpuDevice());
            
            Z = dlarray(Z,'CB');
            % fprintf("SH encoding takes: %f \n",toc(start_tic));
        end
    end

end