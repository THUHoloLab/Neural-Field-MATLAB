function [net_E, net_D] = create_NGP_sepa(parsers)

arguments
    parsers.bound_box;
    parsers.n_levels = 8;
    parsers.log2_hashmap_size = 19;
    parsers.base_resolution = 16;
    parsers.finest_resolution = 4096;
    parsers.mlp_width = 64;
end

n_features_per_level = 4;
SH_encode_size = 4;

% create Hash encoder layers
tempNet = [
    featureInputLayer(3,"Name","xyz-input");
    Hash_EncodeingLayer( ...
        "base_res",     parsers.base_resolution,...
        "device",       gpuDevice(),...
        "bounding_box", parsers.bound_box,...
        "feature_len",  n_features_per_level,...
        "high_res",     parsers.finest_resolution,...
        "level",        parsers.n_levels,...
        "log2_hashmap_size",parsers.log2_hashmap_size...
    );
];
net_E = dlnetwork(tempNet);

% create decoding layers
net_D = dlnetwork;
input_feature_dim = parsers.n_levels * n_features_per_level;
tempNet = [
    featureInputLayer(input_feature_dim,"Name","Hash-input");
    FC_SimpleLayer(parsers.mlp_width,"Name","fc_1");
    reluLayer();
    % FC_SimpleLayer(parsers.mlp_width,"Name","fc_2");
    % reluLayer();
    FC_SimpleLayer(16,"Name","fc_3");
    reluLayer();
    functionLayer(@(x) deal(x(1,:),x(2:end,:)),...
                "Name", 'function',...
                "Description", '@(x) deal(x(1,:),x(2:end,:))', ...
                "NumOutputs", 2, ...
                "OutputNames", {'out1' 'out2'}) ...
];

net_D = addLayers(net_D,tempNet);

tempNet = [
    featureInputLayer(3,"Name","dir-input")
    SH_EncodingLayer(SH_encode_size,"SH-encoding")];
net_D = addLayers(net_D,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","concat")
    FC_SimpleLayer(parsers.mlp_width,"Name","fc_4")
    reluLayer()
    % FC_SimpleLayer(parsers.mlp_width,"Name","fc_5")
    % reluLayer()
    FC_SimpleLayer(parsers.mlp_width,"Name","fc_6")
    reluLayer()
    FC_SimpleLayer(3,"Name","fc_8")
    depthConcatenationLayer(2,"Name","concat_out")
];

net_D = addLayers(net_D,tempNet);
net_D = connectLayers(net_D,"SH-encoding","concat/in2");
net_D = connectLayers(net_D,"function/out2","concat/in1");
net_D = connectLayers(net_D,"function/out1","concat_out/in2");
net_D = initialize(net_D);
end

