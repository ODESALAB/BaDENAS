def get_params(param_str):

    if param_str == 'meta_standard':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        params = {'ensemble_params':[metanet_params for _ in range(5)]}

    elif param_str == 'meta_diverse':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ensemble_params = [
            {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ]

        params = {'ensemble_params':ensemble_params}
    else:
        print('Invalid meta neural net params: {}'.format(param_str))
        raise NotImplementedError()

    return params