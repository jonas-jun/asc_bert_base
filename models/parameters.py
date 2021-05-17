def get_parameters(model):
    '''
    @return
    get_parameters[0]: total params
    get_parameters[1]: trainable ones
    '''

    total = [p for p in model.parameters()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('{:,} total parameters in this model'.format(sum(p.numel() for p in total)))
    print('{:,} trainable parameters in this model'.format(sum(p.numel() for p in trainable)))
    return total, trainable

