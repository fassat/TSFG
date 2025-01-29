def restore_parameters(model, best_model):

    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param