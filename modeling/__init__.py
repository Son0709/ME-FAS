from .mefas import MEFas


def BuildModel(cfg):

    model = MEFas(cfg)

    return model

