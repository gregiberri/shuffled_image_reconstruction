from ml.solvers.siamese_solver import SiameseSolver


def get_solver(config, args):
    if config.env.solver == 'siamese-solver':
        return SiameseSolver(config, args)

    else:
        raise ValueError(f'Wrong solver config: {config.env.solver}')
