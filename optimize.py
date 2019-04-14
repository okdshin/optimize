import optuna

import subprocess


def gen_objective(command, suggest_params):
    def _objective(trial):
        def make_arg(trial, name, suggest_param):
            type_ = suggest_param[0]
            min_ = suggest_param[1]
            max_ = suggest_param[2]
            if type_ == 'u':
                param = trial.suggest_uniform(name, min_, max_)
            elif type_ == 'lu':
                param = trial.suggest_loguniform(name, min_, max_)
            elif type_ == 'i':
                param = trial.suggest_int(name, min_, max_)
            else:
                raise RuntimeError(
                    'unsupported suggest type: {}'.format(type_))
            return (name, param)

        params = [make_arg(trial, k, v) for k, v in suggest_params.items()]
        args = []
        for name, param in params:
            args.append(['--{}'.format(name), str(param)])
        args = sum(args, [])
        cmd = command.split() + args
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, b''):
            out = line.rstrip().decode("utf8")
            print(out)
            if 'OPTUNA_STEP' in out:
                splitted_out = out.split()
                step = int(splitted_out[1])
                intermediate_value = float(splitted_out[2])
                trial.report(intermediate_value, step)
                if trial.should_prune(step):
                    raise optuna.structs.TrialPruned()
            elif 'OPTUNA_FINISHED' in out:
                p.terminate()
                return float(out.split()[1])

    return _objective


def main(command, params):
    study = optuna.create_study()
    study.optimize(gen_objective(command, params), n_trials=5)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.structs.TrialState.COMPLETE
    ]

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
