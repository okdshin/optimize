Example

'''
python optimize.py --command 'python train_mnist.py' --params '{'lr': ('lu', 0.001, 0.1), 'momentum': ('u', 0.8, 0.99)}'
'''

The executed command must send below texts to interact with optuna

`OPTUNA_STEP <step> <intermediate_score>`

`OPTUNA_FINISHED <final_score>`

WARN: As `OPTUNA_FINISHED` is given, command process is terminated immediately

Users have to finish necessary operations (e.g. model exporting) before printing `OPTUNA_FINISHED`
