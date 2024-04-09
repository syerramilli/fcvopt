import os
import torch
import argparse
import gpytorch
from fcvopt.models import GPR

parser = argparse.ArgumentParser(description='Generate torchscript models from saved data')
parser.add_argument('--orig_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, 'true_cv_models')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

DATASETS = os.listdir(args.orig_dir)

class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        return self.gp.predict(x, return_std=True)
    
    
def trace_model(model):
    test_x = model.train_inputs[0][:5, :]
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
        model.eval()
        pred = model(test_x)  # Do precomputation
        traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)

    return traced_model

for dataset in DATASETS:
    print(dataset)
    training = torch.load(os.path.join(args.orig_dir, dataset,'model_train.pt'))
    model = GPR(
        train_x=training['train_x'],
        train_y=training['train_y'],
    )

    model.load_state_dict(
        torch.load(os.path.join(args.orig_dir, dataset,'model_state.pth'))
    )

    traced_model = trace_model(model)

    # save traced model
    traced_model.save(os.path.join(SAVE_DIR, dataset + '.pt'))