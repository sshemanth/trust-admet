import torch
from clearml import Task
from rdkit import Chem

task = Task.init(
    project_name="TRUST-ADMET",
    task_name="setup_test"
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

mol = Chem.MolFromSmiles("CCO")
print("RDKit molecule atoms:", mol.GetNumAtoms())

task.close()
