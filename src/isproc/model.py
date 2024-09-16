import torch,importlib

class Discriminator(torch.nn.Module):

    def __init__(self, module_name, model_name, input_key, **kwargs):
        super().__init__()
        module = importlib.import_module(module_name)
        self.model = getattr(module,model_name)(**kwargs)
        self._input_key = input_key

    def forward(self,input_data):

        return self.model(input_data[self._input_key])


        
