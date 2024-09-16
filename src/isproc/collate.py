import numpy as np
import torch

class SGOverlayCollate:

    def __init__(self, input_key, signal_key, background_key, regressor_target=[], weight_key=None, compute_weight=False):

        self._in_key = input_key
        self._sg_key = signal_key
        self._bg_key = background_key
        self._wt_key = weight_key
        self._regressor_target=list(regressor_target)
        self._recompute = compute_weight

        if self._wt_key is None and self._recompute:
            raise ValueError(f'[CollateSGOverlay] cannot have compute_weight==True and weight_key==None')

    def __call__(self,batch):

        input_data = dict()
        for key in batch[0].keys():
            if hasattr(batch[0][key],'__len__'):
                input_data[key] = np.stack([d[key] for d in batch])
            else:
                input_data[key] = np.array([d[key] for d in batch])

        if self._in_key in input_data.keys():
            raise KeyError(f'input_key label "{self._in_key}" is already present in the loaded data dictionary {input_data.keys()}')

        input_data[self._in_key] = torch.as_tensor(input_data[self._sg_key] + input_data[self._bg_key])

        target_data = dict(classifier=torch.as_tensor((input_data[self._sg_key]>0).squeeze(),dtype=int))
        
        if len(self._regressor_target):
            regressor_tensors=[]
            for key in self._regressor_target:
                regressor_tensors.append(input_data[key])
            regressor_tensors = torch.as_tensor(np.column_stack(regressor_tensors))
            target_data['regressor'] = regressor_tensors

        if self._wt_key and self._recompute:
                raise NotImplementedError

        return input_data, target_data


