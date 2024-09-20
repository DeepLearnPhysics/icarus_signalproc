import torch
from torcher.util import instantiate

class SGOverlayLoss(torch.nn.Module):

    def __init__(self, Weights=[1.,1.], RegressorMask=[], EnableStatWeight=True, RegressorLoss=None, ClassifierLoss=None):
        super().__init__()
        self.rg_criterion, self.cl_criterion = None, None
        self.weights=torch.as_tensor(Weights)
        self.weight_stats=bool(EnableStatWeight)
        self.regressor_mask=list(RegressorMask)
        if RegressorLoss:
            self.rg_criterion=instantiate(RegressorLoss)
        if ClassifierLoss:
            self.cl_criterion=instantiate(ClassifierLoss)

        if len(self.regressor_mask) and self.cl_criterion is None:
            raise NotImplementedError(f'[SGOverlayLoss] must provide ClassifierLoss to apply RegressorMask')
    
    def forward(self,prediction,target):

        result = dict()
        loss = None

        if self.cl_criterion:
            if not 'classifier' in prediction:
                raise KeyError(f'[SGOverlayLoss] "classifier" keyword not found in the prediction')
            if not 'classifier' in target:
                raise KeyError(f'[SGOverlayLoss] "classifier" keyword not found in the target')

            pred_classifier   = prediction['classifier']
            target_classifier = target['classifier']

            if self.weight_stats:
                stat_weights = torch.zeros_like(target_classifier,dtype=torch.float32)
                for class_type in torch.unique(target_classifier):
                    mask = target_classifier == class_type
                    weight_factor = (1./torch.sqrt(mask.sum())).type(stat_weights.dtype)
                    stat_weights[mask] = weight_factor
                    
                stat_weights = stat_weights / stat_weights.sum() * torch.prod(torch.as_tensor(stat_weights.shape))

                loss = (self.cl_criterion(pred_classifier,target_classifier) * stat_weights).mean()

            else:
                
                loss = self.cl_criterion(pred_classifier,target_classifier).mean()

            if len(self.weights):
                loss = loss * self.weights[0]
            result['classifier_loss'] = loss

        if self.rg_criterion:
            if not 'regressor' in prediction:
                raise KeyError(f'[SGOverlayLoss] "regressor" keyword not found in the prediction')
            if not 'regressor' in target:
                raise KeyError(f'[SGOverlayLoss] "regressor" keyword not found in the target')

            pred_regressor   = prediction['regressor']
            target_regressor = target['regressor']
            assert pred_regressor.shape == target_regressor.shape

            if len(self.regressor_mask)<1:
                reg_loss = self.rg_criterion(pred_regressor,target_regressor)
            else:
                if not hasattr(self,'reg_mask'):
                    self.reg_mask = torch.zeros_like(pred_regressor,dtype=bool)
                self.reg_mask[:]=False
                for val in self.regressor_mask:
                    self.reg_mask = torch.logical_or(self.reg_mask,(target_classifier==val))
                result['reg_mask']=self.reg_mask

                if self.reg_mask.sum()<1:
                    reg_loss = 0
                else:
                    reg_loss = self.rg_criterion(pred_regressor[self.reg_mask],target_regressor[self.reg_mask])
                    #print(pred_regressor[self.reg_mask].sum(),target_regressor[self.reg_mask].sum())
                    #print((pred_regressor-target_regressor).sum(),(pred_regressor-target_regressor)[self.reg_mask].sum())    

            result['regressor_loss']=reg_loss
            if len(self.weights):
                loss = loss + self.weights[1]*reg_loss
            else:
                loss = loss + reg_loss

        result['loss']=loss
        return result


        

