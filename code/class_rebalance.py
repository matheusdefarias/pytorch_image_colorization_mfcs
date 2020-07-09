from torch.autograd import Function

class ClassRebalance(Function):
    @staticmethod
    def forward(ctx, outputs, prior_boost_nongray):
        
        ctx.save_for_backward(prior_boost_nongray)
        
        return outputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        
        prior_boost_nongray, = ctx.saved_tensors
        
        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * prior_boost_nongray
        
        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None