# Colorful Image Colorization(2016) paper
#     - Project website: https://richzhang.github.io/colorization/
#     - GitHub Repository: https://github.com/richzhang/colorization
#     - Caffe implementation: https://github.com/richzhang/colorization/tree/caffe
#
# Pytorch Implementation of Colorful Image Colorization(2016) paper 
#     - By: Matheus de Farias Cavalcanti Santos
#     - Repository: https://github.com/matheusdefarias/pytorch_image_colorization_mfcs

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