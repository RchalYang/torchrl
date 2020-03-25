import collections
import copy
import torch
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Normal

from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np

from utils.torch_utils import device,Tensor, Tensor_zeros_like
import utils.math_utils as math_utils
from .a2c_agent import A2CAgent

# #From Pytorch Official Github Repo
# def _check_param_device(param, old_param_device):
#     r"""This helper function is to check if the parameters are located
#     in the same device. Currently, the conversion between model parameters
#     and single vector form is not supported for multiple allocations,
#     e.g. parameters in different GPUs, or mixture of CPU/GPU.
#     Arguments:
#         param ([Tensor]): a Tensor of a parameter of a model
#         old_param_device (int): the device where the first parameter of a
#                                 model is allocated.
#     Returns:
#         old_param_device (int): report device for the first time
#     """

#     # Meet the first parameter
#     if old_param_device is None:
#         old_param_device = param.get_device() if param.is_cuda else -1
#     else:
#         warn = False
#         if param.is_cuda:  # Check if in same GPU
#             warn = (param.get_device() != old_param_device)
#         else:  # Check if in CPU
#             warn = (old_param_device != -1)
#         if warn:
#             raise TypeError('Found two parameters on different devices, '
#                             'this is currently not supported.')
#     return old_param_device


# #From Pytorch Official Github Repo
# def parameters_to_vector(parameters):
#     r"""Convert parameters to one vector
#     Arguments:
#         parameters (Iterable[Tensor]): an iterator of Tensors that are the
#             parameters of a model.
#     Returns:
#         The parameters represented by a single vector
#     """
#     # Flag for the device where the parameter is located
#     param_device = None

#     vec = []
#     for param in parameters:
#         # Ensure the parameters are located in the same device
#         if param.grad is not None:
#             param_device = _check_param_device(param, param_device)
#             vec.append(param.view(-1))

#     return torch.cat(vec)

# #From Pytorch Official Github Repo
# def parameters_to_vector_grad(parameters):
#     r"""Convert parameters to one vector
#     Arguments:
#         parameters (Iterable[Tensor]): an iterator of Tensors that are the
#             parameters of a model.
#     Returns:
#         The parameters represented by a single vector
#     """
#     # Flag for the device where the parameter is located
#     param_device = None

#     vec = []
#     for param in parameters:
#         # Ensure the parameters are located in the same device
#         if param is not None:
#             param_device = _check_param_device(param, param_device)
#             vec.append(param.view(-1))

#     return torch.cat(vec)

# #From Pytorch Official Github Repo
# def vector_to_parameters(vec, parameters):
#     r"""Convert one vector to the parameters
#     Arguments:
#         vec (Tensor): a single vector represents the parameters of a model.
#         parameters (Iterable[Tensor]): an iterator of Tensors that are the
#             parameters of a model.
#     """
#     # Ensure vec of type Tensor
#     if not isinstance(vec, torch.Tensor):
#         raise TypeError('expected torch.Tensor, but got: {}'
#                         .format(torch.typename(vec)))
#     # Flag for the device where the parameter is located
#     param_device = None

#     # Pointer for slicing the vector for each parameter
#     pointer = 0
#     for param in parameters:
#         if param.grad is not None:
#             # Ensure the parameters are located in the same device
#             param_device = _check_param_device(param, param_device)

#             # The length of the parameter
#             num_param = param.numel()
#             # Slice the vector, reshape it, and replace the old data of the parameter
#             param.data = vec[pointer:pointer + num_param].view_as(param).data

#             # Increment the pointer
#             pointer += num_param

# def padding_non_grad( params ):
#     for param in params :
#         if param.grad is None:
#             param.grad = Tensor_zeros_like ( param )
#             # print(grad)
#     # print(grads)

class TRPOAgent(A2CAgent):
    # def __init__(self,args,env_wrapper, continuous):
    def __init__(self, args, model, optim, env, data_generator, memory, continuous):
        """
        Instantiate a TRPO agent
        """
        super(TRPOAgent, self).__init__(args, model, optim, env, data_generator, memory, continuous)
                                        
        self.max_kl = args.max_kl
        self.cg_damping = args.cg_damping
        self.cg_iters = args.cg_iters
        self.residual_tol = args.residual_tol

        self.algo="trpo"

    def mean_kl_divergence(self, model):
        """
        Returns an estimate of the average KL divergence between a given model and self.policy_model
        """
        # actprob = model(self.observations).detach() + 1e-8
        # old_actprob = self.model(self.observations)

        def normal_distribution_kl_divergence(mean_old, std_old, mean_new, std_new):
            return torch.mean(torch.sum((torch.log(std_new) - torch.log(std_old) \
                                        + (std_old * std_old + (mean_old - mean_new) * (mean_old - mean_new)) \
                                        / (2.0 * std_new * std_new) \
                                        - 0.5), 1))

        if self.continuous:
            mean_new, std_new, _ = model( self.obs )
            mean_old, std_old, _ = self.model( self.obs )

            mean_new = mean_new.detach()
            std_new = std_new.detach()

            kl = normal_distribution_kl_divergence( mean_old, std_old, mean_new, std_new )

        else:

            probs_new, _ = model(self.obs)
            probs_old, _ = self.model(self.obs)

            probs_new = probs_new.detach()

            probs_new = F.softmax( probs_new, dim = 1 )
            probs_old = F.softmax( probs_old, dim = 1 )

            kl = torch.sum(probs_old * torch.log( probs_old / (probs_new + 1e-8 ) ), 1).mean()

        return kl

    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        """
        self.model.zero_grad()
        mean_kl_div = self.mean_kl_divergence(self.model)
        
        # mean_kl_div.backward( retain_graph=True, create_graph=True )
        kl_grad_vector = torch.autograd.grad(mean_kl_div, self.model.policy_parameters(), create_graph=True )
        
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad_vector])
        grad_vector_product = torch.sum(kl_grad_vector * vector)

        second_order_grad = torch.autograd.grad(grad_vector_product, self.model.policy_parameters())
        
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in second_order_grad])

        return fisher_vector_product + self.cg_damping * vector.detach()

    def conjugate_gradient(self, b):
        """
        Returns F^(-1) b where F is the Hessian of the KL divergence
        """
        p = b.clone()
        r = b.clone()
        x = Tensor_zeros_like(p)
        rdotr = r.double().dot(r.double())
            
        for _ in range(self.cg_iters):
            z = self.hessian_vector_product(p).squeeze(0)
            v = (rdotr / p.double().dot(z.double())).float()

            x += v * p
            r -= v * z

            newrdotr = r.double().dot(r.double())
            mu = newrdotr / rdotr
            
            p = r + mu.float() * p
            rdotr = newrdotr
            if rdotr < self.residual_tol:
                break
        return x

    def surrogate_loss(self, theta):
        """
        Returns the surrogate loss w.r.t. the given parameter vector theta
        """
        theta = theta.detach()
        new_model = copy.deepcopy(self.model)
        # for param in new_model.parameters():
        #     print(param)
        vector_to_parameters(theta, new_model.policy_parameters())

        if self.continuous:
            mean_new, std_new, _ = new_model( self.obs )
            mean_old, std_old, _ = self.model( self.obs )
                
            dis_new = Normal( mean_new, std_new )
            dis_old = Normal( mean_old, std_old )
            
            log_prob_new = dis_new.log_prob( self.acts ).sum( -1, keepdim=True )
            log_prob_old = dis_old.log_prob( self.acts ).sum( -1, keepdim=True )

            ratio = torch.exp( log_prob_new - log_prob_old ).detach()
        else:

            probs_new, _ = new_model(self.obs)
            probs_old, _ = self.model(self.obs)

            dis_new = F.softmax( probs_new, dim = 1 )
            dis_old = F.softmax( probs_old, dim = 1 )

            probs_new = dis_new.gather( 1, self.acts ).detach()
            probs_old = dis_old.gather( 1, self.acts ).detach() + 1e-8

            ratio = probs_new / probs_old

        return -torch.mean( ratio * self.advs )

    def linesearch(self, x, fullstep, expected_improve_rate):
        """
        Returns the parameter vector given by a linesearch
        """
        accept_ratio = .1
        max_backtracks = 10
        fval = self.surrogate_loss(x)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            print("Search number {}...".format(_n_backtracks + 1))
            stepfrac = float(stepfrac)
            xnew     = x + stepfrac * fullstep
            newfval  = self.surrogate_loss(xnew)
            actual_improve = fval - newfval

            expected_improve = expected_improve_rate * stepfrac
            
            ratio = actual_improve / expected_improve
            
            if ratio > accept_ratio and actual_improve > 0:
                return xnew
        return x.detach()

    def _optimize(self, obs, acts, advs, est_rs):
        
        self.obs, self.acts, self.advs, self.est_rs = obs, acts, advs, est_rs

        self.obs  = Tensor( self.obs )
        self.acts = Tensor( self.acts )
        self.advs = Tensor( self.advs ).unsqueeze(1)
        self.est_rs = Tensor( self.est_rs ).unsqueeze(1)

        # Calculate Advantage & Normalize it
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

        # Surrogate loss with Entropy

        if self.continuous:
            mean, std, values = self.model( self.obs )

            dis = Normal(mean, std)
            
            log_prob = dis.log_prob( self.acts ).sum( -1, keepdim=True )

            ent = dis.entropy().sum( -1, keepdim=True )

            probs_new = torch.exp( log_prob )
            probs_old = probs_new.detach() + 1e-8

        else:

            probs, values = self.model( self.obs)

            dis = F.softmax( probs, dim = 1 )

            self.acts = self.acts.long()

            probs_new = dis.gather( 1, self.acts )
            probs_old = probs_new + 1e-8 

            ent = -( dis.log() * dis ).sum(-1)


        ratio = probs_new / probs_old

        surrogate_loss = - torch.mean( ratio * self.advs ) - self.entropy_para * ent.mean()

        # criterion = torch.nn.MSELoss()
        # empty_value_loss = criterion( values, values.detach() )

        # Calculate the gradient of the surrogate loss
        self.model.zero_grad()
        surrogate_loss.backward()
        policy_gradient = parameters_to_vector([p.grad for p in self.model.policy_parameters()]).squeeze(0).detach()
        
        # ensure gradient is not zero
        if policy_gradient.nonzero().size()[0]:
            # Use Conjugate gradient to calculate step direction
            step_direction = self.conjugate_gradient(-policy_gradient)
            # line search for step 
            shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction))
            
            lm = torch.sqrt(shs / self.max_kl)
            fullstep = step_direction / lm

            gdotstepdir = -policy_gradient.dot(step_direction)
            theta = self.linesearch(parameters_to_vector(self.model.policy_parameters()).detach(), fullstep, gdotstepdir / lm)
            # Update parameters of policy model
            old_model = copy.deepcopy(self.model)
            old_model.load_state_dict(self.model.state_dict())

            if any(np.isnan(theta.cpu().detach().numpy())):
                print("NaN detected. Skipping update...")
            else:
                # for param in self.model.policy_parameters():
                #     print(param)
                vector_to_parameters(theta, self.model.policy_parameters())

            kl_old_new = self.mean_kl_divergence(old_model)
            print( 'KL:{:10} , Entropy:{:10}'.format(kl_old_new.item(), ent.mean().item()))

        else:
            print("Policy gradient is 0. Skipping update...")
            print(policy_gradient.shape)


        self.model.zero_grad()

        if self.continuous:
            _, _, values = self.model( self.obs )
        else:
            _, values = self.model( self.obs)

        criterion = torch.nn.MSELoss()
        critic_loss = self.value_loss_coeff * criterion(values, self.est_rs )
        critic_loss.backward()
        self.optim.step()
        print("MSELoss for Value Net:{}".format(critic_loss.item()))