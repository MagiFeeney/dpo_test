import torch
from torch.distributions.kl import kl_divergence
from torch.autograd import Variable
import numpy as np
import scipy.optimize
from .retrace import set_flat_params_to, get_flat_params_from, get_flat_grad_from

class OnTheFly(object):
    def __init__(self, model, obs, rhx, masks, actions, returns, old_dist, old_action_log_probs, advantages):
        self.model     = model
        self.obs       = obs
        self.rhx       = rhx
        self.masks     = masks
        self.actions   = actions
        self.returns   = returns
        self.old_dist  = model._dist(old_dist)
        self.advantages = advantages
        self.old_action_log_probs = old_action_log_probs

        
    def linesearch(self,
                   x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = self.get_loss(True).data
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.model.base.actor, xnew)
            newfval = self.get_loss(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

    def get_loss_grad(self, loss):
        grads = torch.autograd.grad(loss, self.model.base.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        return loss_grad     

    def get_loss(self, volatile=False):
        if volatile:
            with torch.no_grad():
                _, action_log_probs, _ = self.model.evaluate_actions(self.obs, self.rhx, self.masks, self.actions)            
        else:
            _, action_log_probs, _ = self.model.evaluate_actions(self.obs, self.rhx, self.masks, self.actions)
            
        ratio = torch.exp(action_log_probs - self.old_action_log_probs)
        action_loss = -(ratio * self.advantages).mean()
            
        return action_loss

    def get_value_loss(self, flat_params):
        set_flat_params_to(self.model.base.critic, torch.Tensor(flat_params))
        for param in self.model.base.critic.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values, _, _ = self.model.evaluate_actions(self.obs, self.rhx, self.masks, self.actions)

        value_loss = (self.returns - values).pow(2).mean()
            
        # weight decay
        for param in self.model.base.critic.parameters():
            value_loss += param.pow(2).sum() * self.l2_reg
        value_loss.backward()
        return (value_loss.data.cpu().double().numpy(), get_flat_grad_from(self.model.base.critic).data.cpu().double().numpy())


    def get_kl(self, volatile=False):
        if volatile:
            with torch.no_grad():
                curr_dist = self.model.get_dist(self.obs, self.rhx, self.masks)
        else:
            curr_dist = self.model.get_dist(self.obs, self.rhx, self.masks)
        return kl_divergence(self.old_dist, curr_dist).mean()

    def get_entropy(self, volatile=False):
        if volatile:
            with torch.no_grad():
                dist_entropy = self.model.get_entropy(self.obs, self.rhx, self.masks)
        else:
            dist_entropy = self.model.get_entropy(self.obs, self.rhx, self.masks)
        return dist_entropy


    def Fvp(self, v):
        kl = self.get_kl()

        grads = torch.autograd.grad(kl, self.model.base.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.model.base.actor.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * self.damping

    def conjugate_gradients(self, Avp, b, nsteps, device, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def step(self, max_kl, damping, l2_reg, device):
        self.l2_reg  = l2_reg
        self.damping = damping
        
        action_loss = self.get_loss()

        # update value network using LBFGS
        flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss, get_flat_params_from(self.model.base.critic).cpu().double().numpy(), maxiter=25)
        set_flat_params_to(self.model.base.critic, torch.Tensor(flat_params).to(device))
    
        loss_grad = self.get_loss_grad(action_loss)
        
        stepdir = self.conjugate_gradients(self.Fvp, -loss_grad, 10, device)

        shs = 0.5 * (stepdir * self.Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]
        
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

        prev_params = get_flat_params_from(self.model.base.actor)
        success, new_params = self.linesearch(prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        set_flat_params_to(self.model.base.actor, new_params)

        value_loss = None # just to align outputs
        action_loss  = self.get_loss(True)        
        dist_entropy = self.get_entropy(True)

        return value_loss, action_loss, dist_entropy
