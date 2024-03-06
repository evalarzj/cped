from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd

class FLOWTrainer(TorchTrainer):
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        target_policy,
        flow,
        discriminator,
        batch_size,
        prior,
        f_div,
        model_type,
        flowgan_train_epoch,
        env_name,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.5,
        

        discount=0.99,
        reward_scale=1.0,

        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        
        policy_and_target_update_period=2,
        tau=0.005,
        qf_criterion=None,

        #flow specific params
        margin_prob_thresh=0,
        like_reg=1
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.policy = policy
        self.target_policy = target_policy
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        
        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion


        self.flow=flow
        self.discriminator=discriminator
        self.prior=prior
        self.f_div=f_div
        self.model_type=model_type
        self.flowgan_train_epoch=flowgan_train_epoch*1000
        self.env_name=env_name

        self.like_reg=like_reg
        self.batch_size=batch_size
        
        self.alpha=1000


        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )


        self.flow_optimizer = optimizer_class(
            self.flow.parameters(),
            lr=1e-4,
            betas=(0.9,0.01),
            eps=1e-4,
        )
        self.d_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=1e-4,
        )

        

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

        self.margin_prob_thresh=margin_prob_thresh
        
        for name, param in self.flow.named_parameters():
           if param.requires_grad:
              print (name)
              
        for name, param in self.discriminator.named_parameters():
           if param.requires_grad:
              print (name)
              
              

    

    def compute_log_density_x(self,z, sum_log_det_jacobians, prior):

        zs = list(z.size())
        if len(zs) == 4:
            K = zs[1] * zs[2] * zs[3]  # dimension of the Gaussian distribution
            z = torch.reshape(z, (-1, K))
        else:
            K = zs[1]
        log_density_z = 0
        if prior == "gaussian":
            log_density_z = -0.5 * torch.sum(torch.square(z), 1) - 0.5 * K * np.log(2 * np.pi)
        elif prior == "logistic":
            m = nn.Softplus()
            temp=m(z)
            log_density_z = -torch.sum(-z + 2 * temp, 1)
        elif prior == "uniform":
            log_density_z = 0
        log_density_x = log_density_z + sum_log_det_jacobians
        
        return log_density_x

    def log_likelihood(self,z, sum_log_det_jacobians, prior):
        return -torch.sum(self.compute_log_density_x(z, sum_log_det_jacobians, prior))
        
    
    def calculate_gradient_penalty(self,model, real, fake):
    # Random weight term for interpolation between real and fake data
        alpha = ptu.randn((real.size(0), 1))

    # Get random interpolation between real and fake data
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates,model_logit = model(interpolates)
        grad_outputs = ptu.ones(model_interpolates.size(), requires_grad=False)

    # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
        outputs=model_logit,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
        gradients =gradients.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        

        
        """
        Learn the distribution of the behavior policy
        """
        
        # The time(epoch) varying constrain parameter alpha(Lagrange multiplier) used in the policy optimization objective.
        if 'antmaze' in self.env_name:
           if self._n_train_steps_total>800000:
              self.alpha=0.08
           elif self._n_train_steps_total>500000 and self._n_train_steps_total<=800000:
              self.alpha=0.1
           elif self._n_train_steps_total>300000 and self._n_train_steps_total<=500000:
              self.alpha=0.25
           elif self._n_train_steps_total>=100000 and self._n_train_steps_total<=300000:
              self.alpha=0.5
           else:
              self.alpha=5
        else:
            self.alpha=1000
            if self._n_train_steps_total>(self.flowgan_train_epoch+100000):
              if self._n_train_steps_total>1000000:
                 self.alpha=1.5
              else:
                 self.alpha=10-2.5*((self._n_train_steps_total-self.flowgan_train_epoch-100000)//100000)
        
        
        
        if self._n_train_steps_total%100000==0:
           print(self.alpha)
           
        
        if self._n_train_steps_total<self.flowgan_train_epoch:

          # calculate the MLE
           gen_para, jac = self.flow(obs, actions)

           log_li = self.log_likelihood(gen_para, jac, self.prior) / self.batch_size
        
        


          # calculate the loss for G and D, adversorial loss

          # sample z from the prior
        
           np.random.seed()   
           

           if self.prior == "uniform":
              sample_z = np.random.uniform(-1, 1, size=(self.batch_size, gen_para.size()[1]))
           elif self.prior == "logistic":
              sample_z = np.random.logistic(loc=0., scale=1., size=(self.batch_size, gen_para.size()[1]))
           elif self.prior == "gaussian":
              sample_z = np.random.normal(0.0, 1.0, size=(self.batch_size, gen_para.size()[1]))
           else:
              print("ERROR: Unrecognized prior...exiting")
              exit(-1)

        
           sample_z=ptu.from_numpy(sample_z)
           sample_z=sample_z.to(torch.float32)
        
        

        # calculate the forward pass of G and D network
           G = self.flow.inverse(sample_z)
        
           real_data=torch.cat((obs, actions), dim=1)
           #real_data=real_data+ptu.normal(0.0, 1.0, size=real_data.size())
           real_data=real_data.to(torch.float32)

        
           D, D_logits = self.discriminator(real_data)
           D_, D_logits_ = self.discriminator(G.detach())

      
           advloss = nn.BCELoss(reduction='mean')

        ### Vanilla gan loss
           if self.f_div == 'ce':
              d_loss_real = torch.mean(
                  advloss(D_logits, torch.ones_like(D)))
              d_loss_fake = torch.mean(
                  advloss(D_logits_, torch.zeros_like(D_)))
           else:
            ### other gan losses
              if self.f_div == 'hellinger':
                  d_loss_real = torch.mean(torch.exp(-D_logits))
                  d_loss_fake = torch.mean(torch.exp(D_logits_) - 2.)
                
              elif self.f_div == 'rkl':
                  d_loss_real = torch.mean(torch.exp(D_logits))
                  d_loss_fake = torch.mean(-D_logits_ - 1.)
                
              elif self.f_div == 'kl':
                  d_loss_real = torch.mean(-D_logits)
                  d_loss_fake = torch.mean(torch.exp(D_logits_ - 1.))
                
              elif self.f_div == 'tv':
                  d_loss_real = torch.mean(-0.5 * torch.tanh(D_logits))
                  d_loss_fake = torch.mean(0.5 * torch.tanh(D_logits_))
                
              elif self.f_div == 'lsgan':
                  d_loss_real = 0.5 * torch.mean((D_logits - 0.9) ** 2)
                  d_loss_fake = 0.5 * torch.mean(D_logits_ ** 2)
                
              elif self.f_div == "wgan":
                  d_loss_real = -torch.mean(D_logits)
                  d_loss_fake = torch.mean(D_logits_)
                  real=torch.cat((obs, actions), dim=1)
                  fake=G.detach().data
                  gradient_penalty = self.calculate_gradient_penalty(self.discriminator,real.data, fake.data)
        
           if self.f_div=="wgan":
             d_loss = d_loss_real+d_loss_fake+gradient_penalty * 0.5
           else:
             d_loss = d_loss_real+d_loss_fake
        
          #update discriminator network
           if self._n_train_steps_total%1==0:
             if not torch.isnan(d_loss).any() and not torch.isinf(d_loss).any():
                 self.d_optimizer.zero_grad()
                 d_loss.backward()
                 self.d_optimizer.step()
        
        
           D1_, D_logits1_ = self.discriminator(G)

        
          ### Vanilla gan loss
           if self.f_div == 'ce':
              g_loss = torch.mean(
                  advloss(D_logits1_, torch.ones_like(D1_)*0.9))
           else:
              ### other gan losses
              if self.f_div == 'hellinger':
                  g_loss = torch.mean(torch.exp(-D_logits1_))
              elif self.f_div == 'rkl':
                  g_loss = -torch.mean(-D_logits1_ - 1.)
              elif self.f_div == 'kl':
                  g_loss = torch.mean(-D_logits1_)
              elif self.f_div == 'tv':
                  g_loss = torch.mean(-0.5 * torch.tanh(D_logits1_))
              elif self.f_div == 'lsgan':
                  g_loss = 0.5 * torch.mean((D_logits1_ - 0.9) ** 2)
              elif self.f_div == "wgan":
                  g_loss = -torch.mean(D_logits1_)
                
           if self._n_train_steps_total %400==0:
             print(d_loss_real.data,d_loss_fake.data,d_loss.data)
             print(log_li)
             print(self.log_likelihood(sample_z, jac, self.prior) / self.batch_size)

           if self.like_reg > 0:
             g_loss = g_loss/self.like_reg+log_li

         ### update the flow(G) network
           if self._n_train_steps_total%5==0:
             if not torch.isnan(g_loss).any() and not torch.isinf(g_loss).any():
                    self.flow_optimizer.zero_grad()
                    g_loss.backward()
                    self.flow_optimizer.step()



        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()
        
        
        
        """
        Update critic Networks
        """
        if self._n_train_steps_total >= (self.flowgan_train_epoch+30000):
           self.qf1_optimizer.zero_grad()
           qf1_loss.backward()
           self.qf1_optimizer.step()

           self.qf2_optimizer.zero_grad()
           qf2_loss.backward()
           self.qf2_optimizer.step()
        
        """
        Actor Training
        """
        
        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output1 = self.qf1(obs, policy_actions)
            q_output2 = self.qf2(obs, policy_actions)
            q_output=torch.min(q_output1,q_output2)
            fakez, jacc = self.flow(obs, policy_actions)
            fakez1, jacc1 = self.flow(obs, actions)
            neg_li = -self.compute_log_density_x(fakez, jacc, self.prior)
            log_li_r = -self.compute_log_density_x(fakez1, jacc1, self.prior)
            self.margin_prob_thresh=log_li_r.detach()
            
            
            
            if self._n_train_steps_total >= self.flowgan_train_epoch+50000:
             policy_loss = - q_output.mean()+self.alpha *(neg_li - self.margin_prob_thresh).mean()
            else:
             policy_loss = self.alpha *(neg_li - self.margin_prob_thresh).mean()
             #policy_loss = (neg_li - self.margin_prob_thresh).mean()
             
            if self._n_train_steps_total >= self.flowgan_train_epoch:
              if not torch.isnan(policy_loss).any() and not torch.isinf(policy_loss).any() and not policy_loss>1e+7:
                 self.policy_optimizer.zero_grad()
                 policy_loss.backward()
                 self.policy_optimizer.step()
        
        

        
        """
        Update target critic networks
        """
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)
        
        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'likelihood Loss',
                ptu.get_numpy(neg_li)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))

        
        self._n_train_steps_total += 1
       
    
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            trained_policy=self.policy,
            target_policy=self.target_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
