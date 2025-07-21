import time

import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import nn


from model import function


from model.noise import GammaNetwork, PredefinedNoiseSchedule
from typing import Union



class EDM(torch.nn.Module):
    def __init__(
            self,
            dynamics_scaffold, dynamics_rgroup,
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),

    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:  # 使用 PredefinedNoiseSchedule 根据指定的调度类型和时间步长预定义噪声。
            # self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)
            self.gamma_scaffold =PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=timesteps,precision=1e-5,task_type='scaffold')
            self.gamma_rgroup = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=timesteps,precision=1e-5,task_type='rgroup')

        self.dynamics_scaffold = dynamics_scaffold
        self.dynamics_rgroup = dynamics_rgroup

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases


    def forward(self, x, h, node_mask, fragment_mask, remaining_mask, rgroup_mask, scaffold_mask,
            edge_mask, context=None,
              task_id =None,pocket_mask=None):
        """
        根据当前训练阶段，进行不同区域的噪声加噪：
          - skeleton: 生成骨架 remain 部分（fragment 保持不变）
          - rgroup: 对 rgroup 部分加噪，保持 scaffold 不变
          - joint: 联合训练，采用 rgroup 分支（条件中融合口袋信息）
        # """

        if task_id == 0:
            dynamics = self.dynamics_scaffold
        elif task_id == 1:
            dynamics = self.dynamics_rgroup
        else:
            raise NotImplementedError("Unknown task_id")

        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # 2. 根据阶段选择对应的掩码及条件信息
        if task_id == 0:
            # 骨架阶段：目标是生成 remain 部分，保留 fragment 部分
            noise_mask = remaining_mask  # 仅对 remain 部分加噪
        elif task_id == 1:
            # R基团阶段：目标是生成 rgroup 部分，保持 scaffold 不变
            noise_mask = rgroup_mask  # 仅对 rgroup 部分加噪
        else:
            raise NotImplementedError("Unknown training stage: %s" % self.training_stage)

        delta_log_px = self.delta_log_px(noise_mask).mean()

        # 生成一个形状为 (batch_size, 1) 的张量，每个元素是从 [0, self.T] 范围内随机抽取的整数
        t_int = torch.randint(0, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1

        # 将离散时间步 t_int 和 s_int 归一化到 [0, 1] 范围内。
        t = t_int / self.T
        s = s_int / self.T
        # 创建掩码，标记哪些样本的 t_int 为 0。
        t_is_zero = (t_int == 0).float().squeeze()
        t_is_not_zero = 1 - t_is_zero

        # # 5. 计算扩散参数 gamma, alpha, sigma                 self.gamma(t)得到每个时间步的噪声调度参数  γ-t
        # gamma_t = self.inflate_batch_array(self.gamma(t), x)  # 将返回的 γ 值扩展到与目标张量 x 的形状匹配，
        # gamma_s = self.inflate_batch_array(self.gamma(s), x)
        # alpha_t = self.alpha(gamma_t, x)  # 根据 γ 值计算α
        # sigma_t = self.sigma(gamma_t, x)   # 根据 γ 值计算 σ

        if task_id == 0:
            gamma_t = self.inflate_batch_array(self.gamma_scaffold(t), x)
            gamma_s = self.inflate_batch_array(self.gamma_scaffold(s), x)
        else:
            gamma_t = self.inflate_batch_array(self.gamma_rgroup(t), x)
            gamma_s = self.inflate_batch_array(self.gamma_rgroup(s), x)

        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)


        eps_t = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), mask=noise_mask)


        # 7. 生成加噪样本 z_t
        z_t = alpha_t * xh + sigma_t * eps_t

        if task_id == 0:
            z_t = xh * fragment_mask + z_t * noise_mask + xh * pocket_mask
        else :
            z_t = xh * scaffold_mask + z_t * rgroup_mask + xh * pocket_mask

        eps_t_hat = dynamics.forward(
                xh=z_t, t=t, node_mask=node_mask, noise_mask=noise_mask, context=context, edge_mask=edge_mask,
                task_id=task_id
            )

        eps_t_hat = eps_t_hat * noise_mask

        error_t = self.sum_except_batch((eps_t - eps_t_hat) ** 2)
        normalization = (self.n_dims + self.in_node_nf) * self.numbers_of_nodes(noise_mask)
        l2_loss = (error_t / normalization).mean()


        kl_prior = self.kl_prior(x, noise_mask, task_id).mean()

        SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        loss_term_t = self.T * 0.5 * SNR_weight * error_t
        loss_term_t = (loss_term_t * t_is_not_zero).sum() / (t_is_not_zero.sum() + 1e-8)
        noise_norm = torch.norm(eps_t_hat, dim=[1, 2])
        noise_t_val = (noise_norm * t_is_not_zero).sum() / (t_is_not_zero.sum() + 1e-8)

        if t_is_zero.sum() > 0:
            neg_log_constants = -self.log_constant_of_p_x_given_z0(x, noise_mask, task_id)
            loss_term_0 = -self.log_p_xh_given_z0_without_constants(h, z_t, gamma_t, eps_t, eps_t_hat, noise_mask)
            loss_term_0 = loss_term_0 + neg_log_constants
            loss_term_0 = (loss_term_0 * t_is_zero).sum() / (t_is_zero.sum() + 1e-8)
            noise_0 = (noise_norm * t_is_zero).sum() / (t_is_zero.sum() + 1e-8)
        else:
            loss_term_0 = 0.
            noise_0 = 0.

        return delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t_val, noise_0,z_t


    @torch.no_grad()
    def sample_scaf_chain(self, x, h, node_mask, fragment_mask, edge_mask, context, keep_frames=None,
                    rem_mask=None, pocket_mask=None,id=None,node_mask_1=None):
        print("开始采样")
        n_samples = x.size(0)
        n_nodes = x.size(1)

        # Normalization and concatenation
        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        fragment_mask  = fragment_mask
        rem_mask = rem_mask

        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, mask=rem_mask)
        z = xh * fragment_mask + z * rem_mask+xh * pocket_mask

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt_only_scaf(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                fragment_mask=fragment_mask,
                rem_mask=rem_mask,
                pocket_mask=pocket_mask,
                edge_mask=edge_mask,
                context=context,
                id=id,
                node_mask_1=node_mask_1
            )
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_scaf(
            z_0=z,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            rem_mask=rem_mask,
            pocket_mask=pocket_mask,
            edge_mask=edge_mask,
            context=context,
            id=id,
            node_mask_1=node_mask_1
        )
        chain[0] = torch.cat([x, h], dim=2)

        return chain

    def sample_p_zs_given_zt_only_scaf(self, s, t, z_t, node_mask, fragment_mask, rem_mask, pocket_mask=None,
                                       edge_mask=None, context=None, id=None,node_mask_1=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only rgroup features and coords"""
        gamma_s = self.gamma_scaffold(s)
        gamma_t = self.gamma_scaffold(t)


        # 计算条件标准噪声差和缩放因子
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)


        sigma_s = self.sigma(gamma_s, target_tensor=z_t)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t)


        # 神经网络预测噪声，仅针对 rem 部分
        eps_hat = self.dynamics_scaffold.forward(
           xh=z_t,
            t=t,
            node_mask=node_mask,
            task_id=id,
            noise_mask=rem_mask,
            edge_mask=edge_mask,
           context=context
        )
        eps_hat = eps_hat * rem_mask


        # 计算 p(z_s | z_t) 的均值 mu
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # 计算 p(z_s | z_t) 的标准差 sigma
        sigma = sigma_t_given_s * sigma_s / sigma_t


        # 根据参数采样 z_s
        z_s = self.sample_normal(mu, sigma, rem_mask)
        z_s = z_t * fragment_mask + z_s * rem_mask + z_t * pocket_mask


        return z_s



    def sample_p_xh_given_z0_only_scaf(self, z_0, node_mask, fragment_mask, rem_mask, pocket_mask=None, edge_mask=None,
                                       context=None, id=None, node_mask_1=None):
        """Samples x ~ p(x|z0). Samples only rgroup features and coords"""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma_scaffold(zeros)

        #  torch.exp(-gamma)
        # 计算标准差
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        # 神经网络预测噪声，仅针对 rem 部分
        eps_hat= self.dynamics_scaffold.forward(
            t=zeros,
            xh=z_0,
            node_mask=node_mask,
            task_id=id,
            noise_mask=rem_mask,
            edge_mask=edge_mask,
            context=context
        )  # 形状 [N_rem, feats_dim + pos_dim]
        eps_hat = eps_hat * rem_mask

        # 计算 x 的均值
        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)

        # 采样 xh
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=rem_mask)

        xh = z_0 * fragment_mask + xh * rem_mask + z_0 * pocket_mask

        # 分离 x 和 h 并反归一化
        x, h = xh[:, :, :self.n_dims], xh[:, :, self.n_dims:]
        x_new, h_new = self.unnormalize(x, h)
        x, h = x_new, h_new  # 更新 x 和 h

        # 将 h 转换为独热编码
        h = F.one_hot(torch.argmax(h, dim=2), self.in_node_nf).float()  # 确保输出为浮点型张量
        # 确保 node_mask 的形状匹配
        node_mask1 = node_mask.expand_as(h)  # [batch_size, num_atoms, self.in_node_nf]
        h = h * node_mask1  # 应用掩码


        return x, h

    @torch.no_grad()
    def sample_chain(self, x, h, node_mask, scaffold_mask, rgroup_mask,pocket_mask=None,edge_mask=None, context=None, keep_frames=None,id=None):
        n_samples = x.size(0)
        n_nodes = x.size(1)

        # Normalization and concatenation
        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initial rgroup sampling from N(0, I)
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, mask=rgroup_mask)


        z = xh * scaffold_mask + z * rgroup_mask +xh*pocket_mask


        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        # 这里 range(0, self.T) 生成从 0 到 self.T - 1 的整数序列。
        # reversed(...) 使得循环从最后一步（self.T - 1）开始，到第 0 步结束。
        # Sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z.device)  # 为当前时间步 s 创建一个形状为 (n_samples, 1) 的张量，每个样本对应的时间步都为 s。
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt_only_rgroup(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                scaffold_mask=scaffold_mask,
                rgroup_mask=rgroup_mask,
                pocket_mask=pocket_mask,
                edge_mask=edge_mask,
                context=context,
                id=id
            )
            # 根据  𝑠计算保存的索引 write_index（这里通过线性映射），并将经过去归一化处理后的  z 存入 chain。
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_rgroup(
            z_0=z,
            node_mask=node_mask,
            scaffold_mask=scaffold_mask,
            rgroup_mask=rgroup_mask,
            pocket_mask=pocket_mask,
            edge_mask=edge_mask,
            context=context,
            id=id
        )
        chain[0] = torch.cat([x, h], dim=2)

        return chain

    # 在采样过程中，根据当前状态Zt计算并且采样出上一个时间步Zs的状态，条件分布
    def sample_p_zs_given_zt_only_rgroup(self, s, t, z_t, node_mask, scaffold_mask, rgroup_mask,  pocket_mask, edge_mask, context,id=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only rgroup features and coords"""
        gamma_s = self.gamma_rgroup(s)
        gamma_t = self.gamma_rgroup(t)

        # 这个函数根据 gamma_t 和 gamma_s 以及Zt计算条件标准噪声差和缩放因子
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = self.sigma(gamma_s, target_tensor=z_t)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t)

        # Neural net prediction.
        eps_hat = self.dynamics_rgroup.forward(
            xh=z_t,
            t=t,
            node_mask=node_mask,
            noise_mask=rgroup_mask,
            context=context,
            edge_mask=edge_mask,
            task_id=id
        )
        eps_hat = eps_hat * rgroup_mask

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # Compute sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        z_s = self.sample_normal(mu, sigma, rgroup_mask)
        z_s = z_t * scaffold_mask + z_s * rgroup_mask+z_t * pocket_mask

        return z_s

    def sample_p_xh_given_z0_only_rgroup(self, z_0, node_mask, scaffold_mask, rgroup_mask,pocket_mask, edge_mask, context,id=None):
        """Samples x ~ p(x|z0). Samples only rgroup features and coords"""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma_rgroup(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        eps_hat = self.dynamics_rgroup.forward(
            t=zeros,
            xh=z_0,
            node_mask=node_mask,
           noise_mask=rgroup_mask,
            edge_mask=edge_mask,
            context=context,
            task_id=id

        )
        eps_hat = eps_hat * rgroup_mask

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=rgroup_mask)
        xh = z_0 * scaffold_mask + xh * rgroup_mask +z_0 * pocket_mask

        x, h = xh[:, :, :self.n_dims], xh[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        h = F.one_hot(torch.argmax(h, dim=2), self.in_node_nf) * node_mask

        return x, h


    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def kl_prior(self, xh, mask,task):
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T
        # xh：输入数据张量，形状为 (batch_size, n_nodes, n_dims + in_node_nf)，包含位置 (x) 和特征 (h) 两部分。
        ones = torch.ones((xh.size(0), 1), device=xh.device)# 形状为 (batch_size, 1) 的全一张量，表示归一化的时间步
        if task == 0:
            gamma=self.gamma_scaffold
        else:
            gamma=self.gamma_rgroup
        # self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)
        gamma_T =gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T, mu_T_x).view(-1)  # Remove inflate, only keep batch dimension for x-part
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = self.gaussian_kl(mu_T_h, sigma_T_h, zeros, ones)

        # Compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        d = self.dimensionality(mask)
        kl_distance_x = self.gaussian_kl_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=d)

        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, x, mask,task_id):
        batch_size = x.size(0)
        degrees_of_freedom_x = self.dimensionality(mask)
        zeros = torch.zeros((batch_size, 1), device=x.device)
        if task_id == 0:
            gamma_x = self.gamma_scaffold
        else:
            gamma_x = self.gamma_rgroup
        gamma_0 = gamma_x(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_0
        z_h = z_0[:, :, self.n_dims:]

        # Take only part over x
        eps_x = eps[:, :, :self.n_dims]
        eps_hat_x = eps_hat[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0) * self.norm_values[1]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        log_p_x_given_z_without_constants = -0.5 * self.sum_except_batch((eps_x - eps_hat_x) ** 2)

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=2, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z = self.sum_except_batch(log_probabilities * h * mask)

        # Combine log probabilities for x and h
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, mask):
        # 其功能是为图中节点的位置（坐标）和特征生成高斯噪声，并且只对掩码（mask）标记的部分生成噪声。
        z_x = function.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = function.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        eps = self.sample_combined_position_feature_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps

    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(2) == self.n_dims + self.in_node_nf
        x, h = z[:, :, :self.n_dims], z[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=2)

    def delta_log_px(self, mask):
        return -self.dimensionality(mask) * np.log(self.norm_values[0])

    def dimensionality(self, mask): # 结果是  有效节点数目乘以维度数。
        return self.numbers_of_nodes(mask) * self.n_dims

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s_scaf(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor,s):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        Added value clamping to prevent numerical instability and control variance growth.
        """
        # 限制 gamma 的范围以防止数值不稳定
        gamma_t = gamma_t.clamp(min=-10., max=10.)
        gamma_s = gamma_s.clamp(min=-10., max=10.)

        # 计算 log(alpha_t^2) 和 log(alpha_s^2)
        log_alpha2_t = F.logsigmoid(-gamma_t)  # shape: (batch,)
        log_alpha2_s = F.logsigmoid(-gamma_s)  # shape: (batch,)

        # 计算 log((alpha_t/alpha_s)^2)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s  # shape: (batch,)

        # 计算 (alpha_t/alpha_s)^2 并添加约束
        alpha2_t_given_s = torch.exp(log_alpha2_t_given_s)  # (alpha_t/alpha_s)^2
        alpha2_t_given_s = torch.clamp(alpha2_t_given_s, min=1e-6, max=1.0)  # 防止值过小或大于1

        # 计算 sigma^2 = 1 - alpha^2 并添加下限
        sigma2_t_given_s = 1.0 - alpha2_t_given_s
        sigma2_t_given_s = torch.clamp(sigma2_t_given_s, min=1e-10)  # 防止负数或过小值

        # 扩展到目标张量形状
        sigma2_t_given_s = self.inflate_batch_array(sigma2_t_given_s, target_tensor)
        alpha_t_given_s = torch.sqrt(self.inflate_batch_array(alpha2_t_given_s, target_tensor))
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)


        max_sigma = 1.0 * (s / self.T) if hasattr(self, 'T') else 1.0  # 动态或固定
        sigma_t_given_s = torch.clamp(sigma_t_given_s, max=max_sigma)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
          alpha t given s = alpha t / alpha s,
          sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    def numbers_of_nodes(mask):
        if mask.dim() == 3:
            mask = mask.squeeze(2)  # (B, N, 1) -> (B, N)
        elif mask.dim() != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        return torch.sum(mask, dim=1)  # (B, N) -> (B,)

    @staticmethod
    def inflate_batch_array(array, target):
        # 函数的作用是将一个批次的张量 array 扩展到与目标形状 target 匹配的形状，扩展的部分是维度为 1 的空维度
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    @staticmethod
    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma):
        """
        Computes the KL distance between two normal distributions.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        kl = torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2) - 0.5
        return EDM.sum_except_batch(kl)

    @staticmethod
    def gaussian_kl_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
        """
        Computes the KL distance between two normal distributions taking the dimension into account.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
            d: dimension
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        mu_norm_2 = EDM.sum_except_batch((q_mu - p_mu) ** 2)
        return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm_2) / (p_sigma ** 2) - 0.5 * d