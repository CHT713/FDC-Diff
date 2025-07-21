import torch
import torch.nn.functional as F
import math
import numpy as np


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
     # clip_value裁剪的最小值（默认为0.001），用于限制步与步之间的变化
     #该函数通过裁剪 alpha_t / alpha_{t-1} 的比率来提高采样的稳定性
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0) # 在输入的 alphas2 数组前面拼接一个值为1的元素

    alphas_step = (alphas2[1:] / alphas2[:-1]) # alphas_step 表示每个时间步相对于前一步的变化率
 # 例如：若 alphas2 = [1., 0.9, 0.8, 0.7]，则 alphas_step = [0.9/1., 0.8/0.9, 0.7/0.8] = [0.9, 0.888..., 0.875]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
# 使用 np.clip 将 alphas_step 的值限制在 [clip_value, 1.] 范围内,例如，若 alphas_step = [0.9, 0.0005, 0.875]，且 clip_value=0.001，则变为 [0.9, 0.001, 0.875]
    alphas2 = np.cumprod(alphas_step, axis=0)
# 对 alphas_step 计算累积乘积，生成新的 alphas2

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1  # 1001
    x = np.linspace(0, steps, steps)  # 生成一个包含 0到1001个均匀分布数字的数组
    alphas2 = (1 - np.power(x / steps, power)) ** 2  # 计算的是 (x / steps) 的 power 次方。

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)  # 一个经过裁剪和调整的噪声调度数组

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.004, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        positive_weight = F.softplus(self.weight)
        return F.linear(x, positive_weight, self.bias)


def cosine_beta_schedule_smooth(timesteps, s=0.01, raise_to_power: float = 1):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=1e-6, a_max=0.999)  # 控制betas的范围
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod, betas


class PredefinedNoiseSchedule1(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision, task_type=None):
        super(PredefinedNoiseSchedule1, self).__init__()
        self.timesteps = timesteps
        self.task_type = task_type

        # 统一为所有任务类型使用更稳定的调度实现
        if noise_schedule == 'cosine':
            # 使用更稳定的cosine调度
            steps = timesteps + 2
            x = np.linspace(0, steps, steps)
            # 添加s参数使曲线更平滑
            s = 0.008  # 更大的s值使曲线在端点更平滑
            alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas2 = alphas_cumprod
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            power = float(splits[1]) if len(splits) == 2 else 3.0
            # 确保多项式调度平滑过渡
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        # 确保数值稳定性
        alphas2 = np.clip(alphas2, a_min=1e-7, a_max=0.9999)
        sigmas2 = 1 - alphas2
        sigmas2 = np.clip(sigmas2, a_min=1e-7, a_max=0.9999)

        # 使用log空间计算更稳定
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # 存储更多信息以便采样时使用
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

        # 额外保存alpha和sigma值，避免重复计算
        self.log_alphas2 = torch.nn.Parameter(
            torch.from_numpy(log_alphas2).float(),
            requires_grad=False)
        self.log_sigmas2 = torch.nn.Parameter(
            torch.from_numpy(log_sigmas2).float(),
            requires_grad=False)

        # 保存原始的alpha和sigma值
        self.alphas = torch.nn.Parameter(
            torch.from_numpy(np.sqrt(alphas2)).float(),
            requires_grad=False)
        self.sigmas = torch.nn.Parameter(
            torch.from_numpy(np.sqrt(sigmas2)).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

    def get_alpha_sigma(self, t):
        """返回特定时间步的alpha和sigma值"""
        t_int = torch.round(t * self.timesteps).long()
        alpha = self.alphas[t_int]
        sigma = self.sigmas[t_int]
        return alpha, sigma

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
# self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)
    def __init__(self, noise_schedule, timesteps, precision,task_type=None):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps
        self.task_type = task_type
        # print(precision) =1e-05
        if self.task_type == 'scaffold':
            # Use polynomial schedule with power=2 (linear-like) for scaffold task
            if noise_schedule =='cosine':
                alphas2 = cosine_beta_schedule(timesteps)
            elif 'polynomial' in noise_schedule:  # polynomial_2
                splits = noise_schedule.split('_')
                assert len(splits) == 2
                power = float(splits[1])
                alphas2 = polynomial_schedule(timesteps, s=precision,
                                              power=power)  # polynomial_schedule 函数的目的是生成一个基于多项式的噪声调度
        else:
            if noise_schedule == 'cosine':
                alphas2 = cosine_beta_schedule(timesteps)
            elif 'polynomial' in noise_schedule:  # polynomial_2
                splits = noise_schedule.split('_')
                assert len(splits) == 2
                power = float(splits[1])
                alphas2 = polynomial_schedule(timesteps, s=precision, power=power) # polynomial_schedule 函数的目的是生成一个基于多项式的噪声调度数组
            else:
                raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma
