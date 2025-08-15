import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import random
from collections import deque


def read_LST_data(LST_name_list_dir,start_year,end_year):
    LST_name_list_dir = LST_name_list_dir
    LST_name_list = []
    for i in range(start_year,end_year):
        LST_name_list.append(LST_name_list_dir + '//' + f'MYD21C2_LST_{i}_03.tif')
    print(LST_name_list)

    # 读取第一个文件获取空间信息
    ds_first = gdal.Open(LST_name_list[0])
    geo_transform = ds_first.GetGeoTransform()
    projection = ds_first.GetProjection()
    num_bands = ds_first.RasterCount
    y_size, x_size = ds_first.RasterYSize, ds_first.RasterXSize
    # 计算坐标轴
    x_coords = np.linspace(
        geo_transform[0] + geo_transform[1]/2, 
        geo_transform[0] + geo_transform[1]*(x_size - 0.5), 
        x_size
    )

    y_coords = np.linspace(
        geo_transform[3] + geo_transform[5]/2, 
        geo_transform[3] + geo_transform[5]*(y_size - 0.5), 
        y_size
    )[::-1]  # 反转y轴使坐标从北到南递增

    # 初始化数据容器
    data_list = []
    years = []

    # 读取所有文件数据
    for file_path in LST_name_list:
        # 提取年份
        year = int(file_path.split('_')[-2])
        years.append(year)
        
        # 读取文件
        ds = gdal.Open(file_path)
        data = np.zeros((num_bands, y_size, x_size))
        
        for band_idx in range(num_bands):
            band = ds.GetRasterBand(band_idx+1)
            data[band_idx, :, :] = band.ReadAsArray()
        
        data_list.append(data)

    # 创建多维数组
    all_data = np.stack(data_list, axis=0)

    # 创建xarray DataArray
    LST_xr = xr.DataArray(
        data=all_data,
        dims=['time', 'band', 'y', 'x'],
        coords={
            'time': years,
            'band': ['LST'],
            'y': y_coords,
            'x': x_coords
        }
    )
    return LST_xr

# SM_xr = read_SM_data(SM_name_list_dir,start_year,end_year)

def read_NDVI_data(NDVI_name_list_dir,start_year,end_year):
    NDVI_name_list_dir = NDVI_name_list_dir
    NDVI_name_list = []
    for i in range(start_year,end_year):
        NDVI_name_list.append(NDVI_name_list_dir + '//' + f'NDVI_{i}_03.tif')
    print(NDVI_name_list)

    # 读取第一个文件获取空间信息
    ds_first = gdal.Open(NDVI_name_list[0])
    geo_transform = ds_first.GetGeoTransform()
    projection = ds_first.GetProjection()
    num_bands = ds_first.RasterCount
    y_size, x_size = ds_first.RasterYSize, ds_first.RasterXSize
    # 计算坐标轴
    x_coords = np.linspace(
        geo_transform[0] + geo_transform[1]/2, 
        geo_transform[0] + geo_transform[1]*(x_size - 0.5), 
        x_size
    )

    y_coords = np.linspace(
        geo_transform[3] + geo_transform[5]/2, 
        geo_transform[3] + geo_transform[5]*(y_size - 0.5), 
        y_size
    )[::-1]  # 反转y轴使坐标从北到南递增

    # 初始化数据容器
    data_list = []
    years = []

    # 读取所有文件数据
    for file_path in NDVI_name_list:
        # 提取年份
        year = int(file_path.split('_')[-2])
        years.append(year)
        
        # 读取文件
        ds = gdal.Open(file_path)
        data = np.zeros((num_bands, y_size, x_size))
        
        for band_idx in range(num_bands):
            band = ds.GetRasterBand(band_idx+1)
            data[band_idx, :, :] = band.ReadAsArray()
        
        data_list.append(data)

    # 创建多维数组
    all_data = np.stack(data_list, axis=0)

    # 创建xarray DataArray
    NDVI_xr = xr.DataArray(
        data=all_data,
        dims=['time', 'band', 'y', 'x'],
        coords={
            'time': years,
            'band': ['NDVI'],
            'y': y_coords,
            'x': x_coords
        }
    )
    return NDVI_xr

# NDVI_xr = read_NDVI_data(NDVI_name_list_dir,start_year,end_year)

def read_SM_data(SM_name_list_dir,start_year,end_year):
    SM_name_list_dir = SM_name_list_dir
    SM_name_list = []
    for i in range(start_year,end_year):
        SM_name_list.append(SM_name_list_dir + '//' + f'ERA5_SoilWater_Avg_{i}.tif')
    print(SM_name_list)

    # 读取第一个文件获取空间信息
    ds_first = gdal.Open(SM_name_list[0])
    geo_transform = ds_first.GetGeoTransform()
    projection = ds_first.GetProjection()
    num_bands = ds_first.RasterCount
    y_size, x_size = ds_first.RasterYSize, ds_first.RasterXSize
    # 计算坐标轴
    x_coords = np.linspace(
        geo_transform[0] + geo_transform[1]/2, 
        geo_transform[0] + geo_transform[1]*(x_size - 0.5), 
        x_size
    )

    y_coords = np.linspace(
        geo_transform[3] + geo_transform[5]/2, 
        geo_transform[3] + geo_transform[5]*(y_size - 0.5), 
        y_size
    )[::-1]  # 反转y轴使坐标从北到南递增

    # 初始化数据容器
    data_list = []
    years = []

    # 读取所有文件数据
    for file_path in SM_name_list:
        # 提取年份
        year = int(file_path.split('_')[-1][0:4])
        years.append(year)
        
        # 读取文件
        ds = gdal.Open(file_path)
        data = np.zeros((num_bands, y_size, x_size))
        
        for band_idx in range(num_bands):
            band = ds.GetRasterBand(band_idx+1)
            data[band_idx, :, :] = band.ReadAsArray()
        
        data_list.append(data)

    # 创建多维数组
    all_data = np.stack(data_list, axis=0)

    # 创建xarray DataArray
    SM_xr = xr.DataArray(
        data=all_data,
        dims=['time', 'band', 'y', 'x'],
        coords={
            'time': years,
            'band': ['avr_sm'],
            'y': y_coords,
            'x': x_coords
        }
    )
    return SM_xr
# SM_xr = read_SM_data(SM_name_list_dir,start_year,end_year)

# 定义环境类
class FunctionOptimizationEnv():
    def __init__(self,LST_xr,SM_xr,NDVI_xr):
        self.state = np.array([8, 280, -15, 310], dtype=np.float32)  # 初始状态
        self.max_steps = 5000
        self.current_step = 0
        self.sm_xr = SM_xr.isel(band=0)
        self.lst_xr = LST_xr.isel(band=0)
        self.ndvi_xr = NDVI_xr.isel(band=0)
        
    def reset(self):
        """重置环境到初始状态"""
        # self.state = np.array([8, 280, -15, 310], dtype=np.float32)
        self.state = np.array([
        np.random.uniform(-10, 10),   # x1
        np.random.uniform(270, 295),  # x2
        np.random.uniform(-10, 10),   # x3
        np.random.uniform(310, 330)   # x4
    ], dtype=np.float32)
        self.current_step = 0
        return self.state.copy()

    def calculate_tvdi(self, LST_xr, NDVI_xr, x1, x2, x3, x4):
        """
        计算 Temperature Vegetation Dryness Index (TVDI)
        
        参数:
            LST_xr (xarray.DataArray): 地表温度数据，形状(time, y, x)，波段名为'LST'
            NDVI_xr (xarray.DataArray): NDVI数据，形状(time, y, x)，波段名为'NDVI'
            x1 (float): 湿边斜率
            x2 (float): 湿边截距
            x3 (float): 干边斜率
            x4 (float): 干边截距
            
        返回:
            xarray.DataArray: TVDI计算结果，形状(time, y, x)
        """
        lst = LST_xr
        ndvi = NDVI_xr
        
        # 计算湿边和干边的LST值
        LST_wet = x1 * ndvi + x2  # 湿边方程: LST_wet = x1 * NDVI + x2
        LST_dry = x3 * ndvi + x4  # 干边方程: LST_dry = x3 * NDVI + x4
        
        # 计算TVDI
        TVDI = (lst - LST_wet) / (LST_dry - LST_wet)
        
        # 处理可能的异常值
        TVDI = xr.where((LST_dry - LST_wet) == 0, np.nan, TVDI)  # 避免除以0
        TVDI = TVDI.clip(0, 1)  # TVDI理论上应在0-1之间
        
        return TVDI

    @staticmethod
    def calculate_r(xr1,xr2):
        corr_da = xr.corr(xr1, xr2, dim='time')               # 沿时间维度计算
        mean_r=np.nanmean(corr_da)
        
        return mean_r

    def reward_f(self, x1, x2, x3, x4):
        LST_xr = self.lst_xr
        NDVI_xr = self.ndvi_xr
    
        TVDI_xr = self.calculate_tvdi(LST_xr, NDVI_xr, x1, x2, x3, x4)
        
        r2 = self.calculate_r(TVDI_xr,self.sm_xr)**2
        print(r2,end='')
    
        return r2

    
    def step(self, action):
        """执行动作，返回新状态、奖励、是否结束和额外信息"""
        # 根据动作更新状态
        delta = np.zeros(4, dtype=np.float32)
        if action == 0:   # x1 + 0.01
            delta[0] = 0.01
        elif action == 1: # x1 - 0.01
            delta[0] = -0.01
        elif action == 2: # x2 + 0.01
            delta[1] = 0.01
        elif action == 3: # x2 - 0.01
            delta[1] = -0.01
        elif action == 4: # x3 + 0.01
            delta[2] = 0.01
        elif action == 5: # x3 - 0.01
            delta[2] = -0.01
        elif action == 6: # x4 + 0.01
            delta[3] = 0.01
        elif action == 7: # x4 - 0.01
            delta[3] = -0.01

        # 分别设置每个参数的上下限
        low_bounds = np.array([-10, 270, -10, 310], dtype=np.float32)
        high_bounds = np.array([10, 295, 10, 330], dtype=np.float32)
        self.state = np.clip(self.state + delta, low_bounds, high_bounds)


        # # 计算奖励
        reward = self.reward_f(x1=self.state[0],
                               x2=self.state[1],
                               x3=self.state[2],
                               x4=self.state[3],)

        # 更新步数
        self.current_step += 1

        # 是否结束
        done = (self.current_step >= self.max_steps) #or np.all(self.state >= 3.999)

        return self.state.copy(), reward, done, {}


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state):
        """前向传播，返回动作概率分布"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def act(self, state):
        """根据状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        """前向传播，返回状态值"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# PPO 算法实现
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 初始化策略网络和价值网络
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        
        # 旧策略网络，用于重要性采样
        self.old_policy = PolicyNetwork(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
    
    def update(self, states, actions, log_probs, rewards, dones):
        """更新网络参数"""
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs).detach()
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        
        # 归一化折扣奖励
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 多次更新网络
        for _ in range(self.K_epochs):
            # 计算新策略的动作概率
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算状态值
            state_values = self.value_net(states).squeeze()
            
            # 计算优势函数
            advantages = discounted_rewards - state_values.detach()
            
            # 重要性采样比率
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 裁剪目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = self.mse_loss(state_values, discounted_rewards)
            
            # 总损失
            entropy_loss = dist.entropy().mean()  # 计算熵
            loss = policy_loss + 0.5 * value_loss - 4.5e-4 * entropy_loss  # 加入熵项

            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新旧策略网络
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.old_policy.load_state_dict(self.policy.state_dict())


# 训练函数
def train_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year):
    LST_name_list_dir = LST_name_list_dir
    SM_name_list_dir = SM_name_list_dir
    NDVI_name_list_dir = NDVI_name_list_dir
    start_year = start_year
    end_year = end_year

    LST_xr = read_LST_data(LST_name_list_dir,start_year,end_year)
    SM_xr = read_SM_data(SM_name_list_dir,start_year,end_year)
    NDVI_xr = read_NDVI_data(NDVI_name_list_dir,start_year,end_year)


    # 初始化环境和PPO
    env = FunctionOptimizationEnv(LST_xr,SM_xr,NDVI_xr)
    state_dim = 4
    action_dim = 8
    agent = PPO(state_dim, action_dim)
    
    # 训练参数
    num_episodes = 50
    max_steps = 5000
    update_interval = 2000  # 每2000步更新一次
    
    # 训练统计
    best_reward = -float('inf')
    reward_history = []
    
    # 经验缓冲区
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    
    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob = agent.old_policy.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            # 定期更新网络
            if len(states) >= update_interval:
                agent.update(states, actions, log_probs, rewards, dones)
                # 清空缓冲区
                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
            
            if done:
                break
        
        # 记录奖励
        reward_history.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('best_ppo_model.pth')
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f'Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}')
    
    # 返回训练历史
    return reward_history


def test_ppo(LST_name_list_dir,SM_name_list_dir,NDVI_name_list_dir,start_year,end_year):
    LST_name_list_dir = LST_name_list_dir
    SM_name_list_dir = SM_name_list_dir
    NDVI_name_list_dir = NDVI_name_list_dir
    start_year = start_year
    end_year = end_year

    LST_xr = read_LST_data(LST_name_list_dir,start_year,end_year)
    SM_xr = read_SM_data(SM_name_list_dir,start_year,end_year)
    NDVI_xr = read_NDVI_data(NDVI_name_list_dir,start_year,end_year)

    # 初始化环境和PPO
    env = FunctionOptimizationEnv(LST_xr,SM_xr,NDVI_xr)
    state_dim = 4
    action_dim = 8
    agent = PPO(state_dim, action_dim)
    
    # 加载最佳模型
    agent.load_model('best_ppo_model.pth')
    
    # 测试模型
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    # 添加记录最佳表现的变量
    best_reward = -float('inf')
    best_state = None
    best_step = 0
    
    while not done and steps < env.max_steps:
        action, _ = agent.policy.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # 更新最佳表现记录
        if reward > best_reward:
            best_reward = reward
            best_state = state.copy()
            best_step = steps
        
        print(f"Step {steps}: State: {np.round(state,4)}, Reward: {reward}, Total Reward: {total_reward}")
    
    # 打印最终结果和最佳结果
    print("\n=== Final Results ===")
    print(f"Final state: {np.round(state, 4)}")
    print(f"Final reward: {total_reward}")
    print(f"Final product: {np.prod(state)}")
    
    print("\n=== Best Performance ===")
    print(f"Best state (step {best_step}): {np.round(best_state, 4)}")
    print(f"Best single-step reward: {best_reward}")
    print(f"Product at best state: {np.prod(best_state)}")
    
    # 返回结果供进一步分析
    return {
        'final_state': state,
        'final_reward': total_reward,
        'best_state': best_state,
        'best_reward': best_reward,
        'best_step': best_step
    }

