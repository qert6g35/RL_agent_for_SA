# źródło https://www.youtube.com/watch?v=MEt6rrxH8W4&t=220s
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
import random
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from SA_ENV import SA_env
from PPO.PPO_Model import PPO_NN
import gymnasium as gym

#niżej struktura zapropomowana przez co-pilota (bardzo nie chciał się do tego przyznać)

# # Define the neural network architecture
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim, std=0.0):
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#         )
#         self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
#         self.apply(self.init_weights)

class PPO:
    def __init__(self):
        #params
        # parametry związane GAE
        self.use_gae = True
        self.gamma = 0.99
        self.gae_lambda = 0.95
        # parametry związane z lr i jego updatem
        self.starting_lr = 0.001 
        self.update_lr = True
        # podstawowe okreslające uczenie
        self.seed = 1
        self.num_envs = 5
        self.num_steps = 128 # ilość kroków na 1 batch 
        self.total_timesteps = int(10 * SA_env().max_steps / self.num_envs)
        self.batch_size = int(self.num_envs * self.num_steps)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("We will use ",self.device," to learn")
        # TRY NOT TO MODIFY: ja nawet nie wiem co te linijki tu robią ;-; 
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.envs = gym.vector.SyncVectorEnv(
            [ SA_env for _ in range(self.num_envs)]
        )
        
        self.agent = PPO_NN(self.envs).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr = self.starting_lr, eps=1e-5)
        
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # print("next obs shape",self.next_obs.shape)
        # print("agent.getValue(next obs)",self.agent.get_value(self.next_obs))
        # print("agent.getValue(next obs) shape",self.agent.get_value(self.next_obs).shape)
        # print()
        # print("agent.get_action_and_value(self.next_obs)",self.agent.get_action_and_value(self.next_obs))

    def run_learning(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        nxt_obs,_ = self.envs.reset()
        next_obs = torch.Tensor(nxt_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size

        # główna pętla ucząca
        for update in range(1,num_updates+1):
            # zmiana / dostosowanie lr
            if self.update_lr:
                updated_lr = self.starting_lr * (1.0 - (update - 1.0)/ num_updates)
                self.optimizer.param_groups[0]["lr"] = updated_lr

            for step in range(0, self.num_steps): # tutaj wykonujemy tyle stepów ile przypada na 1 batch
                global_step += 1 * self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done,_, info = self.envs.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)


            # # bootstrap value if not done
            with torch.no_grad(): # z generalnego zrozumienia dwa warianty poniżej to różne sposoby obliczania wartości ADVENTAGE która służy do określenia jak dobra była podjęta akcja. (Służy do wzoru uczącego w PPO) 
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.use_gae:#                   GeneralAdventageEstimation   to technika, która pomaga wyliczać advantage z mniejszą wariancją, co skutkuje bardziej stabilnym i efektywnym treningiem w algorytmach takich jak PPO.
                    advantages = torch.zeros_like(self.rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values
                else: # TemporalDivrence, inna "common" metoda na obliczanie wartości Adventage i returns 
                    returns = torch.zeros_like(self.rewards).to(self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                    advantages = returns - self.values
            twoja_mama = 1/0 #! skończyliśmy na 18:00 https://www.youtube.com/watch?v=MEt6rrxH8W4&t=220s

            # # flatten the batch
            # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            # b_logprobs = logprobs.reshape(-1)
            # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            # b_advantages = advantages.reshape(-1)
            # b_returns = returns.reshape(-1)
            # b_values = values.reshape(-1)

            # # Optimizing the policy and value network
            # b_inds = np.arange(args.batch_size)
            # clipfracs = []
            # for epoch in range(args.update_epochs):
            #     np.random.shuffle(b_inds)
            #     for start in range(0, args.batch_size, args.minibatch_size):
            #         end = start + args.minibatch_size
            #         mb_inds = b_inds[start:end]

            #         _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            #         logratio = newlogprob - b_logprobs[mb_inds]
            #         ratio = logratio.exp()

            #         with torch.no_grad():
            #             # calculate approx_kl http://joschu.net/blog/kl-approx.html
            #             old_approx_kl = (-logratio).mean()
            #             approx_kl = ((ratio - 1) - logratio).mean()
            #             clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            #         mb_advantages = b_advantages[mb_inds]
            #         if args.norm_adv:
            #             mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            #         # Policy loss
            #         pg_loss1 = -mb_advantages * ratio
            #         pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            #         pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            #         # Value loss
            #         newvalue = newvalue.view(-1)
            #         if args.clip_vloss:
            #             v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            #             v_clipped = b_values[mb_inds] + torch.clamp(
            #                 newvalue - b_values[mb_inds],
            #                 -args.clip_coef,
            #                 args.clip_coef,
            #             )
            #             v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            #             v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            #             v_loss = 0.5 * v_loss_max.mean()
            #         else:
            #             v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            #         entropy_loss = entropy.mean()
            #         loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            #         optimizer.zero_grad()
            #         loss.backward()
            #         nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            #         optimizer.step()

            #     if args.target_kl is not None:
            #         if approx_kl > args.target_kl:
            #             break

            # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            # var_y = np.var(y_true)
            # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # # TRY NOT TO MODIFY: record rewards for plotting purposes
            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print(global_step)

    def show_agent(self):
        print(self.agent)
