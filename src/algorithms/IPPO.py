import numpy as np
import torch


class IPPO:
    # We use lam instead of lambda as lambda keyword create a lambda function in python
    def __init__(self, policy, buffer, learning_rate, gamma, lam, epsilon, c_1, c_2, epochs, n_minibatches, total_iters):
        self.policy = policy
        self.buffer = buffer

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.c_1 = c_1
        self.c_2 = c_2
        
        self.epochs = epochs
        self.n_minibatches = n_minibatches

        self.DEVICE = next(self.policy.parameters()).device

    def update(self):

        # Data collection, correction of shape mismatch, i.e (T, E) -> (T, E, 1)
        alives            = self.buffer.alives
        observations      = self.buffer.observations
        available_actions = self.buffer.available_actions
        actions           = self.buffer.actions
        log_probs         = self.buffer.log_probs
        values            = self.buffer.values
        next_values       = self.buffer.next_values
        rewards           = self.buffer.rewards[..., None]
        terminated        = self.buffer.terminated[..., None]
        truncated         = self.buffer.truncated[..., None]
        done              = terminated | truncated

        # metrics will be used to construct diagrams
        metrics = {
            "PPO/policy loss"       : [],
            "PPO/value loss"        : [],
            "PPO/entropy"           : [],
            "PPO/approx kl"         : [],
            "PPO/clip frac"         : []
            }

        # Delta is directly computed into A
        A = (rewards + self.gamma * next_values * ~terminated - values) # We give dead agents the teams reward to encourage sacrifice
        for t in reversed(range(len(A) - 1)):
            A[t] += self.gamma * self.lam * A[t + 1] * ~done[t]

        R = A + values

        # We flatten everything we give to DEVICE
        # we use keep boolean mask to delete dead agent data
        keep = alives.reshape(-1)

        observations      = torch.as_tensor(observations).flatten(0, 2)[keep].to(self.DEVICE)
        available_actions = torch.as_tensor(available_actions).flatten(0, 2)[keep].to(self.DEVICE)
        actions           = torch.as_tensor(actions).flatten(0, 2)[keep].to(self.DEVICE)
        log_probs_old     = torch.as_tensor(log_probs).flatten(0, 2)[keep].to(self.DEVICE)
        A                 = torch.as_tensor(A).flatten(0, 2)[keep].to(self.DEVICE)
        R                 = torch.as_tensor(R).flatten(0, 2)[keep].to(self.DEVICE)

        V_old              = torch.as_tensor(values).flatten(0, 2)[keep].to(self.DEVICE)
        explained_variance = (1 - (R - V_old).var() / R.var()).item()

        A = (A - A.mean()) / (A.std(correction=0) + 1e-8)

        for _ in range(self.epochs):
            for i in torch.randperm(len(actions), device=self.DEVICE).chunk(self.n_minibatches):

                dist, V = self.policy(observations[i], available_actions[i]) # Run the new policy at each step
                # r_t(theta) doesnt work as variable names cannot have parenthesis
                # we call it rho
                log_rho = dist.log_prob(actions[i]) - log_probs_old[i]
                rho = torch.exp(log_rho)
                L_clip = torch.min(rho * A[i], rho.clamp(1 - self.epsilon, 1 + self.epsilon) * A[i]).mean() 
                L_vf   = ((V - R[i]) ** 2).mean() 
                H      = dist.entropy().mean()
                loss   = -L_clip + self.c_1 * L_vf - self.c_2 * H
                
                self.optimizer.zero_grad() # Delete old gradient
                loss.backward()      # Compute new gradient
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0) # normalise to avoid too big gradient
                self.optimizer.step() # update the network

                with torch.no_grad():
                    metrics["PPO/value loss"].append(L_vf.item())
                    metrics["PPO/policy loss"].append(-L_clip.item())
                    metrics["PPO/entropy"].append(H.item())
                    metrics["PPO/approx kl"].append(((rho - 1) - log_rho).mean().item())
                    metrics["PPO/clip frac"].append(((rho - 1).abs() > self.epsilon).float().mean().item())

        learning_rate = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        self.buffer.clear()
        return {name: np.mean(scalar) for name, scalar in metrics.items()} | {
            "PPO/learning rate"     : learning_rate,
            "PPO/explained variance": explained_variance,
        }
