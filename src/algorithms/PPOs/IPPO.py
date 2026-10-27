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


        # Delta is directly computed into A
        A = alives * (rewards + self.gamma * next_values * ~terminated - values)
        for t in reversed(range(len(A) - 1)):
            A[t] += self.gamma * self.lam * A[t + 1] * ~done[t]

        R = A + values

        # We flatten everything we gives to DEVICE
        # we use keep boolean mask to delete dead agent data
        keep = alives.reshape(-1)

        observations      = torch.as_tensor(observations).flatten(0, 2)[keep].to(self.DEVICE)
        available_actions = torch.as_tensor(available_actions).flatten(0, 2)[keep].to(self.DEVICE)
        actions           = torch.as_tensor(actions).flatten(0, 2)[keep].to(self.DEVICE)
        log_probs_old     = torch.as_tensor(log_probs).flatten(0, 2)[keep].to(self.DEVICE)
        A                 = torch.as_tensor(A).flatten(0, 2)[keep].to(self.DEVICE)
        R                 = torch.as_tensor(R).flatten(0, 2)[keep].to(self.DEVICE)

        A = (A - A.mean()) / (A.std(correction=0) + 1e-8)

        for _ in range(self.epochs):
            for i in torch.randperm(len(actions), device=self.DEVICE).chunk(self.n_minibatches):

                dist, V = self.policy(observations[i], available_actions[i]) # Run the new policy at each step
                rho = torch.exp(dist.log_prob(actions[i]) - log_probs_old[i])
                
                # For L_clip, L_vf and H we compute the average, 
                # because the gradient of the average is equals to the average of all gradient 
                L_clip = torch.min(rho * A[i], rho.clamp(1 - self.epsilon, 1 + self.epsilon) * A[i]).mean() 
                L_vf   = ((V - R[i]) ** 2).mean() 
                H      = dist.entropy().mean()
                loss   = -L_clip + self.c_1 * L_vf - self.c_2 * H
                
                self.optimizer.zero_grad() # Delete old gradient
                loss.backward()      # Compute new gradient
                self.optimizer.step() # update the network
                self.scheduler.step() 

        self.buffer.clear()
