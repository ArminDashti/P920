import torch

# Example tensor
x = torch.tensor([100.0, 200.0, 3.0])

# Calculate log-sum-exp
log_sum_exp = torch.logsumexp(x, dim=0)

print("Log-Sum-Exp:", log_sum_exp)

log_sum_exp = torch.exp(x)
print("Log-Sum-Exp:", log_sum_exp)