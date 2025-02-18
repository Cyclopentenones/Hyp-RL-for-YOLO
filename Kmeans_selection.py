import torch

class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4, device="cpu"):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
        self.means = None  # Centroids 

    def fit(self, data):
        data = data.to(self.device)


        idx = torch.randperm(data.shape[0])[:self.k]
        self.means = data[idx].clone()

        for i in range(self.max_iters):
            distances = torch.cdist(data, self.means, p=2)  # shape: (num_samples, k)
            clusters = torch.argmin(distances, dim=1)  

            new_means = torch.stack([data[clusters == j].mean(dim=0) if (clusters == j).sum() > 0 
                                     else self.means[j] for j in range(self.k)])

            if torch.norm(self.means - new_means) < self.tol:
                break
            
            self.means = new_means

    def predict(self, data):
        data = data.to(self.device)
        distances = torch.cdist(data, self.means, p=2)
        return torch.argmin(distances, dim=1)

    def get_means(self):
        return self.means.cpu().detach().numpy()
