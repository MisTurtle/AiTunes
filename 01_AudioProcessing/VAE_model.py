import torch
from torch import nn

class VAE_Model(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_Model, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Couche d'entrée
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)

        )
        
        # Paramètres latents
        self.mu = nn.Linear(64, latent_dim)      # Moyenne de la distribution latente
        self.logvar = nn.Linear(64, latent_dim)  # Logarithme de la variance


        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  # Couche d'entrée pour décodeur
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),   # Couche de sortie pour reconstruire l'entrée
            nn.Sigmoid()                 # Normalisaer 
        )

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)      
        return mu + eps * std            

    def forward(self, x):
        x = self.encoder(x)              
        mu = self.mu(x)                  
        logvar = self.logvar(x)          
        z = self.reparameterization(mu, logvar)  
        return self.decoder(z), mu, logvar  