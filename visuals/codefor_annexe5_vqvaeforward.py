def forward(self, x, training=False):
    # Dans les commentaires qui suivent:
    #   B: taille du lot
    #   H et W: hauteur et largeur dans l'espace latent
    #   N: dimension des vecteurs caractéristiques 
    #   n: nombre de vecteurs discrets dans le dictionnaire
    
    # Redimensionner l'entrée pour isoler la taille du lot et celle du code latent,
    # récupérant une liste de vecteurs caractéristiques de la même taille que la table discrète
    flat_x = x.permute(0, 2, 3, 1)  # B, H, W, N
    flat_x = flat_x.reshape((flat_x.shape[0], -1, flat_x.shape[-1]))  # [B, H x W, N]
    
    # Calcul de la distance entre chaque vecteur discret et chaque vecteur caractéristique
    lookup_weights = self.embeddings.weight[None, :].repeat((flat_x.shape[0], 1, 1))
    distances = torch.cdist(flat_x, lookup_weights)  # [B, H x W, n]
    
    # Extrait l'id des vecteurs discrets qui sont les plus proches de chacun des vecteur caractéristique 
    encoding_indices = distances.argmin(dim=-1)  # B, H x W
    # Etape de quantification
    quantized = torch.index_select(self.embeddings.weight, 0, encoding_indices.view(-1))  # B x H x W, N 
    flat_x = flat_x.reshape((-1, flat_x.shape[-1]))  # B x H x W, Embedding dim
    
    # Calcul des pertes pour mettre à jour le réseau
    # Codebook loss: Les vecteurs discrets se dirigent vers le résultat de l'encodeur
    self.last_codebook_loss = None if self.ema else torch.mean((quantized - flat_x.detach()) ** 2)
    # Commitment loss: L'encodeur devrait produire des résultats plus proches des vecteurs discrets
    self.last_commitment_loss = torch.mean((flat_x - quantized.detach()) ** 2)

    # Préparatifs pour historiser le degré d'utilisation des vecteurs discrets
    encodings = F.one_hot(encoding_indices.view(-1), num_classes=self.num_embeddings).float()  # B x H x W, n
    cluster_count = encodings.sum(dim=0)
    self.overall_embedding_usage += cluster_count  # historisation

    if self.ema and training:
        # Entraînement manuel des Sonnet EMA
        with torch.no_grad():
            # Calcul de l'utilisation de chaque vecteur discret
            updated_ema_cluster_size = self.EMA_cluster_counts(cluster_count)
            dw = torch.matmul(flat_x.T, encodings)
            updated_ema_dw = self.EMA_embeddings(dw)
            
            # Mise à jour progressive de la table de vecteurs discrets
            n = updated_ema_cluster_size.sum()
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            normalised_updated_ema_w = updated_ema_dw / updated_ema_cluster_size.unsqueeze(0)
            self.embeddings.weight.copy_(normalised_updated_ema_w.t())
            
            # Technique du Random Codebook Restart
            if self.random_restart > 0:
                self.used_embeddings += cluster_count
                self.restart_in -= 1
                if self.restart_in == 0:
                    self.restart_in = self.random_restart
                    dead_indices = torch.where(self.used_embeddings < self.restart_threshold)[0]
                    if dead_indices.numel() > 0:
                        # Réinitialisation des entrées qui ne sont pas suffisamment utilisées avec des valeurs du batch actuel
                        rand_indices = torch.randint(0, flat_x.size(0), (dead_indices.shape[0], ))
                        new_entries = flat_x[rand_indices]
                        self.embeddings.weight[dead_indices] = new_entries
                    # Réinitialisation des données d'utilisation
                    self.used_embeddings = torch.zero_(self.used_embeddings)

    # Straight Through Estimation: quantized ~= x
    quantized = flat_x + (quantized - flat_x).detach()
    quantized = quantized.reshape(x.shape)  # Batch Size, Channels, Height, Width
    
    # Perplexity computation: évaluation de la qualité de l'utilisation du codebook
    avg_probs = torch.mean(encodings, 0)
    self.last_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    return quantized
