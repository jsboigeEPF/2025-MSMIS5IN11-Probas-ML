import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# ==========================================
# 1. LE MODÈLE INTELLIGENT (GAT - Attention)
# ==========================================
class BioGNN(torch.nn.Module):
    """
    Utilise le mécanisme d'Attention (GAT).
    - Plus lourd en calcul.
    - Permet de générer la Heatmap (Explicabilité).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.3):
        super(BioGNN, self).__init__()
        torch.manual_seed(12345)
        self.heads = 4 # Nombre de "points de vue" simultanés
        
        # Couches GAT (Attention)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=self.heads, dropout=0.1)
        self.conv2 = GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, dropout=0.1)
        self.conv3 = GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, dropout=0.1)
        
        self.lin = torch.nn.Linear(hidden_channels * self.heads, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch, return_attention=False):
        # 1. Couche 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)

        # 2. Couche 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)

        # 3. Couche 3 (Avec option d'espionnage pour la Heatmap)
        if return_attention:
            # On demande à PyTorch de renvoyer les poids alpha
            x, (edge_index_att, alpha) = self.conv3(x, edge_index, return_attention_weights=True)
            return edge_index_att, alpha
        else:
            # Mode normal (Entraînement)
            x = self.conv3(x, edge_index)
        
        # 4. Pooling (Moyenne des nœuds -> 1 vecteur par protéine)
        x = global_mean_pool(x, batch)
        
        # 5. Classification finale
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.lin(x)
        
        return x


# ==========================================
# 2. LE MODÈLE RAPIDE (GCN - Classique)
# ==========================================
class SimpleGCN(torch.nn.Module):
    """
    Utilise la Convolution simple (GCN).
    - Très rapide.
    - Pas d'attention (Pas de Heatmap possible).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.3):
        super(SimpleGCN, self).__init__()
        torch.manual_seed(12345)
        
        # Couches GCN classiques
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch, return_attention=False):
        # Sécurité : Si on demande une heatmap à ce modèle, on prévient que c'est impossible
        if return_attention:
            raise TypeError("⚠️ ERREUR : Le modèle 'SimpleGCN' n'a pas d'Attention. Impossible de générer la Heatmap. Utilise 'BioGNN' à la place.")

        # 1. Couche 1
        x = self.conv1(x, edge_index)
        x = F.relu(x) # ReLU est standard pour GCN (vs ELU pour GAT)
        x = F.dropout(x, p=self.dropout_rate, training=True)

        # 2. Couche 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)

        # 3. Couche 3
        x = self.conv3(x, edge_index)
        
        # 4. Pooling
        x = global_mean_pool(x, batch)
        
        # 5. Classification
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.lin(x)
        
        return x