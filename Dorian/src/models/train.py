import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset_loader import load_bio_data
from model import BioGNN
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. CHARGEMENT
# ==========================================
# Charge PROTEINS (Assure-toi que dataset_loader charge bien PROTEINS)
train_loader, test_loader, num_features, num_classes = load_bio_data(batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔬 Analyse en cours sur : {device}")

# Modèle GAT (Attention) - BioGNN
model = BioGNN(in_channels=num_features, hidden_channels=128, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

loss_history = []

# ==========================================
# 2. ENTRAÎNEMENT
# ==========================================
print("\n🧬 Début de l'apprentissage...")
model.train()

for epoch in range(1, 151):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

# ==========================================
# 3. TEST UNITAIRE (Bayésien)
# ==========================================
print("\n💊 Simulation : Test sur 10 PROTEINES inconnues...")
print(f"{'ID':<8} | {'Prédiction':<12} | {'Confiance':<10} | {'Incertitude':<15}")
print("-" * 55)

model.eval()

for i, data in enumerate(test_loader):
    if i >= 10: break 
    data = data.to(device)
    
    probs_active = []
    
    # Monte Carlo Dropout
    with torch.no_grad():
        for _ in range(50):
            out = model(data.x, data.edge_index, data.batch)
            prob = torch.softmax(out, dim=1)[:, 1].item()
            probs_active.append(prob)
    
    mean_prob = np.mean(probs_active)
    uncertainty = np.std(probs_active)
    
    if mean_prob > 0.5:
        pred_label = "ENZYME"
        real_confidence = mean_prob 
    else:
        pred_label = "NON ENZYME"
        real_confidence = 1.0 - mean_prob 
    
    print(f"Prot_{i:<3} | {pred_label:<12} | {real_confidence:.2f}       | {uncertainty:.4f}")

print("-" * 55)

# ==========================================
# 4. BILAN GLOBAL
# ==========================================
print("\n📊 Bilan de performance global :")
y_true = []
y_pred = []

for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1).item()
        y_true.append(data.y.item())
        y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
cm_val = confusion_matrix(y_true, y_pred)

print(f"Précision Globale : {acc*100:.1f}%")
print("Matrice de Confusion :")
print(f"[{cm_val[0][0]}  {cm_val[0][1]}]")
print(f"[{cm_val[1][0]}  {cm_val[1][1]}]")

# ==========================================
# 5. GRAPHIQUE LOSS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Loss')
plt.title("Apprentissage BioGNN (PROTEINS)")
plt.savefig('loss_curve.png')
print("\n📈 'loss_curve.png' sauvegardé.")

# ==========================================
# 6. VISUALISATION SIMPLE
# ==========================================
print("🖼️ Génération du visuel simple...")
data_mol = test_loader.dataset[7] 
g = to_networkx(data_mol, to_undirected=True)
plt.figure(figsize=(8, 6))
pos = nx.kamada_kawai_layout(g) 
nx.draw(g, pos, with_labels=False, node_size=200, node_color='skyblue', edge_color='gray')
plt.title(f"Structure Protéine 7 (Classe: {data_mol.y.item()})")
plt.savefig('protein_structure.png')
print("✅ 'protein_structure.png' sauvegardé !")

# ==========================================
# 7. ANALYSE STATISTIQUE (TITRES CORRIGÉS)
# ==========================================
print("\n📊 Génération des Histogrammes Comparatifs (Stats)...")

features_enzyme = []
features_non_enzyme = []

for data in train_loader.dataset:
    avg_feat = data.x.mean(dim=0).numpy()
    if data.y.item() == 1:
        features_enzyme.append(avg_feat)
    else:
        features_non_enzyme.append(avg_feat)

features_enzyme = np.array(features_enzyme)
features_non_enzyme = np.array(features_non_enzyme)

plt.figure(figsize=(15, 5))

# Feature 1 : Hélice Alpha
plt.subplot(1, 3, 1)
plt.hist(features_enzyme[:, 0], bins=20, alpha=0.5, label='Enzymes', color='red')
plt.hist(features_non_enzyme[:, 0], bins=20, alpha=0.5, label='Non-Enzymes', color='blue')
plt.xlabel("Proportion (0 à 1)")
plt.ylabel("Nombre de Protéines")
plt.title("Feature 1 : Hélices Alpha")
plt.legend()

# Feature 2 : Feuillet Bêta
plt.subplot(1, 3, 2)
plt.hist(features_enzyme[:, 1], bins=20, alpha=0.5, label='Enzymes', color='red')
plt.hist(features_non_enzyme[:, 1], bins=20, alpha=0.5, label='Non-Enzymes', color='blue')
plt.xlabel("Proportion (0 à 1)")
plt.title("Feature 2 : Feuillets Bêta")

# Feature 3 : Tours / Autres
plt.subplot(1, 3, 3)
plt.hist(features_enzyme[:, 2], bins=20, alpha=0.5, label='Enzymes', color='red')
plt.hist(features_non_enzyme[:, 2], bins=20, alpha=0.5, label='Non-Enzymes', color='blue')
plt.xlabel("Proportion (0 à 1)")
plt.title("Feature 3 : Tours/Boucles (Sparse)")

plt.tight_layout()
plt.savefig('stats_comparison.png')
print("✅ Image 'stats_comparison.png' générée avec les bons titres biologiques !")
plt.show()

# ==========================================
# 8. HEATMAP D'ATTENTION AVANCEE (Nœuds Colorés)
# ==========================================
print("\n🔥 Génération de la 'Heatmap' Biologique Complète...")
import matplotlib.patches as mpatches # Pour la légende personnalisée

target_idx = 0
found = False
for i, data in enumerate(test_loader):
    if data.y.item() == 1: # Enzyme
        target_idx = i
        found = True
        break

if not found:
    print("Pas d'enzyme trouvée.")
else:
    data_mol = test_loader.dataset[target_idx].to(device)
    model.eval()

    try:
        # 1. Récupération des poids d'attention (Arêtes)
        edge_index_att, alpha = model(data_mol.x, data_mol.edge_index, data_mol.batch, return_attention=True)
        att_weights = alpha.mean(dim=1).cpu().detach().numpy()
        
        # 2. Récupération des Types de Nœuds (Structures)
        # data_mol.x est de forme [Nb_Noeuds, 3]. On prend l'indice du max (0, 1 ou 2)
        node_types = data_mol.x.argmax(dim=1).cpu().numpy()
        
        # Dictionnaire de couleurs pour les nœuds
        # 0: Hélice (Vert), 1: Feuillet (Bleu), 2: Autre (Jaune)
        color_map = {0: '#2ecc71', 1: '#3498db', 2: '#f1c40f'}
        node_colors = [color_map[t] for t in node_types]

        # 3. Préparation du Graphe
        g = to_networkx(data_mol, to_undirected=False)
        
        # --- DESSIN ---
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(g) 
        
        # Gestion des arêtes (Attention)
        w_min, w_max = att_weights.min(), att_weights.max()
        att_dict = {}
        rows, cols = edge_index_att.cpu().numpy()
        for k in range(len(rows)):
            att_dict[(rows[k], cols[k])] = att_weights[k]
        
        edge_colors = []
        edge_widths = []
        for u, v in g.edges():
            weight = att_dict.get((u, v), att_dict.get((v, u), 0))
            norm_w = (weight - w_min) / (w_max - w_min + 1e-9)
            edge_colors.append(norm_w)
            edge_widths.append(1 + norm_w * 6)
        
        # Dessin des Nœuds (Avec les couleurs biologiques !)
        nx.draw_networkx_nodes(g, pos, ax=ax, 
                               node_size=500, 
                               node_color=node_colors, 
                               edgecolors='black')
        
        # Dessin des Arêtes (Attention en Rouge)
        edges_draw = nx.draw_networkx_edges(g, pos, ax=ax,
                               width=edge_widths, 
                               edge_color=edge_colors, 
                               edge_cmap=plt.cm.Reds, 
                               edge_vmin=0, edge_vmax=1)
        
        # --- LÉGENDES ---
        # 1. Barre de couleur pour l'Attention (Arêtes)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Intensité de l'Attention (Site Actif)")
        
        # 2. Légende pour les Nœuds (Structures)
        legend_patches = [
            mpatches.Patch(color='#2ecc71', label='Hélice Alpha'),
            mpatches.Patch(color='#3498db', label='Feuillet Bêta'),
            mpatches.Patch(color='#f1c40f', label='Boucle/Tour')
        ]
        ax.legend(handles=legend_patches, loc='upper right', title="Structure 3D")
        
        ax.set_title(f"Visualisation Biologique Complète (Enzyme)\n(Structure + Fonction)")
        ax.axis('off')
        
        plt.savefig('heatmap_biologique.png', dpi=150)
        print("✅ Image 'heatmap_biologique.png' générée ! C'est magnifique.")
        plt.show()

    except TypeError:
        print("❌ ERREUR : Utilise BioGNN.")
    except Exception as e:
        print(f"❌ Erreur : {e}")