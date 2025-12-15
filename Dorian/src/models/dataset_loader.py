import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_bio_data(batch_size=64, test_split=0.2):
    print("--- Chargement du dataset 'PROTEINS' (Enzymes) ---")
    
    # C'est ICI que ça change : on appelle PROTEINS
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    
    print(f"✅ Données chargées : {len(dataset)} protéines")
    print(f"   - Nœuds moyens : {dataset.data.num_nodes / len(dataset):.1f}")
    print(f"   - Features : {dataset.num_features}")
    print(f"   - Classes : {dataset.num_classes} (0=Non-Enzyme, 1=Enzyme)")

    # Mélange indispensable
    dataset = dataset.shuffle()
    
    # Split 80% Train / 20% Test
    train_size = int(len(dataset) * (1 - test_split))
    test_size = len(dataset) - train_size
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"📊 Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    # Batch size plus gros (64) car on a plus de données
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, dataset.num_features, dataset.num_classes

if __name__ == "__main__":
    load_bio_data()