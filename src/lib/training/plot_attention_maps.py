import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random


def plot_attention_maps(output_dir, attention_maps_dict):
    """
    Sauvegarde les cartes d'attention des layers 1 et 2
    pour la première vidéo, la dernière vidéo, et 8 vidéos aléatoires dans des fichiers PNG.

    attention_maps_dict : dict avec clés:
        - "layer1_attention": tensor [B,1,T,H,W]
        - "layer2_attention": tensor [B,1,T,H,W]
    """

    attn_layer1 = attention_maps_dict["layer1"].cpu().numpy()  # [B,1,T,H,W]
    attn_layer2 = attention_maps_dict["layer2"].cpu().numpy()

    total_videos = attn_layer1.shape[0]

    # Sélectionner la première, dernière et 8 aléatoires distinctes
    random_indices = random.sample(range(1, total_videos - 1), min(8, total_videos - 2))
    selected_indices = [0, total_videos - 1] + random_indices

    for idx in selected_indices:
        try:
            num_frames = attn_layer1.shape[2]

            # On ne prend que la première et dernière frame
            selected_frames = [0, num_frames - 1]

            fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2 lignes, 2 colonnes
            plt.suptitle(f"Attention Maps - Video {idx}", fontsize=16)

            for i, t in enumerate(selected_frames):
                # Attention du Layer 1
                axes[0, i].imshow(attn_layer1[idx, 0, t], cmap="jet")
                axes[0, i].set_title(f"Layer1 - Frame {t}")
                axes[0, i].axis("off")

                # Attention map du Layer 2
                axes[1, i].imshow(attn_layer2[idx, 0, t], cmap="jet")
                axes[1, i].set_title(f"Layer2 - Frame {t}")
                axes[1, i].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Sauvegarde le plot en PNG clairement nommé
            plt.savefig(os.path.join(output_dir, f"attention_video_{idx}.png"))
            plt.close(fig)

        except Exception as e:
            print(f"Erreur lors du traitement de la vidéo {idx}: {e}")
            continue  # Continue avec la prochaine vidéo
