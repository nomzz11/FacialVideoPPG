import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random


def plot_attention_maps(output_dir, attention_maps_dict):
    """
    Sauvegarde les cartes d'attention des layers 1 et 2 pour chaque vidéo disponible dans le batch.
    Pour chaque vidéo, affiche la première et la dernière frame.
    """
    # Vérification que les clés attendues sont présentes
    if "layer1" not in attention_maps_dict or "layer2" not in attention_maps_dict:
        print(
            "Erreur: les clés 'layer1' et 'layer2' sont requises dans attention_maps_dict"
        )
        return

    attn_layer1 = attention_maps_dict["layer1"].cpu().numpy()  # [B,1,T,H,W]
    attn_layer2 = attention_maps_dict["layer2"].cpu().numpy()

    # Vérification des dimensions
    if len(attn_layer1.shape) != 5 or len(attn_layer2.shape) != 5:
        print(
            f"Erreur: Les tenseurs d'attention doivent être 5D. Formes actuelles: layer1 {attn_layer1.shape}, layer2 {attn_layer2.shape}"
        )
        return

    total_videos = attn_layer1.shape[0]
    print(f"Traitement de {total_videos} vidéo(s)")

    # Traiter chaque vidéo disponible dans le batch
    for idx in range(total_videos):
        try:
            num_frames = attn_layer1.shape[2]
            print(f"Vidéo {idx}: {num_frames} frames")

            # Si une seule frame, on l'affiche
            # Sinon on prend la première et la dernière
            if num_frames < 2:
                selected_frames = [0]
            else:
                selected_frames = [0, num_frames - 1]

            fig, axes = plt.subplots(
                2, len(selected_frames), figsize=(4 * len(selected_frames), 8)
            )
            if len(selected_frames) == 1:
                axes = axes.reshape(2, 1)  # Pour uniformiser l'accès aux axes

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
            import traceback

            traceback.print_exc()  # Affiche la stacktrace complète pour aider au debug
            continue  # Continue avec la prochaine vidéo
