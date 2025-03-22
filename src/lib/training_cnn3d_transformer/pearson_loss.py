import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        loss = 0

        for i in range(batch_size):
            # Extraire les vecteurs pour chaque exemple
            p = pred[i]
            t = target[i]

            # Vérifier s'il y a de la variance dans les deux vecteurs
            if torch.var(p) > 1e-5 and torch.var(t) > 1e-5:
                # Empiler les vecteurs pour torch.corrcoef
                stacked = torch.stack((p, t))
                corr_matrix = torch.corrcoef(stacked)
                # Vérifier si le résultat contient des NaN
                if not torch.isnan(corr_matrix[0, 1]):
                    loss += 1 - corr_matrix[0, 1]
                else:
                    print("Présencede NaN")
                    # Fallback: utiliser 0 de corrélation (perte de 1)
                    loss += 1.0
            else:
                # Aucune variance = aucune corrélation possible
                print("Aucune variance")
                loss += 1.0

        return loss / batch_size
