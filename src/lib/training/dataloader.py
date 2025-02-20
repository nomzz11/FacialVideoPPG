from torch.utils.data import DataLoader


def dataloader(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader
