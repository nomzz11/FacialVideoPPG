from torch.utils.data import DataLoader


def dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Batch size utilisé : {64}")
    return train_dataloader, val_dataloader, test_dataloader
