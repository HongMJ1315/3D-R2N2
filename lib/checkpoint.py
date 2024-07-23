import os
import torch

'''
data format:
{
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch_losses': epoch_losses, # list of epoch losses
    'last_file': last_file, # last file name
    'train_loss': train_loss, # list of sub train losses
    'val_loss': val_loss, # list of sub val losses
}

'''

def save_checkpoint(state, filename):
    torch.save(state, filename)



def load_checkpoint(filename, model, optimizer, device):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        try:
            checkpoint = torch.load(filename, map_location=device)
        except Exception as e:
            print(f"Failed to load checkpoint '{filename}': {e}")
            return 0, [], None, [], []

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        train_log = checkpoint.get('epoch_losses', [])
        last_file = checkpoint.get('last_file', None)
        train_loss = checkpoint.get('train_loss', [])
        val_loss = checkpoint.get('val_loss', [])
        print(train_loss, val_loss)
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch, train_log, last_file, train_loss, val_loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, [], None, [], []
