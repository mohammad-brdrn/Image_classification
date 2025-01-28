from tqdm import tqdm
import torch

def register_checkpoints(epoch:int, train_losses: list[float],val_losses: list[float], model:torch.nn.Module, optimizer:torch.optim.Optimizer, model_name:str):
    check_point = {'epoch': epoch,
                   'train_losses': train_losses,
                   'val_losses': val_losses,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()
                   }
    torch.save(check_point, f'checkpoints/check_point_{model_name}.pth')

def train(epochs:int, model, train_loader, val_loader, optimizer, criterion, device, model_name, scheduler=None):
    """ This function trains the model by getting the needed items. """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data , labels in tqdm(train_loader, desc=f'Training in epoch {epoch}'):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for data, labels in tqdm(val_loader, desc=f'Validation in epoch {epoch}'):
                data = data.to(device)
                labels = labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
        scheduler.step()

        print(f'epoch{epoch}, train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/best_{model_name}.pth')
    register_checkpoints(epochs, train_losses, val_losses, model, optimizer, model_name)
