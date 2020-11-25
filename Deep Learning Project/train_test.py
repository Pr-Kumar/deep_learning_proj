import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def train_model(model, criterion, optimizer, epochs, device, train_loader, val_loader, save_model):
    counter = 0
    print_every = 500
    valid_loss_min = np.Inf

    model.train()
    for i in range(epochs):
        train_losses = []
        val_losses = []
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output.squeeze(), labels)
            train_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            if counter%print_every == 0:
                curr_val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    inp, lab = inp.to(device), lab.to(device)
                    out = model(inp)
                    val_loss = criterion(out.squeeze(), lab)
                    curr_val_losses.append(val_loss.item())

                val_losses += curr_val_losses
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                      
                if np.mean(curr_val_losses) <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(curr_val_losses)))
                    valid_loss_min = np.mean(curr_val_losses)
                    torch.save(model.state_dict(), save_model)

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def test_model(model, criterion, device, test_loader, load_model):
    model.load_state_dict(torch.load(load_model))
    test_losses = []
    num_correct = 0

    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        test_loss = criterion(output.squeeze(), labels)
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
            
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))


def train_lstm(model, criterion, optimizer, epochs, device, train_loader, val_loader, save_model):
    counter = 0
    print_every = 500
    valid_loss_min = np.Inf
    batch_size = train_loader.batch_size

    train_losses = []
    val_losses = []
    model.train()
    for i in range(epochs):
        h = model.init_hidden(batch_size)
        
        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels)
            train_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            if counter%print_every == 0:
                val_h = model.init_hidden(batch_size)
                curr_val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab)
                    curr_val_losses.append(val_loss.item())
                    
                val_losses += curr_val_losses
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

                if np.mean(curr_val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), save_model)
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(curr_val_losses)))
                    valid_loss_min = np.mean(curr_val_losses)

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def test_lstm(model, criterion, device, test_loader, load_model):
    model.load_state_dict(torch.load(load_model))
    test_losses = []
    num_correct = 0
    batch_size = test_loader.batch_size
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels)
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
            
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))

def train_cnn(model, criterion, optimizer, epochs, device, train_loader, val_loader, save_model):
    counter = 0
    print_every = 500
    valid_loss_min = np.Inf

    train_losses = []
    val_losses = []
    model.train()
    for i in range(epochs):
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output.squeeze(1), labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if counter%print_every == 0:
                curr_val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    inp, lab = inp.to(device), lab.to(device)
                    out = model(inp)
                    val_loss = criterion(out.squeeze(), lab)
                    curr_val_losses.append(val_loss.item())

                val_losses += curr_val_losses 
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(curr_val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), save_model)
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(curr_val_losses)))
                    valid_loss_min = np.mean(curr_val_losses)
    
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def test_cnn(model, criterion, device, test_loader, load_model):
    model.load_state_dict(torch.load(load_model))
    test_losses = []
    num_correct = 0

    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
            
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))