import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, cuda, train_mode, metrics = []):

    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    if train_mode:
        train(train_loader, model, loss_fn, optimizer, n_epochs, cuda, metrics)
    else:
        test(val_loader, model, cuda, metric)


def train(train_loader, model, loss_fn, optimizer, num_epochs, cuda, metrics):

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            if not type(images) in (tuple,list):
                images = (images,)
            images = tuple(np.swapaxes(image, 1, -1 for image in images)
            images = tuple(image.float() for image in images)
             
            labels = labels.long() if len(label)>0 else None
            
            if cuda:
                images = tuple(image.cuda() for image in images)  # to(device)
                if labels in not None:
                    labels = labels.cuda()  # to(device)
            #print(labels)
            
            outputs = model(*images)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            
            if labels is not None:
                labels = (labels,)
                loss_inputs += labels	


            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
  
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for metric in metrics:
                metric(outputs, target, loss_outputs)

    
            if (i + 1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test(test_loader, model, cuda, metric):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if not type(images) in (tuple,list):
                images = (images,)
            images = tuple(np.swapaxes(image, 1, -1 for image in images)
            images = tuple(image.float() for image in images)
             
            labels = labels.long() if len(label)>0 else None
            
            if cuda:
                images = tuple(image.cuda() for image in images)  # to(device)
                if labels in not None:
                    labels = labels.cuda()  # to(device)
            #print(labels)
            
            outputs = model(*images)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            
            if labels is not None:
                labels = (labels,)
                loss_inputs += labels	


            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            
            for metric in metrics:
                metric(outputs, target, loss_outputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))

