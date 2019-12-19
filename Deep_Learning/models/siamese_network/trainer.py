import torch
import numpy as np
from metrics import Metric,  AccumulatedAccuracyMetric

def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, cuda, train_mode, metrics = [AccumulatedAccuracyMetric()]):

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
        test(val_loader, model, loss_fn, cuda, metrics)


def train(train_loader, model, loss_fn, optimizer, num_epochs, cuda, metrics):

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            if not type(images) in (tuple,list):
                images = (images,)
            images = tuple(np.swapaxes(image, 1, -1) for image in images)
            images = tuple(image.float() for image in images)
             
            labels = labels.long() if len(labels)>0 else None
            #print(labels.type())   
            if cuda:
                images = tuple(image.cuda() for image in images)  # to(device)
                if labels is not None:
                    labels = labels.cuda()  # to(device)
            print('labels', labels)
            
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
                metric(outputs, labels, loss_outputs)

    
            if (i + 1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test(test_loader, model, loss_fn, cuda, metrics):
    print('testing')
    '''
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if not type(images) in (tuple,list):
                images = (images,)
            images = tuple(np.swapaxes(image, 1, -1) for image in images)
            images = tuple(image.float() for image in images)
             
            labels = labels.long() if len(labels)>0 else None
            
            if cuda:
                images = tuple(image.cuda() for image in images)  # to(device)
                if labels is not None:
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
     '''
     
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(np.swapaxes(image, 1, -1) for image in data)
            data = tuple(image.float() for image in data)
            
            target = target.long()

            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                print(metric(outputs, target, loss_outputs))



    return val_loss, metrics

          
