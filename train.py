## For pretty print of tensors.
## Must be located at the first line except the comments.
from __future__ import print_function

## Import the basic modules.
import argparse
import numpy as np
import time
## Import the PyTorch modules.
import torch
import torch.nn as nn
import dataload
import Vocab_builder
import torch.nn.functional as functional
import torch.optim as optim
from   torchvision import transforms
from   torch.autograd import Variable
from   torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
## You are supposed to implement the following four source codes:
## {softmax.py, twolayernn.py, convnet.py, mymodel.py

import model 
#import EncoderCNN, DecoderRNN

#Ankit Initialization Start

alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
print_freq = 1 #test parameters
epochs = 120  #test parameters
grad_clip = 5.  # clip gradients at an absolute value of
global split
#Ankit Initialization Ends


## Initilize a command-line option parser.
parser = argparse.ArgumentParser(description='Flickr8k')

## Add a list of command-line options that users can specify.
## Shall scripts (.sh) files for specifying the proper options are provided.
parser.add_argument('--lr', type=float, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', help='number of epochs to train')
parser.add_argument('--model',
                    choices=['Pretrained'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int, help='number of hidden features/activations')
parser.add_argument('--embed-dim', type=int, help='embed layer dimensions')
parser.add_argument('--kernel-size', type=int, help='size of convolution kernels/filters')
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'adadelta'], help='which optimizer')

## Add more command-line options for other configurations.
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
#parser.add_argument('--flickr8k', default='data',
                   # help='directory that contains cifar-10-batches-py/ (downloaded automatically if necessary)')

## Parse the command-line option.
args = parser.parse_args()

## CUDA will be supported only when user wants and the machine has GPU devices.
args.cuda = not args.no_cuda and torch.cuda.is_available()

## Change the random seed.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## Set the device-specific arguments if CUDA is available.
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

im_size = (3, 224, 224)
n_classes = 10


from os import listdir
from os.path import join, isdir
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
train_root = "/content/drive/My Drive/IDS 576 Milestone/Processed Data/train"
test_root="/content/drive/My Drive/IDS 576 Milestone/Processed Data/test"
val_root="/content/drive/My Drive/IDS 576 Milestone/Processed Data/dev"


train_data_url = [d for d in listdir(train_root+"/images")]
val_data_url=[d for d in listdir(val_root+"/images")]
test_data_url = [ d for d in listdir(test_root+"/images")]

x=open("/content/drive/My Drive/IDS 576 Milestone/Processed Data/train/captions.txt","r")
train_captions_text=x.readlines()
x=open("/content/drive/My Drive/IDS 576 Milestone/Processed Data/dev/captions.txt","r")
val_captions_text=x.readlines()
x=open("/content/drive/My Drive/IDS 576 Milestone/Processed Data/test/captions.txt","r")
test_captions_text=x.readlines()

## Normalize each image by subtracdting the mean color and divde by standard deviation.
## For convenience, per channel mean color and standard deviation are provided.

flickr_mean_color = [0.44005906636983594, 0.4179391194853607, 0.3848489007106294]
flickr_std_color = [0.28628332805186396, 0.2804168453671926, 0.29043924853401465]
transform = transforms.Compose([
    #transforms.Scale(224,224)
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(flickr_mean_color, flickr_std_color),
])


global vocab
print('Building Vocabulary')
caption_dict = dataload.loadcaptions(train_captions_text)
vocab = Vocab_builder.Vocab_builder(caption_dict = caption_dict, threshold = 0)   
global vocab_size
vocab_size = vocab.index

## Load training, validation, and test data separately.
## Apply the normalizing transform uniformly across three dataset.
train_dataset = dataload.DATALOAD(train_root,train_data_url[:3000],train_captions_text,vocab, split='train', download=False, transform=transform)
val_dataset = dataload.DATALOAD(val_root,val_data_url[:500],val_captions_text,vocab, split='val', download=False, transform=transform)
test_dataset = dataload.DATALOAD(test_root,test_data_url[:500],test_captions_text,vocab, split='test', download=False, transform=transform)

#Start of Ankit
#attention_dim, decoder_dim, encoder_dim

attention_dim = 512  #Dimension of attention linear layers
decoder_dim   = 512  #Dimension of decoder RNN
encoder_dim   = 2048 #Dimension of encoder RNN
emb_dim       = 512  # dimension of word embeddings
#End of Ankit

#Ankit below line
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

## DataLoaders provide various ways to get batches of examples.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.val_collate, **kwargs)
## Load the proper neural network model.
if args.model == 'Pretrained':
    # Problem 2 (no hidden layer, input -> output)
    model.encoder = model.EncoderCNN(10)
    model.decoder = model.DecoderRNN(encoder_dim = 2048, decoder_dim=512, attention_dim=512, embed_size = 512, hidden_size = args.hidden_dim, vocab_size = vocab_size,num_layers=1,max_seq_length=15)
#elif args.model == 'resnet_common':
    # Problem 5 (multiple hidden layers, input -> hidden layers -> output)
 #   print("sruthi check 1")
  #  model = models.resnetcommon.ResnetCommon(im_size, args.hidden_dim, args.kernel_size, n_classes)

else:
    raise Exception('Unknown model {}'.format(args.model))

## Deinfe the loss function as cross-entropy.
## This is the softmax loss function (i.e., multiclass classification).
criterion = functional.cross_entropy

## Activate CUDA if specified and available.
if args.cuda:    
    model.cuda()
    
params = list(model.encoder.linear.parameters()) + list(model.decoder.parameters())

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    #optimizer = torch.optim.Adam(params, args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, args.lr, betas=(0.9,args.momentum), weight_decay=args.weight_decay)
elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), args.lr, rho=0.9, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,weight_decay = args.weight-decay)
pass


#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
#ANKIT GRADIENT CLIP CODE BEGINS

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


#ANKIT GRADIENT CLIP CODE ENDS


#ANKIT AVERAGE METER CODE BEGINS
"""
class AverageMeter(object):
    """"""
    Keeps track of most recent, average, sum, and count of a metric.
    """"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
"""
"""
def adjust_learning_rate(optimizer, shrink_factor):
    """"""
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """"""

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

"""
"""
def accuracy(scores, targets, k):
    """"""
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """"""

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
"""
#ANKIT AVERAGE METER CODE ENDS

## Function: train the model just one iteration.
def train(epoch):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in train mode as opposed to eval mode, so it knows which one to use.
    print("sruthi check 2")
    
    model.encoder.train()
    print(" check lalala")
    model.decoder.train()
    print("sruthi check 6")
    # print(model.fc)
    #Ankit begins
    """
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    
    """
    #Ankit ends

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                             lr=decoder_lr)
        #encoder = Encoder()
        #encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                             lr=encoder_lr)
    
    # For each batch of training images,
    cum_train_loss=0
    cum_val_loss=0

    # For each batch of training images,
    for batch_idx, batch in enumerate(train_loader):
        
        # Read images and their target labels in the current batch.
        #print(batch[0].size())
        #print(batch[1].size())
        
        images,captions,lengths = Variable(batch[0]),batch[1],batch[2]
        #print(lengths)
        captions_new=captions
        #captions_new=dataload.tokenize(vocab,captions)
       # lengths = [len(cap) for cap in captions_new]

        #print("TARGET INPUTS")
        #print(type(lengths), lengths)
        #print(captions)
        captions_new = captions_new[:, 1:]
        #print(type(captions_new), captions_new.size())
        #print("captions new ",captions_new)
        targets = pack_padded_sequence(captions_new, lengths, batch_first=True)[0]
        #print("TARGETSSSSSSSSSSSSSS")
        #print(targets.size(),targets)
        targets = targets[targets.nonzero()]
        targets = targets.squeeze(1)
        #print(targets.size(), targets)
        #Find the index of 0 in tensor
        # Load the current training example in the CUDA core if available.
        if args.cuda:
            images= images.cuda()

        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.      #
        # This only requires a couple lines of code.                                #
        #############################################################################
       # optimizer.zero_grad()
        # images=torch.nn.Upsample(scale_factor=2,mode='nearest')
        #torch.distributed.init_process_group(backend="cpu")#Ankit

        features = model.encoder(images)
       
        output, caps_sorted, decode_lengths, alphas  = model.decoder(features,captions,lengths)
        #print("PACK INPUTS")
        #output = output.squeeze(0)
       # print(output.size())
        #print(type(decode_lengths), decode_lengths)
        
        output = pack_padded_sequence(output, decode_lengths, batch_first=True)[0]
        #targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        #print("output", output.size())
        #print(output.size())
        #print(targets.size())

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, targets) 
  
        #Ankit start

         # Add doubly stochastic attention regularization
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        
        # Update weights
        #decoder_optimizer.step()
        #if encoder_optimizer is not None:
        #    encoder_optimizer.step()
        #loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Keep track of metrics
        #top5 = accuracy(output, targets, 5)
        #losses.update(loss.item(), sum(decode_lengths))
        #top5accs.update(top5, sum(decode_lengths))
        #batch_time.update(time.time() - start)

        #start = time.time()
    
        # Print status
        #if batch_idx % print_freq == 0:
        #    print('Epoch: [{0}][{1}/{2}]\t'
        #          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch_idx, len(train_loader),
                                                                         # batch_time=batch_time,
                                                                         # data_time=data_time, loss=losses,
                                                                         # top5=top5accs))

        #print("ok till now")
        #Ankit end

        #OLD CODE BEGINS - Commented

        model.decoder.zero_grad()
        model.encoder.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        cum_train_loss +=loss
        
        # Print out the loss and accuracy on the first 4 batches of the validation set.
        # You can adjust the printing frequency by changing --log-interval option in the command-line.
        if batch_idx % args.log_interval == 0:
            # Compute the average validation loss and accuracy.
            val_loss=evaluate('val', n_batches=10)
            #print("check 11")
            # Compute the training loss.
            train_loss = loss.data.item()
            #print(train_loss)
            # Compute the number of examples in this batch.
            examples_this_epoch = batch_idx * len(images)

            # Compute the progress rate in terms of the batch.
            epoch_progress = 100. * batch_idx / len(train_loader)
            #print(epoch_progress)
            # Print out the training loss, validation loss, and accuracy with epoch information.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss:{:.6f}\t'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss))
        cum_val_loss+=val_loss
    avg_val_loss=cum_val_loss/(batch_idx+1)
    avg_train_loss=cum_train_loss/(batch_idx+1)
    print('Train Epoch: {}\t'
           'Avg Train Loss: {:.6f}\t Val Loss:{:.6f}\t'.format(epoch,avg_train_loss,avg_val_loss))
  
        # Print out the loss and accuracy on the first 4 batches of the validation set.
        # You can adjust the printing frequency by changing --log-interval option in the command-line.
        #if batch_idx % args.log_interval == 0:
            # Compute the average validation loss and accuracy.
         #   val_loss, val_acc = evaluate('val', n_batches=4)

            # Compute the training loss.
          #  train_loss = loss.data.item()

            # Compute the number of examples in this batch.
           # examples_this_epoch = batch_idx * len(images)

            # Compute the progress rate in terms of the batch.
            #epoch_progress = 100. * batch_idx / len(train_loader)

            # Print out the training loss, validation loss, and accuracy with epoch information.
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
            #     'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
            #      epoch, examples_this_epoch, len(train_loader.dataset),
            #    epoch_progress, train_loss, val_loss, val_acc))

        #OLD CODE ENDS - Commented
## Function: evaluate the learned model on either validation or test data.
def evaluate(split, verbose=False, n_batches=None):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in eval mode as opposed to train mode, so it knows which one to use.
    model.encoder.eval()
    model.decoder.eval()
    # Initialize cumulative loss and the number of correctly predicted examples.
    loss = 0
    correct = 0
    n_examples = 0
       
    # Load the correct dataset between validation and test data based on the split option.
    if split == 'val':
        loader = val_loader

    elif split == 'test':
        loader = test_loader
    
    if (split == 'val'):
      # For each batch in the loaded dataset,
      with torch.no_grad():
          for batch_i, batch in enumerate(loader):

              data,caption,lengths = Variable(batch[0]),batch[1],batch[2]

              #data,captions = Variable(batch[0]),batch[1]
       
              #caption=captions
              #captions_new=dataload.tokenize(vocab,captions)
              #lengths = [len(cap) for cap in caption]

              caption = caption[:, 1:]

              targets = pack_padded_sequence(caption, lengths, batch_first=True)[0]
              targets = targets[targets.nonzero()]
              targets = targets.squeeze(1)          
              
              #data,caption,lengths = batch[0],batch[1],batch[2]
              #lengths = [len(cap) for cap in caption]
              
              #lengths = [len(cap) for cap in caption]

              #targets = pack_padded_sequence(caption, lengths, batch_first=True)[0]
              #print(caption)
              # Load the current training example in the CUDA core if available.
              
              if args.cuda:
                  data,caption = data.cuda(), caption.cuda()

              # Read images and their target labels in the current batch.
              #data,caption = Variable(data),caption
              # Measure the output results given the data.
              features = model.encoder(data)
              #output=model.decoder(features,caption,lengths)
              output, caps_sorted, decode_lengths, alphas = model.decoder(features,caption,lengths)
              # Accumulate the loss by comparing the predicted output and the true target labels.
              output = pack_padded_sequence(output, decode_lengths, batch_first=True)[0]
              #output = output.squeeze(0)
            
              loss += criterion(output, targets).data
            
              # Skip the rest of evaluation if the number of batches exceed the n_batches.
              if n_batches and (batch_i >= n_batches):
                  break

      # Compute the average loss per example.
      loss /= (batch_i+1)

    
      # If verbose is True, then print out the average loss and accuracy.
      if verbose:
          print('\n{} set: Average loss: {:.4f}'.format(
              split, loss))
      return loss

    if (split == 'test'):
      
      # For each batch in the loaded dataset,
      with torch.no_grad():
          for batch_i, batch in enumerate(loader):
          
            # print(batch.size)
              #data,caption = batch[0],batch[1] #Ankit commented
              #lengths = [len(cap) for cap in caption] #Ankit commented

              #Ankit Start
              
              data,caption,lengths = batch[0],batch[1],batch[2]
                   
              #data,caption = Variable(batch[0]),batch[1]
              caption_or = caption[:]
              #lengths = [len(cap) for cap in caption]
              #caption = caption[:, 1:]
              #Ankit ends
  
              #targets = pack_padded_sequence(caption, lengths, batch_first=True)[0]
          
              # Load the current training example in the CUDA core if available.
              if args.cuda:
                  data,caption = data.cuda(), caption.cuda()

              # Read images and their target labels in the current batch.
              #data,caption = Variable(data),caption
      
              # Measure the output results given the data.
              features = model.encoder(data)

              #Start of Ankit
              caption = 0
              lengths = 0
              
              output=model.decoder(features,caption,lengths)
              output = output.squeeze(1)
              #print(output.size())
              ground_truth = []
              predicted = []

              for i in range(len(output)):
                sampled_seq =vocab.get_sentence(output[i])
                predicted.append(sampled_seq)

              print(predicted)

              for i in range(len(caption_or)):

                targets = [c[0:-1] for c in caption_or[i]]

                ground_truth.append(targets)

              print(ground_truth)

              print("bleu 4",corpus_bleu(ground_truth, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
              print("bleu 1",corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))
            #End of Ankit


            # Accumulate the loss by comparing the predicted output and the true target labels.
            #loss += criterion(output, targets).data

            # Predict the class by finding the argmax of the log-probabilities among all classes.
            #pred = output.data.max(1, keepdim=True)[1]

            # Add the number of correct classifications in each class.
            #correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

            # Keep track of the total number of predictions.
            #n_examples += pred.size(0)

            # Skip the rest of evaluation if the number of batches exceed the n_batches.
            #if n_batches and (batch_i >= n_batches):
            #    break

    # Compute the average loss per example.
    #loss /= n_examples

    # Compute the average accuracy in terms of percentile.
    #acc = 100.* correct / n_examples

    # If verbose is True, then print out the average loss and accuracy.
    #if verbose:
    #    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #        split, loss, correct, n_examples, acc))
    #return loss, acc


## Train the model one epoch at a time.
for epoch in range(1, args.epochs + 1):
    print("check 4")
    train(epoch)
    save_every=1
    if epoch % save_every == 0:
            torch.save(model.encoder.state_dict(), 'new_%dencoder.pt'%(epoch))
            torch.save(model.decoder.state_dict(), 'new_%ddecoder.pt'%(epoch))

#print("check 5")
## Evaluate the model on the test data and print out the average loss and accuracy.
## Note that you should use every batch for evaluating on test data rather than just the first four batches.
evaluate('test', verbose=True)

## Save the model (architecture and weights)
#torch.save(model, args.model + '.pt')
print("COMPLETED")

"""
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details
"""