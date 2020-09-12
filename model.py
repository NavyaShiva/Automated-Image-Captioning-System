import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)

        #features = self.linear(features)
        #features = self.bn(features)
        return features

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, decoder_dim, encoder_dim, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        
        #Start of Ankit
        #        decoder_dim = 512
        
        self.vocab_size = vocab_size
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.sigmoid = nn.Sigmoid()
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.dropout = 0.5
        self.dropout = nn.Dropout(p=self.dropout)
        #End of Ankit

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.lstm_test = nn.LSTM(2560, 512, batch_first=True)
        #self.lstm_test = nn.LSTM(embed_size + encoder_dim, decoder_dim, batch_first=True)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, vocab_size)
        self.max_seg_length = max_seq_length
    
    #Ankit init_hidden_state START
    def init_hidden_state(self, features):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #print("features size")
        #print(features.size())
        mean_encoder_out = features.mean(dim=1)
        #print(mean_encoder_out.size())
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    #Ankit init_hidden_state END
          
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        if (lengths != 0):  
          
          #Ankit Additions
          batch_size  = features.size(0)
          encoder_dim = features.size(-1)
          vocab_size  = self.vocab_size
          #import torch
          #lengths = torch.tensor(vocab_size) 
          # Flatten image
          #print(features.size())
          #print("LENGTHSSSSSSSSSSSSS")
          #print(lengths)
          features   = features.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
          num_pixels = features.size(1)
          #print(features.size())
          # Sort input data by decreasing lengths; why? apparent below
          #caption_lengths, sort_ind = lengths.squeeze(1).sort(dim=0, descending=True)
          #features = features[sort_ind]
          #captions = captions[sort_ind]

          #Ankit additions end
          embeddings = self.embed(captions)
          #print("embedding")
          #print(embeddings.size())
          #Begin of Ankit - Part 2
          # Initialize LSTM state
          hiddens, states = self.init_hidden_state(features)  # (batch_size, decoder_dim)
          #h = (torch.zeros(2,1,512), torch.zeros(2,1,512))

          # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
          # So, decoding lengths are actual lengths - 1
          
          decode_lengths = (lengths[0] - 1)
          decode_lengths = [l-1 for l in lengths]
          #print(decode_lengths)

          # Create tensors to hold word predicion scores and alphas
          predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
          alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)
          #print("ok")
          #print("FEATURESSSSSSSSSSS")
          #print(features)
          #print(features[:1])

          for t in range(max(decode_lengths)):
            #print("DECODEEE LENGTHSSSSSSSS")
            #print(decode_lengths)
            batch_size_t = sum([l > t for l in decode_lengths])
            #print("BATCH SIZE TTTTTT")
            #print(batch_size_t)
            attention_weighted_encoding, alpha = self.attention(features[:batch_size_t],
                                                                  hiddens[:batch_size_t])          
            #print("attention", attention_weighted_encoding)
            gate = self.sigmoid(self.f_beta(hiddens[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            #print(features.size())
            #embeddings = torch.cat((features.unsqueeze(1), attention_weighted_encoding), dim=1)
            #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            #hiddens, _ = self.lstm(packed)
            #outputs = self.linear(hiddens[0])
            #predictions[:batch_size_t, t, :] =outputs
            #print("reached")
            #print((embeddings[:batch_size_t, t, :]).size())
            #print(attention_weighted_encoding.size())
            #print((hiddens[:batch_size_t]).size())
            #print((states[:batch_size_t]).size())
      
            hiddens, states = self.lstm(
                  torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                  (hiddens[:batch_size_t], states[:batch_size_t]))  # (batch_size_t, decoder_dim)
            #print("completed")
            #print(h.size())
            preds = self.fc(self.dropout(hiddens))  # (batch_size_t, vocab_size)
            #print("TRAIN PHASE")
            #print(preds.size())
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
          
          #print("PREDICTIONSSSSSSS")
          #x=pack_padded_sequence(predictions,lengths,batch_first=True)[0]

          #print(predictions.size())
          #print(x.size())
          return predictions, captions, decode_lengths, alphas
          #End of Ankit - Part 2


          #embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
          #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
          #hiddens, _ = self.lstm(packed)
          #outputs = self.linear(hiddens[0])

          #return outputs

        if (lengths == 0):

          # Tensor to store top k previous words at each step; now they're just <start>
          #k_prev_words = torch.LongTensor([[word_map['<start>']]] * k)  # (k, 1)

          #print(k_prev_words)
          #Beam size
          #k = 10
          #print("TEST BEGINS")
          encoder_out = features
          print("Encoder_out", encoder_out.size())
          batch_size  = features.size(0)
          encoder_dim = features.size(-1)
          vocab_size  = self.vocab_size

          encoder_out = encoder_out.unsqueeze(1)
          #encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
          num_pixels = encoder_out.size(1)
          print("New Encoder Size", encoder_out.size())
          hiddens, states = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

          batch_size_t = 1
          sampled_ids = []
          print("Hidden", hiddens.size())
          # Create tensors to hold word predicion scores and alphas
          #predictions = torch.zeros(batch_size, self.max_seg_length)
         # predictions = torch.zeros(batch_size, self.max_seg_length, vocab_size)
          for t in range(self.max_seg_length):
            
            batch_size_t = (encoder_out.size())[1]
            
            attention_weighted_encoding, alpha = self.attention(encoder_out, hiddens)          
            
            gate = self.sigmoid(self.f_beta(hiddens))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding 

            hiddens, states = self.lstm(
                    torch.cat([hiddens, attention_weighted_encoding], dim=1),
                    (hiddens, states))  # (batch_size_t, decoder_dim)
            
            preds = self.fc(hiddens.squeeze(1))
            #preds = self.fc(self.dropout(hiddens))  # (batch_size_t, vocab_size)
            #print("TEST PHASE")
            #print(preds.size())
            _,preds = preds.max(1)
           # preds = torch.tensor(preds).to(torch.int64)
           # print(predictions.size())
            #print(preds.size())
            #predictions[:batch_size_t, t] = preds
            #print(predictions.size())
            #alphas[:batch_size_t, t, :] = alpha

            #preds = self.fc(self.dropout(hiddens))  # (batch_size_t, vocab_size)
            #print("preds1", len(preds))
            #preds = preds.max(1)
            #print("preds2", len(preds))
            #preds = torch.tensor(preds).to(torch.int64)
            #print("preds3", len(preds))
            #alphas[:batch_size_t, t, :] = alpha

            sampled_ids.append(preds)

            #inputs = self.embed(preds)
            #inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
          #print(sampled_ids)
          #sampled_ids=pack_padded_sequence(sampled_ids,batch_first=True)
   
          sampled_ids = torch.stack(sampled_ids, -1)                # sampled_ids: (batch_size, max_seq_length)
          #print("samp",sampled_ids)
          #sampled_ids=sampled_ids.transpose(0,1)
          #print(sampled_ids)
          return sampled_ids
          