# EEG_Project Progress

# Team Members
1. Aung Zar Lin (st121956)
2. Lin Tun Naing (st122403)
3. Min Khant Soe (st122277)
4. Win Win Phyo (st122314)

## Final 

### Files/ Folders check list

00. Visualize ERP
01. Load_data_pool.ipynb
01. Load_data
02. BN3 subject specific
03. BN3 train on one subject and test on another subject
04. BN3 regional channel
05. BN3+LSTM subject specific
06. BN3+LSTM train on one subject and test on another subject
07. BN3+LSTM regional channel
08. CNN subject specific
09. CNN train on one subject and test on another subject
10. CNN regional channel
11. BN3 regional channel P Region Test
12. BN3+LSTM regional channel P Region Test
13. CNN regional channel P Region Test
14. BN3 subject pool
15. BN3+LSTM subject pool
16. CNN subject pool
17. LDA train on one subject and test on another subject
18. SVM train on one subject and test on another subject

-- 'data' Folder contains '_epo.fif' files

### Data Pipeline

We loaded all data from .csv files and turned into "_epo.fif" formats. Each coding process includes in "Load_data.ipynb" file. The following preprocessing steps are done.
    
    1. Load csv
    2. Convert to raw - set montage, eeg channels, targets and sampling frequency
    3. Filter - electrical band, -0.1 to 0.7 band pass
    4. Epoching - epoch "Target" and "Non-target"
    5. save - as "_epo.fif" format
    
"Load_data_pool.ipynb" loads all the "_epo.fif" files and convert each and every of them into numpy arrays and concatenate each others into X and y. The sample shape becomes (34236, 32, 410) for BN3 Conv1D or (34236, 1, 32, 410) for CNN Conv2D.
    
### Models

We used total of 3 models named BN3 (Batchnormalization Conv1D), BN3+LSTM and CNN Conv2D.

### Experiment

The experiment includes the following.

    1. Trained and test on one subject and test on another subject.
         - Classical machine learning models: SVM and LDA as baselines
         - BN3, BN3+LSTM, CNN Conv2D
    2. Subject specific
         - BN3, BN3+LSTM, CNN Conv2D
    3. Regional channel
         - BN3, BN3+LSTM, CNN Conv2D
    4. Subject Pool
         - BN3, BN3+LSTM, CNN Conv2D
         
## 15- 21 November

- Data are saved as (.fif) file in folder in order to save time when we want to run the data in model again.
- Build CNN model and recorded the accuracy in excels for both 1 subject and each channels of that subject.
- Compared accuracy result with ML models.


Expected to finish next week
- Improve CNN model as possible as to beat the accuracy of ML
- Build LSTM+CNN model

## 8- 14 November

modeling - 80 %
In this week, we do P300 analysis and continue working on classification models.

Expected to do next week
-inter-brain synchrony


## 1- 7 November

modeling - 50 %
In this week, we fit the data with 4 models (ML, LSTM, CNN with 1d and 2d) and compare the accuracy.

Expecting to work on the next week
- p300 Analysis

## 25 - 31 Ocotober

preprocessing and modeling - 25%

We have done in this week.
- ICA (difficult to differentiate bettween eye,muscle artifacts and signals)
- Epoching

Expecting to work on the next week
- modeling

## 18 - 24 October

modeling - 7%
loading the dataset - 100%
creating the github - 100%
reading paper - 90 % 

we read the following paper

 - MEG and EEG data analysis with MNE-Python 
 (https://www.frontiersin.org/articles/10.3389/fnins.2013.00267/full)
 
 - Universal Joint Feature Extraction for P300 EEG Classification Using Multi-Task Autoencoder
 (https://ieeexplore.ieee.org/abstract/document/8723080)
 
 - Lawhern, Vernon J. EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces
 (https://www.researchgate.net/publication/310953136_EEGNet_A_Compact_Convolutional_Network_for_EEG-based_Brain-Computer_Interfaces)
 
 - Asscement of preprocessing of classifiers on using P300 paradigm
 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4462003)
 
 - A P300 event-related potential brain–computer interface (BCI): The effects of matrix size and inter stimulus interval on performance
 (https://www.sciencedirect.com/science/article/pii/S0301051106001396)

We done this steps in data processing:
  - transform data in to raw mne object
  - notch filter 
  - band pass filter
      
## Expected to finish in next week

  - got error in notch filter coding and try to fix
  - ICA
  - epoching
  
## 11 - 17 October

task-based
modeling - 0%
loading the dataset - 100%
creating the github - 100%
reading paper - 40 % 

we read the following paper 
  
  - Experimental procedures for this dataset 
  (https://hal.archives-ouvertes.fr/hal-02173958/document?fbclid=IwAR2HuX-mjDmMsokUg2zYyPkHnI-WyfX0oIRWgKAffLgJ7yO0pbyM9mNY7Q8) 
  
  - A novel P300 BCI speller based on the Triple RSVP paradigm 
  (https://www.nature.com/articles/s41598-018-21717-y)
  
  - The Changing Face of P300 BCIs: A Comparison of Stimulus Changes in a P300 BCI Involving Faces, Emotion, and Movement
  (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049688)
  
  - Progress in neural networks for EEG signal recognition in 2021
  (https://arxiv.org/ftp/arxiv/papers/2103/2103.15755.pdf )

  - Trends in EEG-BCI for daily-life: Requirements for artifact removal
  (https://www.sciencedirect.com/science/article/pii/S1746809416301318#bib0610)

 - Analysis of P300 Related Target Choice in Oddball Paradigm
 (https://www.koreascience.or.kr/article/JAKO201120661418465.page)

 - A Novel P300 Classification Algorithm Based on a Principal Component Analysis-Convolutional Neural Network
 (https://www.mdpi.com/2076-3417/10/4/1546/htm)
 
 
 
 
 
 
 
 
 
 Beam_search chatbot.py
 
 
 
 import torch
from torch import nn
from train import trainIters
from torch import optim
from voc import SOS_token, EOS_token, MAX_LENGTH, Voc, loadPrepareData, trimRareWords, normalizeString
from seq2seq import EncoderRNN, LuongAttnDecoderRNN
from test import evaluateInput,evaluate

from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


CUDA = torch.cuda.is_available()
device = torch.device("cuda:1" if CUDA else "cpu")


# Load/Assemble Voc and pairs

datafile = 'chatDataset.txt'
voc, pairs = loadPrepareData(datafile)

print("\npairs:")
for pair in pairs[:10]:
    print(pair)



MIN_COUNT = 3    # Minimum word count threshold for trimming

# Trim vocabulary and pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)


testpairs = pairs[45000:]
pairs  = pairs[:45000]

class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) 
                continue
            idxes = self.sentence_idxes[:] 
            scores = self.sentence_scores[:] 
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
       
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

    def __repr__(self):
        res = f"Sentence with indices {self.sentence_idxes} "
        res += f"and scores {self.sentence_scores}"
        return res


def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)
           
        
        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []
        

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]


class BeamSearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, voc, beam_size=10):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc
        self.beam_size = beam_size

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        sentences = beam_decode(self.decoder, decoder_hidden, encoder_outputs, self.voc, self.beam_size, max_length)
        
        
        all_tokens = [torch.tensor(self.voc.word2index.get(w, 0)) for w in sentences[0][0]]
        return all_tokens, None

    def __str__(self):
        res = f"BeamSearchDecoder with beam size {self.beam_size}"
        return res

model_name = 'cb_model'
attn_model = 'dot'

hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 4
dropout = 0.5
batch_size = 256 

loadFilename = None
localFilename = 'content/cb_model/Chat/2-4_512/8000_checkpoint.tar'
checkpoint = torch.load(localFilename)
#print('Got checkpoint with keys', checkpoint.keys())

voc = Voc()
voc.restoreCheckpoint(checkpoint['voc_dict'])

embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(checkpoint['embedding'])

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
encoder = encoder.to(device)
encoder.load_state_dict(checkpoint['en'])
encoder.eval()

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
decoder = decoder.to(device)
decoder.load_state_dict(checkpoint['de'])
decoder.eval()

#Restore from checkpoint rather than training again?


corpus_name="Chat"


searcher = BeamSearchDecoder(encoder, decoder, voc, 10)

# evaluateInput(encoder, decoder, searcher, voc)



gram1_bleu_score = []
gram2_bleu_score = []

for i in range(0,len(testpairs),1):
    input_sentence = testpairs[i][0]
  
    reference = testpairs[i][1:]
    templist = []
    for k in range(len(reference)):
        if(reference[k]!=''):
            temp = reference[k].split(' ')
            templist.append(temp)
  
    input_sentence = normalizeString(input_sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    chencherry = SmoothingFunction()
    score1 = sentence_bleu(templist,output_words,weights=(1, 0, 0, 0) ,smoothing_function=chencherry.method1)
    score2 = sentence_bleu(templist,output_words,weights=(0.5, 0.5, 0, 0),smoothing_function=chencherry.method1) 
    gram1_bleu_score.append(score1)
    gram2_bleu_score.append(score2)
    if i%1000 == 0:
        print(i,sum(gram1_bleu_score)/len(gram1_bleu_score),sum(gram2_bleu_score)/len(gram2_bleu_score))
print("Total Bleu Score for 1 grams on testing pairs: ", sum(gram1_bleu_score)/len(gram1_bleu_score))  
print("Total Bleu Score for 2 grams on testing pairs: ", sum(gram2_bleu_score)/len(gram2_bleu_score))  


