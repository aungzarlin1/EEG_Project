# EEG_Project Progress

# Team Members
1. Aung Zar Lin (st121956)
2. Lin Tun Naing (st122403)
3. Min Khant Soe (st122277)
4. Win Win Phyo (st122314)

## 15- 21 November
- Data are saved as numpy file in folder in order to save time when we want to run the data in model again.
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

 - MEG and EEG data analysis with MNE-Python (https://www.frontiersin.org/articles/10.3389/fnins.2013.00267/full)
 - Universal Joint Feature Extraction for P300 EEG Classification Using Multi-Task Autoencoder (https://ieeexplore.ieee.org/abstract/document/8723080)
 - Lawhern, Vernon J. EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces(https://www.researchgate.net/publication/310953136_EEGNet_A_Compact_Convolutional_Network_for_EEG-based_Brain-Computer_Interfaces)
 - Asscement of preprocessing of classifiers on using P300 paradigm(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4462003)
 - A P300 event-related potential brain–computer interface (BCI): The effects of matrix size and inter stimulus interval on performance(https://www.sciencedirect.com/science/article/pii/S0301051106001396)

We done this steps in data processing:
  - transform data in to raw mne object
  - notch filter 
  - band pass filter
      
# Expected to finish in next week
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
  - Experimental procedures for this dataset (https://hal.archives-ouvertes.fr/hal-02173958/document?fbclid=IwAR2HuX-mjDmMsokUg2zYyPkHnI-WyfX0oIRWgKAffLgJ7yO0pbyM9mNY7Q8) 
  - A novel P300 BCI speller based on the Triple RSVP paradigm (https://www.nature.com/articles/s41598-018-21717-y)
  - The Changing Face of P300 BCIs: A Comparison of Stimulus Changes in a P300 BCI Involving Faces, Emotion, and Movement (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049688)
  - Progress in neural networks for EEG signal recognition in 2021
https://arxiv.org/ftp/arxiv/papers/2103/2103.15755.pdf 

  - Trends in EEG-BCI for daily-life: Requirements for artifact removal
https://www.sciencedirect.com/science/article/pii/S1746809416301318#bib0610

 - Analysis of P300 Related Target Choice in Oddball Paradigm
https://www.koreascience.or.kr/article/JAKO201120661418465.page



