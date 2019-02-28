# socceRNN

## Description

1. Used soccer players and ball position data to train a LSTM to predict future states of play from previous sequences. Inspired by CharRNN by A.Karpathy.
2. Special attention is given to player-ball interactions by training a fully connected layer inspired by Neural Physics Engine.
3. Modified the above network to predict future video frames by taking RESNET features of the previous frames as input.

Original Game Sequence          |  Next Game State Prediction    |  Generated Sequence from Single Frame
:------------------------------:|:------------------------------:|:--------------------------------------:
![](https://github.com/vishwasuppoor/socceRNN/blob/master/viz/actual.gif)  |  ![](https://github.com/vishwasuppoor/socceRNN/blob/master/viz/snet_nf.gif)  |  ![](https://github.com/vishwasuppoor/socceRNN/blob/master/viz/snet_gen.gif)
