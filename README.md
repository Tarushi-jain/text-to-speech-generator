# text-to-speech-generator
The code you provided is a script for text-to-speech (TTS) using the Tacotron2 and WaveRNN models in torchaudio. It generates spectrograms and waveforms from input text.

Here's a breakdown of the code:

Importing necessary libraries: IPython, matplotlib, torch, and torchaudio.
Setting the figure size for matplotlib plots.
Checking the versions of torch and torchaudio.
Defining the symbols and look-up table for text processing.
Implementing the text_to_sequence function to convert text into a sequence of symbols.
Getting the text processor for the TACOTRON2_WAVERNN_CHAR_LJSPEECH bundle and processing the input text.
Printing the processed text and its lengths.
Getting the text processor for the TACOTRON2_WAVERNN_PHONE_LJSPEECH bundle and processing the input text.
Printing the processed text and its lengths.
Generating a spectrogram using the TACOTRON2 model and displaying it using matplotlib.
Generating multiple spectrograms and displaying them using matplotlib.
Initializing the TACOTRON2 and WaveRNN models from the TACOTRON2_WAVERNN_PHONE_LJSPEECH bundle.
Processing the input text and generating the spectrogram using the TACOTRON2 model.
Displaying the spectrogram using matplotlib.
Generating waveforms using the WaveRNN vocoder and displaying them using matplotlib.
Using IPython to play the generated audio.
