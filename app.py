import tkinter as tk
from tkinter import ttk
import IPython
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchaudio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

root = tk.Tk()
root.title("Text-to-Speech")

# Text input field
text_entry = ttk.Entry(root)
text_entry.pack()

# Spectrogram display
spectrogram_fig = plt.Figure(figsize=(10, 3), dpi=100)
spectrogram_ax = spectrogram_fig.add_subplot(111)
spectrogram_canvas = FigureCanvasTkAgg(spectrogram_fig, master=root)
spectrogram_canvas.draw()
spectrogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Waveform display
waveform_fig = plt.Figure(figsize=(10, 3), dpi=100)
waveform_ax = waveform_fig.add_subplot(111)
waveform_canvas = FigureCanvasTkAgg(waveform_fig, master=root)
waveform_canvas.draw()
waveform_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def generate_spectrogram():
    text = text_entry.get()

    # Code for generating the spectrogram
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)

    text = input("Text to speech:")

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, _, _ = tacotron2.infer(processed, lengths)


    _ = plt.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")

    fig, ax = plt.subplots(3, 1, figsize=(16, 4.3 * 3))
    for i in range(3):
        with torch.inference_mode():
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        print(spec[0].shape)
        ax[i].imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    plt.show()

    # Update the spectrogram plot
    spectrogram_ax.clear()
    spectrogram_ax.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    spectrogram_canvas.draw()

def generate_waveform():
    text = text_entry.get()

    # Code for generating the waveform
    # ...
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    text = "Hello world! Text to speech!"

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9))
    ax1.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    ax2.plot(waveforms[0].cpu().detach())

    IPython.display.Audio(waveforms[0:1].cpu(), rate=vocoder.sample_rate)

    # Update the waveform plot
    waveform_ax.clear()
    waveform_ax.plot(waveforms[0].cpu().detach())
    waveform_canvas.draw()
    IPython.display.Audio(waveforms[0:1].cpu(), rate=vocoder.sample_rate)

# Generate spectrogram button
spectrogram_button = ttk.Button(root, text="Generate Spectrogram", command=generate_spectrogram)
spectrogram_button.pack()

# Generate waveform button
waveform_button = ttk.Button(root, text="Generate Waveform", command=generate_waveform)
waveform_button.pack()

root.mainloop()



