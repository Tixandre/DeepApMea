from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import h5py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from joblib import Parallel, delayed
import joblib
from tkinter.messagebox import showinfo,showerror
import tkinter as tk


if os.path.exists("denoising_Dense_1.h5"):
    autoencoder = tf.keras.models.load_model('denoising_Dense_1.h5')
    output_layer = (autoencoder.layers[-11].output)
    encoder = Model(autoencoder.input, output_layer)
else:
    showerror(message='Autoencoder not found')

if os.path.exists("Classifier_cnn.h5"):
    model = tf.keras.models.load_model('Classifier_cnn.h5')
else:
    showerror(message='Classifier not found')


def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("H5 files", "*.h5*"),
                                                       ("all files", "*.*")))
    global filePath
    filePath = filename


    start_button = ttk.Button(
        window,
        text='Start filter',
        command=runFilter
    )
        # place the progressbar
    pb.grid(column=1, row=4, columnspan=2, padx=10, pady=20)
    value_label.grid(column=1, row=5, columnspan=2)

    start_button.grid(column=1, row=6, padx=10, pady=10)
    label_file_explorer.configure(text="File Opened: "+filename)

def update_progress_label():
    return f"Current Progress: {round(pb['value'])}%"

def progress():
    if pb['value'] < 100:
        window.update_idletasks()
        pb['value'] += 3.125
        value_label['text'] = update_progress_label()
    else:
        showinfo(message='The progress completed!')
      
def runFilter():
    N_ELECTRODES = 32
    CUT_OFF = 120

    window.update_idletasks()
    pb['value'] = 0
    value_label['text'] = update_progress_label()

    with h5py.File(filePath, "r+") as f:
        for i in range(N_ELECTRODES):
            dset = f[f'SpikeWindow-0.{i}']
            stmp = f[f'SpikeTimestamp-0.{i}']

            if(dset.shape[0] != 0):
                sp = np.empty((dset.shape[0],CUT_OFF),dset.dtype)

                for idx,spke in enumerate(dset):
                    sp[idx] = (spke[0:CUT_OFF])
            
                test = pd.DataFrame(sp)
                test_encode = encoder.predict(test)
                predict = model.predict(test_encode).round()
                print(f'Num electrode : {i}')
                print(f'Spike predicted : {np.count_nonzero(predict == 1)}')
                print(f'Noise predicted : {np.count_nonzero(predict == 0)}')

                res = np.where(predict == 1)[0].tolist()

                indx = 0
                test_spike_windows = np.empty((len(res),dset.shape[1]),dset.dtype)
                for idx,row in enumerate(dset):
                    if(idx in res):
                        test_spike_windows[indx] = row
                        indx += 1

                indx = 0
                test_spike_timestamp = np.empty((len(res),stmp.shape[1]),stmp.dtype)
                for idx,row in enumerate(stmp):
                    if(idx in res):
                        test_spike_timestamp[indx] = row
                        indx += 1

                del f[f'SpikeWindow-0.{i}']
                f.create_dataset(f'SpikeWindow-0.{i}',data=test_spike_windows)

                del f[f'SpikeTimestamp-0.{i}']
                f.create_dataset(f'SpikeTimestamp-0.{i}',data=test_spike_timestamp)
            progress()
        f.close()
    progress()


window = Tk()
window.title('DEEPAPMEA - Filter file')
window.geometry("700x300")
window.config(background = "white")
  
pb = ttk.Progressbar(
    window,
    orient='horizontal',
    mode='determinate',
    length=280
)
value_label = ttk.Label(window, text=update_progress_label())
# Create a File Explorer label
label_file_explorer = Label(window,text = "",width = 100, height = 4)
    
mainmenu = Menu(window)
 
# Menu 1
filemenu = Menu(mainmenu, tearoff = 0)
filemenu.add_command(label = "Open", command = browseFiles)
filemenu.add_separator()
filemenu.add_command(label = "Exit", command = window.destroy)
mainmenu.add_cascade(label="File", menu=filemenu)

window.config(menu = mainmenu)
label_file_explorer.grid(column = 1, row = 1)

window.mainloop()