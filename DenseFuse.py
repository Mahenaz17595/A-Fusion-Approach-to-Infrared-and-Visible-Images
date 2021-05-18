from __future__ import print_function
from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import numpy as np
from train_recons import train_recons
from generate import generate
from utils import list_images
import cv2

main = tkinter.Tk()
main.title("DenseFuse: A Fusion Approach to Infrared and Visible Images") #designing main screen
main.geometry("800x700")

global filename
SSIM_WEIGHTS = [1, 10, 100, 1000]
MODEL_SAVE_PATHS = [
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e1.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e3.ckpt',
]


def upload(): #function to upload
    global filename
    filename = filedialog.askdirectory(initialdir="testImages")
    textarea.insert(END,filename+" loaded\n")
    
def denseFuse():
    ssim_weight = SSIM_WEIGHTS[0]
    model_path = MODEL_SAVE_PATHS[2]
    infrared = filename + '/IR.png'
    visible = filename + '/VIS.png'
    fusion_type = 'addition'
    output_save_path = 'outputs'
    generate(infrared, visible, model_path, None, ssim_weight, 0, False, False, type = fusion_type, output_path = output_save_path)
    cv2.imshow("Infrared Image",cv2.imread(infrared))
    cv2.imshow("Visible Image",cv2.imread(visible))
    cv2.waitKey(0)


def exit():
    global main
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='DenseFuse: A Fusion Approach to Infrared and Visible Images', justify=LEFT)
title.config(bg='lavender blush', fg='maroon')
title.config(font=font)           
title.config(height=3, width=220)
title.place(x=250,y=8)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Upload Visible & IR Image", command=upload)
model.place(x=200,y=100)
model.config(font=font1)  

uploadimage = Button(main, text="Generate DenseFuse Image", command=denseFuse)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1) 

exitapp = Button(main, text="Exit", command=exit)
exitapp.place(x=200,y=200)
exitapp.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=35,width=300)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=300)
textarea.config(font=font1)

main.config(bg='MistyRose4')
main.mainloop()
