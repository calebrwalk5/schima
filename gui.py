import tkinter as tk
from threading import Thread
import matplotlib
matplotlib.use("TkAgg")
from tkinter import ttk
from train import train_gan, generate_shapes, create_gan
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import queue

class GANMonitor(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("GAN Monitor")
        self.geometry("500x500")

        self.gan, self.generator, self.discriminator = create_gan()
        self.running = False
        self.train_queue = queue.Queue()
        self.train_thread = Thread(target=self.train, args=(self.train_queue,))

        self.start_button = ttk.Button(self, text="Start", command=self.start)
        self.start_button.pack()

        self.stop_button = ttk.Button(self, text="Stop", command=self.stop)
        self.stop_button.pack()

        self.step_label = ttk.Label(self, text="Step: ")
        self.step_label.pack()

        self.d_loss_label = ttk.Label(self, text="D loss: ")
        self.d_loss_label.pack()

        self.d_acc_label = ttk.Label(self, text="D acc: ")
        self.d_acc_label.pack()

        self.g_loss_label = ttk.Label(self, text="G loss: ")
        self.g_loss_label.pack()

        self.shape_button = ttk.Button(self, text="Generate shape", command=self.generate_shape)
        self.shape_button.pack()

        self.shape_label = ttk.Label(self, text="Shape: ")
        self.shape_label.pack()

        # View the shape
        self.figure = Figure(figsize=(5, 5))
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()

    def start(self):
        self.running = True
        self.train_thread.start()
        self.after(1000, self.update_train_queue, self.train_queue)
    def stop(self):
        self.running = False

    def train(self, train_queue):
        while self.running:
            step, d_loss, d_acc, g_loss = train_gan(self.gan, self.generator, self.discriminator)
            train_queue.put((step, d_loss, d_acc, g_loss))

    def update_train_queue(self, train_queue):
        try:
            step, d_loss, d_acc, g_loss = train_queue.get(0)
            self.step_label.config(text="Step: {}".format(step))
            self.d_loss_label.config(text="D loss: {}".format(d_loss))
            self.d_acc_label.config(text="D acc: {}".format(d_acc))
            self.g_loss_label.config(text="G loss: {}".format(g_loss))
            if self.running:
                self.after(1000, self.update_train_queue, train_queue)
        except queue.Empty:
            if self.running:
                self.after(1000, self.update_train_queue, train_queue)


    def generate_shape(self):
        shape = generate_shapes(self.generator, "circle")
        self.shape_label.config(text="Shape: {}".format(shape))
        self.axes.clear()
        self.axes.scatter(shape[0], shape[1], c=shape[2])
        self.canvas.draw()

if __name__ == "__main__":
    app = GANMonitor()
    app.mainloop()



if __name__ == "__main__":
    app = GANMonitor()
    app.mainloop()
