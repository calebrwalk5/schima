import tkinter as tk
from threading import Thread
import matplotlib
matplotlib.use("TkAgg")
from tkinter import ttk
from train import train_gan, generate_shapes, create_gan, generate_data
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
        self.title("Schima | GAN Monitor")
        self.geometry("1000x1000")

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
        self.shape_figure = Figure(figsize=(5, 5))
        self.shape_axes = self.shape_figure.add_subplot(111)
        self.shape_canvas = FigureCanvasTkAgg(self.shape_figure, self)
        self.shape_canvas.get_tk_widget().pack()

        # View the data
        self.data_figure = Figure(figsize=(5, 5))
        self.data_axes = self.data_figure.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, self)
        self.data_canvas.get_tk_widget().pack()

    def display_shapes(self):
        shape = generate_shapes(self.generator, "trapezoid")
        self.plot_shape(shape)
        self.display_data()
        self.plot_data()
        if self.running:
            self.after(50, self.display_shapes)

    def display_data(self):
        data, labels = generate_data()
        # Code for displaying data
        if self.running:
            self.after(50, self.display_data)
    
    def plot_data(self):
        data, labels = generate_data()
        self.data_axes.clear()
        self.data_axes.scatter(data[:, 0], data[:, 1], c=(1, 0, 1))
        self.data_canvas.draw()
        self.data_axes.set_xlim(-2, 2)
        self.data_axes.set_ylim(-1, 1)

    def add_data_plot(self):
        self.data_figure = Figure(figsize=(5, 5))
        self.data_axes = self.data_figure.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, self)
        self.data_canvas.get_tk_widget().pack()

    def plot_shape(self, shape):
        self.shape_axes.clear()
        self.shape_axes.scatter(shape[0], shape[1], c=(1, 0, 1))
        self.shape_canvas.draw()

    def start(self):
        self.running = True
        self.train_thread.start()
        self.display_shapes()
        self.add_data_plot()
        self.plot_data()
        self.display_data()
        self.after(50, self.update_train_queue, self.train_queue)

    def stop(self):
        self.running = False
        self.train_thread.join()

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
                self.after(50, self.update_train_queue, train_queue)
        except queue.Empty:
            if self.running:
                self.after(50, self.update_train_queue, train_queue)

    def generate_shape(self):
        shape = generate_shapes(self.generator, "trapezoid")
        self.plot_shape(shape)
        x, y = shape[0], shape[1]
        color = np.full(x.shape, 'r')
        self.shape_label.config(text="Shape: {}".format(shape))
        self.axes.clear()
        self.axes.scatter(shape[0], shape[1], c=(1, 0, 1, 1))
        self.canvas.draw()

if __name__ == "__main__":
    app = GANMonitor()
    app.mainloop()
