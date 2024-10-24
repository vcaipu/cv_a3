import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image

def numeric_gradient(f, x, h=1e-6):
    """
    Computes a numeric gradient of a function that takes an array argument.

    Inputs:
    - f: A function of the form y = f(x) where x is a numpy array and y is
      a Python float
    - x: The point at which to compute a numeric gradient
    - h: The step size to use for computing the numeric gradient

    Returns:
    - grad_x: Numpy array of the same shape as x giving a numeric approximation
      to the gradient of f at the point x.
    """
    grad = np.zeros_like(x)
    grad_flat = grad.reshape(-1)
    x_flat = x.reshape(-1)
    for i in range(grad_flat.shape[0]):
        old_val = x_flat[i]
        x_flat[i] = old_val + h
        pos = f(x)
        x_flat[i] = old_val - h
        neg = f(x)
        grad_flat[i] = (pos - neg) / (2.0 * h)
        x_flat[i] = old_val
    return grad


class Logger:
    def __init__(self, P, P_prime, print_every=50):
        self.P = P
        self.P_prime = P_prime
        self.print_every = print_every
        self.iterations = []
        self.losses = []
        self.predictions = []

    def log(self, iteration, loss, prediction):
        if iteration % self.print_every == 0:
            print(f'Iteration {iteration}, loss = {loss}')
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.predictions.append(prediction)

    def save_loss_plot(self, filename):
        plt.plot(self.iterations, self.losses, 'o')
        plt.savefig(filename)
        plt.clf()
        print(f'Saved loss plot to {filename}')

            
    def save_animated_gif(self, filename, show_every=50):
        print(filename)
        imgs = []
        factors = (self.iterations, self.losses, self.predictions)
        for i, loss, pred in zip(*factors):
            if i % show_every != 0:
                continue
            plt.scatter(self.P[:, 0], self.P[:, 1], label="P")
            plt.scatter(self.P_prime[:, 0], self.P_prime[:, 1], label="P'")
            plt.scatter(pred[:, 0], pred[:, 1], label='prediction')
            plt.legend()
            plt.title(f'Iteration {i}, loss = {loss}')
            fig = plt.gcf()
            fig.canvas.draw() 
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            imgs.append(img)
            plt.clf()
        imageio.mimwrite(filename, imgs)
        print(f'Saved animated gif to {filename}')


        #img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
