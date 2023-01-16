import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate a dataset of images of hexagons and squares
def generate_data():
    hexagons = []
    squares = []
    for i in range(1000):
        # Generate a random hexagon
        angle = np.random.uniform(0, 2*np.pi)
        x = np.cos(angle)
        y = np.sin(angle)
        hexagons.append([x, y])
        # Generate a random square
        x, y = np.random.randn(2)
        if abs(x) <= 1 and abs(y) <= 1:
            squares.append([x, y])
    data = np.concatenate((hexagons, squares), axis=0)
    labels = np.concatenate((np.zeros(len(hexagons)), np.ones(len(squares))))
    return data, labels

# Create the GAN
def create_gan():
    # Create the generator
    generator = tf.keras.Sequential()
    generator.add(tf.keras.layers.Dense(2, input_dim=2))
    generator.add(tf.keras.layers.LeakyReLU())
    # Create the discriminator
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Dense(10, input_dim=2))
    discriminator.add(tf.keras.layers.LeakyReLU())
    discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile the discriminator
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Create the GAN
    gan = tf.keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan, generator, discriminator

# Train the GAN
def train_gan(gan, generator, discriminator):
    data, labels = generate_data()
    for i in range(1000):
        # Generate fake data
        noise = np.random.randn(len(data), 2)
        fake_data = generator.predict(noise)
        # Train the discriminator
        d_loss = discriminator.train_on_batch(fake_data, np.zeros(len(data)))
        d_loss = discriminator.train_on_batch(data, labels)
        # Train the generator
        noise = np.random.randn(len(data), 2)
        g_loss = gan.train_on_batch(noise, np.ones(len(data)))
        print(f'Step {i}, D loss: {d_loss[0]}, D acc: {d_loss[1]}, G loss: {g_loss}')

# Use the generator to generate new shapes
def generate_shapes(generator, prompt, noise):
    noise = noise
    if prompt == 'hexagon':
        noise[0, 0] = abs(noise[0, 0])
        noise[0, 1] = abs(noise[0, 1])
    elif prompt == 'square':
        noise[0, 0] = noise[0, 0]
    shape = generator.predict(noise)[0]
    return shape

if __name__ == '__main__':
    gan, generator, discriminator = create_gan()
    train_gan(gan, generator, discriminator)
    noise = np.random.randn(1, 2)
    shape = generate_shapes(generator, noise)
    print(shape)