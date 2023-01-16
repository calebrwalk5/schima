import tensorflow as tf
import numpy as np

# Generate a dataset of images of trapezoids and squares
def generate_data():
    trapezoids = []
    a, b, c = 2, 3, -1
    for i in range(10000): # Generate 10000 trapezoids
        x, y = np.random.rand(2) # Random numbers between 0 and 1
        x = x*2-1
        y = (a/b)*x + c
        trapezoids.append([x, y])
    data = np.array(trapezoids)
    labels = np.ones(len(trapezoids))
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
def generate_shapes(generator, prompt):
    noise = np.random.randn(1, 2)
    if prompt == 'trapezoid':
        noise[0, 0] = abs(noise[0, 0])
        noise[0, 1] = abs(noise[0, 1])
    shape = generator.predict(noise)[0]
    return shape

if __name__ == '__main__':
    gan, generator, discriminator = create_gan()
    train_gan(gan, generator, discriminator)
    shape = generate_shapes(generator, 'trapezoid')
    print(shape)
