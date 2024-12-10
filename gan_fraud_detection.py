# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %%
data = pd.read_csv(r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\GAN CreditCard\creditcard.csv')
# %%
fraud_data = data[data['Class'] == 1]
normal_data = data[data['Class'] == 0]
# %%
fraud_data.shape
# %%
scaler = StandardScaler()
fraud_data_scaled = scaler.fit_transform(fraud_data.drop('Class', axis=1))
normal_data_scaled = scaler.fit_transform(normal_data.drop('Class', axis=1))
input_dim = fraud_data_scaled.shape[1]
# %%
fraud_data_scaled.shape
print(type(input_dim))
# %%
def create_generator(input_dim, output_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(input_dim,)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_dim, activation='tanh')
    ])
# %%
def create_discriminator(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(input_dim,)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
# %%
class StableGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(StableGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(StableGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, real_data):
        real_data = tf.cast(real_data, tf.float32)
        batch_size = tf.shape(real_data)[0]
        noise_dim = self.generator.input_shape[1]

        # Add noise to labels for label smoothing
        real_labels = tf.random.uniform((batch_size, 1), 0.8, 1.0, dtype=tf.float32)
        fake_labels = tf.random.uniform((batch_size, 1), 0.0, 0.2, dtype=tf.float32)

        # Train discriminator
        d_loss_total = 0
        for _ in range(2):
            noise = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)

            with tf.GradientTape() as disc_tape:
                # Generate fake samples
                generated_data = self.generator(noise, training=True)

                # Get discriminator outputs
                real_output = self.discriminator(real_data, training=True)
                fake_output = self.discriminator(generated_data, training=True)

                # Calculate discriminator loss
                real_loss = self.loss_fn(real_labels, real_output)
                fake_loss = self.loss_fn(fake_labels, fake_output)
                disc_loss = real_loss + fake_loss

                # Add gradient penalty
                epsilon = tf.random.uniform([batch_size, 1], 0, 1, dtype=tf.float32)
                interpolated = real_data + epsilon * (generated_data - real_data)
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    interp_output = self.discriminator(interpolated, training=True)
                grads = gp_tape.gradient(interp_output, interpolated)
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
                gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

                disc_loss += 10.0 * gradient_penalty

            # Apply discriminator gradients
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(
                gradients_of_discriminator, self.discriminator.trainable_variables))

            d_loss_total += disc_loss

        d_loss_total /= 2

        # Train Generator
        noise = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)

        with tf.GradientTape() as gen_tape:
            # Generate fake samples
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            # Generator loss
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        # Apply generator gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)

        # Clip gradients
        gradients_of_generator = [tf.clip_by_norm(g, 1.0) for g in gradients_of_generator]

        self.g_optimizer.apply_gradients(zip(
            gradients_of_generator, self.generator.trainable_variables))

        return {"d_loss": d_loss_total, "g_loss": gen_loss}
# %%
def create_and_compile_gan(input_dim):
    generator = create_generator(input_dim, input_dim)
    discriminator = create_discriminator(input_dim)

    gan = StableGAN(generator, discriminator)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    gan.compile(
        g_optimizer=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        loss_fn=loss_fn
    )

    return gan
# %%
def train_gan(gan, data, epochs=10000, batch_size=128):
    # Convert data to float32
    data = tf.cast(data, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=1000).batch(batch_size)

    for epoch in range(epochs):
        losses = {"d_loss": [], "g_loss": []}

        for batch in dataset:
            batch_losses = gan.train_step(batch)
            losses["d_loss"].append(float(batch_losses["d_loss"]))
            losses["g_loss"].append(float(batch_losses["g_loss"]))

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, '
                  f'D Loss: {np.mean(losses["d_loss"]):.4f}, '
                  f'G Loss: {np.mean(losses["g_loss"]):.4f}')

        # Early stopping if generator loss is too high
        if np.mean(losses["g_loss"]) > 10:
            print("Early stopping due to high generator loss")
            break
# %%
# Assuming fraud_data_scaled is your preprocessed data
fraud_data_scaled = tf.cast(fraud_data_scaled, tf.float32)
input_dim = fraud_data_scaled.shape[1]
gan = create_and_compile_gan(input_dim)
train_gan(gan, fraud_data_scaled)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)
# %%
@tf.function(reduce_retracing=True)
def train_gan(generator, discriminator, data, epochs=10000, batch_size=128):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Sample random real data from input
        idx = tf.random.uniform(shape=[half_batch], minval=0, maxval=data.shape[0], dtype=tf.int32)
        real_data = tf.gather(data, idx)

        # Generate fake data
        noise = tf.random.normal(shape=(half_batch, input_dim))
        fake_fraud = generator(noise)

        # Train discriminator
        dlossReal = discriminator.train_on_batch(real_data, tf.ones((half_batch, 1)))
        dlossFake = discriminator.train_on_batch(fake_fraud, tf.zeros((half_batch, 1)))
        dloss = 0.5 * (dlossReal + dlossFake)

        # Train generator
        noise = tf.random.normal(shape=(batch_size, input_dim))
        valid_y = tf.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        # Log the progress
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Discriminator Loss: {dloss.numpy()}, Generator Loss: {g_loss.numpy()}')
# %%
def train_gan_with_graphs(gan, data , epochs=10000, batch_size = 128):
    data = tf.cast(data , tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size = 1000).batch(batch_size)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        epoch_d_losses = []
        epoch_g_losses = []

        for batch in dataset:
            batch_losses = gan.train_step(batch)
            epoch_d_losses.append(batch_losses["d_loss"])
            epoch_g_losses.append(batch_losses["g_loss"])
        d_losses.append(np.mean(epoch_d_losses))
        g_losses.append(np.mean(epoch_g_losses))
        if epoch%1000 == 00:
            print(f"for epoch {epoch} generator loss is {np.float32(epoch_g_losses)} and discriminator loss is {np.float32(epoch_d_losses)}")

    plt.figure(figsize=(10,5))
    plt.plot(d_losses , label="Discriminator Loss")
    plt.plot(g_losses , label = "Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

# %%
train_gan_with_graphs(gan , fraud_data_scaled)
# %%
def generate_synthetic_data(card_number, generator , discriminator , scaler ,input_dim):
    synthetic_features = np.random.normal(0,1,(1, input_dim))

    scaled_input = scaler.inverse_transform(synthetic_features)
    prediction = discriminator.predict(synthetic_features)

    is_fraud = prediction[0,0]>0.5

    print(f"Generated Transaction {scaled_input} from Card Number {card_number}")
    print(f"Fraud Prediction Score: {prediction[0,0]:.4f}")
    print(f"Transaction is {'Fraudulent' if is_fraud else 'Legitimate'}")

# %%
card_number = input("Enter Card Number")
generate_synthetic_data(card_number ,gan.generator,gan.discriminator,scaler,input_dim)
# %%
generator_path = r"C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\GAN CreditCard\generator.keras"
gan.generator.save(generator_path)
print(f"saved at {generator_path}")
# %%
