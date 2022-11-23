import tensorflow as tf
model = tf.keras.models.load_model("static/trained_cnn2")
print(model.summary())