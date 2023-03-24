# DL_cheatsheet

model = tf.keras.models.Sequential([
tf.keras.layers.Dense(10,activation="relu"),
tf.keras.layers.Dense(1,activation="sigmoid"),


])
model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(lr= 0.01),
              metrics = ["accuracy"])

history = model.fit(tf.constant(X),y,epochs=100)

plot_decision_boundary(model, X, y)
