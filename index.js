//import * as tf from '@tensorflow/tfjs';
const tf = require('@tensorflow/tfjs');

// 1. Definiendo modelo de regresion lineal
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// 2. Preparando el modelo para el entrenamiento
// meanSquaredError => Raiz media cuadrada
// sgd => GradientDescentOptimier
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// 3. Generando datos para el entrenamiento
// Tensores monodimensionales, [4, 1] => indica 4 columnas y 1 fila
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// 4. Entrenamos el modelo usando datos
// El entrenamiento genera una promea que cuando se conolide permitira predecir valores
// model.fit(xs, ys, {epochs: 10});

// 5. Generamos la prediccion con datos nuevos una vez el modelo ete entrenado
// Nos suscribimos a la promesa e imprimimos la predicción para el tensor escalar 5 ([1,1], indica 1 columna, una fila)
model.fit(xs, ys, {epochs: 1000}).then(() => {
    model.predict(tf.tensor2d([5], [1, 1])).print();
});
