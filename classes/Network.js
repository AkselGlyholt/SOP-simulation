class NeuralNetwork {
  constructor(neuronCounts) {
    this.levels = [];

    for (let i = 0; i < neuronCounts.length - 1; i++) {
      const isOutputLayer = i === neuronCounts.length - 2;
      // Skjulte lag: ReLU. Output-lag: lineær (Q-værdier skal kunne være alle reelle tal)
      const activation = isOutputLayer ? "linear" : "relu";
      this.levels.push(
        new Level(neuronCounts[i], neuronCounts[i + 1], activation),
      );
    }
  }

  /**
   * Kør input gennem hele netværket, lag for lag
   * @returns {number[]} Q-værdierne (en per mulig handling)
   */
  forward(inputs) {
    let outputs = inputs;

    for (const level of this.levels) {
      outputs = level.forward(outputs);
    }

    return outputs;
  }

  /**
   * Kør gradienten baglæns gennem hele netværket, lag for lag i omvendt rækkefølge
   * @param {number[]} outputGradient - dL/da for netværkets ENDELIGE output
   *    (typisk 0 for alle handlinger undtagen den der blev valgt, hvor det er -2(y - Q(s, a)))
   */
  backward(outputGradient) {
    let gradient = outputGradient;

    for (let i = this.levels.length - 1; i >= 0; i--) {
      gradient = this.levels[i].backward(gradient);
    }
  }

  zeroGradients() {
    for (const level of this.levels) level.zeroGradients();
  }

  applyGradients(learningRate, batchSize) {
    for (const level of this.levels) {
      level.applyGradients(learningRate, batchSize);
    }
  }

  /**
   * Dyb kopi - bruges til at oprette target-netværket
   */
  clone() {
    const clone = Object.create(NeuralNetwork.prototype);
    clone.levels = this.levels.map((level) => level.clone());

    return clone;
  }

  /**
   * Hard update: kopier vægtene fra sourceNetwork ind i dette netværk (target-sync)
   */
  copyFrom(sourceNetwork) {
    for (let i = 0; i < this.levels.length; i++) {
      const src = sourceNetwork.levels[i];
      const dst = this.levels[i];

      for (let a = 0; a < src.inputCount; a++) {
        dst.weights[a] = src.weights[a].slice();
      }

      dst.biases = src.biases.slice();
    }
  }
}

class Level {
  /**
   *
   * @param {number} inputCount
   * @param {number} outputCount
   * @param {"relu"|"linear"} activation - "relu" for hidden layers, "linear" for the output layers (raw Q-values)
   */
  constructor(inputCount, outputCount, activation = "relu") {
    this.inputCount = inputCount;
    this.outputCount = outputCount;
    this.activation = activation;

    // weights[i][j] = the weight from input i to the output-neuron j
    this.weights = [];
    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount);
    }
    this.biases = new Array(outputCount);

    Level.#randomize(this);

    // Saved from last forward pass. Required for backward()
    this.lastInputs = new Array(inputCount).fill(0);
    this.lastZ = new Array(outputCount).fill(0);
    this.lastOutputs = new Array(outputCount).fill(0);

    // Gradients collected over a whole batch (reset with zeroGradients()).
    this.weightGradients = [];
    for (let i = 0; i < inputCount; i++) {
      this.weightGradients[i] = new Array(outputCount).fill(0);
    }

    this.biasGradients = new Array(outputCount).fill(0);
  }

  static #randomize(level) {
    // He-initialisering: skaleret efter antal inputs, så variansen af z
    // holdes fornuftig gennem laget, uanset netværkets bredde.
    // Vigtigt for ReLU-netværk - undgår at outputs "dør" (bliver 0) eller
    // eksploderer i de første mange træningsskridt.
    const scale = Math.sqrt(2 / level.inputCount);

    for (let i = 0; i < level.inputCount; i++) {
      for (let j = 0; j < level.outputCount; j++) {
        level.weights[i][j] = (Math.random() * 2 - 1) * scale;
      }
    }

    // Bias initialiseres til 0, modsat vægte, er der ingen grund til at
    // starte bias tilfældigt; det lærer sig selv hen mod den rigtige værdi
    for (let j = 0; j < level.biases.length; j++) {
      level.biases[j] = 0;
    }
  }

  #activate(z) {
    if (this.activation === "relu") return Math.max(0, z);
    return z; // lineær aktivering (bruges i output-laget)
  }

  #activateDerivative(z) {
    if (this.activation === "relu") return z > 0 ? 1 : 0;
    return 1; // d/dz [z] = 1 for lineær aktivering
  }

  /**
   * Forward pass: z = W·x + b, a = f(z)
   * Gemmer inputs, z og a - de skal bruges igen i backward().
   */
  forward(inputs) {
    this.lastInputs = inputs.slice();

    for (let j = 0; j < this.outputCount; j++) {
      let sum = this.biases[j];
      for (let i = 0; i < this.inputCount; i++) {
        sum += inputs[i] * this.weights[i][j];
      }
      this.lastZ[j] = sum;
      this.lastOutputs[j] = this.#activate(sum);
    }

    return this.lastOutputs;
  }

  /**
   * Backward pass - kædereglen, ét lag ad gangen.
   *
   * @param {number[]} outputGradient - dL/da for hver af dette lags outputs
   * @returns {number[]} dL/dx - gradienten ift. dette lags INPUTS, som skal
   *                     sendes videre til det foregående lags backward()
   *
   * Gradienterne akkumuleres (lægges til), de nulstilles IKKE her - det er
   * hvad der gør mini-batch træning muligt: kald backward() én gang per
   * eksempel i batchen, og de summeres automatisk. Kald zeroGradients() før
   * hver ny batch, og applyGradients() til sidst for at opdatere vægtene.
   */
  backward(outputGradient) {
    const inputGradient = new Array(this.inputCount).fill(0);

    for (let j = 0; j < this.outputCount; j++) {
      // Led 1+2 fra kædereglen: dL/da_j * da_j/dz_j
      const dA_dZ = this.#activateDerivative(this.lastZ[j]);
      const dL_dZ = outputGradient[j] * dA_dZ;

      // dL/db_j = dL/dz_j (bias' "input" er altid 1)
      this.biasGradients[j] += dL_dZ;

      for (let i = 0; i < this.inputCount; i++) {
        // Led 3: dL/dw_ij = dL/dz_j * dz_j/dw_ij = dL/dz_j * x_i
        this.weightGradients[i][j] += dL_dZ * this.lastInputs[i];

        // Gradient der skal videre "baglæns" til forrige lag:
        // dL/dx_i = sum over j af (dL/dz_j * dz_j/dx_i) = sum over j af (dL/dz_j * w_ij)
        inputGradient[i] += dL_dZ * this.weights[i][j];
      }
    }

    return inputGradient;
  }

  zeroGradients() {
    for (let i = 0; i < this.inputCount; i++) {
      this.weightGradients[i].fill(0);
    }

    this.biasGradients.fill(0);
  }

  /**
   * Gradient descent-opdateringen: w <- w - eta * gennemsnitlig_gradient
   */
  applyGradients(learningRate, batchSize) {
    for (let i = 0; i < this.inputCount; i++) {
      for (let j = 0; j < this.outputCount; j++) {
        this.weights[i][j] -=
          learningRate * (this.weightGradients[i][j] / batchSize);
      }
    }

    for (let j = 0; j < this.outputCount; j++) {
      this.biases[j] -= learningRate * (this.biasGradients[j] / batchSize);
    }
  }

  clone() {
    const level = new Level(this.inputCount, this.outputCount, this.activation);

    for (let i = 0; i < this.inputCount; i++) {
      level.weights[i] = this.weights[i].slice();
    }

    level.biases = this.biases.slice();
    return level;
  }
}
