/**
 * Genererer det diskrete action-space ud fra de 4 uafhængige styresignaler
 * (forward, left, right, reverse), og fjerner de kombinationer der er
 * fysisk meningsløse (speede og bakke samtidig, dreje begge veje samtidig).
 *
 * Ændr selv listen af udelukkelser her, hvis du ønsker et andet action-space.
 */
function buildActionSpace() {
  const actions = [];

  for (let mask = 0; mask < 16; mask++) {
    const forward = Boolean(mask & 1);
    const left = Boolean(mask & 2);
    const right = Boolean(mask & 4);
    const reverse = Boolean(mask & 8);

    if (forward && reverse) continue; // meningsløst: speeder og bakker samtidig
    if (left && right) continue; // meningsløst: drejer til begge sider samtidig

    actions.push({ forward, left, right, reverse });
  }

  return actions; // 10 gyldige kombinationer ud af de oprindelige 16
}

class DQNAgent {
  constructor({
    stateSize,
    hiddenSize = 12,
    actions = buildActionSpace(),
    gamma = 0.95,
    learningRate = 0.01,
    epsilonStart = 1.0,
    epsilonMin = 0.05,
    epsilonDecay = 0.0005, // hvor hurtigt epsilon falder, pr. TRÆNINGSSKRIDT (ikke pr. frame)
    bufferCapacity = 10000,
    batchSize = 32,
    targetSyncEvery = 1000, // C - antal træningsskridt mellem hver hard update
  }) {
    this.actions = actions;
    this.gamma = gamma;
    this.learningRate = learningRate;

    this.epsilon = epsilonStart;
    this.epsilonMin = epsilonMin;
    this.epsilonDecay = epsilonDecay;

    this.batchSize = batchSize;
    this.targetSyncEvery = targetSyncEvery;

    this.mainNetwork = new NeuralNetwork([
      stateSize,
      hiddenSize,
      actions.length,
    ]);
    this.targetNetwork = this.mainNetwork.clone();

    this.replayBuffer = new ReplayBuffer(bufferCapacity);
    this.trainStepCount = 0;
  }

  /**
   * Epsilon-greedy: vælg en tilfældig handling med sandsynlighed epsilon,
   * ellers den handling med højest Q-værdi ifølge hovednetværket.
   * @param {number[]} state
   * @returns {number} index i this.actions
   */
  chooseActionIndex(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actions.length);
    }

    const qValues = this.mainNetwork.forward(state);
    let bestIndex = 0;
    for (let i = 1; i < qValues.length; i++) {
      if (qValues[i] > qValues[bestIndex]) bestIndex = i;
    }
    return bestIndex;
  }

  remember(state, actionIndex, reward, nextState, done) {
    this.replayBuffer.add({ state, actionIndex, reward, nextState, done });
  }

  /**
   * Eksponentielt aftagende epsilon - kaldes én gang per træningsskridt (ikke per frame).
   */
  #decayEpsilon() {
    this.epsilon =
      this.epsilonMin +
      (1 - this.epsilonMin) *
        Math.exp(-this.epsilonDecay * this.trainStepCount);
  }

  /**
   * Sampler én batch fra replay bufferen og udfører ét gradient descent-skridt.
   * @returns {number|null} gennemsnitlig loss for batchen, eller null hvis der
   *   endnu ikke er nok data i bufferen til at træne på.
   */
  train() {
    if (this.replayBuffer.size() < this.batchSize) return null;

    const batch = this.replayBuffer.sample(this.batchSize);
    this.mainNetwork.zeroGradients();

    let totalLoss = 0;

    for (const { state, actionIndex, reward, nextState, done } of batch) {
      // Forward pass på hovednetværket - genererer også lastZ/lastInputs,
      // som backward() bruger til DENNE specifikke transition.
      const qValues = this.mainNetwork.forward(state);

      // Bellman-target: y = r hvis episoden er slut, ellers r + gamma * max Q(s', a')
      let targetValue;
      if (done) {
        targetValue = reward;
      } else {
        const nextQValues = this.targetNetwork.forward(nextState);
        targetValue = reward + this.gamma * Math.max(...nextQValues);
      }

      const prediction = qValues[actionIndex];
      const error = targetValue - prediction; // (y - Q(s,a))
      totalLoss += error * error;

      // dL/da er kun forskellig fra 0 for den handling, der faktisk blev
      // valgt - de øvrige Q-værdier havde ingen target-værdi denne omgang.
      const outputGradient = new Array(this.actions.length).fill(0);
      outputGradient[actionIndex] = -2 * error;

      this.mainNetwork.backward(outputGradient);
    }

    this.mainNetwork.applyGradients(this.learningRate, batch.length);

    this.trainStepCount++;
    this.#decayEpsilon();

    if (this.trainStepCount % this.targetSyncEvery === 0) {
      this.targetNetwork.copyFrom(this.mainNetwork);
    }

    return totalLoss / batch.length;
  }
}
