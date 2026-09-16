class ReplayBuffer {
  /**
   * @param {number} capacity - maks antal transitions bufferen kan indeholde,
   *   før den begynder at overskrive de ældste (FIFO)
   */
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = [];
    this.position = 0; // hvor i bufferen den NÆSTE overskrivning skal ske
  }

  /**
   * @param {{state:number[], actionIndex:number, reward:number, nextState:number[], done:boolean}} transition
   */
  add(transition) {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
    } else {
      // Bufferen er fuld - overskriv den ældste transition (cirkulær kø)
      this.buffer[this.position] = transition;
      this.position = (this.position + 1) % this.capacity;
    }
  }

  /**
   * Tilfældig sampling MED tilbagelægning - simplere at implementere korrekt
   * end uden, og gør intet reelt fra eller til, når bufferen er stor
   * (tusindvis af transitions) ift. en lille batch (32-64).
   */
  sample(batchSize) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const index = Math.floor(Math.random() * this.buffer.length);
      batch.push(this.buffer[index]);
    }
    return batch;
  }

  size() {
    return this.buffer.length;
  }
}
