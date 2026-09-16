class Car {
  constructor(x, y, width, height, maxSpeed = 3) {
    this.startX = x;
    this.startY = y;
    this.width = width;
    this.height = height;
    this.maxSpeed = maxSpeed;

    this.acceleration = 0.2;
    this.friction = 0.05;

    this.sensor = new Sensor(this);
    this.controls = new Controls();

    // NB: intet "brain" her længere - DQNAgent ejer netværket og
    // fortæller bilen hvad den skal gøre via applyAction().

    this.reset();
  }

  /**
   * Nulstiller bilen til udgangspositionen - kaldes ved starten af hver
   * episode (både den allerførste, og efter hvert crash).
   */
  reset() {
    this.x = this.startX;
    this.y = this.startY;
    this.speed = 0;
    this.angle = 0;
    this.crashed = false;
    this.polygon = this.#createPolygon();

    // Bruges til at beregne per-frame reward (fremdrift siden sidste frame)
    this.previousY = this.y;
  }

  /**
   * Sætter styresignalerne ud fra en handling valgt af agenten.
   * @param {{forward:boolean, left:boolean, right:boolean, reverse:boolean}} action
   */
  applyAction(action) {
    this.controls.forward = action.forward;
    this.controls.left = action.left;
    this.controls.right = action.right;
    this.controls.reverse = action.reverse;
  }

  /**
   * Kør simulationen ét tidsskridt frem. Returnerer IKKE state/reward selv -
   * det håndteres af trænings-loopet via getState()/getReward(), så
   * rækkefølgen (state -> action -> step -> reward) er tydelig ét sted.
   */
  update(roadBorders) {
    if (this.crashed) return;

    this.previousY = this.y;

    this.#move();
    this.polygon = this.#createPolygon();
    this.crashed = this.#assesDamage(roadBorders);

    this.sensor.update(roadBorders);
  }

  /**
   * State-vektor til netværket: 7 sensor-aflæsninger (0 = intet i sigte,
   * 1 = forhindring lige op ad bilen) + normaliseret hastighed.
   * @returns {number[]} længde 8
   */
  getState() {
    const sensorReadings = this.sensor.readings.map((reading) =>
      reading == null ? 0 : 1 - reading.offset,
    );
    const normalizedSpeed = this.speed / this.maxSpeed; // typisk i [-0.25, 1]
    return [...sensorReadings, normalizedSpeed];
  }

  /**
   * Reward for det frame der lige er kørt. Kald EFTER update().
   * v1: fremdrift (hvor langt op ad vejen bilen kom) minus stor straf ved crash.
   */
  getReward() {
    if (this.crashed) return -100;
    // Bilen bevæger sig i negativ y-retning når den kører fremad (angle=0),
    // så fremdrift = hvor meget y er faldet siden sidste frame.
    return this.previousY - this.y;
  }

  isDone() {
    return this.crashed;
  }

  #assesDamage(roadBorders) {
    for (let i = 0; i < roadBorders.length; i++) {
      if (polysIntersect(this.polygon, roadBorders[i])) return true;
    }
    return false;
  }

  #createPolygon() {
    const points = [];
    const rad = Math.hypot(this.width, this.height) / 2;
    const alpha = Math.atan2(this.width, this.height);

    points.push({
      x: this.x - Math.sin(this.angle - alpha) * rad,
      y: this.y - Math.cos(this.angle - alpha) * rad,
    });

    points.push({
      x: this.x - Math.sin(this.angle + alpha) * rad,
      y: this.y - Math.cos(this.angle + alpha) * rad,
    });

    points.push({
      x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad,
    });

    points.push({
      x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad,
    });

    return points;
  }

  #move() {
    if (this.controls.forward) this.speed += this.acceleration;
    if (this.controls.reverse) this.speed -= this.acceleration;

    if (this.speed > this.maxSpeed) this.speed = this.maxSpeed;
    if (this.speed < -this.maxSpeed / 4) this.speed = -this.maxSpeed / 4;

    // Turning is only possible whilst driving
    if (this.speed != 0) {
      if (this.controls.left) this.angle += 0.03;
      if (this.controls.right) this.angle -= 0.03;
    }

    if (this.speed > 0) this.speed -= this.friction;
    if (this.speed < 0) this.speed += this.friction;

    if (Math.abs(this.speed) < this.friction) this.speed = 0;

    this.x -= Math.sin(this.angle) * this.speed;
    this.y -= Math.cos(this.angle) * this.speed;
  }

  draw(ctx, color) {
    ctx.fillStyle = color;

    ctx.beginPath();
    ctx.moveTo(this.polygon[0].x, this.polygon[0].y);

    for (let i = 1; i < this.polygon.length; i++) {
      ctx.lineTo(this.polygon[i].x, this.polygon[i].y);
    }

    ctx.fill();

    if (this.sensor) {
      this.sensor.draw(ctx);
    }
  }
}
