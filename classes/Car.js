class Car {
  constructor(x, y, width, height, maxSpeed = 3) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;

    this.speed = 0;
    this.acceleration = 0.2;
    this.maxSpeed = maxSpeed;
    this.friction = 0.05;
    this.angle = 0;
    this.crashed = false;

    this.sensor = new Sensor(this);

    this.controls = new Controls();
  }

  update(roadBorders) {
    // No reason to update, when crashed
    if (this.crashed) return;

    this.#move();
    this.polygon = this.#createPolygon();
    this.crashed = this.#assesDamage(roadBorders);

    if (this.sensor) {
      this.sensor.update(roadBorders);

      const offsets = this.sensor.readings.map((s) =>
        s == null ? 0 : 1 - s.offset,
      );
    }
  }

  #assesDamage(roadBorders) {
    for (let i = 0; i < roadBorders.length; i++) {
      if (polysIntersect(this.polygon, roadBorders[i])) return true;
    }
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
