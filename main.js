const simCanvas = document.getElementById("simulation");
simCanvas.width = 200;

const simCtx = simCanvas.getContext("2d");
const road = new Road(simCanvas.width / 2, simCanvas.width * 0.9);

const car = new Car(road.getLaneCenter(1), 100, 30, 50);

let then = Date.now();

animate();

function animate(time) {
  // We want the simulation to run at 60hz, across all devices.
  // since requestAnimationFrame() runs on every frame,
  // a screen running 240 hz, will run this 240 times a second
  // which can throw off our timings. Therefore we limit to 60

  requestAnimationFrame(animate);

  const now = Date.now();
  elapsed = now - then;

  if (elapsed >= 1000 / 60) {
    then = now;
  } else {
    // We're running inbetween our 60hz, therefore return
    return;
  }

  simCanvas.height = window.innerHeight;

  // Update car position
  car.update();

  // Position camera acording to car
  simCtx.save();
  simCtx.translate(0, -car.y + simCanvas.height * 0.7);

  road.draw(simCtx);
  car.draw(simCtx, "blue");
}
