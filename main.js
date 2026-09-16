const simCanvas = document.getElementById("simulation");
simCanvas.width = 200;
const simCtx = simCanvas.getContext("2d");

const road = new Road(simCanvas.width / 2, simCanvas.width * 0.9);
const car = new Car(road.getLaneCenter(1), 100, 30, 50);
const stats = document.getElementById("stats");

// stateSize = 7 sensorer + 1 hastighed
const agent = new DQNAgent({ stateSize: 8, hiddenSize: 12 });

const RENDER_EVERY_N_EPISODES = 1000; // tegn kun indimellem - træning skal ikke vente på skærmen
const MAX_STEPS_PER_EPISODE = 2000; // sikkerhedsnet, hvis bilen aldrig crasher

let episodeCount = 0;
let stepInEpisode = 0;
let renderThisEpisode = false;
let episodeReward = 0;
let lastEpisodeReward = 0;
let lastEpisodeDistance = 0;
let lastEpisodeSteps = 0;
let latestLoss = null;

// Sensoren skal opdateres før state læses. Ellers indeholder den første
// transition kun hastigheden, hvilket giver NaN-vægte når den samples igen.
car.sensor.update(road.borders);
let currentState = car.getState();

function trainingStep() {
  const actionIndex = agent.chooseActionIndex(currentState);
  car.applyAction(agent.actions[actionIndex]);

  car.update(road.borders);

  const reward = car.getReward();
  const done = car.isDone() || stepInEpisode + 1 >= MAX_STEPS_PER_EPISODE;
  const nextState = car.getState();

  agent.remember(currentState, actionIndex, reward, nextState, done);
  const loss = agent.train();
  if (loss !== null) latestLoss = loss;

  currentState = nextState;
  stepInEpisode++;
  episodeReward += reward;

  if (done) {
    episodeCount++;
    lastEpisodeReward = episodeReward;
    lastEpisodeDistance = car.startY - car.y;
    lastEpisodeSteps = stepInEpisode;
    episodeReward = 0;
    stepInEpisode = 0;
    car.reset();
    car.sensor.update(road.borders);
    currentState = car.getState();
    renderThisEpisode = episodeCount % RENDER_EVERY_N_EPISODES === 0;

    if (episodeCount % 100 === 0) {
      console.log(`Episode ${episodeCount} slut - epsilon: ${agent.epsilon}`);
    }
  }

  return done;
}

function updateStats() {
  const loss =
    latestLoss === null
      ? "warming up"
      : Number.isFinite(latestLoss)
        ? latestLoss
        : "invalid";

  stats.textContent = [
    `episode: ${episodeCount}`,
    `epsilon: ${agent.epsilon}`,
    `last reward: ${lastEpisodeReward.toFixed(1)}`,
    `last distance: ${lastEpisodeDistance.toFixed(1)}`,
    `last steps: ${lastEpisodeSteps}`,
    `loss: ${loss}`,
  ].join("\n");
}

function trainingLoop() {
  // Kør mange steps per "tick" uden rendering - det er her du undgår at
  // være låst til 60 steps/sekund. Juster tallet efter hvor hurtig din
  // maskine er, og om browseren begynder at føles fastfrosset.
  const stepsThisTick = renderThisEpisode ? 1 : 200;

  for (let i = 0; i < stepsThisTick; i++) {
    if (trainingStep()) break;
  }

  if (renderThisEpisode) {
    simCanvas.height = window.innerHeight;
    simCtx.save();
    simCtx.translate(0, -car.y + simCanvas.height * 0.7);
    road.draw(simCtx);
    car.draw(simCtx, "blue");
    simCtx.restore();
  }

  updateStats();

  requestAnimationFrame(trainingLoop);
}

trainingLoop();
