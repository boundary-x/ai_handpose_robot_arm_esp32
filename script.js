import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// --- [설정] 하드웨어 및 통신 설정 ---
const UUID_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UUID_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"; 

// 스무딩 & 필터 설정
const SMOOTHING = 0.1; 
const FILTER_SIZE = 3; 
const MIN_CHANGE = 1.5; 

// --- [변수] ---
let handLandmarker = undefined;
let webcam = null;
let canvas, ctx;
let lastVideoTime = -1;
let results = undefined;

let bluetoothDevice, rxCharacteristic;
let isConnected = false;
let isSendingData = false;

let targetAngles = { b: 90, s: 90, e: 90, g: 0 };
let currentAngles = { b: 90, s: 90, e: 90 }; 
let lastSentAngles = { b: -999, s: -999, e: -999, g: -1 };
let angleQueue = { b: [], s: [], e: [] };

// DOM 요소
const modelStatus = document.getElementById("model-status");
const statusBt = document.getElementById("bt-status");
const packetLog = document.getElementById("packet-log");
const connectBtn = document.getElementById("connect-btn");
const disconnectBtn = document.getElementById("disconnect-btn");

const uiBars = { b: document.getElementById("bar-base"), s: document.getElementById("bar-shoulder"), e: document.getElementById("bar-elbow") };
const uiVals = { b: document.getElementById("val-base"), s: document.getElementById("val-shoulder"), e: document.getElementById("val-elbow"), g: document.getElementById("val-gripper") };

// [NEW] 매핑 설정 DOM 요소 가져오기
const configUI = {
    b: { min: document.getElementById("min-base"), max: document.getElementById("max-base"), rev: document.getElementById("rev-base") },
    s: { min: document.getElementById("min-shoulder"), max: document.getElementById("max-shoulder"), rev: document.getElementById("rev-shoulder") },
    e: { min: document.getElementById("min-elbow"), max: document.getElementById("max-elbow"), rev: document.getElementById("rev-elbow") }
};

// --- [1] AI 초기화 ---
async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
    runningMode: "VIDEO", numHands: 1 
  });
  modelStatus.innerText = "AI 모델 준비 완료";
  modelStatus.classList.add("ready");
  startWebcam();
}

// --- [2] 웹캠 ---
function startWebcam() {
  webcam = document.getElementById("webcam");
  canvas = document.getElementById("output_canvas");
  ctx = canvas.getContext("2d");
  const constraints = { video: { width: 1280, height: 720 } };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    webcam.srcObject = stream;
    webcam.addEventListener("loadeddata", predictWebcam);
  });
}

// --- [3] 메인 루프 ---
async function predictWebcam() {
  if (canvas.width !== webcam.videoWidth) { canvas.width = webcam.videoWidth; canvas.height = webcam.videoHeight; }
  let startTimeMs = performance.now();
  if (lastVideoTime !== webcam.currentTime) { lastVideoTime = webcam.currentTime; results = handLandmarker.detectForVideo(webcam, startTimeMs); }

  ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height); ctx.restore();

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    calculateRobotAngles(landmarks); 
    drawSkeleton(landmarks);         
  } else {
    // 손 없으면 원점 복귀
    targetAngles.b = 90; targetAngles.s = 90; targetAngles.e = 90; targetAngles.g = 0;
    angleQueue = { b: [], s: [], e: [] };
  }

  smoothMove(); updateUI(); sendPacket(); 
  window.requestAnimationFrame(predictWebcam);
}

// --- [4] 각도 계산 (UI 설정값 적용) ---
function calculateRobotAngles(lm) {
    // UI에서 현재 설정값 읽어오기
    // 체크박스가 켜져 있으면(Reverse), Min과 Max를 서로 바꿔서 매핑함
    
    // 1. Base (좌우) : 입력 x (0~1)
    let bMin = parseInt(configUI.b.min.value) || 0;
    let bMax = parseInt(configUI.b.max.value) || 180;
    // 반전 체크 시: 입력0 -> Max, 입력1 -> Min
    let bOutMin = configUI.b.rev.checked ? bMax : bMin;
    let bOutMax = configUI.b.rev.checked ? bMin : bMax;
    
    let x = 1 - lm[0].x; 
    let baseRaw = map(x, 0, 1, bOutMin, bOutMax);

    // 2. Shoulder (거리) : 입력 size (0.05~0.25)
    let sMin = parseInt(configUI.s.min.value) || 20;
    let sMax = parseInt(configUI.s.max.value) || 160;
    let sOutMin = configUI.s.rev.checked ? sMax : sMin;
    let sOutMax = configUI.s.rev.checked ? sMin : sMax;

    let size = getDistance(lm[0], lm[9]);
    let shoulderRaw = map(size, 0.05, 0.25, sOutMin, sOutMax);

    // 3. Elbow (상하) : 입력 y (0~1)
    let eMin = parseInt(configUI.e.min.value) || 20;
    let eMax = parseInt(configUI.e.max.value) || 160;
    let eOutMin = configUI.e.rev.checked ? eMax : eMin;
    let eOutMax = configUI.e.rev.checked ? eMin : eMax;

    let y = lm[0].y;
    let elbowRaw = map(y, 0, 1, eOutMin, eOutMax);

    // 4. 이동 평균 및 Gripper
    let baseAvg = getMovingAverage(angleQueue.b, baseRaw);
    let shoulderAvg = getMovingAverage(angleQueue.s, shoulderRaw);
    let elbowAvg = getMovingAverage(angleQueue.e, elbowRaw);

    let pinchDist = getDistance(lm[4], lm[8]);
    let gripState = (pinchDist < 0.05) ? 0 : 1; 

    // 최종 목표값
    targetAngles.b = constrain(baseAvg, 0, 180);
    targetAngles.s = constrain(shoulderAvg, 0, 180);
    targetAngles.e = constrain(elbowAvg, 0, 180);
    targetAngles.g = gripState;
}

function getMovingAverage(queue, newValue) {
    queue.push(newValue); 
    if (queue.length > FILTER_SIZE) queue.shift();
    return queue.reduce((a, b) => a + b, 0) / queue.length;
}

// --- [5] 유틸리티 ---
function smoothMove() {
    currentAngles.b += (targetAngles.b - currentAngles.b) * SMOOTHING;
    currentAngles.s += (targetAngles.s - currentAngles.s) * SMOOTHING;
    currentAngles.e += (targetAngles.e - currentAngles.e) * SMOOTHING;
}
function getDistance(p1, p2) { return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2)); }
function map(value, inMin, inMax, outMin, outMax) { return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin; }
function constrain(val, min, max) { return Math.min(Math.max(val, min), max); }

function drawSkeleton(lm) {
    const w = canvas.width; const h = canvas.height;
    ctx.fillStyle = "#00E676"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    [0, 4, 8, 9].forEach(i => { ctx.beginPath(); ctx.arc((1-lm[i].x)*w, lm[i].y*h, 6, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); });
    let tX = (1-lm[4].x)*w; let tY = lm[4].y*h; let iX = (1-lm[8].x)*w; let iY = lm[8].y*h;
    ctx.beginPath(); ctx.moveTo(tX, tY); ctx.lineTo(iX, iY);
    ctx.strokeStyle = targetAngles.g === 0 ? "#FF1744" : "#00E676"; ctx.lineWidth = 4; ctx.stroke();
}

function updateUI() {
    let b = Math.round(currentAngles.b); let s = Math.round(currentAngles.s); let e = Math.round(currentAngles.e);
    uiVals.b.innerText = `${b}°`; uiVals.s.innerText = `${s}°`; uiVals.e.innerText = `${e}°`;
    uiBars.b.style.width = `${(b/180)*100}%`; uiBars.s.style.width = `${(s/180)*100}%`; uiBars.e.style.width = `${(e/180)*100}%`;
    uiVals.g.innerText = targetAngles.g === 0 ? "CLOSE" : "OPEN"; uiVals.g.style.color = targetAngles.g === 0 ? "#FF1744" : "#00E676";
}

// --- [6] 통신 ---
async function sendPacket() {
    if (!isConnected || !rxCharacteristic || isSendingData) return;

    let b = Math.round(currentAngles.b);
    let s = Math.round(currentAngles.s);
    let e = Math.round(currentAngles.e);
    let g = targetAngles.g;

    let diffB = Math.abs(b - lastSentAngles.b);
    let diffS = Math.abs(s - lastSentAngles.s);
    let diffE = Math.abs(e - lastSentAngles.e);
    let diffG = Math.abs(g - lastSentAngles.g);

    if (diffB < MIN_CHANGE && diffS < MIN_CHANGE && diffE < MIN_CHANGE && diffG === 0) return;

    let packet = `B${String(b).padStart(3,'0')}S${String(s).padStart(3,'0')}E${String(e).padStart(3,'0')}G${String(g).padStart(3,'0')}`;
    packetLog.innerText = packet;

    try {
        isSendingData = true;
        const encoder = new TextEncoder();
        await rxCharacteristic.writeValue(encoder.encode(packet + "\r\n"));
        lastSentAngles = { b, s, e, g };
    } catch (err) { } finally { isSendingData = false; }
}

connectBtn.addEventListener('click', async () => {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({ filters: [{ namePrefix: "BBC micro:bit" }], optionalServices: [UUID_SERVICE] });
    bluetoothDevice.addEventListener('gattserverdisconnected', onDisc);
    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UUID_SERVICE);
    rxCharacteristic = await service.getCharacteristic(UUID_RX);
    isConnected = true; statusBt.innerText = "연결됨: " + bluetoothDevice.name; statusBt.classList.add("status-connected");
    connectBtn.classList.add("hidden"); disconnectBtn.classList.remove("hidden");
  } catch (error) { alert("연결 실패: " + error); }
});
function onDisc() { isConnected = false; statusBt.innerText = "연결 해제됨"; statusBt.classList.remove("status-connected"); connectBtn.classList.remove("hidden"); disconnectBtn.classList.add("hidden"); }
disconnectBtn.addEventListener('click', () => { if(bluetoothDevice && bluetoothDevice.gatt.connected) { bluetoothDevice.gatt.disconnect(); } });

createHandLandmarker();


