const imageInput = document.getElementById('imageInput');
const imageCanvas = document.getElementById('imageCanvas');
const ctx = imageCanvas.getContext('2d');
let points = [];

imageInput.addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            imageCanvas.width = img.width;
            imageCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(file);
}

imageCanvas.addEventListener('click', handleCanvasClick);

function handleCanvasClick(event) {
    const rect = imageCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    points.push({ x, y });
    drawPoint(x, y);
    if (points.length === 2) {
        calculateRealDimensions();
    }
}

function drawPoint(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = 'red';
    ctx.fill();
}

function calculateRealDimensions() {
    const knownDimensions = parseFloat(document.getElementById('knownDimensions').value);
    const distance = parseFloat(document.getElementById('distance').value);
    const pixelDistance = Math.sqrt(Math.pow(points[1].x - points[0].x, 2) + Math.pow(points[1].y - points[0].y, 2));
    const focalLength = (pixelDistance * distance) / knownDimensions;
    const realDistance = (knownDimensions * focalLength) / pixelDistance;
    document.getElementById('result').innerText = `Real-world dimensions: ${realDistance} meters`;
}
