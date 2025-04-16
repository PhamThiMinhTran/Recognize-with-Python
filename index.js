let videoStream;
let captureInterval;
let isCapturing = false;
let isDone = false;
let imagesBatch = [];
let lastStudentID = null;
let trainingStatusInterval = null;
let lastTrainingStatus = "";
let recognizeInterval = null;
let recognizedNames = new Set();

async function startCamera() {
    const videoElement = document.getElementById('video');
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = videoStream;
    } catch (err) {
        showSystemMessage("Không thể truy cập camera: " + err.message);
    }
}

async function callAPI(endpoint) {
    const studentID = document.getElementById('studentID').value.trim();
    if (endpoint === '/capture') {
        if (!studentID) {
            showSystemMessage("Vui lòng nhập tên sinh viên!");
            return;
        }
        if (studentID !== lastStudentID) {
            isDone = false;
            lastStudentID = studentID;
            document.getElementById('chatLog').innerHTML = '';
        }
        if (isCapturing) {
            showSystemMessage("Đang trong quá trình chụp ảnh xin chờ trong ít phút...");
            return;
        }
        if (isDone) {
            showSystemMessage("Quá trình chụp ảnh đã hoàn tất.");
            return;
        }
        showSystemMessage("Đang thực hiện chụp ảnh và xử lý, bạn chờ hệ thống thêm chút thời gian nhé...");
        isCapturing = true;
        captureInterval = setInterval(async () => {
            const base64Image = await captureImageAsBase64();
            imagesBatch.push(base64Image);
            if (imagesBatch.length > 5) {
                fetch('http://127.0.0.1:5000/api/capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        images: imagesBatch,
                        student_id: studentID
                    })
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            showSystemMessage(data.message);
                        } else if (data.error) {
                            showSystemMessage("Lỗi: " + data.error);
                        }
                        if (data.done) {
                            clearInterval(captureInterval);
                            isCapturing = false;
                            isDone = true;
                            showSystemMessage("Đã hoàn tất lưu ảnh cho sinh viên!");
                        }
                    })
                    .catch(error => {
                        showSystemMessage("Lỗi khi gửi ảnh: " + error.message);
                        clearInterval(captureInterval);
                        isCapturing = false;
                    });
                imagesBatch = [];
            }
        }, 300);
    } else if (endpoint == "/training") {
        fetch('http://127.0.0.1:5000/api/training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_training: true })
        })
            .then(res => res.json())
            .then(data => {
                if (data.message) {
                    showSystemMessage(data.message);
                    startTrainingStatusPolling();
                } else {
                    showSystemMessage("Lỗi: " + (data.error || "Lỗi không rõ."));
                }
            })
            .catch(error => {
                showSystemMessage("Lỗi API: " + error.message);
            });
    }
}

function showSystemMessage(msg) {
    const chatLog = document.getElementById('chatLog');
    chatLog.innerHTML += `<div class="msg system">${msg}</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function captureImageAsBase64() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const width = 640;
    const height = 480;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, width, height);
    return canvas.toDataURL('image/jpeg', 0.9);
}

function sendChat() {
    const msg = document.getElementById('chatBox').value;
    if (msg) {
        const chatLog = document.getElementById('chatLog');
        chatLog.innerHTML += `<div class="msg user">${msg}</div>`;
        document.getElementById('chatBox').value = '';
    }
}

function startTrainingStatusPolling() {
    if (trainingStatusInterval) clearInterval(trainingStatusInterval);
    trainingStatusInterval = setInterval(() => {
        fetch('http://127.0.0.1:5000/api/training_status')
            .then(res => res.json())
            .then(data => {
                const status = data.training_status;
                if (status !== lastTrainingStatus) {
                    showSystemMessage("Chú ý: " + status);
                    lastTrainingStatus = status;

                    // Nếu training đã hoàn tất hoặc lỗi, ngưng polling
                    if (status.startsWith("Hoàn tất") || status.startsWith("Lỗi")) {
                        clearInterval(trainingStatusInterval);
                        trainingStatusInterval = null;
                    }
                }
            })
            .catch(err => {
                showSystemMessage("Lỗi khi kiểm tra trạng thái training: " + err.message);
                clearInterval(trainingStatusInterval);
                trainingStatusInterval = null;
            });
    }, 3000);
}

function startRecognizing() {
    if (recognizeInterval) return;
    showSystemMessage("Bắt đầu nhận diện khuôn mặt...");
    recognizeInterval = setInterval(async () => {
        const base64Image = await captureImageAsBase64();
        fetch('http://127.0.0.1:5000/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        })
            .then(res => res.json())
            .then(data => {
                const recognizedResults = data.results?.results;
                if (recognizedResults && recognizedResults.length > 0) {
                    recognizedResults.forEach(result => {
                        const label = result.label;
                        if (label !== "Unknown" && label !== "Processing" && !recognizedNames.has(label)) {
                            recognizedNames.add(label);
                            showSystemMessage(`Nhận diện: ${label}`);
                        }
                    });
                } else {
                    showSystemMessage(data.message || "Không nhận diện được khuôn mặt.");
                }
            })
            .catch(err => {
                showSystemMessage("Lỗi nhận diện: " + err.message);
            });
    }, 1500);
}

function stopRecognizing() {
    clearInterval(recognizeInterval);
    recognizeInterval = null;
    recognizedNames.clear();
    showSystemMessage("Đã dừng nhận diện.");
}

function exportData(endpoint) {
    showSystemMessage("Đang tải dữ liệu điểm danh...");
    fetch('http://127.0.0.1:5000/api/download_attendance')
        .then(response => {
            if (!response.ok) throw new Error("Không thể tải file.");
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'attendance.xlsx';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showSystemMessage("Đã tải xuống file attendance.xlsx");
        })
        .catch(err => showSystemMessage("Lỗi khi tải file: " + err.message));
}

window.onload = startCamera;