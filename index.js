let videoStream = null;
let captureInterval;
let isCapturing = false;
let isDone = false;
let imagesBatch = [];
let lastStudentID = null;
let trainingStatusInterval = null;
let lastTrainingStatus = "";
let recognizeInterval = null;
let videoTrack = null;
let zoomSlider = null;
let zoomValueDisplay = null;
let recognizedNames = new Set();

async function listCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === "videoinput");

    const select = document.getElementById("cameraSelect");
    select.innerHTML = ""; // clear trước

    videoDevices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index}`;
        select.appendChild(option);
    });
}

function switchCamera() {
    const select = document.getElementById("cameraSelect");
    const deviceId = select.value;
    startCamera(deviceId);
}

async function startCamera(deviceId = null) {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    const constraints = {
        video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            deviceId: deviceId ? { exact: deviceId } : undefined
        }
    };
    // try {
    //     videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    //     const video = document.getElementById("video");
    //     video.srcObject = videoStream;
    //     const videoTrack = videoStream.getVideoTracks()[0];
    //     const capabilities = videoTrack.getCapabilities();

    //     if (capabilities.zoom) {
    //         const zoomLevel = capabilities.zoom.max;
    //         await videoTrack.applyConstraints({
    //             advanced: [{ zoom: zoomLevel }]
    //         });
    //         console.log("Zoom được hỗ trợ");
    //     } else {
    //         console.warn("Camera không hỗ trợ zoom.");
    //     }
    // } catch (err) {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById("video");
        video.srcObject = videoStream;

        videoTrack = videoStream.getVideoTracks()[0];
        const capabilities = videoTrack.getCapabilities();

        zoomSlider = document.getElementById('zoomSlider');
        zoomValueDisplay = document.getElementById('zoomValue');

        if (capabilities.zoom) {
            const zoomMin = capabilities.zoom.min;
            const zoomMax = capabilities.zoom.max;
            const zoomStep = capabilities.zoom.step || 0.1;
            const zoomDefault = (zoomMin + zoomMax) / 2;

            zoomSlider.min = zoomMin;
            zoomSlider.max = zoomMax;
            zoomSlider.step = zoomStep;
            zoomSlider.value = zoomDefault.toFixed(2);

            // Áp dụng zoom mặc định
            await videoTrack.applyConstraints({
                advanced: [{ zoom: zoomDefault }]
            });
            zoomValueDisplay.textContent = `${parseFloat(zoomSlider.value).toFixed(1)}x`;

            zoomSlider.oninput = async () => {
                const zoomLevel = parseFloat(zoomSlider.value);
                await videoTrack.applyConstraints({
                    advanced: [{ zoom: zoomLevel }]
                });
                zoomValueDisplay.textContent = `${zoomLevel.toFixed(1)}x`;
            };

            zoomSlider.disabled = false;
            document.getElementById('zoomControl').style.display = 'block';
            console.log("Zoom được hỗ trợ");
        } else {
            // Nếu không hỗ trợ zoom thì ẩn slider
            zoomSlider.disabled = true;
            document.getElementById('zoomControl').style.display = 'none';
            console.warn("Camera không hỗ trợ zoom.");
        }
    } catch (err) {
        showSystemMessage("Không thể bật camera: " + err.message);
    }
}

async function captureImageAsBase64() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const width = video.videoWidth;
    const height = video.videoHeight;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, width, height);
    return canvas.toDataURL('image/jpeg', 0.9);
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
                const recognizedResults = data.results;
                if (recognizedResults && recognizedResults.length > 0) {
                    recognizedResults.forEach(result => {
                        const label = result.label;
                        const prob = (result.probability * 100).toFixed(1);
                        const displayText = (label !== "Unknown")
                            ? `${label} (${prob}%)`
                            : `Unknown (${prob}%)`;

                        if (!recognizedNames.has(displayText)) {
                            recognizedNames.add(displayText);
                            showSystemMessage(`Nhận diện: ${displayText}`);
                        }
                    });
                } else {
                    showSystemMessage(data.message || "Không nhận diện được khuôn mặt.");
                }
            })
            .catch(err => {
                showSystemMessage("Lỗi nhận diện: " + err.message);
            });
    }, 3000);
}

function stopRecognizing() {
    clearInterval(recognizeInterval);
    recognizeInterval = null;
    recognizedNames.clear();
    showSystemMessage("Đã dừng nhận diện.");
}

function exportData() {
    showSystemMessage("Đang lấy dữ liệu điểm danh tạm...");
    fetch('http://127.0.0.1:5000/api/preview_attendance')
        .then(res => {
            if (!res.ok) throw new Error("Không có dữ liệu để xuất.");
            return res.json();
        })
        .then(data => {
            renderPreviewModal(data);

            const modalEl = document.getElementById('attendancePreviewModal');
            const modalInstance = bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl);
            if (!modalEl.classList.contains('show')) {
                modalInstance.show();
            }
        })
        .catch(err => {
            showSystemMessage("Lỗi: " + err.message);
        });
}

function deleteStudent(name) {
    const safeName = name.replace(/'/g, "\\'");
    if (!confirm(`Bạn có chắc muốn xoá "${name}" khỏi danh sách điểm danh?`)) return;
    showSystemMessage(`Đang xoá "${name}"...`);

    fetch('http://127.0.0.1:5000/api/remove_attendance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names: [name] })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            showSystemMessage(data.message);
            exportData();
        })
        .catch(err => {
            showSystemMessage("Lỗi khi xoá sinh viên: " + err.message);
        });
}

function renderPreviewModal(data) {
    const container = document.getElementById("attendancePreviewList");
    if (!Array.isArray(data) || data.length === 0) {
        container.innerHTML = "<p class='text-danger'>Danh sách điểm danh trống.</p>";
        return;
    }

    const keys = Object.keys(data[0]);
    let html = `<table class="table table-bordered table-sm"><thead><tr>`;
    keys.forEach(k => html += `<th>${k}</th>`);
    html += `<th>Hành động</th></tr></thead><tbody>`;

    data.forEach(row => {
        html += "<tr>";
        keys.forEach(k => html += `<td>${sanitizeHTML(row[k])}</td>`);
        const safeNameAttr = encodeURIComponent(row['Name']);
        html += `<td><button class="btn btn-danger btn-sm" onclick="deleteStudent(decodeURIComponent('${safeNameAttr}'))">Xoá</button></td>`;
        html += "</tr>";
    });

    html += "</tbody></table>";
    container.innerHTML = html;
}

function confirmSaveAndDownload() {
    showSystemMessage("Đang lưu dữ liệu...");
    fetch('http://127.0.0.1:5000/api/save_attendance', {
        method: 'POST'
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            showSystemMessage(data.message);
            return fetch('http://127.0.0.1:5000/api/download_attendance');
        })
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

            const modal = bootstrap.Modal.getInstance(document.getElementById('attendancePreviewModal'));
            modal.hide();
        })
        .catch(err => showSystemMessage("Lỗi: " + err.message));
}

function appendMessage(content, type = 'system') {
    const chatLog = document.getElementById('chatLog');
    chatLog.innerHTML += `<div class="msg ${type}">${content}</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;
}

function chatBot() {
    const input = document.getElementById('chatBox');
    const message = input.value.trim();
    if (!message) return;
    appendMessage(message, 'user');
    input.value = '';
    fetch('http://localhost:5000/api/chatBot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    })
        .then(res => res.json())
        .then(data => {
            if (data.response) {
                appendMessage(data.response, 'bot');
            } else {
                appendMessage("Chatbot không phản hồi đúng định dạng.", 'system');
            }
        })
        .catch(error => {
            appendMessage("Lỗi kết nối chatbot: " + error.message, 'system');
        });
}

function sanitizeHTML(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

window.onload = async () => {
    await listCameras();
    const select = document.getElementById("cameraSelect");
    if (select.options.length > 0) {
        startCamera(select.options[0].value); // mở camera đầu tiên
    }
};