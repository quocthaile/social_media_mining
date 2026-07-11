document.addEventListener("DOMContentLoaded", () => {
    // Navigation Tabs
    const tabs = document.querySelectorAll(".nav-tab");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            document.getElementById(`pane-${target}`).classList.add("active");
        });
    });

    // Model Selector & Stats Loading
    const modelBadge = document.getElementById("model-name-badge");
    const statsModel = document.getElementById("stats-model-name");
    const statsEpochs = document.getElementById("stats-epochs");
    const statsMaxLen = document.getElementById("stats-max-len");
    const statsLr = document.getElementById("stats-lr");
    
    const modelDropdown = document.getElementById("model-dropdown");
    const dropdownTrigger = document.getElementById("dropdown-trigger");
    const dropdownSelectedText = document.getElementById("dropdown-selected-text");
    const dropdownOptionsList = document.getElementById("dropdown-options-list");

    const statsOverlay = document.getElementById("stats-loading-overlay");

    function loadModelStats() {
        if (statsOverlay) {
            statsOverlay.classList.remove("hidden");
        }
        return fetch("/api/stats")
            .then(res => res.json())
            .then(data => {
                const shortName = data.model_name.split("/").pop();
                modelBadge.textContent = `${shortName} (${data.model_type.toUpperCase()})`;
                statsModel.textContent = data.model_name;
                statsModel.title = data.model_name;
                statsEpochs.textContent = data.epochs;
                statsMaxLen.textContent = `${data.max_len} tokens`;
                statsLr.textContent = data.lr;

                // Cập nhật chỉ số đánh giá động
                if (data.metrics) {
                    // Hỗ trợ cả cấu trúc lồng nhau (accuracy_optimized/f1_optimized) lẫn cấu trúc phẳng
                    let metricsObj = data.metrics;
                    if (data.metrics.accuracy_optimized) {
                        metricsObj = data.metrics.accuracy_optimized;
                    }

                    const accValue = metricsObj.accuracy || 0;
                    const f1Value = metricsObj.f1_macro || 0;
                    const cm = metricsObj.confusion_matrix;

                    document.getElementById("stats-accuracy").textContent = `${(accValue * 100).toFixed(2)}%`;
                    document.getElementById("stats-f1").textContent = `${(f1Value * 100).toFixed(2)}%`;
                    
                    if (cm && cm.length === 3) {
                        for (let r = 0; r < 3; r++) {
                            const rowTotal = cm[r].reduce((sum, val) => sum + val, 0) || 1;
                            for (let c = 0; c < 3; c++) {
                                const cell = document.getElementById(`cm-${r}-${c}`);
                                cell.textContent = cm[r][c];
                                cell.classList.add("heatmap-cell");
                                
                                const ratio = cm[r][c] / rowTotal;
                                if (r === c) {
                                    // Correct predictions (diagonal) -> Turquoise glow
                                    cell.style.backgroundColor = `rgba(0, 242, 254, ${Math.max(0.06, ratio * 0.45)})`;
                                } else {
                                    // Incorrect predictions (misses) -> Red/rose glow
                                    cell.style.backgroundColor = `rgba(244, 63, 94, ${Math.max(0, ratio * 0.45)})`;
                                }
                            }
                        }
                    }
                } else {
                    document.getElementById("stats-accuracy").textContent = "N/A";
                    document.getElementById("stats-f1").textContent = "N/A";
                    for (let r = 0; r < 3; r++) {
                        for (let c = 0; c < 3; c++) {
                            const cell = document.getElementById(`cm-${r}-${c}`);
                            cell.textContent = "-";
                            cell.style.backgroundColor = "";
                        }
                    }
                }
            })
            .catch(err => {
                console.error("Lỗi load stats:", err);
                modelBadge.textContent = "Offline";
            })
            .finally(() => {
                if (statsOverlay) {
                    statsOverlay.classList.add("hidden");
                }
            });
    }

    let selectedModelKey = "";

    function loadAvailableModels() {
        fetch("/api/models")
            .then(res => res.json())
            .then(data => {
                dropdownOptionsList.innerHTML = "";
                if (data.models.length === 0) {
                    dropdownSelectedText.textContent = "Không tìm thấy model nào";
                    return;
                }

                data.models.forEach(m => {
                    const div = document.createElement("div");
                    div.className = "dropdown-option";
                    div.dataset.value = m.key;
                    div.textContent = m.name;
                    
                    if (m.key === data.active_model) {
                        div.classList.add("selected");
                        dropdownSelectedText.textContent = m.name;
                        selectedModelKey = m.key;
                    }

                    div.addEventListener("click", () => {
                        selectModel(m.key, m.name);
                    });

                    dropdownOptionsList.appendChild(div);
                });
            })
            .catch(err => {
                console.error("Lỗi tải danh sách model:", err);
            });
    }

    // Toggle dropdown open
    dropdownTrigger.addEventListener("click", (e) => {
        e.stopPropagation();
        if (modelDropdown.classList.contains("disabled")) return;
        modelDropdown.classList.toggle("open");
    });

    // Close dropdown on click outside
    window.addEventListener("click", () => {
        modelDropdown.classList.remove("open");
    });

    function selectModel(modelKey, modelName) {
        if (modelKey === selectedModelKey) return;
        
        modelDropdown.classList.add("disabled");
        modelDropdown.classList.remove("open");
        modelBadge.textContent = "Đang chuyển model...";
        dropdownSelectedText.textContent = "Đang tải mô hình...";

        fetch("/api/select_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_key: modelKey })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            selectedModelKey = modelKey;
            
            // Cập nhật selected option trong list UI
            document.querySelectorAll(".dropdown-option").forEach(opt => {
                if (opt.dataset.value === modelKey) {
                    opt.classList.add("selected");
                    dropdownSelectedText.textContent = opt.textContent;
                } else {
                    opt.classList.remove("selected");
                }
            });

            loadModelStats().finally(() => {
                modelDropdown.classList.remove("disabled");
            });
        })
        .catch(err => {
            console.error(err);
            alert("Lỗi kết nối khi chuyển mô hình!");
            modelDropdown.classList.remove("disabled");
            loadModelStats();
        });
    }

    // Khởi tạo
    loadAvailableModels();
    loadModelStats();

    // -------------------------------------------------------------
    // CHỨC NĂNG PHÂN TÍCH ĐƠN LẺ (SINGLE COMMENT)
    // -------------------------------------------------------------
    const textarea = document.getElementById("comment-textarea");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnClear = document.getElementById("btn-clear");
    const resultPanel = document.getElementById("single-result-panel");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");

    let confidenceChart = null;

    // Character Counter
    const charCount = document.getElementById("char-count");
    function updateCharCount() {
        const len = textarea.value.length;
        charCount.textContent = len;
        if (len >= 450) {
            charCount.style.color = "var(--color-hate)";
        } else if (len >= 350) {
            charCount.style.color = "var(--color-offensive)";
        } else {
            charCount.style.color = "var(--text-secondary)";
        }
    }
    textarea.addEventListener("input", updateCharCount);

    // Quick suggestions
    document.querySelectorAll(".example-tag").forEach(tag => {
        tag.addEventListener("click", () => {
            textarea.value = tag.dataset.text;
            updateCharCount();
            btnAnalyze.click();
        });
    });

    btnAnalyze.addEventListener("click", () => {
        const text = textarea.value.trim();
        if (!text) return;

        // Trạng thái Loading
        btnAnalyze.disabled = true;
        btnText.textContent = "Đang quét...";
        btnSpinner.classList.remove("hidden");

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Hiện panel kết quả
            resultPanel.classList.remove("empty");
            resultPanel.querySelector(".empty-state").classList.add("hidden");
            resultPanel.querySelector(".result-content").classList.remove("hidden");

            // Cập nhật banner phán quyết
            const banner = document.getElementById("verdict-banner");
            const val = document.getElementById("verdict-value");
            const icon = document.getElementById("verdict-icon");

            banner.className = "verdict-banner";
            if (data.label_id === 0) {
                banner.classList.add("verdict-clean");
                val.textContent = "SẠCH (CLEAN)";
                icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            } else if (data.label_id === 1) {
                banner.classList.add("verdict-offensive");
                val.textContent = "XÚC PHẠM (OFFENSIVE)";
                icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            } else {
                banner.classList.add("verdict-hate");
                val.textContent = "THÙ ĐỊCH (HATE SPEECH)";
                icon.innerHTML = '<i class="fa-solid fa-hand-holding-hand"></i>';
            }

            // Đánh dấu từ độc hại
            document.getElementById("highlighted-text-p").innerHTML = data.highlighted_text;

            // Render detected bad words tags
            const detectedWrapper = document.getElementById("detected-words-wrapper");
            const detectedTags = document.getElementById("detected-tags");
            detectedTags.innerHTML = "";
            if (data.detected_bad_words && data.detected_bad_words.length > 0) {
                detectedWrapper.classList.remove("hidden");
                data.detected_bad_words.forEach(word => {
                    const span = document.createElement("span");
                    span.className = "detected-tag";
                    span.textContent = word;
                    detectedTags.appendChild(span);
                });
            } else {
                detectedWrapper.classList.add("hidden");
            }

            // In biểu đồ độ tin cậy
            renderConfidenceChart(data.probs);

            // Cập nhật Meta-features
            renderMetaMetrics(data.meta_features);
        })
        .catch(err => {
            console.error(err);
            alert("Đã xảy ra lỗi khi phân tích bình luận!");
        })
        .finally(() => {
            btnAnalyze.disabled = false;
            btnText.textContent = "Phân tích";
            btnSpinner.classList.add("hidden");
        });
    });

    btnClear.addEventListener("click", () => {
        textarea.value = "";
        updateCharCount();
        resultPanel.classList.add("empty");
        resultPanel.querySelector(".empty-state").classList.remove("hidden");
        resultPanel.querySelector(".result-content").classList.add("hidden");
        document.getElementById("detected-words-wrapper").classList.add("hidden");
    });

    function renderConfidenceChart(probs) {
        const ctx = document.getElementById("confidence-chart").getContext("2d");
        const chartData = [probs.clean * 100, probs.offensive * 100, probs.hate * 100];

        if (confidenceChart) {
            confidenceChart.destroy();
        }

        confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Sạch', 'Xúc phạm', 'Thù địch'],
                datasets: [{
                    label: 'Độ tin cậy (%)',
                    data: chartData,
                    backgroundColor: [
                        'rgba(0, 230, 118, 0.45)',
                        'rgba(255, 234, 0, 0.45)',
                        'rgba(255, 23, 68, 0.45)'
                    ],
                    borderColor: [
                        '#00e676',
                        '#ffea00',
                        '#ff1744'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9aa5bf' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f1f3f9', font: { weight: 'bold' } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function renderMetaMetrics(meta) {
        const grid = document.getElementById("metrics-grid");
        grid.innerHTML = "";

        // Danh sách các meta key muốn hiển thị trực quan
        const displayKeys = {
            "feat_bad_word_density": "Mật độ từ tục",
            "feat_emoji_density": "Mật độ Emoji",
            "feat_exclamation_density": "Mật độ kích động (!/?)",
            "feat_allcaps_ratio": "Tỷ lệ viết hoa (la hét)",
        };

        Object.entries(displayKeys).forEach(([key, name]) => {
            const val = meta[key] || 0;
            // Chuyển đổi sang % để vẽ thanh bar
            const percentage = Math.min(val * 100, 100).toFixed(1);

            const card = document.createElement("div");
            card.className = "metric-bar";
            card.innerHTML = `
                <div class="metric-header">
                    <span>${name}</span>
                    <strong>${percentage}%</strong>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            grid.appendChild(card);
        });
    }

    // -------------------------------------------------------------
    // CHỨC NĂNG XỬ LÝ HÀNG LOẠT (BATCH CSV PROCESS)
    // -------------------------------------------------------------
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("csv-file-input");
    const uploadActions = document.getElementById("upload-actions");
    const selectedFileName = document.getElementById("selected-file-name");
    const btnBatchProcess = document.getElementById("btn-batch-process");
    const batchResultPanel = document.getElementById("batch-result-panel");
    const downloadLink = document.getElementById("download-link");
    const previewTableBody = document.querySelector("#preview-table tbody");

    let batchChart = null;

    // Kéo thả files
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    ["dragleave", "drop"].forEach(event => {
        dropZone.addEventListener(event, () => dropZone.classList.remove("dragover"));
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleSelectedFile(fileInput.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.name.endsWith(".csv")) {
            alert("Vui lòng chọn file định dạng .csv");
            return;
        }
        selectedFileName.textContent = file.name;
        uploadActions.classList.remove("hidden");
    }

    btnBatchProcess.addEventListener("click", () => {
        const file = fileInput.files[0];
        if (!file) return;

        btnBatchProcess.disabled = true;
        btnBatchProcess.textContent = "Đang xử lý...";

        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/predict_batch", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Hiện panel kết quả batch
            batchResultPanel.classList.remove("empty");
            batchResultPanel.querySelector(".empty-state").classList.add("hidden");
            batchResultPanel.querySelector(".result-content").classList.remove("hidden");

            // Thiết lập link tải xuống
            downloadLink.href = data.download_url;

            // Cập nhật các thẻ số liệu thống kê
            const total = data.results.length;
            let clean = 0, offensive = 0, hate = 0;
            data.results.forEach(r => {
                if (r.label_id === 0) clean++;
                else if (r.label_id === 1) offensive++;
                else if (r.label_id === 2) hate++;
            });
            
            document.getElementById("batch-total").textContent = total;
            document.getElementById("batch-clean").textContent = `${clean} (${(clean / total * 100).toFixed(1)}%)`;
            document.getElementById("batch-offensive").textContent = `${offensive} (${(offensive / total * 100).toFixed(1)}%)`;
            document.getElementById("batch-hate").textContent = `${hate} (${(hate / total * 100).toFixed(1)}%)`;

            // Render biểu đồ phân phối tròn (Donut chart)
            renderBatchDonutChart(data.results);

            // Hiện bản xem trước 10 dòng đầu
            renderPreviewTable(data.results.slice(0, 10));
        })
        .catch(err => {
            console.error(err);
            alert("Lỗi khi gửi tệp đi phân tích!");
        })
        .finally(() => {
            btnBatchProcess.disabled = false;
            btnBatchProcess.textContent = "Bắt đầu xử lý";
        });
    });

    function renderBatchDonutChart(results) {
        let clean = 0, offensive = 0, hate = 0;
        results.forEach(r => {
            if (r.label_id === 0) clean++;
            else if (r.label_id === 1) offensive++;
            else if (r.label_id === 2) hate++;
        });

        const ctx = document.getElementById("batch-distribution-chart").getContext("2d");

        if (batchChart) {
            batchChart.destroy();
        }

        batchChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Sạch', 'Xúc phạm', 'Thù địch'],
                datasets: [{
                    data: [clean, offensive, hate],
                    backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                cutout: '70%'
            }
        });
    }

    function renderPreviewTable(items) {
        previewTableBody.innerHTML = "";
        items.forEach(item => {
            const tr = document.createElement("tr");

            // Xác định class badge cho label
            let badgeClass = "badge-success";
            if (item.label_id === 1) badgeClass = "badge-warning";
            else if (item.label_id === 2) badgeClass = "badge-danger";

            // Tìm độ tự tin cao nhất
            const maxProb = Math.max(item.prob_clean, item.prob_offensive, item.prob_hate);
            const confidence = (maxProb * 100).toFixed(1);

            // Cắt ngắn text preview nếu quá dài
            const display_text = item.text.length > 70 ? item.text.substring(0, 70) + "..." : item.text;

            tr.innerHTML = `
                <td>${display_text}</td>
                <td><span class="badge ${badgeClass}">${item.label_name}</span></td>
                <td><strong>${confidence}%</strong></td>
            `;
            previewTableBody.appendChild(tr);
        });
    }
});
