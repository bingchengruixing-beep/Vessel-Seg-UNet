(function () {
    // State
    const state = {
        isTraining: false,
        pollIntervalId: null,
        charts: {
            loss: null,
            metrics: null
        },
        inferenceImageBase64: null,
        inferenceFiles: [],
        config: null
    };

    // DOM Elements
    const els = {
        // Header
        sysStatusBadge: document.getElementById('system-status-badge'),
        trainingIndicator: document.getElementById('training-indicator'),
        trainingIndicatorText: document.getElementById('training-indicator-text'),
        
        // Tabs
        tabBtns: document.querySelectorAll('.tab-btn[data-tab]'),
        tabContents: document.querySelectorAll('.tab-content'),
        
        // Dashboard
        metricEpoch: document.getElementById('metric-epoch'),
        metricDice: document.getElementById('metric-dice'),
        metricTrainLoss: document.getElementById('metric-train-loss'),
        metricValLoss: document.getElementById('metric-val-loss'),
        chartLoss: document.getElementById('chart-loss'),
        chartMetrics: document.getElementById('chart-metrics'),
        btnStartTraining: document.getElementById('btn-start-training'),
        btnStopTraining: document.getElementById('btn-stop-training'),
        currentLr: document.getElementById('current-lr'),
        epochProgress: document.getElementById('epoch-progress'),
        epochProgressText: document.getElementById('epoch-progress-text'),
        trainingLog: document.getElementById('training-log'),
        
        // Config
        cfgTrainImageDir: document.getElementById('cfg-train-image-dir'),
        cfgTrainMaskDir: document.getElementById('cfg-train-mask-dir'),
        cfgValImageDir: document.getElementById('cfg-val-image-dir'),
        cfgValMaskDir: document.getElementById('cfg-val-mask-dir'),
        cfgImgSize: document.getElementById('cfg-img-size'),
        cfgNumWorkers: document.getElementById('cfg-num-workers'),
        cfgElasticTransform: document.getElementById('cfg-elastic-transform'),
        cfgModelName: document.getElementById('cfg-model-name'),
        cfgEncoderName: document.getElementById('cfg-encoder-name'),
        cfgEncoderGroup: document.getElementById('cfg-encoder-group'),
        cfgInChannels: document.getElementById('cfg-in-channels'),
        cfgOutChannels: document.getElementById('cfg-out-channels'),
        cfgPretrained: document.getElementById('cfg-pretrained'),
        cfgDeepSupervision: document.getElementById('cfg-deep-supervision'),
        cfgDeepSupervisionGroup: document.getElementById('cfg-deep-supervision-group'),
        cfgBatchSize: document.getElementById('cfg-batch-size'),
        cfgLearningRate: document.getElementById('cfg-learning-rate'),
        cfgWeightDecay: document.getElementById('cfg-weight-decay'),
        cfgEpochs: document.getElementById('cfg-epochs'),
        cfgOptimizer: document.getElementById('cfg-optimizer'),
        cfgScheduler: document.getElementById('cfg-scheduler'),
        cfgUseAmp: document.getElementById('cfg-use-amp'),
        cfgBceWeight: document.getElementById('cfg-bce-weight'),
        cfgDiceWeight: document.getElementById('cfg-dice-weight'),
        cfgBceWeightValue: document.getElementById('cfg-bce-weight-value'),
        cfgDiceWeightValue: document.getElementById('cfg-dice-weight-value'),
        cfgClDiceGroup: document.getElementById('cfg-cldice-group'),
        cfgClDiceWeight: document.getElementById('cfg-cldice-weight'),
        cfgClDiceWeightValue: document.getElementById('cfg-cldice-weight-value'),
        cfgLossWeightHint: document.getElementById('cfg-loss-weight-hint'),
        cfgPatience: document.getElementById('cfg-patience'),
        cfgSaveDir: document.getElementById('cfg-save-dir'),
        cfgSaveBestOnly: document.getElementById('cfg-save-best-only'),
        btnSaveConfig: document.getElementById('btn-save-config'),
        btnResetConfig: document.getElementById('btn-reset-config'),
        
        // Inference
        infDropzone: document.getElementById('inference-dropzone'),
        infFileInput: document.getElementById('inference-file-input'),
        infOriginal: document.getElementById('inference-original'),
        infResult: document.getElementById('inference-result'),
        infCheckpoint: document.getElementById('inference-checkpoint'),
        infThreshold: document.getElementById('inference-threshold'),
        infThresholdValue: document.getElementById('inference-threshold-value'),
        infProcessing: document.getElementById('inference-processing'),
        infMinComponent: document.getElementById('inference-min-component'),
        infMaxHole: document.getElementById('inference-max-hole'),
        infMorphKernel: document.getElementById('inference-morph-kernel'),
        btnRunInference: document.getElementById('btn-run-inference'),
        thresholdScanValues: document.getElementById('threshold-scan-values'),
        btnThresholdScan: document.getElementById('btn-threshold-scan'),
        thresholdScanStatus: document.getElementById('threshold-scan-status'),
        thresholdScanBest: document.getElementById('threshold-scan-best'),
        thresholdScanTableWrap: document.getElementById('threshold-scan-table-wrap'),
        thresholdScanTableBody: document.getElementById('threshold-scan-table-body'),
        batchInferenceResults: document.getElementById('batch-inference-results'),
        batchInferenceGrid: document.getElementById('batch-inference-grid'),
        
        // System
        sysGpuName: document.getElementById('sys-gpu-name'),
        sysGpuMemory: document.getElementById('sys-gpu-memory'),
        sysPython: document.getElementById('sys-python'),
        sysTorch: document.getElementById('sys-torch'),
        checkpointTableBody: document.getElementById('checkpoint-table-body'),
        dsTrainCount: document.getElementById('dataset-train-count'),
        dsValCount: document.getElementById('dataset-val-count'),
        dsTrainPath: document.getElementById('dataset-train-path'),
        dsValPath: document.getElementById('dataset-val-path')
    };

    // 1. Toast Notification System
    const toastContainer = document.createElement('div');
    toastContainer.style.position = 'fixed';
    toastContainer.style.top = '20px';
    toastContainer.style.right = '20px';
    toastContainer.style.zIndex = '9999';
    toastContainer.style.display = 'flex';
    toastContainer.style.flexDirection = 'column';
    toastContainer.style.gap = '10px';
    document.body.appendChild(toastContainer);

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.padding = '12px 20px';
        toast.style.borderRadius = '4px';
        toast.style.color = '#fff';
        toast.style.fontSize = '14px';
        toast.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';
        toast.style.transition = 'all 0.3s ease';
        toast.style.minWidth = '250px';

        if (type === 'success') {
            toast.style.backgroundColor = '#238636';
            toast.style.borderLeft = '4px solid #2ea043';
        } else if (type === 'error') {
            toast.style.backgroundColor = '#da3633';
            toast.style.borderLeft = '4px solid #f85149';
        } else {
            toast.style.backgroundColor = '#1f6feb';
            toast.style.borderLeft = '4px solid #58a6ff';
        }

        toastContainer.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        });

        // Auto remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, 3000);
    }

    // Utility
    function formatNumber(num, decimals = 4) {
        return num != null ? Number(num).toFixed(decimals) : '-';
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function formatDate(dateString) {
        if (!dateString) return '-';
        const d = new Date(dateString);
        return d.toLocaleString('zh-CN', { hour12: false });
    }

    function formatLR(lr) {
        if (!lr) return '-';
        return Number(lr).toExponential(2);
    }

    // 2. Initialization
    async function init() {
        setupTabs();
        setupCharts();
        setupEventListeners();
        
        try {
            await fetchSystemInfo();
            await fetchDatasetInfo();
            await fetchConfig();
            await fetchCheckpoints();
            await checkTrainingStatus();
        } catch (error) {
            console.error('Initialization error:', error);
            showToast('初始化数据加载失败', 'error');
        }
    }

    // 3. Tab Navigation
    function setupTabs() {
        els.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                
                els.tabBtns.forEach(b => b.classList.remove('active'));
                els.tabContents.forEach(c => c.classList.remove('active'));
                
                btn.classList.add('active');
                const targetContent = document.getElementById('tab-' + targetTab);
                if (targetContent) targetContent.classList.add('active');
            });
        });
    }

    // 4. API Calls
    async function fetchSystemInfo() {
        try {
            const res = await fetch('/api/system');
            if (!res.ok) throw new Error('API request failed');
            const data = await res.json();
            
            if (els.sysStatusBadge) {
                els.sysStatusBadge.textContent = data.gpu_available ? data.gpu_name : 'CPU';
                els.sysStatusBadge.style.color = data.gpu_available ? '#2ea043' : '#8b949e';
            }
            if (els.sysGpuName) els.sysGpuName.textContent = data.gpu_name || 'N/A';
            if (els.sysGpuMemory) els.sysGpuMemory.textContent = data.gpu_memory || 'N/A';
            if (els.sysPython) els.sysPython.textContent = data.python_version || 'N/A';
            if (els.sysTorch) els.sysTorch.textContent = data.torch_version || 'N/A';
        } catch (e) {
            console.error(e);
        }
    }

    async function fetchDatasetInfo() {
        try {
            const res = await fetch('/api/dataset/info');
            if (!res.ok) throw new Error('API request failed');
            const data = await res.json();
            
            if (els.dsTrainCount) els.dsTrainCount.textContent = data.train.count;
            if (els.dsValCount) els.dsValCount.textContent = data.val.count;
            if (els.dsTrainPath) els.dsTrainPath.textContent = data.train.path;
            if (els.dsValPath) els.dsValPath.textContent = data.val.path;
        } catch (e) {
            console.error(e);
        }
    }

    async function fetchConfig() {
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error('API request failed');
            const cfg = await res.json();
            state.config = cfg;
            populateConfigForm(cfg);
        } catch (e) {
            console.error(e);
            showToast('获取配置失败', 'error');
        }
    }

    async function fetchCheckpoints() {
        try {
            const res = await fetch('/api/checkpoints');
            if (!res.ok) throw new Error('API request failed');
            const data = await res.json();
            
            if (els.checkpointTableBody) {
                els.checkpointTableBody.innerHTML = '';
                data.forEach(cp => {
                    const tr = document.createElement('tr');
                    [cp.name, formatBytes(cp.size), formatDate(cp.modified)].forEach(value => {
                        const td = document.createElement('td');
                        td.style.padding = '8px';
                        td.textContent = value;
                        tr.appendChild(td);
                    });
                    const actionCell = document.createElement('td');
                    actionCell.style.padding = '8px';
                    const button = document.createElement('button');
                    button.className = 'btn btn-sm';
                    button.style.cssText = 'background: #da3633; color: white; border: none; padding: 4px 8px; cursor: pointer; border-radius: 4px;';
                    button.textContent = '删除';
                    button.addEventListener('click', () => window.deleteCheckpoint(cp.name));
                    actionCell.appendChild(button);
                    tr.appendChild(actionCell);
                    els.checkpointTableBody.appendChild(tr);
                });
            }

            if (els.infCheckpoint) {
                const currentSelection = els.infCheckpoint.value;
                els.infCheckpoint.innerHTML = '';
                if (data.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.textContent = '无可用模型';
                    els.infCheckpoint.appendChild(opt);
                } else {
                    let found = false;
                    data.forEach(cp => {
                        const opt = document.createElement('option');
                        opt.value = cp.name;
                        opt.textContent = cp.name;
                        els.infCheckpoint.appendChild(opt);
                        if (cp.name === currentSelection) found = true;
                    });
                    if (found) {
                        els.infCheckpoint.value = currentSelection;
                    }
                }
            }
        } catch (e) {
            console.error(e);
        }
    }

    window.deleteCheckpoint = async function(name) {
        if (!confirm(`确定要删除模型 ${name} 吗？`)) return;
        try {
            const res = await fetch(`/api/checkpoints/${encodeURIComponent(name)}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Delete failed');
            showToast(`已删除 ${name}`, 'success');
            fetchCheckpoints();
        } catch (e) {
            console.error(e);
            showToast(`删除失败: ${name}`, 'error');
        }
    };

    // 5. Config Form Handling
    function populateConfigForm(cfg) {
        if (!cfg) return;
        const setVal = (el, val) => { if (el) { el.value = val; } };
        const setCb = (el, val) => { if (el) { el.checked = val; } };

        setVal(els.cfgTrainImageDir, cfg.dataset?.train_image_dir || '');
        setVal(els.cfgTrainMaskDir, cfg.dataset?.train_mask_dir || '');
        setVal(els.cfgValImageDir, cfg.dataset?.val_image_dir || '');
        setVal(els.cfgValMaskDir, cfg.dataset?.val_mask_dir || '');
        setVal(els.cfgImgSize, cfg.dataset?.img_size ?? 512);
        setVal(els.cfgNumWorkers, cfg.dataset?.num_workers ?? 0);
        setCb(els.cfgElasticTransform, cfg.dataset?.augmentation?.elastic_transform ?? false);

        setVal(els.cfgModelName, cfg.model?.name || 'unet_baseline');
        setVal(els.cfgEncoderName, cfg.model?.encoder_name || 'resnet34');
        setVal(els.cfgInChannels, cfg.model?.in_channels || 1);
        setVal(els.cfgOutChannels, cfg.model?.out_channels || 1);
        setCb(els.cfgPretrained, cfg.model?.pretrained ?? true);
        setCb(els.cfgDeepSupervision, cfg.model?.deep_supervision ?? true);

        setVal(els.cfgBatchSize, cfg.training?.batch_size || 8);
        setVal(els.cfgLearningRate, cfg.training?.learning_rate || 0.001);
        setVal(els.cfgWeightDecay, cfg.training?.weight_decay || 0.0001);
        setVal(els.cfgEpochs, cfg.training?.epochs || 100);
        setVal(els.cfgOptimizer, cfg.training?.optimizer || 'adamw');
        setVal(els.cfgScheduler, cfg.training?.scheduler || 'cosine');
        setCb(els.cfgUseAmp, cfg.training?.use_amp);

        const bce_w = cfg.training?.loss?.bce_weight ?? 0.5;
        const dice_w = cfg.training?.loss?.dice_weight ?? 0.5;
        setVal(els.cfgBceWeight, bce_w);
        setVal(els.cfgDiceWeight, dice_w);
        if (els.cfgBceWeightValue) els.cfgBceWeightValue.textContent = bce_w;
        if (els.cfgDiceWeightValue) els.cfgDiceWeightValue.textContent = dice_w;
        const cldice_w = cfg.training?.loss?.cldice_weight ?? 0.0;
        setVal(els.cfgClDiceWeight, cldice_w);
        if (els.cfgClDiceWeightValue) els.cfgClDiceWeightValue.textContent = Number(cldice_w).toFixed(2);
        updateModelSpecificControls(cfg.model?.name || 'unet_baseline');

        setVal(els.cfgPatience, cfg.training?.early_stopping?.patience || 10);
        setVal(els.cfgSaveDir, cfg.training?.checkpoint?.save_dir || 'checkpoints');
        setCb(els.cfgSaveBestOnly, cfg.training?.checkpoint?.save_best_only);
    }

    function buildConfigFromForm() {
        const getVal = (el) => el ? el.value : '';
        const getNum = (el) => el ? Number(el.value) : 0;
        const getCb = (el) => el ? el.checked : false;

        return {
            dataset: {
                train_image_dir: getVal(els.cfgTrainImageDir),
                train_mask_dir: getVal(els.cfgTrainMaskDir),
                val_image_dir: getVal(els.cfgValImageDir),
                val_mask_dir: getVal(els.cfgValMaskDir),
                img_size: getNum(els.cfgImgSize),
                num_workers: getNum(els.cfgNumWorkers),
                augmentation: {
                    ...(state.config?.dataset?.augmentation || {}),
                    elastic_transform: getCb(els.cfgElasticTransform)
                },
                pin_memory: state.config?.dataset?.pin_memory ?? true,
                keep_aspect_ratio: state.config?.dataset?.keep_aspect_ratio ?? true
            },
            model: {
                name: getVal(els.cfgModelName),
                in_channels: getNum(els.cfgInChannels),
                out_channels: getNum(els.cfgOutChannels),
                encoder_name: getVal(els.cfgEncoderName) || 'resnet34',
                pretrained: getCb(els.cfgPretrained),
                deep_supervision: getCb(els.cfgDeepSupervision)
            },
            training: {
                batch_size: getNum(els.cfgBatchSize),
                epochs: getNum(els.cfgEpochs),
                learning_rate: getNum(els.cfgLearningRate),
                weight_decay: getNum(els.cfgWeightDecay),
                optimizer: getVal(els.cfgOptimizer),
                scheduler: getVal(els.cfgScheduler),
                use_amp: getCb(els.cfgUseAmp),
                deep_supervision_weights: state.config?.training?.deep_supervision_weights || [0.3, 0.2],
                loss: {
                    name: getVal(els.cfgModelName) === 'resunet_aspp' ? 'BCEDiceClDiceLoss' : 'BCEDiceLoss',
                    bce_weight: getNum(els.cfgBceWeight),
                    dice_weight: getNum(els.cfgDiceWeight),
                    cldice_weight: getVal(els.cfgModelName) === 'resunet_aspp' ? getNum(els.cfgClDiceWeight) : 0.0,
                    dice_smooth: state.config?.training?.loss?.dice_smooth ?? 1e-6,
                    skeleton_iterations: state.config?.training?.loss?.skeleton_iterations ?? 5
                },
                early_stopping: {
                    patience: getNum(els.cfgPatience)
                },
                checkpoint: {
                    save_dir: getVal(els.cfgSaveDir),
                    save_best_only: getCb(els.cfgSaveBestOnly),
                    save_interval: state.config?.training?.checkpoint?.save_interval ?? 10
                }
            },
            evaluation: state.config?.evaluation ?? { threshold: 0.5, apply_postprocess: false },
            inference: {
                ...(state.config?.inference ?? {
                threshold: 0.5,
                img_size: null,
                postprocess: { enabled: true, min_component_size: 50, max_hole_size: 100, morph_close_kernel: 3 }
                })
            }
        };
    }

    async function saveConfig() {
        const payload = buildConfigFromForm();
        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error('Failed to save config');
            showToast('配置已保存', 'success');
            await fetchConfig();
        } catch (e) {
            console.error(e);
            showToast('保存配置失败', 'error');
        }
    }

    // 6. Charts Setup
    function setupCharts() {
        if (!window.Chart) {
            console.warn('Chart.js not found');
            return;
        }

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#c9d1d9' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#8b949e' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#8b949e' }
                }
            }
        };

        if (els.chartLoss) {
            const ctxLoss = els.chartLoss.getContext('2d');
            state.charts.loss = new Chart(ctxLoss, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Train Loss', data: [], borderColor: '#58a6ff', tension: 0.1, fill: false },
                        { label: 'Val Loss', data: [], borderColor: '#f0883e', tension: 0.1, fill: false }
                    ]
                },
                options: commonOptions
            });
        }

        if (els.chartMetrics) {
            const ctxMetrics = els.chartMetrics.getContext('2d');
            state.charts.metrics = new Chart(ctxMetrics, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Dice', data: [], borderColor: '#2ea043', tension: 0.1, fill: false },
                        { label: 'IoU', data: [], borderColor: '#bc8cff', tension: 0.1, fill: false }
                    ]
                },
                options: {
                    ...commonOptions,
                    scales: {
                        ...commonOptions.scales,
                        y: {
                            ...commonOptions.scales.y,
                            min: 0,
                            max: 1
                        }
                    }
                }
            });
        }
    }

    function updateCharts(history) {
        if (!history || !Array.isArray(history)) return;
        const epochs = history.map(h => h.epoch);
        const trainLoss = history.map(h => h.train_loss);
        const valLoss = history.map(h => h.val_loss);
        const dice = history.map(h => h.dice);
        const iou = history.map(h => h.iou);

        if (state.charts.loss) {
            state.charts.loss.data.labels = epochs;
            state.charts.loss.data.datasets[0].data = trainLoss;
            state.charts.loss.data.datasets[1].data = valLoss;
            state.charts.loss.update();
        }

        if (state.charts.metrics) {
            state.charts.metrics.data.labels = epochs;
            state.charts.metrics.data.datasets[0].data = dice;
            state.charts.metrics.data.datasets[1].data = iou;
            state.charts.metrics.update();
        }
    }

    // 7. Training Control & Polling
    async function startTraining() {
        if (state.isTraining) return;
        try {
            const res = await fetch('/api/train/start', { method: 'POST' });
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.message || 'Start failed');
            }
            const data = await res.json();
            if (data && !data.success) {
                throw new Error(data.message || 'Start failed');
            }
            showToast('训练已启动', 'success');
            setTrainingState(true);
            startPolling();
        } catch (e) {
            console.error(e);
            showToast(`启动训练失败: ${e.message}`, 'error');
        }
    }

    async function stopTraining() {
        if (!state.isTraining) return;
        if (els.btnStopTraining) els.btnStopTraining.textContent = '停止中...';
        try {
            const res = await fetch('/api/train/stop', { method: 'POST' });
            if (!res.ok) throw new Error('Stop failed');
            showToast('正在停止训练...', 'info');
            // Polling will detect when it actually stops
        } catch (e) {
            console.error(e);
            showToast('停止训练失败', 'error');
            if (els.btnStopTraining) els.btnStopTraining.textContent = '停止训练';
        }
    }

    function setTrainingState(isRunning) {
        state.isTraining = isRunning;
        
        if (els.trainingIndicator) {
            if (isRunning) els.trainingIndicator.classList.add('active');
            else els.trainingIndicator.classList.remove('active');
        }
        
        if (els.trainingIndicatorText) {
            els.trainingIndicatorText.textContent = isRunning ? '训练中' : '空闲';
        }
        
        if (els.btnStartTraining) els.btnStartTraining.disabled = isRunning;
        if (els.btnStopTraining) {
            els.btnStopTraining.disabled = !isRunning;
            els.btnStopTraining.textContent = '停止训练';
        }
    }

    function startPolling() {
        if (state.pollIntervalId) clearInterval(state.pollIntervalId);
        pollStatus(); // immediate call
        state.pollIntervalId = setInterval(pollStatus, 2000);
    }

    function stopPolling() {
        if (state.pollIntervalId) {
            clearInterval(state.pollIntervalId);
            state.pollIntervalId = null;
        }
    }

    async function checkTrainingStatus() {
        try {
            const res = await fetch('/api/train/status');
            if (!res.ok) return;
            const data = await res.json();
            
            updateDashboard(data);
            
            if (data.running) {
                setTrainingState(true);
                startPolling();
            } else {
                setTrainingState(false);
            }
        } catch (e) {
            console.error(e);
        }
    }

    async function pollStatus() {
        try {
            const res = await fetch('/api/train/status');
            if (!res.ok) throw new Error('Poll failed');
            const data = await res.json();
            
            updateDashboard(data);
            
            if (!data.running && state.isTraining) {
                // Training just stopped
                setTrainingState(false);
                stopPolling();
                showToast('训练已结束', 'success');
                fetchCheckpoints(); // Refresh models
            }
        } catch (e) {
            console.error(e);
        }
    }

    function updateDashboard(data) {
        if (!data) return;
        
        if (els.metricEpoch) {
            els.metricEpoch.textContent = `${data.epoch || 0} / ${data.total_epochs || 0}`;
        }
        if (els.metricDice) els.metricDice.textContent = formatNumber(data.best_dice || data.dice);
        if (els.metricTrainLoss) els.metricTrainLoss.textContent = formatNumber(data.train_loss);
        if (els.metricValLoss) els.metricValLoss.textContent = formatNumber(data.val_loss);
        if (els.currentLr) els.currentLr.textContent = formatLR(data.lr);
        
        if (els.epochProgress && data.total_epochs > 0) {
            const pct = Math.min(100, Math.max(0, ((data.epoch || 0) / data.total_epochs) * 100));
            els.epochProgress.style.width = pct + '%';
        }
        if (els.epochProgressText && data.total_epochs > 0) {
            const pct = Math.min(100, Math.max(0, ((data.epoch || 0) / data.total_epochs) * 100));
            els.epochProgressText.textContent = pct.toFixed(1) + '%';
        }

        if (data.history) {
            updateCharts(data.history);
            
            // Generate log from history if we want
            if (els.trainingLog && data.history.length > 0) {
                const logs = data.history.map(h => 
                    `Epoch [${h.epoch}/${data.total_epochs}] - Train Loss: ${formatNumber(h.train_loss)}, Val Loss: ${formatNumber(h.val_loss)}, Dice: ${formatNumber(h.dice)}, IoU: ${formatNumber(h.iou)}`
                );
                const currentText = els.trainingLog.textContent;
                const logPathLine = data.log_path ? `日志文件: ${data.log_path}\n\n` : '';
                const newText = logPathLine + logs.join('\n');
                if (currentText !== newText) {
                    els.trainingLog.textContent = newText;
                    els.trainingLog.scrollTop = els.trainingLog.scrollHeight;
                }
            } else if (els.trainingLog && data.message) {
                // 没有历史记录时显示状态消息（包括错误信息）
                els.trainingLog.textContent = data.message;
            }
        } else if (els.trainingLog && data.message) {
            els.trainingLog.textContent = data.message;
        }
    }

    // 8. Inference
    function handleFileDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (els.infDropzone) els.infDropzone.style.borderColor = '#30363d';
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) processInferenceFiles(e.dataTransfer.files);
    }

    function handleFileSelect(e) {
        if (e.target.files && e.target.files.length > 0) {
            processInferenceFiles(e.target.files);
        }
    }

    function readInferenceFile(file) {
        return new Promise((resolve, reject) => {
            if (!file.type.startsWith('image/')) {
                reject(new Error(`${file.name} 不是图像文件`));
                return;
            }
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, dataUrl: reader.result, image_base64: reader.result.split(',')[1] });
            reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
            reader.readAsDataURL(file);
        });
    }

    async function processInferenceFiles(files) {
        try {
            state.inferenceFiles = await Promise.all(Array.from(files).slice(0, 32).map(readInferenceFile));
            state.inferenceImageBase64 = state.inferenceFiles[0]?.image_base64 || null;
            const first = state.inferenceFiles[0];
            if (first && els.infOriginal) {
                els.infOriginal.src = first.dataUrl;
                els.infOriginal.style.display = 'block';
                const ph = document.getElementById('inf-orig-placeholder');
                if (ph) ph.style.display = 'none';
            }
            if (els.infResult) els.infResult.style.display = 'none';
            if (els.batchInferenceResults) els.batchInferenceResults.style.display = state.inferenceFiles.length > 1 ? 'block' : 'none';
            showToast(`已选择 ${state.inferenceFiles.length} 张图像`, 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    }

    function processInferenceFile(file) {
        processInferenceFiles([file]);
    }

    async function runInference() {
        if (!state.inferenceImageBase64) {
            showToast('请先上传图片', 'error');
            return;
        }
        
        const cp = els.infCheckpoint ? els.infCheckpoint.value : '';
        if (!cp) {
            showToast('请选择模型权重', 'error');
            return;
        }
        
        const threshold = els.infThreshold ? parseFloat(els.infThreshold.value) : 0.5;
        const processingOptions = getProcessingOptions();
        const bodyOptions = {
            min_component_size: Number(els.infMinComponent?.value || 50),
            max_hole_size: Number(els.infMaxHole?.value || 100),
            morph_close_kernel: Number(els.infMorphKernel?.value || 3)
        };

        if (els.btnRunInference) {
            els.btnRunInference.disabled = true;
            els.btnRunInference.textContent = '推理中...';
        }

        try {
            const endpoint = state.inferenceFiles.length > 1 ? '/api/inference-batch' : '/api/inference';
            const body = state.inferenceFiles.length > 1 ? {
                images: state.inferenceFiles.map(item => ({ name: item.name, image_base64: item.image_base64 })),
                checkpoint: cp,
                threshold: threshold,
                ...processingOptions,
                ...bodyOptions
            } : {
                image_base64: state.inferenceImageBase64,
                checkpoint: cp,
                threshold: threshold,
                ...processingOptions,
                ...bodyOptions
            };
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || 'Inference failed');
            }
            const data = await res.json();
            
            if (state.inferenceFiles.length > 1 && data.results) {
                renderBatchResults(data.results);
                showToast(`批量推理完成，共 ${data.count} 张`, 'success');
            } else if (els.infResult && data.mask_base64) {
                els.infResult.src = 'data:image/png;base64,' + data.mask_base64;
                els.infResult.style.display = 'block';
                const ph = document.getElementById('inf-result-placeholder');
                if(ph) ph.style.display = 'none';
                showToast('推理完成', 'success');
            } else if (data.error) {
                throw new Error(data.error);
            }
        } catch (e) {
            console.error(e);
            showToast(`推理失败: ${e.message}`, 'error');
        } finally {
            if (els.btnRunInference) {
                els.btnRunInference.disabled = false;
                els.btnRunInference.textContent = '开始推理';
            }
        }
    }

    function getProcessingOptions() {
        return { processing: els.infProcessing?.value || 'config' };
    }

    function renderBatchResults(results) {
        if (!els.batchInferenceGrid || !els.batchInferenceResults) return;
        els.batchInferenceGrid.innerHTML = '';
        results.forEach(item => {
            const box = document.createElement('div');
            box.className = 'infer-result-box glass-panel';
            const title = document.createElement('div');
            title.className = 'result-title';
            title.textContent = item.name;
            const wrap = document.createElement('div');
            wrap.className = 'result-img-wrap';
            const image = document.createElement('img');
            image.alt = `${item.name} 预测掩膜`;
            image.src = `data:image/png;base64,${item.mask_base64}`;
            wrap.appendChild(image);
            box.append(title, wrap);
            els.batchInferenceGrid.appendChild(box);
        });
        els.batchInferenceResults.style.display = 'block';
    }

    function updateModelSpecificControls(modelName) {
        const topologyModel = modelName === 'resunet_aspp';
        const resnetModel = modelName === 'unet_resnet';
        if (els.cfgEncoderGroup) els.cfgEncoderGroup.style.display = resnetModel ? '' : 'none';
        if (els.cfgClDiceGroup) els.cfgClDiceGroup.style.display = topologyModel ? '' : 'none';
        if (els.cfgDeepSupervisionGroup) els.cfgDeepSupervisionGroup.style.display = topologyModel ? '' : 'none';
        if (els.cfgLossWeightHint) {
            els.cfgLossWeightHint.textContent = topologyModel
                ? 'BCE + Dice + clDice 权重总和为 1.0（自动联动）'
                : 'BCE 与 Dice 权重之和为 1.0（自动联动）';
        }
    }

    function updateRegionWeights(changed) {
        const topologyModel = els.cfgModelName?.value === 'resunet_aspp';
        const cldice = topologyModel ? Number(els.cfgClDiceWeight?.value || 0) : 0;
        const regionTotal = Math.max(0, 1 - cldice);
        if (changed === 'bce') {
            const bce = Math.min(regionTotal, Math.max(0, Number(els.cfgBceWeight.value)));
            els.cfgBceWeight.value = bce.toFixed(2);
            els.cfgDiceWeight.value = (regionTotal - bce).toFixed(2);
        } else {
            const dice = Math.min(regionTotal, Math.max(0, Number(els.cfgDiceWeight.value)));
            els.cfgDiceWeight.value = dice.toFixed(2);
            els.cfgBceWeight.value = (regionTotal - dice).toFixed(2);
        }
        if (els.cfgBceWeightValue) els.cfgBceWeightValue.textContent = Number(els.cfgBceWeight.value).toFixed(2);
        if (els.cfgDiceWeightValue) els.cfgDiceWeightValue.textContent = Number(els.cfgDiceWeight.value).toFixed(2);
        if (els.cfgClDiceWeightValue) els.cfgClDiceWeightValue.textContent = cldice.toFixed(2);
    }

    async function scanThresholds() {
        const checkpoint = els.infCheckpoint?.value;
        if (!checkpoint) {
            showToast('请选择模型权重', 'error');
            return;
        }
        const thresholds = (els.thresholdScanValues?.value || '').split(',').map(value => Number(value.trim())).filter(value => Number.isFinite(value));
        if (!thresholds.length || thresholds.length > 21 || thresholds.some(value => value < 0 || value > 1)) {
            showToast('阈值必须是 0 到 1 之间的数字，最多 21 个', 'error');
            return;
        }
        if (els.btnThresholdScan) {
            els.btnThresholdScan.disabled = true;
            els.btnThresholdScan.textContent = '扫描中...';
        }
        if (els.thresholdScanStatus) els.thresholdScanStatus.textContent = '正在读取验证集并计算各阈值...';
        try {
            const res = await fetch('/api/threshold-scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    checkpoint,
                    thresholds,
                    ...getProcessingOptions(),
                    min_component_size: Number(els.infMinComponent?.value || 50),
                    max_hole_size: Number(els.infMaxHole?.value || 100),
                    morph_close_kernel: Number(els.infMorphKernel?.value || 3)
                })
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || '阈值扫描失败');
            if (els.thresholdScanStatus) els.thresholdScanStatus.textContent = `已完成：${data.samples} 张验证图像，设备 ${data.device}`;
            if (els.thresholdScanBest) {
                els.thresholdScanBest.textContent = `最佳阈值 ${Number(data.best_threshold).toFixed(2)} | Dice ${formatNumber(data.best_dice)}`;
                els.thresholdScanBest.style.display = 'block';
            }
            if (els.thresholdScanTableBody) {
                els.thresholdScanTableBody.innerHTML = '';
                data.results.forEach(item => {
                    const row = document.createElement('tr');
                    [Number(item.threshold).toFixed(2), formatNumber(item.dice), formatNumber(item.iou), formatNumber(item.precision), formatNumber(item.recall)].forEach(value => {
                        const cell = document.createElement('td');
                        cell.textContent = value;
                        row.appendChild(cell);
                    });
                    els.thresholdScanTableBody.appendChild(row);
                });
            }
            if (els.thresholdScanTableWrap) els.thresholdScanTableWrap.style.display = 'block';
            showToast(`阈值扫描完成，最佳 Dice ${formatNumber(data.best_dice)}`, 'success');
        } catch (error) {
            if (els.thresholdScanStatus) els.thresholdScanStatus.textContent = '扫描失败';
            showToast(`阈值扫描失败: ${error.message}`, 'error');
        } finally {
            if (els.btnThresholdScan) {
                els.btnThresholdScan.disabled = false;
                els.btnThresholdScan.textContent = '📊 批量测试阈值';
            }
        }
    }

    // 9. Event Listeners Setup
    function setupEventListeners() {
        // Config Form Buttons
        if (els.btnSaveConfig) {
            els.btnSaveConfig.addEventListener('click', saveConfig);
        }
        
        if (els.btnResetConfig) {
            els.btnResetConfig.addEventListener('click', () => {
                if (state.config) populateConfigForm(state.config);
                showToast('配置已重置', 'info');
            });
        }

        // Linked Weights
        if (els.cfgBceWeight && els.cfgDiceWeight) {
            els.cfgBceWeight.addEventListener('input', () => updateRegionWeights('bce'));
            els.cfgDiceWeight.addEventListener('input', () => updateRegionWeights('dice'));
        }
        if (els.cfgClDiceWeight) {
            els.cfgClDiceWeight.addEventListener('input', () => updateRegionWeights('bce'));
        }
        if (els.cfgModelName) {
            els.cfgModelName.addEventListener('change', () => {
                updateModelSpecificControls(els.cfgModelName.value);
                updateRegionWeights('bce');
            });
        }

        // Training Control
        if (els.btnStartTraining) {
            els.btnStartTraining.addEventListener('click', startTraining);
        }
        if (els.btnStopTraining) {
            els.btnStopTraining.addEventListener('click', stopTraining);
        }

        // Inference Drag & Drop
        if (els.infDropzone) {
            els.infDropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                els.infDropzone.style.borderColor = '#58a6ff';
            });
            
            els.infDropzone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.stopPropagation();
                els.infDropzone.style.borderColor = '#30363d';
            });
            
            els.infDropzone.addEventListener('drop', handleFileDrop);
            
            els.infDropzone.addEventListener('click', () => {
                if (els.infFileInput) els.infFileInput.click();
            });
        }

        if (els.infFileInput) {
            els.infFileInput.addEventListener('change', handleFileSelect);
        }

        if (els.btnRunInference) {
            els.btnRunInference.addEventListener('click', runInference);
        }
        if (els.btnThresholdScan) els.btnThresholdScan.addEventListener('click', scanThresholds);

        if (els.infThreshold) {
            els.infThreshold.addEventListener('input', (e) => {
                if (els.infThresholdValue) els.infThresholdValue.textContent = e.target.value;
            });
        }
    }

    // Start
    document.addEventListener('DOMContentLoaded', init);

})();
