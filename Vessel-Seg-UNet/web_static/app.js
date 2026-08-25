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
        cfgPersistentWorkers: document.getElementById('cfg-persistent-workers'),
        cfgPrefetchFactor: document.getElementById('cfg-prefetch-factor'),
        cfgCacheSize: document.getElementById('cfg-cache-size'),
        cfgElasticTransform: document.getElementById('cfg-elastic-transform'),
        cfgDomainBalanceEnabled: document.getElementById('cfg-domain-balance-enabled'),
        cfgDomainBalanceGroup: document.getElementById('cfg-domain-balance-group'),
        cfgDomainTargetProbability: document.getElementById('cfg-domain-target-probability'),
        cfgCrossValidationEnabled: document.getElementById('cfg-cross-validation-enabled'),
        cfgCrossValidationGroup: document.getElementById('cfg-cross-validation-group'),
        cfgNumFolds: document.getElementById('cfg-num-folds'),
        cfgFoldIndex: document.getElementById('cfg-fold-index'),
        cfgTemporalEnabled: document.getElementById('cfg-temporal-enabled'),
        cfgFrangiEnabled: document.getElementById('cfg-frangi-enabled'),
        cfgFrangiGroup: document.getElementById('cfg-frangi-group'),
        cfgTrainFrangiDir: document.getElementById('cfg-train-frangi-dir'),
        cfgValFrangiDir: document.getElementById('cfg-val-frangi-dir'),
        cfgFrangiSigmas: document.getElementById('cfg-frangi-sigmas'),
        cfgFrangiMethod: document.getElementById('cfg-frangi-method'),
        btnGenerateFrangi: document.getElementById('btn-generate-frangi'),
        btnStopFrangi: document.getElementById('btn-stop-frangi'),
        btnClearFrangi: document.getElementById('btn-clear-frangi'),
        frangiGenerateStatus: document.getElementById('frangi-generate-status'),
        cfgPatchEnabled: document.getElementById('cfg-patch-enabled'),
        cfgPatchSize: document.getElementById('cfg-patch-size'),
        cfgPatchForegroundProbability: document.getElementById('cfg-patch-foreground-probability'),
        cfgPatchStride: document.getElementById('cfg-patch-stride'),
        cfgPatchGroup: document.getElementById('cfg-patch-group'),
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
        infInputMode: document.getElementById('inference-input-mode'),
        infThreshold: document.getElementById('inference-threshold'),
        infThresholdValue: document.getElementById('inference-threshold-value'),
        infProcessing: document.getElementById('inference-processing'),
        infProcessingHint: document.getElementById('inference-processing-hint'),
        btnRunInference: document.getElementById('btn-run-inference'),
        thresholdScanValues: document.getElementById('threshold-scan-values'),
        thresholdEvaluationSplit: document.getElementById('threshold-evaluation-split'),
        thresholdScanMode: document.getElementById('threshold-scan-mode'),
        thresholdCoarseStep: document.getElementById('threshold-coarse-step'),
        thresholdFineStep: document.getElementById('threshold-fine-step'),
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
        dsValPath: document.getElementById('dataset-val-path'),
        dsSplitStrategy: document.getElementById('dataset-split-strategy')
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
            await checkFrangiStatus();
        } catch (error) {
            console.error('Initialization error:', error);
            showToast('初始化数据加载失败', 'error');
        }
    }

    // 页面加载时检查 Frangi 生成状态
    async function checkFrangiStatus() {
        try {
            const res = await fetch('/api/frangi/status');
            const data = await res.json();
            if (data.running) {
                // 有正在运行的生成任务，显示停止按钮并开始轮询
                if (els.btnGenerateFrangi) {
                    els.btnGenerateFrangi.disabled = true;
                    els.btnGenerateFrangi.textContent = '⏳ 生成中...';
                }
                if (els.btnStopFrangi) {
                    els.btnStopFrangi.style.display = '';
                }
                if (els.frangiGenerateStatus) {
                    els.frangiGenerateStatus.textContent = data.progress || '处理中...';
                }
                pollFrangiProgress();
            }
        } catch (e) {
            // 忽略
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
            if (els.dsSplitStrategy) {
                els.dsSplitStrategy.textContent = data.split_strategy || '按当前训练/验证路径使用，不自动重分层';
            }
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
                const currentSelections = new Set(
                    Array.from(els.infCheckpoint.selectedOptions).map(option => option.value)
                );
                els.infCheckpoint.innerHTML = '';
                if (data.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.textContent = '无可用模型';
                    els.infCheckpoint.appendChild(opt);
                } else {
                    let found = false;
                    data.forEach((cp, index) => {
                        const opt = document.createElement('option');
                        opt.value = cp.name;
                        opt.textContent = cp.name;
                        opt.selected = currentSelections.has(cp.name) || (
                            currentSelections.size === 0 && index === 0
                        );
                        els.infCheckpoint.appendChild(opt);
                        if (currentSelections.has(cp.name)) found = true;
                    });
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
        setCb(els.cfgPersistentWorkers, cfg.dataset?.loader?.persistent_workers ?? true);
        setVal(els.cfgPrefetchFactor, cfg.dataset?.loader?.prefetch_factor ?? 2);
        setVal(els.cfgCacheSize, cfg.dataset?.loader?.cache_size ?? 32);
        setCb(els.cfgElasticTransform, cfg.dataset?.augmentation?.elastic_transform ?? false);
        setCb(els.cfgDomainBalanceEnabled, cfg.dataset?.domain_balance?.enabled ?? true);
        setVal(els.cfgDomainTargetProbability, cfg.dataset?.domain_balance?.target_probability ?? 0.4);
        if (els.cfgDomainBalanceGroup) {
            els.cfgDomainBalanceGroup.style.display = (cfg.dataset?.domain_balance?.enabled ?? true) ? '' : 'none';
        }
        setCb(els.cfgCrossValidationEnabled, cfg.dataset?.cross_validation?.enabled ?? false);
        setVal(els.cfgNumFolds, cfg.dataset?.cross_validation?.num_folds ?? 3);
        setVal(els.cfgFoldIndex, (cfg.dataset?.cross_validation?.fold_index ?? 0) + 1);
        if (els.cfgFoldIndex) els.cfgFoldIndex.max = cfg.dataset?.cross_validation?.num_folds ?? 3;
        if (els.cfgCrossValidationGroup) {
            els.cfgCrossValidationGroup.style.display = (cfg.dataset?.cross_validation?.enabled ?? false) ? '' : 'none';
        }
        setCb(els.cfgTemporalEnabled, cfg.dataset?.temporal_2_5d?.enabled ?? false);
        setCb(els.cfgFrangiEnabled, cfg.dataset?.frangi?.enabled ?? false);
        setVal(els.cfgTrainFrangiDir, cfg.dataset?.frangi?.train_frangi_dir || '');
        setVal(els.cfgValFrangiDir, cfg.dataset?.frangi?.val_frangi_dir || '');
        setVal(els.cfgFrangiSigmas, (cfg.dataset?.frangi?.sigmas || [1.0,2.0,3.0,4.0,5.0]).join(','));
        setVal(els.cfgFrangiMethod, cfg.dataset?.frangi?.method || 'hessian');
        if (els.cfgFrangiGroup) {
            els.cfgFrangiGroup.style.display = (cfg.dataset?.frangi?.enabled ?? false) ? '' : 'none';
        }
        setCb(els.cfgPatchEnabled, cfg.dataset?.patch?.enabled ?? false);
        setVal(els.cfgPatchSize, cfg.dataset?.patch?.size ?? 640);
        setVal(els.cfgPatchForegroundProbability, cfg.dataset?.patch?.foreground_probability ?? 0.7);
        setVal(els.cfgPatchStride, cfg.inference?.patch?.stride ?? 480);

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
        const cldice_w = cfg.training?.loss?.cldice_weight ?? cfg.training?.loss?.cl_dice_weight ?? 0.0;
        setVal(els.cfgClDiceWeight, cldice_w);
        if (els.cfgClDiceWeightValue) els.cfgClDiceWeightValue.textContent = Number(cldice_w).toFixed(2);
        updateModelSpecificControls(cfg.model?.name || 'unet_baseline');

        setVal(els.cfgPatience, cfg.training?.early_stopping?.patience || 10);
        setVal(els.cfgSaveDir, cfg.training?.checkpoint?.save_dir || 'checkpoints');
        setCb(els.cfgSaveBestOnly, cfg.training?.checkpoint?.save_best_only);
        syncAdvancedDataControls();
    }

    function syncAdvancedDataControls() {
        if (els.cfgDomainBalanceGroup) {
            els.cfgDomainBalanceGroup.style.display = els.cfgDomainBalanceEnabled?.checked ? '' : 'none';
        }
        if (els.cfgCrossValidationGroup) {
            els.cfgCrossValidationGroup.style.display = els.cfgCrossValidationEnabled?.checked ? '' : 'none';
        }
        if (els.cfgFoldIndex && els.cfgNumFolds) {
            const folds = Math.min(5, Math.max(3, Number(els.cfgNumFolds.value) || 3));
            els.cfgFoldIndex.max = String(folds);
            els.cfgFoldIndex.value = String(Math.min(folds, Math.max(1, Number(els.cfgFoldIndex.value) || 1)));
        }
        if (els.cfgInChannels) {
            const automaticChannels = els.cfgTemporalEnabled?.checked ? 3 : (els.cfgFrangiEnabled?.checked ? 2 : null);
            const wasAutomatic = els.cfgInChannels.disabled;
            els.cfgInChannels.disabled = automaticChannels !== null;
            if (automaticChannels !== null) els.cfgInChannels.value = String(automaticChannels);
            else if (wasAutomatic) els.cfgInChannels.value = '1';
        }
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
                loader: {
                    persistent_workers: getCb(els.cfgPersistentWorkers),
                    prefetch_factor: getNum(els.cfgPrefetchFactor),
                    cache_size: getNum(els.cfgCacheSize)
                },
                augmentation: {
                    ...(state.config?.dataset?.augmentation || {}),
                    elastic_transform: getCb(els.cfgElasticTransform)
                },
                patch: {
                    enabled: getCb(els.cfgPatchEnabled),
                    size: getNum(els.cfgPatchSize),
                    foreground_probability: getNum(els.cfgPatchForegroundProbability),
                    min_foreground_ratio: state.config?.dataset?.patch?.min_foreground_ratio ?? 0.002
                },
                frangi: {
                    enabled: getCb(els.cfgFrangiEnabled),
                    train_frangi_dir: getVal(els.cfgTrainFrangiDir),
                    val_frangi_dir: getVal(els.cfgValFrangiDir),
                    method: getVal(els.cfgFrangiMethod),
                    sigmas: (getVal(els.cfgFrangiSigmas) || '1.0,2.0,3.0,4.0,5.0').split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n)),
                    beta: state.config?.dataset?.frangi?.beta ?? 0.5,
                    c: state.config?.dataset?.frangi?.c ?? 15.0
                },
                domain_balance: {
                    enabled: getCb(els.cfgDomainBalanceEnabled),
                    target_prefixes: state.config?.dataset?.domain_balance?.target_prefixes || ['dias_train_'],
                    target_probability: getNum(els.cfgDomainTargetProbability),
                    samples_per_epoch: state.config?.dataset?.domain_balance?.samples_per_epoch ?? 0
                },
                cross_validation: {
                    enabled: getCb(els.cfgCrossValidationEnabled),
                    num_folds: getNum(els.cfgNumFolds),
                    fold_index: Math.max(0, getNum(els.cfgFoldIndex) - 1)
                },
                temporal_2_5d: {
                    enabled: getCb(els.cfgTemporalEnabled)
                },
                pin_memory: state.config?.dataset?.pin_memory ?? true,
                keep_aspect_ratio: state.config?.dataset?.keep_aspect_ratio ?? true
            },
            model: {
                name: getVal(els.cfgModelName),
                in_channels: getCb(els.cfgTemporalEnabled) ? 3 : (getCb(els.cfgFrangiEnabled) ? 2 : getNum(els.cfgInChannels)),
                out_channels: getNum(els.cfgOutChannels),
                encoder_name: getVal(els.cfgEncoderName) || 'resnet34',
                pretrained: getCb(els.cfgPretrained),
                deep_supervision: getCb(els.cfgDeepSupervision),
                input_mode: getCb(els.cfgTemporalEnabled) ? 'temporal' : 'grayscale'
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
                    name: getNum(els.cfgClDiceWeight) > 0 ? 'BCEDiceClDiceLoss' : 'BCEDiceLoss',
                    bce_weight: getNum(els.cfgBceWeight),
                    dice_weight: getNum(els.cfgDiceWeight),
                    cl_dice_weight: getNum(els.cfgClDiceWeight),
                    cldice_weight: getNum(els.cfgClDiceWeight),
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
                }),
                patch: {
                    ...(state.config?.inference?.patch || {}),
                    enabled: getCb(els.cfgPatchEnabled),
                    size: getNum(els.cfgPatchSize),
                    stride: getNum(els.cfgPatchStride)
                }
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
            const data = await res.json();
            if (!res.ok || !data.success) {
                throw new Error(data.message || '保存失败');
            }
            showToast(data.note || '配置已保存', 'success');
            await fetchConfig();
        } catch (e) {
            console.error(e);
            showToast('保存配置失败: ' + e.message, 'error');
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

    async function chooseDirectory(button) {
        const target = document.getElementById(button.dataset.directoryTarget || '');
        if (!target) return;
        const originalText = button.textContent;
        button.disabled = true;
        button.textContent = '打开中...';
        try {
            const response = await fetch('/api/select-directory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data.success) {
                if (!data.cancelled) throw new Error(data.message || '选择文件夹失败');
                return;
            }
            target.value = data.path;
            target.dispatchEvent(new Event('change', { bubbles: true }));
        } catch (error) {
            console.error(error);
            showToast(`选择文件夹失败: ${error.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = originalText;
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
            const previewIndex = els.infInputMode?.value === 'temporal' && state.inferenceFiles.length === 3 ? 1 : 0;
            const preview = state.inferenceFiles[previewIndex];
            if (preview && els.infOriginal) {
                els.infOriginal.src = preview.dataUrl;
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
        
        const checkpoints = getSelectedCheckpoints();
        if (!checkpoints.length) {
            showToast('请选择模型权重', 'error');
            return;
        }
        const temporalMode = els.infInputMode?.value === 'temporal';
        if (temporalMode && state.inferenceFiles.length !== 3) {
            showToast('三时相推理需要按前、当前、后顺序选择 3 张图像', 'error');
            return;
        }
        
        const threshold = els.infThreshold ? parseFloat(els.infThreshold.value) : 0.5;
        const processingOptions = getProcessingOptions();

        if (els.btnRunInference) {
            els.btnRunInference.disabled = true;
            els.btnRunInference.textContent = '推理中...';
        }

        try {
            const endpoint = !temporalMode && state.inferenceFiles.length > 1 ? '/api/inference-batch' : '/api/inference';
            const body = temporalMode ? {
                temporal_images: state.inferenceFiles.map(item => item.image_base64),
                image_base64: state.inferenceFiles[1].image_base64,
                checkpoints,
                threshold: threshold,
                ...processingOptions
            } : state.inferenceFiles.length > 1 ? {
                images: state.inferenceFiles.map(item => ({ name: item.name, image_base64: item.image_base64 })),
                checkpoints,
                threshold: threshold,
                ...processingOptions
            } : {
                image_base64: state.inferenceImageBase64,
                checkpoints,
                threshold: threshold,
                ...processingOptions
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

    function getSelectedCheckpoints() {
        if (!els.infCheckpoint) return [];
        return Array.from(els.infCheckpoint.selectedOptions)
            .map(option => option.value)
            .filter(Boolean)
            .slice(0, 5);
    }

    function updateProcessingHint() {
        if (!els.infProcessingHint) return;
        const hints = {
            off: '不进行连通域、孔洞或形态学处理',
            light: '去除很小的噪点，尽量保留细血管',
            config: '使用配置中的标准连通域、孔洞和闭运算参数',
            strong: '去除更多小区域并加强闭运算，结果更平滑但可能丢失细血管'
        };
        els.infProcessingHint.textContent = hints[els.infProcessing?.value] || hints.config;
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
        const resnetModel = ['unet_resnet', 'resunet_aspp', 'vessel_fusion'].includes(modelName);
        if (els.cfgEncoderGroup) els.cfgEncoderGroup.style.display = resnetModel ? '' : 'none';
        // clDice 对所有模型可见
        if (els.cfgClDiceGroup) els.cfgClDiceGroup.style.display = '';
        // Patch 训练对所有模型可见
        if (els.cfgPatchGroup) els.cfgPatchGroup.style.display = '';
        // 深监督对所有模型可见
        if (els.cfgDeepSupervisionGroup) els.cfgDeepSupervisionGroup.style.display = '';
        if (els.cfgLossWeightHint) {
            const cldice = Number(els.cfgClDiceWeight?.value || 0);
            if (cldice > 0) {
                els.cfgLossWeightHint.textContent = 'BCE + Dice + clDice 权重总和为 1.0（自动联动）';
            } else {
                els.cfgLossWeightHint.textContent = 'BCE 与 Dice 权重之和为 1.0（自动联动）';
            }
        }
    }

    function updateRegionWeights(changed) {
        const cldice = Number(els.cfgClDiceWeight?.value || 0);
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
        // 自动联动 in_channels: Frangi 启用时设为 2
        if (els.cfgInChannels) {
            els.cfgInChannels.value = (els.cfgFrangiEnabled && els.cfgFrangiEnabled.checked) ? 2 : 1;
        }
    }

    async function scanThresholds() {
        const checkpoints = getSelectedCheckpoints();
        if (!checkpoints.length) {
            showToast('请选择模型权重', 'error');
            return;
        }
        const rawValues = (els.thresholdScanValues?.value || '').trim();
        let thresholds;
        let scanStart = null;
        let scanEnd = null;
        const scanMode = els.thresholdScanMode?.value || 'coarse_fine';
        const rangeMatch = rawValues.match(/^\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*-\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$/);
        if (rangeMatch) {
            const start = Number(rangeMatch[1]);
            const end = Number(rangeMatch[2]);
            if (start > end) {
                showToast('扫描范围的起点不能大于终点', 'error');
                return;
            }
            scanStart = start;
            scanEnd = end;
            if (scanMode === 'custom') {
                const step = Number(els.thresholdFineStep?.value || 0.01);
                if (!(step > 0)) {
                    showToast('自定义步长必须大于 0', 'error');
                    return;
                }
                thresholds = [];
                for (let value = start; value <= end + 1e-9; value += step) {
                    thresholds.push(Number(value.toFixed(2)));
                }
                if (thresholds[thresholds.length - 1] < end) thresholds.push(Number(end.toFixed(2)));
            } else {
                thresholds = null;
            }
        } else {
            thresholds = rawValues.split(',').map(value => Number(value.trim())).filter(value => Number.isFinite(value));
        }
        if (thresholds && (!thresholds.length || thresholds.length > 101 || thresholds.some(value => value < 0 || value > 1))) {
            showToast('阈值必须是 0 到 1 之间的数字，最多 101 个', 'error');
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
                    checkpoints,
                    evaluation_split: els.thresholdEvaluationSplit?.value || 'dias_external',
                    ...(scanMode === 'coarse_fine' && scanStart !== null ? {
                        adaptive_scan: true,
                        scan_start: scanStart,
                        scan_end: scanEnd,
                        coarse_step: Number(els.thresholdCoarseStep?.value || 0.05),
                        fine_step: Number(els.thresholdFineStep?.value || 0.01)
                    } : { thresholds }),
                    ...getProcessingOptions()
                })
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || '阈值扫描失败');
            const modeText = data.scan_mode === 'coarse_fine'
                ? `粗扫最佳 ${Number(data.coarse_best_threshold).toFixed(2)}，精扫区间 ${data.fine_range.map(value => Number(value).toFixed(2)).join('-')}`
                : '自定义阈值扫描';
            if (els.thresholdScanStatus) els.thresholdScanStatus.textContent = `已完成：${data.evaluation_label}，${data.samples} 张，${modeText}，设备 ${data.device}`;
            if (els.thresholdScanBest) {
                els.thresholdScanBest.textContent = `最佳阈值 ${Number(data.best_threshold).toFixed(2)} | Dice ${formatNumber(data.best_dice)}`;
                els.thresholdScanBest.style.display = 'block';
            }
            if (els.thresholdScanTableBody) {
                els.thresholdScanTableBody.innerHTML = '';
                const bestThreshold = Number(data.best_threshold);
                const nearbyResults = [...data.results]
                    .sort((left, right) => Math.abs(Number(left.threshold) - bestThreshold) - Math.abs(Number(right.threshold) - bestThreshold))
                    .slice(0, 4)
                    .sort((left, right) => Number(left.threshold) - Number(right.threshold));
                nearbyResults.forEach(item => {
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

    let frangiPollId = null;

    async function pollFrangiProgress() {
        if (frangiPollId) clearInterval(frangiPollId);
        frangiPollId = setInterval(async () => {
            try {
                const res = await fetch('/api/frangi/status');
                const data = await res.json();
                if (!data.running) {
                    clearInterval(frangiPollId);
                    frangiPollId = null;
                    // 恢复按钮状态
                    if (els.btnGenerateFrangi) {
                        els.btnGenerateFrangi.disabled = false;
                        els.btnGenerateFrangi.textContent = '🔄 生成 Frangi 增强图';
                    }
                    if (els.btnStopFrangi) {
                        els.btnStopFrangi.style.display = 'none';
                    }
                    if (data.result) {
                        const r = data.result;
                        if (els.frangiGenerateStatus) {
                            els.frangiGenerateStatus.textContent = '✅ 完成！训练集 ' + r.train_count + ' 张，验证集 ' + r.val_count + ' 张';
                        }
                        showToast('Frangi 增强图生成完成', 'success');
                    } else if (data.progress && data.progress.startsWith('失败')) {
                        if (els.frangiGenerateStatus) {
                            els.frangiGenerateStatus.textContent = '❌ ' + data.progress;
                        }
                        showToast('Frangi 生成失败', 'error');
                    } else if (data.progress && data.progress.startsWith('已停止')) {
                        if (els.frangiGenerateStatus) {
                            els.frangiGenerateStatus.textContent = '⏹ ' + data.progress;
                        }
                        showToast('Frangi 生成已停止', 'info');
                    }
                    return;
                }
                if (els.frangiGenerateStatus) {
                    let msg = data.progress || '处理中...';
                    if (data.train_total > 0) {
                        msg += ' 训练集 ' + data.train_done + '/' + data.train_total;
                    }
                    if (data.val_total > 0) {
                        msg += ' 验证集 ' + data.val_done + '/' + data.val_total;
                    }
                    els.frangiGenerateStatus.textContent = msg;
                }
            } catch (e) {
                // 忽略轮询错误
            }
        }, 1000);
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

        document.querySelectorAll('.btn-choose-directory').forEach((button) => {
            button.addEventListener('click', () => chooseDirectory(button));
        });
        if (els.cfgClDiceWeight) {
            els.cfgClDiceWeight.addEventListener('input', () => updateRegionWeights('bce'));
        }
        if (els.cfgModelName) {
            els.cfgModelName.addEventListener('change', () => {
                updateModelSpecificControls(els.cfgModelName.value);
                updateRegionWeights('bce');
            });
        }

        // Frangi 开关切换
        if (els.cfgFrangiEnabled) {
            els.cfgFrangiEnabled.addEventListener('change', () => {
                if (els.cfgFrangiEnabled.checked && els.cfgTemporalEnabled) {
                    els.cfgTemporalEnabled.checked = false;
                }
                if (els.cfgFrangiGroup) {
                    els.cfgFrangiGroup.style.display = els.cfgFrangiEnabled.checked ? '' : 'none';
                }
                syncAdvancedDataControls();
                updateRegionWeights('bce');
            });
        }
        if (els.cfgTemporalEnabled) {
            els.cfgTemporalEnabled.addEventListener('change', () => {
                if (els.cfgTemporalEnabled.checked && els.cfgFrangiEnabled) {
                    els.cfgFrangiEnabled.checked = false;
                    if (els.cfgFrangiGroup) els.cfgFrangiGroup.style.display = 'none';
                }
                syncAdvancedDataControls();
            });
        }
        if (els.cfgDomainBalanceEnabled) {
            els.cfgDomainBalanceEnabled.addEventListener('change', syncAdvancedDataControls);
        }
        if (els.cfgCrossValidationEnabled) {
            els.cfgCrossValidationEnabled.addEventListener('change', syncAdvancedDataControls);
        }
        if (els.cfgNumFolds) {
            els.cfgNumFolds.addEventListener('input', syncAdvancedDataControls);
        }

        // Frangi 生成按钮
        if (els.btnGenerateFrangi) {
            els.btnGenerateFrangi.addEventListener('click', async () => {
                if (els.btnGenerateFrangi) {
                    els.btnGenerateFrangi.disabled = true;
                    els.btnGenerateFrangi.textContent = '⏳ 启动中...';
                }
                if (els.frangiGenerateStatus) {
                    els.frangiGenerateStatus.textContent = '正在启动后台生成...';
                }
                try {
                    const res = await fetch('/api/frangi/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(buildConfigFromForm())
                    });
                    const data = await res.json();
                    if (!res.ok || !data.success) {
                        throw new Error(data.message || '启动失败');
                    }
                    // 显示停止按钮，隐藏生成按钮
                    if (els.btnStopFrangi) els.btnStopFrangi.style.display = '';
                    els.btnGenerateFrangi.textContent = '⏳ 生成中...';
                    // 开始轮询进度
                    pollFrangiProgress();
                } catch (error) {
                    if (els.frangiGenerateStatus) {
                        els.frangiGenerateStatus.textContent = '❌ 失败: ' + error.message;
                    }
                    showToast('生成 Frangi 图失败: ' + error.message, 'error');
                    if (els.btnGenerateFrangi) {
                        els.btnGenerateFrangi.disabled = false;
                        els.btnGenerateFrangi.textContent = '🔄 生成 Frangi 增强图';
                    }
                }
            });
        }

        // Frangi 停止按钮
        if (els.btnStopFrangi) {
            els.btnStopFrangi.addEventListener('click', async () => {
                if (els.btnStopFrangi) els.btnStopFrangi.disabled = true;
                try {
                    const res = await fetch('/api/frangi/stop', { method: 'POST' });
                    const data = await res.json();
                    if (!res.ok && !data.success) {
                        showToast(data.message || '停止失败', 'error');
                        if (els.btnStopFrangi) els.btnStopFrangi.disabled = false;
                    } else {
                        showToast('正在停止...', 'info');
                    }
                } catch (error) {
                    showToast('停止请求失败: ' + error.message, 'error');
                    if (els.btnStopFrangi) els.btnStopFrangi.disabled = false;
                }
            });
        }

        // Frangi 清空按钮
        if (els.btnClearFrangi) {
            els.btnClearFrangi.addEventListener('click', async () => {
                if (!confirm('确定要清空所有已生成的 Frangi 增强图吗？此操作不可撤销。')) return;
                if (els.btnClearFrangi) {
                    els.btnClearFrangi.disabled = true;
                    els.btnClearFrangi.textContent = '⏳ 清空中...';
                }
                try {
                    const res = await fetch('/api/frangi/clear', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(buildConfigFromForm())
                    });
                    const data = await res.json();
                    if (!res.ok || !data.success) {
                        throw new Error(data.message || '清空失败');
                    }
                    if (els.frangiGenerateStatus) {
                        els.frangiGenerateStatus.textContent = '🗑 ' + data.message;
                    }
                    showToast(data.message, 'success');
                } catch (error) {
                    showToast('清空失败: ' + error.message, 'error');
                } finally {
                    if (els.btnClearFrangi) {
                        els.btnClearFrangi.disabled = false;
                        els.btnClearFrangi.textContent = '🗑 清空 Frangi 图';
                    }
                }
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
        if (els.infCheckpoint) {
            els.infCheckpoint.addEventListener('change', () => {
                const selected = Array.from(els.infCheckpoint.selectedOptions);
                if (selected.length > 5) {
                    selected.slice(5).forEach(option => { option.selected = false; });
                    showToast('概率集成最多选择 5 个权重', 'error');
                }
            });
        }
        if (els.infInputMode) {
            els.infInputMode.addEventListener('change', () => {
                const previewIndex = els.infInputMode.value === 'temporal' && state.inferenceFiles.length === 3 ? 1 : 0;
                const preview = state.inferenceFiles[previewIndex];
                if (preview && els.infOriginal) els.infOriginal.src = preview.dataUrl;
            });
        }
        if (els.btnThresholdScan) els.btnThresholdScan.addEventListener('click', scanThresholds);
        if (els.infProcessing) {
            els.infProcessing.addEventListener('change', updateProcessingHint);
            updateProcessingHint();
        }

        if (els.infThreshold) {
            els.infThreshold.addEventListener('input', (e) => {
                if (els.infThresholdValue) els.infThresholdValue.textContent = e.target.value;
            });
        }
    }

    // Start
    document.addEventListener('DOMContentLoaded', init);

})();
