import os
import sys
import yaml
import glob
import time
import base64
import logging
import threading
from io import BytesIO
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory, send_file, abort
from flask_cors import CORS

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
STATIC_FOLDER = os.path.join(PROJECT_ROOT, "web_static")

# 添加模块路径
sys.path.append(PROJECT_ROOT)

# 尝试导入项目模块
try:
    from src.dataset import get_dataloaders
    from src.models import build_model
    from src.losses import BCEDiceLoss
    from src.metrics import calculate_dice, calculate_iou, calculate_precision, calculate_recall
    from src.postprocess import postprocess_mask
    from inference import VesselSegmentor
except ImportError as e:
    logger.error(f"导入项目模块失败，请确保位于正确的工作目录中: {e}")

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)

# 全局训练状态
training_state = {
    "running": False,
    "stop_requested": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "dice": 0.0,
    "iou": 0.0,
    "lr": 0.0,
    "best_dice": 0.0,
    "history": [],
    "message": "就绪"
}

training_thread = None
inference_cache = {
    "model_path": None,
    "segmentor": None
}

# --- 辅助函数 ---

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config_data):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)

# --- 训练线程函数 ---
def run_training():
    global training_state
    
    try:
        config = load_config()
        
        # 初始化训练参数（与 configs/default.yaml 对齐）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_cfg = config.get('training', {})
        model_cfg = config.get('model', {})
        ckpt_cfg  = config.get('checkpoint', {})

        epochs        = train_cfg.get('epochs', 100)
        batch_size    = train_cfg.get('batch_size', 4)
        learning_rate = train_cfg.get('learning_rate', 1e-4)
        weight_decay  = train_cfg.get('weight_decay', 1e-5)

        training_state["total_epochs"] = epochs
        training_state["epoch"] = 0
        training_state["running"] = True
        training_state["stop_requested"] = False
        training_state["history"] = []
        training_state["best_dice"] = 0.0
        training_state["message"] = "准备数据和模型..."

        # 数据加载
        train_loader, val_loader = get_dataloaders(config)

        # 模型（build_model 签名: build_model(model_name, **kwargs)）
        model = build_model(
            model_cfg.get('name', 'unet_baseline'),
            in_channels=model_cfg.get('in_channels', 1),
            out_channels=model_cfg.get('out_channels', 1),
        )
        model = model.to(device)
        
        # 损失函数和优化器
        criterion = BCEDiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_dice = 0.0
        save_dir = os.path.join(PROJECT_ROOT, train_cfg.get('save_dir', 'checkpoints'))
        os.makedirs(save_dir, exist_ok=True)
        
        training_state["message"] = "开始训练..."
        
        for epoch in range(1, epochs + 1):
            if training_state["stop_requested"]:
                training_state["message"] = "训练已由用户终止"
                break
                
            training_state["epoch"] = epoch
            current_lr = optimizer.param_groups[0]['lr']
            training_state["lr"] = current_lr
            
            # --- 训练阶段 ---
            model.train()
            train_loss_total = 0.0
            
            for i, (images, masks) in enumerate(train_loader):
                if training_state["stop_requested"]:
                    break
                    
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item()
                
            train_loss = train_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            
            if training_state["stop_requested"]:
                break
                
            # --- 验证阶段 ---
            model.eval()
            val_loss_total = 0.0
            val_dice_total = 0.0
            val_iou_total = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    if training_state["stop_requested"]:
                        break
                        
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss_total += loss.item()
                    
                    # 预测
                    preds = torch.sigmoid(outputs)
                    preds = (preds > 0.5).float()
                    
                    val_dice_total += calculate_dice(preds, masks)
                    val_iou_total += calculate_iou(preds, masks)
            
            if training_state["stop_requested"]:
                break
                
            val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0
            val_dice = val_dice_total / len(val_loader) if len(val_loader) > 0 else 0
            val_iou = val_iou_total / len(val_loader) if len(val_loader) > 0 else 0
            
            # 更新学习率
            scheduler.step(val_dice)
            
            # 更新状态
            training_state["train_loss"] = train_loss
            training_state["val_loss"] = val_loss
            training_state["dice"] = val_dice
            training_state["iou"] = val_iou
            
            # 保存历史记录
            epoch_data = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "dice": float(val_dice),
                "iou": float(val_iou),
                "lr": float(current_lr)
            }
            training_state["history"].append(epoch_data)
            
            # 保存最佳模型
            if val_dice > best_dice:
                best_dice = val_dice
                training_state["best_dice"] = float(best_dice)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                
            # 保存最新模型
            torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))
            
            # 定期保存检查点
            save_interval = train_cfg.get('save_interval', 10)
            if epoch % save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
                
        if not training_state["stop_requested"]:
            training_state["message"] = "训练完成"
            
    except Exception as e:
        logger.error(f"训练线程中发生错误: {str(e)}", exc_info=True)
        training_state["message"] = f"错误: {str(e)}"
    finally:
        training_state["running"] = False
        training_state["stop_requested"] = False


# --- 路由 ---

@app.route('/')
def index():
    if os.path.exists(os.path.join(STATIC_FOLDER, 'index.html')):
        return send_from_directory(STATIC_FOLDER, 'index.html')
    return "Web静态文件未找到，请检查web_static目录是否存在。", 404

# **配置管理**
@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        config = load_config()
        # JS 直接访问顶层字段，返回扁平化的配置
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        config_data = request.json
        save_config(config_data)
        return jsonify({"success": True, "message": "配置保存成功"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# **数据集信息**
@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    try:
        config = load_config()
        data_cfg = config.get('dataset', {})
        train_img_dir = data_cfg.get('train_image_dir', '')
        val_img_dir   = data_cfg.get('val_image_dir', '')

        def count_images(path):
            if not path or not os.path.isdir(path):
                return 0
            return len([f for f in os.listdir(path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

        return jsonify({
            "train": {"count": count_images(train_img_dir), "path": train_img_dir},
            "val":   {"count": count_images(val_img_dir),   "path": val_img_dir},
        })
    except Exception as e:
        return jsonify({"train": {"count": 0, "path": ""}, "val": {"count": 0, "path": ""}})

@app.route('/api/dataset/preview/<path:filepath>', methods=['GET'])
def dataset_preview(filepath):
    try:
        abs_path = os.path.join(PROJECT_ROOT, filepath)
        if os.path.exists(abs_path):
            return send_file(abs_path)
        else:
            abort(404)
    except Exception as e:
        abort(404)

# **训练控制**
@app.route('/api/train/start', methods=['POST'])
def start_training():
    global training_thread, training_state
    
    if training_state["running"]:
        return jsonify({"success": False, "message": "训练已经在运行中"})
        
    try:
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        return jsonify({"success": True, "message": "训练已开始"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_state
    
    if not training_state["running"]:
        return jsonify({"success": False, "message": "目前没有正在运行的训练"})
        
    training_state["stop_requested"] = True
    return jsonify({"success": True, "message": "已发送停止信号"})

@app.route('/api/train/status', methods=['GET'])
def train_status():
    global training_state
    # JS 直接访问 running/epoch/dice 等字段，返回扁平化状态
    return jsonify(training_state)

# **模型信息**
@app.route('/api/model/info', methods=['GET'])
def model_info():
    try:
        config = load_config()
        model_cfg = config.get('model', {})
        model_name = model_cfg.get('name', 'unet_baseline')
        
        # 尝试构建模型以获取参数量
        model = build_model(
            model_name,
            in_channels=model_cfg.get('in_channels', 1),
            out_channels=model_cfg.get('out_channels', 1)
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return jsonify({
            "success": True,
            "data": {
                "name": model_name,
                "architecture": str(model.__class__.__name__),
                "total_params": total_params,
                "trainable_params": trainable_params
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# **检查点管理**
@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    try:
        config = load_config()
        # 从 training.save_dir 读取
        save_dir = os.path.join(PROJECT_ROOT,
                                config.get('training', {}).get('save_dir', 'checkpoints'))
        if not os.path.exists(save_dir):
            return jsonify([])  # 直接返回空列表

        checkpoints = []
        for file in os.listdir(save_dir):
            if file.endswith('.pth'):
                fp = os.path.join(save_dir, file)
                stat = os.stat(fp)
                checkpoints.append({
                    "name":     file,   # JS 使用 cp.name
                    "size":     stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify(checkpoints)  # 直接返回 list，JS 用 data.forEach
    except Exception as e:
        logger.error(f"列出检查点失败: {e}")
        return jsonify([])

@app.route('/api/checkpoints/<filename>', methods=['DELETE'])
def delete_checkpoint(filename):
    try:
        config = load_config()
        save_dir = os.path.join(PROJECT_ROOT, config.get('training', {}).get('save_dir', 'checkpoints'))
        file_path = os.path.join(save_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True, "message": f"已删除 {filename}"})
        else:
            return jsonify({"success": False, "message": "文件不存在"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# **推理**
@app.route('/api/inference', methods=['POST'])
def run_inference():
    try:
        req = request.json
        if not req or 'image_base64' not in req:
            return jsonify({"error": "未提供 image_base64"}), 400

        image_data = req['image_base64']  # JS 已剥离 data:... 前缀
        checkpoint_name = req.get('checkpoint', 'best_model.pth')
        threshold = float(req.get('threshold', 0.5))

        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('L')  # 灰度

        config = load_config()
        save_dir = os.path.join(PROJECT_ROOT,
                                config.get('training', {}).get('save_dir', 'checkpoints'))
        model_path = os.path.join(save_dir, checkpoint_name)

        global inference_cache
        if not os.path.exists(model_path):
            return jsonify({"error": f"检查点未找到: {checkpoint_name}"}), 404

        if inference_cache["model_path"] != model_path or inference_cache["segmentor"] is None:
            logger.info(f"Loading inference model: {model_path}")
            model_name = config.get('model', {}).get('name', 'unet_baseline')
            inference_cache["segmentor"] = VesselSegmentor(
                model_path=model_path, 
                model_name=model_name, 
                threshold=threshold
            )
            inference_cache["model_path"] = model_path
        else:
            inference_cache["segmentor"].threshold = threshold
            
        segmentor = inference_cache["segmentor"]
        
        import numpy as np
        img_np = np.array(img)
        mask = segmentor.predict_array(img_np)  # 返回 (H,W) uint8

        mask_img = Image.fromarray(mask)
        buf = BytesIO()
        mask_img.save(buf, format='PNG')
        mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # JS 期望 data.mask_base64
        return jsonify({"mask_base64": mask_b64})
    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# **系统信息**
@app.route('/api/system', methods=['GET'])
def system_info():
    try:
        import platform
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else '无 GPU'
        gpu_memory = ''
        if cuda_available:
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory = f'{total // (1024**2):,} MB'

        # 直接返回 JS 访问的字段名（flat）
        return jsonify({
            "gpu_available":  cuda_available,
            "gpu_name":       gpu_name,
            "gpu_memory":     gpu_memory,
            "python_version": platform.python_version(),
            "torch_version":  torch.__version__,
            "os":             platform.system() + ' ' + platform.release(),
        })
    except Exception as e:
        return jsonify({"gpu_available": False, "gpu_name": "未知", "gpu_memory": "",
                        "python_version": "", "torch_version": "", "os": ""})

if __name__ == '__main__':
    port = 5001
    logger.info(f"正在启动训练管理Web服务器 (端口 {port})...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
