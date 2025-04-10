import os
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
from modelscope import AutoModelForImageSegmentation

app = Flask(__name__, static_url_path='', static_folder='static')
# 配置CORS，允许所有来源
CORS(app, resources={r"/*": {"origins": "*"}})

# Use CPU
device = torch.device('cpu')

# 配置常量
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

# 创建上传和输出目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 全局变量用于存储模型
model = None

def allowed_file(filename):
    """检查文件是否为允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """加载图片分割模型"""
    global model
    if model is None:
        print("正在加载模型...")
        model = AutoModelForImageSegmentation.from_pretrained('AI-ModelScope/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        model.to(device)
        model.eval()
        print("模型加载完成！")
    return model

def process_image(image_path):
    """处理单张图片并移除背景"""
    # 记录开始时间
    start_time = time.time()
    
    # 准备模型和转换
    model = load_model()
    # image_size = (1024, 1024)
    image_size = (768, 768)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 获取文件信息
    file_size = os.path.getsize(image_path) / 1024  # KB
    file_name = os.path.basename(image_path)
    
    # 生成唯一输出文件名
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{uuid.uuid4()}_{base_name}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # 打开并处理图片
    with Image.open(image_path) as img:
        original_format = img.format
        original_mode = img.mode
        original_size = img.size
        
        # 转换RGB并处理
        image = img.convert('RGB')
        input_images = transform_image(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        
        # 生成掩码
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        
        # 添加透明通道
        image.putalpha(mask)
        
        # 保存结果为PNG格式（支持透明度）
        image.save(output_path, format='PNG')
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 返回结果
    return {
        'original_filename': file_name,
        'processed_filename': output_filename,
        'original_size': original_size,
        'format': original_format,
        'mode': original_mode,
        'file_size_kb': round(file_size, 2),
        'process_time': round(process_time, 2),
        'output_url': f"/download/{output_filename}"
    }

@app.route('/')
def index():
    """提供前端页面"""
    return app.send_static_file('index.html')

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    """API端点：处理上传的图片并移除背景"""
    print("接收到上传请求")
    
    # 检查是否有文件上传
    if 'files[]' not in request.files:
        print("没有找到上传文件")
        return jsonify({'error': '没有上传文件'}), 400
    
    files = request.files.getlist('files[]')
    print(f"接收到 {len(files)} 个文件")
    
    # 检查是否有文件被选择
    if len(files) == 0:
        return jsonify({'error': '没有选择文件'}), 400
    
    results = []
    for file in files:
        # 检查文件是否有效
        if file and allowed_file(file.filename):
            # 创建唯一文件名并保存上传的文件
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            try:
                print(f"处理文件: {file.filename}")
                # 处理图片
                result = process_image(file_path)
                results.append(result)
                print(f"文件处理完成: {file.filename}")
            except Exception as e:
                print(f"处理文件失败: {file.filename} - {str(e)}")
                results.append({
                    'original_filename': file.filename,
                    'error': str(e)
                })
        else:
            print(f"不支持的文件格式: {file.filename}")
            results.append({
                'original_filename': file.filename,
                'error': '不支持的文件格式'
            })
    
    print(f"所有文件处理完成，返回 {len(results)} 个结果")
    return jsonify({'results': results})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载处理后的图片"""
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    load_model()  # 预加载模型
    print("\n✅ 服务启动成功！请访问 http://localhost:5000 使用图片背景移除工具")
    app.run(debug=False, host='0.0.0.0', port=5000)