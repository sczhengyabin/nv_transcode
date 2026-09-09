# NV Transcoder（NV 转码器）

一个基于 FFmpeg 构建的轻量级 NVIDIA GPU 视频转码工具。
它采用严格的 GPU 视频处理管线：

```text
NVDEC / CUDA 解码 -> CUDA 滤镜 -> NVENC 编码
```

不使用 CPU 视频解码或缩放回退方案。音频流直接拷贝，不重新编码。

## 特性

*   NVIDIA NVDEC/CUDA/NVENC 视频处理管线
    
*   支持 H.264、HEVC 和 AV1 的 NVENC 输出
    
*   支持 CQ（恒定质量）或基于码率的编码
    
*   GPU 缩放、裁剪、旋转和填充（padding）
    
*   并发转码任务
    
*   递归批处理模式
    
*   长时间运行的监听文件夹（watch-folder）模式
    
*   安全的临时文件和原子化输出处理
    
*   当 FFmpeg 不可用时，自动从 BtbN/FFmpeg-Builds 下载
    
*   在 Ctrl+C 和服务终止时优雅关闭
    
*   兼容 Unraid User Scripts
    
*   可选的 Linux 输出属主（ownership）和权限控制
    

## 要求

*   Python 3.10+
    
*   支持 NVENC/NVDEC 的 NVIDIA GPU
    
*   NVIDIA 驱动
    
*   Linux 或 Windows
    

FFmpeg 可通过 `--ffmpeg` 指定。如果在 `PATH` 或脚本所在目录中找不到，会自动下载一个兼容的 BtbN 构建版本。

## 典型用法

### 转码单个文件

```bash
python nv_transcode.py input.mkv \
  --cq 28 \
  --codec hevc \
  -r 1080p
```

### 递归转码一个目录

```bash
python nv_transcode.py /video/source \
  -o /video/output \
  --cq 28 \
  --codec hevc \
  -r 1080p \
  --recursive \
  --workers 3
```

### 监听一个文件夹

```bash
python nv_transcode.py \
  --watch /video/incoming \
  --output /video/converted \
  --temp-dir /tmp/transcode \
  --cq 28 \
  -r 1080p \
  --workers 3 \
  --keep-sub-path
```

监听目录始终会递归扫描。新文件在保持配置好的稳定时间（stability period）不变后才会被加入队列。

### Unraid 示例

```bash
python nv_transcode.py \
  --watch /mnt/user/incoming \
  --output /mnt/user/media \
  --temp-dir /mnt/cache/transcode-temp \
  --cq 28 \
  -r 1080p \
  --workers 3 \
  --keep-sub-path \
  --copy-other-files \
  --keep-src 0 \
  --output-uid 99 \
  --output-gid 100 \
  --output-mode 0775 \
  --output-file-mode 0664
```

## 说明

*   输出视频以 MP4 格式写入。
    
*   只转码第一条视频流；音频流直接拷贝。
    
*   如果某个音频流无法拷贝进 MP4，该任务会失败，而不会转码音频。
    
*   已存在的输出文件会被跳过，除非使用 `--overwrite`。
    
*   Linux 的 UID/GID 和权限选项会影响脚本创建的输出路径。
    
*   使用 `python nv_transcode.py --help` 查看完整选项列表。