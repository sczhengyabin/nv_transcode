# NV Transcoder

[中文说明](README_zh.md)

A lightweight NVIDIA GPU video transcoder built around FFmpeg.

It uses a strict GPU video pipeline:

```text
NVDEC / CUDA decode -> CUDA filters -> NVENC encode
```

No CPU video decode or scaling fallback is used. Audio streams are copied without re-encoding.

## Features

- NVIDIA NVDEC/CUDA/NVENC video pipeline
- H.264, HEVC and AV1 NVENC output
- CQ or bitrate-based encoding
- GPU resize, crop, rotate and padding
- Concurrent transcoding jobs
- Recursive batch mode
- Long-running watch-folder mode
- Safe temporary-file and atomic output handling
- Automatic FFmpeg download from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) when FFmpeg is not available
- Graceful shutdown on Ctrl+C and service termination
- Unraid User Scripts friendly
- Optional Linux output ownership and permission control

## Requirements

- Python 3.10+
- NVIDIA GPU with NVENC/NVDEC support
- NVIDIA driver
- Linux or Windows

FFmpeg can be supplied with `--ffmpeg`. If it is not found in `PATH` or beside the script, a compatible BtbN build is downloaded automatically.

## Typical Usage

### Transcode one file

```bash
python nv_transcode.py input.mkv \
  --cq 28 \
  --codec hevc \
  -r 1080p
```

### Transcode a directory recursively

```bash
python nv_transcode.py /video/source \
  -o /video/output \
  --cq 28 \
  --codec hevc \
  -r 1080p \
  --recursive \
  --workers 3
```

### Watch a folder

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

The watch directory is always scanned recursively. New files are queued after they remain unchanged for the configured stability period.

### Unraid example

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

## Notes

- Output video is written as MP4.
- The first video stream is transcoded; audio streams are copied.
- If an audio stream cannot be copied into MP4, the job fails instead of transcoding the audio.
- Existing outputs are skipped unless `--overwrite` is used.
- Linux UID/GID and permission options affect output paths created by the script.
- Use `python nv_transcode.py --help` for the full option list.
