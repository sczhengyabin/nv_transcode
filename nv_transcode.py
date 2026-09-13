#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Strict NVIDIA batch transcoder with self-bootstrapping FFmpeg.

Pipeline:
    NVDEC/CUDA decode -> scale_cuda -> NVENC encode

No software decode/scale fallback is provided.
All mapped audio streams are copied unchanged.

Examples:
    python nv_transcode.py "D:\Videos\input.mkv" --cq 28 --resolution 1280x720
    python nv_transcode.py "D:\Videos" --cq 28 --resolution 720p --workers 3
    python nv_transcode.py "D:\Videos" --bitrate 4M --resolution 1280x720 --workers 2
"""

from __future__ import annotations

import argparse
import atexit
import errno
import hashlib
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".m4v", ".avi", ".ts", ".mts", ".m2ts",
    ".webm", ".flv", ".wmv", ".mpg", ".mpeg", ".vob", ".3gp"
}

ENCODERS = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "av1": "av1_nvenc",
}

BTBN_RELEASE_BASE = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
)
BTBN_CHECKSUMS = "checksums.sha256"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# NVIDIA CUVID decoders that expose the decoder-side GPU crop option.
CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1": "av1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "mjpeg": "mjpeg_cuvid",
}

_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_PROCESS_LOCK = threading.Lock()
_OUTPUT_DIR_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()

# Graceful shutdown signal captured by the main thread.
# SIGKILL cannot be caught; Linux FFmpeg children are additionally protected
# by PR_SET_PDEATHSIG below.
_RECEIVED_SIGNAL: int | None = None


class ShutdownSignal(KeyboardInterrupt):
    """Raised in the main thread for SIGTERM/SIGHUP/SIGQUIT/SIGINT."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


# Saved POSIX terminal state. This is a final safety net in case an external
# program ever changes TTY flags before being interrupted.
_SAVED_TTY_STATE = None
_SAVED_TTY_FD: int | None = None
_PRINT_LOCK = threading.Lock()


def save_terminal_state() -> None:
    """Save the current POSIX terminal settings, if stdin is a TTY."""
    global _SAVED_TTY_STATE, _SAVED_TTY_FD

    if os.name == "nt":
        return

    try:
        if not sys.stdin.isatty():
            return

        import termios

        fd = sys.stdin.fileno()
        _SAVED_TTY_FD = fd
        _SAVED_TTY_STATE = termios.tcgetattr(fd)
    except Exception:
        _SAVED_TTY_STATE = None
        _SAVED_TTY_FD = None


def restore_terminal_state() -> None:
    """Restore the POSIX terminal settings captured at program startup."""
    if os.name == "nt":
        return

    if _SAVED_TTY_STATE is None or _SAVED_TTY_FD is None:
        return

    try:
        import termios

        termios.tcsetattr(
            _SAVED_TTY_FD,
            termios.TCSANOW,
            _SAVED_TTY_STATE,
        )
    except Exception:
        pass


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with _PRINT_LOCK:
        print(*args, **kwargs)


def safe_print_block(lines: list[str]) -> None:
    """Print a complete progress snapshot atomically."""
    with _PRINT_LOCK:
        for line in lines:
            print(line, flush=True)


def eprint(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with _PRINT_LOCK:
        print(*args, file=sys.stderr, **kwargs)


def clock_text() -> str:
    return time.strftime("%H:%M:%S")


def log_event(
    level: str,
    label: str,
    value: str = "",
    *,
    error: bool = False,
) -> None:
    """NBMiner-style timestamped event line."""
    line = f"{clock_text()} {level:<5} {label:<10}"
    if value:
        line += f": {value}"

    if error:
        eprint(line)
    else:
        safe_print(line)


def print_banner() -> None:
    width = 55
    safe_print("-" * width)
    safe_print("|" + "NV Transcoder v29 - NVIDIA GPU".center(width - 2) + "|")
    safe_print("-" * width)




def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:
        return f"signal {signum}"


def _shutdown_signal_handler(signum, _frame) -> None:
    """
    Convert service-style termination signals into the same unwind path as
    Ctrl+C. Do not perform blocking cleanup inside the signal handler itself.
    """
    global _RECEIVED_SIGNAL

    # Ignore repeated termination signals while the first shutdown is already
    # unwinding. This avoids re-entering cleanup while locks are being released.
    if _RECEIVED_SIGNAL is not None:
        return

    _RECEIVED_SIGNAL = int(signum)
    raise ShutdownSignal(int(signum))


def install_signal_handlers() -> list[str]:
    """
    Install graceful shutdown handlers supported by the current platform.

    POSIX/Unraid:
        SIGINT, SIGTERM, SIGHUP, SIGQUIT

    Windows:
        only signal constants supported by Python on that platform are used.
    """
    installed: list[str] = []

    names = ["SIGINT", "SIGTERM"]
    if os.name != "nt":
        names += ["SIGHUP", "SIGQUIT"]

    for name in names:
        sig = getattr(signal, name, None)
        if sig is None:
            continue

        signal.signal(sig, _shutdown_signal_handler)
        installed.append(name)

    return installed


def _read_proc_cmdline(pid: int) -> str:
    """Best-effort Linux /proc cmdline reader."""
    if not sys.platform.startswith("linux"):
        return ""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return ""

    return raw.replace(b"\0", b" ").decode(
        "utf-8",
        errors="replace",
    ).strip()


def _read_proc_ppid(pid: int) -> int | None:
    """Best-effort Linux /proc PPID reader."""
    if not sys.platform.startswith("linux"):
        return None

    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        if len(fields) >= 4:
            return int(fields[3])
    except Exception:
        pass

    return None


def detect_unraid_userscripts_launcher(
    max_depth: int = 8,
) -> tuple[bool, list[tuple[int, str]]]:
    """
    Detect whether nv_transcode was launched through Unraid User Scripts.

    Current User Scripts uses temporary scripts below:
        /tmp/user.scripts/tmpScripts/

    and launch helpers below:
        /usr/local/emhttp/plugins/user.scripts/

    Walk a few ancestors so this still works whether PHP -> sh -> bash ->
    python contains one shell layer or several.
    """
    if not sys.platform.startswith("linux"):
        return False, []

    markers = (
        "/tmp/user.scripts/tmpscripts/",
        "/usr/local/emhttp/plugins/user.scripts/",
    )

    chain: list[tuple[int, str]] = []
    pid = os.getppid()
    seen: set[int] = set()

    for _ in range(max_depth):
        if pid <= 1 or pid in seen:
            break

        seen.add(pid)
        cmdline = _read_proc_cmdline(pid)
        chain.append((pid, cmdline))

        low = cmdline.lower()
        if any(marker in low for marker in markers):
            return True, chain

        parent = _read_proc_ppid(pid)
        if parent is None or parent == pid:
            break
        pid = parent

    return False, chain


def set_linux_parent_death_signal(signum: int) -> None:
    """Set Linux PR_SET_PDEATHSIG or raise OSError."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("PR_SET_PDEATHSIG is Linux-only")

    import ctypes

    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.prctl(PR_SET_PDEATHSIG, int(signum), 0, 0, 0)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def enable_self_parent_death_guard() -> tuple[bool, int | None]:
    """
    If running under Unraid User Scripts, bind nv_transcode itself to its
    immediate launcher parent with PR_SET_PDEATHSIG=SIGTERM.

    Why this is needed:
      User Scripts Abort sends SIGTERM only to the direct child of its recorded
      runner PID and then SIGKILLs the recorded runner. The Python transcoder
      may be a grandchild behind the temporary shell script. When that shell is
      killed, Python can otherwise be reparented and continue running.

    With this guard:
      parent shell dies -> Linux kernel sends SIGTERM to nv_transcode ->
      existing graceful shutdown handler kills all FFmpeg jobs -> Python exits.

    Returns:
        (enabled, guarded_parent_pid)
    """
    if not sys.platform.startswith("linux"):
        return False, None

    detected, _chain = detect_unraid_userscripts_launcher()
    if not detected:
        return False, None

    expected_parent = os.getppid()

    try:
        set_linux_parent_death_signal(signal.SIGTERM)

        # Close the race where the parent disappears while prctl is being set.
        if os.getppid() != expected_parent:
            # No FFmpeg jobs exist this early in startup, so an immediate
            # service-style exit is safe and avoids becoming an orphan.
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(128 + int(signal.SIGTERM))

        return True, expected_parent

    except Exception as exc:
        log_event(
            "WARN",
            "Parent guard",
            f"failed to enable: {exc}",
            error=True,
        )
        return False, expected_parent


def linux_parent_death_wrap(cmd: list[str]) -> list[str]:
    """
    On Linux, launch FFmpeg through this same script as a tiny exec wrapper.

    The wrapper sets PR_SET_PDEATHSIG=SIGTERM and then execs FFmpeg. Therefore
    if nv_transcode itself is killed abruptly (including SIGKILL), Linux sends
    SIGTERM to the FFmpeg process automatically.

    The wrapper is not used on non-Linux platforms.
    """
    if not sys.platform.startswith("linux"):
        return cmd

    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_ffmpeg-pdeath-wrapper",
        str(os.getpid()),
        "--",
        *cmd,
    ]


def run_linux_pdeath_wrapper(
    expected_parent_pid: int,
    cmd: list[str],
) -> int:
    """
    Internal Linux-only mode. Set PR_SET_PDEATHSIG then exec the real command.

    The getppid() check closes the race where the parent could die immediately
    before/while PR_SET_PDEATHSIG is being installed.
    """
    if not sys.platform.startswith("linux"):
        sys.stderr.write(
            "PDEATH wrapper is only supported on Linux.\\n"
        )
        return 125

    if not cmd:
        sys.stderr.write("PDEATH wrapper: missing command.\\n")
        return 125

    try:
        set_linux_parent_death_signal(signal.SIGTERM)

        # Parent may have died between fork/exec and prctl().
        if os.getppid() != expected_parent_pid:
            return 128 + int(signal.SIGTERM)

        os.execvpe(cmd[0], cmd, os.environ.copy())

    except Exception as exc:
        sys.stderr.write(
            f"PDEATH wrapper failed: {exc}\\n"
        )
        return 125

    return 125  # execvpe never returns on success.


def maybe_run_internal_mode() -> int | None:
    """Handle hidden internal helper modes before normal argparse processing."""
    if len(sys.argv) < 2:
        return None

    if sys.argv[1] != "--_ffmpeg-pdeath-wrapper":
        return None

    if (
        len(sys.argv) < 5
        or sys.argv[3] != "--"
    ):
        sys.stderr.write(
            "Invalid internal PDEATH wrapper invocation.\\n"
        )
        return 125

    try:
        expected_parent_pid = int(sys.argv[2])
    except ValueError:
        sys.stderr.write(
            "Invalid PDEATH wrapper parent PID.\\n"
        )
        return 125

    return run_linux_pdeath_wrapper(
        expected_parent_pid,
        sys.argv[4:],
    )


def script_directory() -> Path:
    """Directory containing this script; auto-downloaded FFmpeg lives here."""
    return Path(__file__).resolve().parent


def ffmpeg_executable_name() -> str:
    return "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"


def local_ffmpeg_path() -> Path:
    return script_directory() / ffmpeg_executable_name()


def detect_btbn_asset() -> str:
    """Return the BtbN latest static GPL archive for this OS/architecture."""
    machine = platform.machine().lower()
    system = platform.system()

    if machine in {"x86_64", "amd64"}:
        arch = "64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        raise RuntimeError(
            "当前 CPU 架构不受 BtbN 自动下载支持: "
            f"{platform.machine() or 'unknown'}。"
            "请使用 --ffmpeg 手动指定可执行文件。"
        )

    if system == "Linux":
        return f"ffmpeg-master-latest-linux{arch}-gpl.tar.xz"

    if system == "Windows":
        return f"ffmpeg-master-latest-win{arch}-gpl.zip"

    raise RuntimeError(
        f"当前系统不支持 FFmpeg 自动下载: {system or sys.platform}。"
        "自动下载仅支持 Windows 64-bit/ARM64 和 Linux x86_64/ARM64；"
        "请使用 --ffmpeg 手动指定。"
    )


def _download_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "nv-transcode/27",
            "Accept": "application/octet-stream,*/*",
        },
    )


def download_text(url: str, timeout: float = 30.0) -> str:
    try:
        with urllib.request.urlopen(
            _download_request(url),
            timeout=timeout,
        ) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"下载失败: {url}\n{exc}") from exc


def expected_sha256(checksums: str, filename: str) -> str:
    pattern = re.compile(
        rf"^([0-9a-fA-F]{{64}})\s+\*?{re.escape(filename)}\s*$",
        re.MULTILINE,
    )
    match = pattern.search(checksums)
    if match is None:
        raise RuntimeError(
            f"BtbN checksums.sha256 中找不到 {filename}，"
            "为安全起见拒绝继续。"
        )
    return match.group(1).lower()


def download_archive(
    url: str,
    destination: Path,
    *,
    expected_hash: str,
    timeout: float = 60.0,
) -> None:
    """Stream an archive to disk, show coarse progress, and verify SHA-256."""
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(
            _download_request(url),
            timeout=timeout,
        ) as response, destination.open("wb") as output:
            raw_length = response.headers.get("Content-Length")
            try:
                total = int(raw_length) if raw_length else 0
            except ValueError:
                total = 0

            downloaded = 0
            next_report = 10

            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = min(100, int(downloaded * 100 / total))
                    if percent >= next_report:
                        log_event(
                            "INFO",
                            "Download",
                            f"{percent:>3}% | "
                            f"{format_bytes(downloaded)} / "
                            f"{format_bytes(total)}",
                        )
                        next_report = (percent // 10 + 1) * 10

    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"FFmpeg 下载失败: {url}\n{exc}") from exc

    actual_hash = digest.hexdigest().lower()
    if actual_hash != expected_hash.lower():
        raise RuntimeError(
            "FFmpeg 下载文件 SHA-256 校验失败。\n"
            f"expected: {expected_hash}\n"
            f"actual:   {actual_hash}"
        )


def _archive_ffmpeg_member(
    names,
    executable_name: str,
) -> str:
    suffix = f"/bin/{executable_name}".lower()

    for name in names:
        normalized = str(name).replace("\\", "/")
        low = normalized.lower()
        if low == f"bin/{executable_name}".lower() or low.endswith(suffix):
            return str(name)

    raise RuntimeError(
        f"下载包中找不到 bin/{executable_name}"
    )


def extract_ffmpeg_binary(
    archive: Path,
    destination: Path,
) -> None:
    """Extract only ffmpeg(.exe), never the whole third-party archive."""
    executable_name = ffmpeg_executable_name()
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, "r") as zf:
                member = _archive_ffmpeg_member(
                    zf.namelist(),
                    executable_name,
                )
                with zf.open(member, "r") as source, temporary.open("wb") as out:
                    shutil.copyfileobj(source, out, length=DOWNLOAD_CHUNK_SIZE)

        else:
            try:
                tf = tarfile.open(archive, "r:xz")
            except (tarfile.TarError, OSError) as exc:
                raise RuntimeError(
                    f"无法识别 FFmpeg 下载包格式: {archive.name}"
                ) from exc

            with tf:
                regular = [
                    member
                    for member in tf.getmembers()
                    if member.isfile()
                ]
                member_name = _archive_ffmpeg_member(
                    [member.name for member in regular],
                    executable_name,
                )
                member = next(
                    item for item in regular
                    if item.name == member_name
                )
                source = tf.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        f"无法读取压缩包成员: {member.name}"
                    )
                with source, temporary.open("wb") as out:
                    shutil.copyfileobj(
                        source,
                        out,
                        length=DOWNLOAD_CHUNK_SIZE,
                    )

        if temporary.stat().st_size <= 0:
            raise RuntimeError("解压得到的 FFmpeg 文件为空")

        if platform.system() != "Windows":
            temporary.chmod(0o755)

        os.replace(temporary, destination)

    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def auto_download_ffmpeg() -> str:
    """
    Download the latest BtbN static GPL FFmpeg into the script directory.

    The archive is temporary. Only ffmpeg(.exe) is retained.
    """
    target = local_ffmpeg_path()
    if target.is_file():
        return str(target)

    asset = detect_btbn_asset()
    base = BTBN_RELEASE_BASE.rstrip("/")
    checksum_url = f"{base}/{BTBN_CHECKSUMS}"
    archive_url = f"{base}/{asset}"
    archive = script_directory() / (
        f".{asset}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )

    log_event(
        "INFO",
        "FFmpeg",
        f"not found | auto-download BtbN {asset}",
    )
    log_event(
        "INFO",
        "Download",
        f"destination: {target}",
    )

    try:
        checksums = download_text(checksum_url)
        expected_hash = expected_sha256(checksums, asset)

        download_archive(
            archive_url,
            archive,
            expected_hash=expected_hash,
        )

        log_event(
            "INFO",
            "Verify",
            f"SHA-256 OK | {expected_hash[:12]}...",
        )

        extract_ffmpeg_binary(archive, target)

        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(
                f"FFmpeg 解压后不存在或为空: {target}"
            )

        log_event(
            "INFO",
            "FFmpeg",
            f"installed: {target}",
        )
        return str(target)

    except PermissionError as exc:
        raise RuntimeError(
            "脚本目录不可写，无法自动安装 FFmpeg: "
            f"{script_directory()}\n"
            "请调整目录权限，或使用 --ffmpeg 手动指定。"
        ) from exc

    finally:
        try:
            archive.unlink(missing_ok=True)
        except OSError:
            pass


def find_ffmpeg(explicit: str | None = None) -> str:
    """
    Resolve FFmpeg with deterministic precedence:

      1. explicit --ffmpeg
      2. ffmpeg from PATH
      3. previously auto-downloaded ffmpeg beside this script
      4. download latest matching BtbN static GPL build beside this script

    An invalid explicit --ffmpeg is treated as a user error and does not
    silently fall through to auto-download.
    """
    if explicit:
        candidate = Path(explicit).expanduser()

        if candidate.parent != Path("."):
            candidate = candidate.resolve()

            if not candidate.is_file():
                raise RuntimeError(f"指定的 FFmpeg 不存在: {candidate}")

            if os.name != "nt" and not os.access(candidate, os.X_OK):
                raise RuntimeError(
                    f"指定的 FFmpeg 没有可执行权限: {candidate}\n"
                    f"可执行: chmod +x {candidate}"
                )
            return str(candidate)

        found = shutil.which(explicit)
        if found:
            return found

        raise RuntimeError(
            f"找不到指定的 FFmpeg 命令: {explicit}\n"
            "请传入完整路径，或确保该命令已加入 PATH。"
        )

    names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg",)
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    local = local_ffmpeg_path()
    if local.is_file():
        if platform.system() != "Windows" and not os.access(local, os.X_OK):
            try:
                local.chmod(local.stat().st_mode | 0o111)
            except OSError as exc:
                raise RuntimeError(
                    f"脚本目录中的 FFmpeg 不可执行: {local}"
                ) from exc
        return str(local)

    return auto_download_ffmpeg()



def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """
    Run a short FFmpeg probe/preflight command under the same lifecycle rules
    as a real transcode.

    v23 used subprocess.run() here, which meant these short-lived FFmpeg
    processes were invisible to terminate_all_ffmpeg(). If an Abort arrived
    while a ThreadPoolExecutor worker was probing a file, that worker could
    remain blocked and keep the Python interpreter alive.

    Every probe is now:
      * protected by Linux PR_SET_PDEATHSIG
      * placed in its own process group/session
      * registered in _ACTIVE_PROCESSES
      * explicitly killed if the calling thread is unwound
    """
    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt

    spawn_cmd = linux_parent_death_wrap(cmd)

    process = subprocess.Popen(
        spawn_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_group_kwargs(),
    )

    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)

    try:
        if _STOP_EVENT.is_set():
            terminate_process_tree(process, force=True)

        try:
            stdout, _ = process.communicate()
        except BaseException:
            # Important for signals delivered to the main thread while it is
            # inside a preflight/probe communicate().
            terminate_process_tree(process, force=False)

            try:
                process.wait(timeout=0.5)
            except Exception:
                terminate_process_tree(process, force=True)

            raise

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=None,
        )

    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)


def probe_encoder_cq_range(
    ffmpeg: str,
    encoder: str,
) -> tuple[float, float] | None:
    """
    Query the selected FFmpeg NVENC encoder for its current -cq range.

    FFmpeg builds/versions can expose different bounds, especially for AV1,
    so public-facing validation should not hard-code a version-specific limit.
    """
    p = run_capture([
        ffmpeg,
        "-hide_banner",
        "-h", f"encoder={encoder}",
    ])

    for line in p.stdout.splitlines():
        if not re.search(r"^\s*-cq\s+", line):
            continue

        m = re.search(
            r"\(from\s+(-?\d+(?:\.\d+)?)\s+"
            r"to\s+(-?\d+(?:\.\d+)?)\)",
            line,
        )
        if m:
            return float(m.group(1)), float(m.group(2))

    return None


def preflight(
    ffmpeg: str,
    encoder: str,
    *,
    need_pad: bool = False,
    need_rotate: bool = False,
) -> None:
    log_event("INFO", "FFmpeg", ffmpeg)

    p = run_capture([ffmpeg, "-hide_banner", "-hwaccels"])
    if p.returncode != 0 or not re.search(r"(?mi)^\s*cuda\s*$", p.stdout):
        raise RuntimeError("当前 FFmpeg 没有 CUDA hwaccel 支持。\n\n" + p.stdout)

    p = run_capture([ffmpeg, "-hide_banner", "-filters"])
    if p.returncode != 0:
        raise RuntimeError("无法读取 FFmpeg filter 列表。\n\n" + p.stdout)

    required_filters = ["scale_cuda"]
    if need_pad:
        required_filters.append("pad_cuda")
    if need_rotate:
        required_filters.append("transpose_cuda")

    missing = [
        name
        for name in required_filters
        if not re.search(rf"(?m)\b{re.escape(name)}\b", p.stdout)
    ]
    if missing:
        raise RuntimeError(
            "当前 FFmpeg 缺少纯 GPU 所需 CUDA filter: "
            + ", ".join(missing)
            + "\n拒绝回退 CPU。\n\n"
            + p.stdout
        )

    p = run_capture([ffmpeg, "-hide_banner", "-encoders"])
    if p.returncode != 0 or not re.search(
        rf"(?m)^\s*V\S*\s+{re.escape(encoder)}\b", p.stdout
    ):
        raise RuntimeError(f"当前 FFmpeg 没有 {encoder}。\n\n" + p.stdout)

    log_event(
        "INFO",
        "GPU chain",
        "NVIDIA decode -> CUDA filter(s) -> " + encoder,
    )


def parse_resolution(
    value: str,
) -> tuple[int | None, int | None] | None:
    """
    Parse --resolution as an OUTPUT UPPER BOUND.

    Return:
        None
            No limit ("source").

        (max_width, max_height)
            Either side may be None, meaning no limit on that side.

    Examples:
        source      -> None
        720p        -> (None, 720)
        1080p       -> (None, 1080)
        1280x720    -> (1280, 720)
        1280x-2     -> (1280, None)
        -2x720      -> (None, 720)

    IMPORTANT:
    This function does NOT build a dynamic scale_cuda expression anymore.
    The actual target dimensions are calculated per source file in Python
    before FFmpeg starts. That makes the "do not upscale" rule explicit and
    avoids the scale_cuda boundary case where the requested upper bound is
    larger than the input.
    """
    v = value.strip().lower()

    if v in {"source", "original", "keep"}:
        return None

    m = re.fullmatch(r"(\d+)p", v)
    if m:
        max_h = int(m.group(1))
        if max_h <= 0:
            raise argparse.ArgumentTypeError(
                "最大分辨率高度必须 > 0"
            )
        return None, max_h

    m = re.fullmatch(
        r"(-?\d+)[xX:](-?\d+)",
        value.strip(),
    )
    if not m:
        raise argparse.ArgumentTypeError(
            "最大分辨率应为 source、720p、1080p、"
            "1280x720 或 1280x-2"
        )

    w, h = map(int, m.groups())

    if w == 0 or h == 0 or w < -2 or h < -2:
        raise argparse.ArgumentTypeError(
            "最大宽高必须为正整数，或使用 -1/-2 "
            "表示该方向不设上限；不能为 0"
        )

    if w < 0 and h < 0:
        raise argparse.ArgumentTypeError(
            "宽和高不能同时为 -1/-2；"
            "若完全保持源分辨率请使用 source"
        )

    max_w = w if w > 0 else None
    max_h = h if h > 0 else None

    return max_w, max_h


def parse_crop_margins(
    crop: str | None,
) -> tuple[int, int, int, int]:
    """Return CUVID crop as (top, bottom, left, right)."""
    if crop is None:
        return 0, 0, 0, 0

    m = re.fullmatch(
        r"(\d+)x(\d+)x(\d+)x(\d+)",
        crop,
    )
    if not m:
        raise RuntimeError(
            f"内部错误：无法解析 crop: {crop}"
        )

    return tuple(map(int, m.groups()))


def floor_even(value: float) -> int:
    """Floor a positive value to an even integer, minimum 2."""
    ivalue = int(value)
    ivalue -= ivalue % 2
    return max(2, ivalue)


def calculate_target_dimensions(
    src_width: int,
    src_height: int,
    resolution_limit: tuple[int | None, int | None] | None,
    crop: str | None = None,
) -> tuple[int, int, int, int, bool]:
    """
    Calculate the exact dimensions passed to scale_cuda.

    The resolution limit is an UPPER BOUND:
      * input below the limit  -> keep input size
      * input above the limit  -> downscale proportionally
      * never upscale

    Crop happens in the CUVID decoder before scale_cuda, so the scale decision
    uses the post-crop dimensions.

    Returns:
        (input_w_after_crop, input_h_after_crop,
         target_w, target_h, needs_resize)
    """
    top, bottom, left, right = parse_crop_margins(crop)

    input_w = src_width - left - right
    input_h = src_height - top - bottom

    if input_w <= 0 or input_h <= 0:
        raise RuntimeError(
            "裁剪后的分辨率无效："
            f"source={src_width}x{src_height}, "
            f"crop={crop}"
        )

    if resolution_limit is None:
        return input_w, input_h, input_w, input_h, False

    max_w, max_h = resolution_limit

    scale = 1.0

    if max_w is not None and input_w > max_w:
        scale = min(scale, max_w / input_w)

    if max_h is not None and input_h > max_h:
        scale = min(scale, max_h / input_h)

    # The defining behavior: never upscale.
    if scale >= 1.0:
        return input_w, input_h, input_w, input_h, False

    target_w = floor_even(input_w * scale)
    target_h = floor_even(input_h * scale)

    # Defensive clamps: rounding must never exceed either configured bound.
    if max_w is not None:
        target_w = min(target_w, max_w - (max_w % 2))
    if max_h is not None:
        target_h = min(target_h, max_h - (max_h % 2))

    if target_w <= 0 or target_h <= 0:
        raise RuntimeError(
            "计算得到无效目标分辨率："
            f"{target_w}x{target_h}"
        )

    return input_w, input_h, target_w, target_h, True


def build_scale_filter_for_source(
    src_width: int,
    src_height: int,
    resolution_limit: tuple[int | None, int | None] | None,
    crop: str | None,
) -> tuple[str, tuple[int, int], tuple[int, int], bool]:
    """
    Build an explicit scale_cuda filter for this specific source.

    If no downscale is needed, use scale_cuda=passthrough=0 without w/h.
    This keeps the strict CUDA-frame pipeline but performs no resize.

    If downscale is needed, pass exact numeric dimensions to scale_cuda.
    """
    (
        input_w,
        input_h,
        target_w,
        target_h,
        needs_resize,
    ) = calculate_target_dimensions(
        src_width,
        src_height,
        resolution_limit,
        crop,
    )

    if not needs_resize:
        return (
            "scale_cuda=passthrough=0",
            (input_w, input_h),
            (target_w, target_h),
            False,
        )

    return (
        "scale_cuda="
        f"w={target_w}:h={target_h}:"
        "interp_algo=lanczos:"
        "passthrough=0",
        (input_w, input_h),
        (target_w, target_h),
        True,
    )



def parse_crop(value: str | None) -> str | None:
    """
    User-facing crop syntax:
        TOP:BOTTOM:LEFT:RIGHT
        TOPxBOTTOMxLEFTxRIGHT

    CUVID expects:
        topxbottomxleftxright
    """
    if value is None:
        return None

    v = value.strip().lower()
    if v in {"", "none", "off", "0"}:
        return None

    m = re.fullmatch(
        r"(\d+)\s*[:x,]\s*(\d+)\s*[:x,]\s*(\d+)\s*[:x,]\s*(\d+)",
        v,
    )
    if not m:
        raise argparse.ArgumentTypeError(
            "--crop 格式应为 TOP:BOTTOM:LEFT:RIGHT，"
            "例如 --crop 140:140:0:0"
        )

    top, bottom, left, right = map(int, m.groups())

    if top == bottom == left == right == 0:
        return None

    return f"{top}x{bottom}x{left}x{right}"


def parse_pad(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    v = value.strip().lower()
    if v in {"", "none", "off", "0"}:
        return None

    m = re.fullmatch(r"(\d+)\s*[x:]\s*(\d+)", v)
    if not m:
        raise argparse.ArgumentTypeError(
            "--pad 格式应为 WIDTHxHEIGHT，例如 --pad 1920x1080"
        )

    width, height = map(int, m.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--pad 宽高必须 > 0")

    return width, height


def parse_pad_position(value: str, axis: str) -> str:
    v = value.strip().lower()

    if v in {"center", "centre", "c"}:
        return "(ow-iw)/2" if axis == "x" else "(oh-ih)/2"

    if not re.fullmatch(r"-?\d+", v):
        raise argparse.ArgumentTypeError(
            f"--pad-{axis} 仅支持整数或 center"
        )

    return v


def validate_pad_color(value: str) -> str:
    """
    Keep the accepted color syntax intentionally conservative so a value cannot
    inject another FFmpeg filter into the GPU-only graph.
    """
    v = value.strip()

    if not v:
        raise argparse.ArgumentTypeError("--pad-color 不能为空")

    if not re.fullmatch(r"[A-Za-z0-9_#.@+-]+", v):
        raise argparse.ArgumentTypeError(
            "--pad-color 仅支持常规颜色名/十六进制颜色，"
            "例如 black、white、0x112233、black@0.5"
        )

    return v


ROTATE_FILTERS = {
    0: None,
    90: "transpose_cuda=dir=clock",
    180: "transpose_cuda=dir=reversal",
    270: "transpose_cuda=dir=cclock",
}


def build_video_filter_chain(
    scale_filter: str,
    *,
    rotate: int,
    pad: tuple[int, int] | None,
    pad_x: str,
    pad_y: str,
    pad_color: str,
) -> str:
    """
    GPU filter order:
        scale_cuda -> transpose_cuda -> pad_cuda

    Cropping is intentionally not here: it happens inside the CUVID decoder
    before CUDA frames enter this filter graph.
    """
    filters = [scale_filter]

    rotate_filter = ROTATE_FILTERS[rotate]
    if rotate_filter:
        filters.append(rotate_filter)

    if pad is not None:
        width, height = pad
        filters.append(
            "pad_cuda="
            f"w={width}:h={height}:"
            f"x={pad_x}:y={pad_y}:"
            f"color={pad_color}"
        )

    return ",".join(filters)


def probe_video_codec(ffmpeg: str, src: Path) -> str:
    """Probe the codec name of the first video stream using ffmpeg itself."""
    p = run_capture([
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i", str(src),
    ])

    m = re.search(
        r"Stream #.*?: Video:\\s*([A-Za-z0-9_]+)",
        p.stdout,
    )
    if not m:
        raise RuntimeError(
            f"无法探测第一个视频流的 codec: {src}\\n\\n{p.stdout}"
        )

    return m.group(1).lower()


def get_available_decoders(ffmpeg: str) -> set[str]:
    p = run_capture([ffmpeg, "-hide_banner", "-decoders"])
    if p.returncode != 0:
        raise RuntimeError(
            "无法读取 FFmpeg decoder 列表。\\n\\n" + p.stdout
        )

    return set(
        re.findall(
            r"(?m)^\\s*V\\S*\\s+([A-Za-z0-9_]+)\\b",
            p.stdout,
        )
    )


def resolve_cuvid_crop_decoder(
    ffmpeg: str,
    src: Path,
    available_decoders: set[str],
) -> tuple[str, str]:
    codec = probe_video_codec(ffmpeg, src)
    decoder = CUVID_DECODERS.get(codec)

    if decoder is None:
        raise RuntimeError(
            f"--crop 需要 CUVID decoder-side GPU crop，但源 codec "
            f"{codec!r} 没有配置对应的 CUVID decoder: {src}"
        )

    if decoder not in available_decoders:
        raise RuntimeError(
            f"--crop 需要 {decoder}，但当前 FFmpeg build 没有这个 decoder: "
            f"{src}\\n"
            "脚本不会退回 CPU crop。"
        )

    return codec, decoder


def validate_bitrate(value: str) -> str:
    v = value.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?[kKmMgG]?", v):
        raise argparse.ArgumentTypeError("bitrate 示例: 4000000、4500k、4M、6.5M")
    m = re.match(r"\d+(?:\.\d+)?", v)
    assert m is not None
    if float(m.group(0)) <= 0:
        raise argparse.ArgumentTypeError("bitrate 必须 > 0")
    return v


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def collect_inputs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if not is_video_file(input_path):
            raise RuntimeError(f"输入文件扩展名不在扫描列表中: {input_path.suffix}")
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise RuntimeError(f"输入路径不存在: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    files = sorted(
        (p.resolve() for p in iterator if is_video_file(p)),
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise RuntimeError(f"没有找到视频文件: {input_path}")
    return files


def resolve_output_root(input_path: Path, output_arg: str | None) -> Path | None:
    if input_path.is_dir():
        if output_arg:
            return Path(output_arg).expanduser().resolve()
        return input_path.resolve().parent / f"{input_path.name}_nvenc"
    return None


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是非负整数") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return parsed


def parse_output_mode(value: str) -> int:
    """Parse chmod-style octal modes such as 775, 0775 or 0o775."""
    raw = value.strip().lower()
    if raw.startswith("0o"):
        raw = raw[2:]

    if not raw or not re.fullmatch(r"[0-7]{1,4}", raw):
        raise argparse.ArgumentTypeError(
            "权限必须是八进制，例如 775、0775、0750 或 0o775"
        )

    mode = int(raw, 8)
    if not 0 <= mode <= 0o7777:
        raise argparse.ArgumentTypeError("权限范围必须是 0000..7777")
    return mode


@dataclass(frozen=True)
class OutputDirPolicy:
    """Linux ownership/mode policy for directories created by this script."""

    uid: int | None = None
    gid: int | None = None
    mode: int | None = None

    @property
    def enabled(self) -> bool:
        return self.uid is not None or self.gid is not None or self.mode is not None

    def describe(self) -> str:
        uid = str(self.uid) if self.uid is not None else "inherit"
        gid = str(self.gid) if self.gid is not None else "inherit"
        mode = f"{self.mode:04o}" if self.mode is not None else "umask"
        return f"uid={uid} gid={gid} mode={mode} | new directories only"


def apply_output_directory_policy(
    directory: Path,
    policy: OutputDirPolicy,
) -> None:
    """Apply ownership first, then exact mode, to one newly created directory."""
    if not policy.enabled:
        return

    if platform.system() != "Linux":
        raise RuntimeError(
            "output uid/gid/mode policy is supported only on Linux"
        )

    try:
        if policy.uid is not None or policy.gid is not None:
            os.chown(
                directory,
                policy.uid if policy.uid is not None else -1,
                policy.gid if policy.gid is not None else -1,
            )

        if policy.mode is not None:
            os.chmod(directory, policy.mode)

    except PermissionError as exc:
        raise RuntimeError(
            f"无法设置新 output 目录的属主/权限: {directory}\n"
            f"requested: {policy.describe()}\n"
            "设置其它 UID/GID 通常需要 root 权限。"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"设置新 output 目录属主/权限失败: {directory}\n"
            f"requested: {policy.describe()}\n{exc}"
        ) from exc


def ensure_output_directory(
    directory: Path,
    policy: OutputDirPolicy,
) -> list[Path]:
    """
    Create an output directory tree and apply policy only to directories that
    were missing before this call.

    Existing directories are intentionally left untouched. A lock makes this
    deterministic with multiple transcoding workers creating sibling paths.
    """
    directory = directory.resolve()

    with _OUTPUT_DIR_LOCK:
        if directory.exists():
            if not directory.is_dir():
                raise RuntimeError(f"output 路径不是目录: {directory}")
            return []

        missing: list[Path] = []
        current = directory

        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent

        directory.mkdir(parents=True, exist_ok=True)

        # Apply deepest-first. This avoids making a parent non-traversable
        # before policy has been applied to its newly created children.
        for created in missing:
            if not created.is_dir():
                raise RuntimeError(
                    f"创建 output 目录后路径状态异常: {created}"
                )
            apply_output_directory_policy(created, policy)

        return missing


def apply_output_file_mode(path: Path, mode: int | None) -> None:
    """Apply an exact chmod mode to one output file on Linux."""
    if mode is None:
        return
    if platform.system() != "Linux":
        raise RuntimeError("--output-file-mode is supported only on Linux")
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise RuntimeError(
            f"设置 output 文件权限失败: {path}\n"
            f"requested mode: {mode:04o}\n{exc}"
        ) from exc


def output_for_file(
    src: Path,
    input_path: Path,
    output_arg: str | None,
    output_root: Path | None,
    recursive: bool,
    output_policy: OutputDirPolicy,
) -> Path:
    if input_path.is_file():
        if output_arg:
            out = Path(output_arg).expanduser()
            if out.exists() and out.is_dir():
                return (out / f"{src.stem}.nvenc.mp4").resolve()
            if str(output_arg).endswith(("/", "\\")):
                ensure_output_directory(out, output_policy)
                return (out / f"{src.stem}.nvenc.mp4").resolve()
            if out.suffix.lower() == ".mp4":
                return out.resolve()
            if not out.suffix:
                ensure_output_directory(out, output_policy)
                return (out / f"{src.stem}.nvenc.mp4").resolve()
            raise RuntimeError("单文件模式下 -o 指定文件名时必须为 .mp4")
        return src.with_name(f"{src.stem}.nvenc.mp4")

    assert output_root is not None
    dest_dir = output_root
    if recursive:
        dest_dir = output_root / src.parent.relative_to(input_path.resolve())
    return dest_dir / f"{src.stem}.mp4"




def path_is_within(path: Path, root: Path) -> bool:
    """Return True if path is root itself or is below root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def watch_output_for_file(
    src: Path,
    watch_root: Path,
    output_root: Path,
    keep_sub_path: bool,
) -> Path:
    if keep_sub_path:
        rel_parent = src.parent.resolve().relative_to(watch_root.resolve())
        dest_dir = output_root / rel_parent
    else:
        dest_dir = output_root

    return dest_dir / f"{src.stem}.mp4"


def copy_sibling_non_video_files(
    src: Path,
    *,
    watch_root: Path,
    output_root: Path,
    keep_sub_path: bool,
    output_policy: OutputDirPolicy,
    output_file_mode: int | None,
) -> int:
    """
    Copy all non-video regular files in the source video's directory.

    Only sibling files are copied; child directories are not recursively copied.
    Existing destination files are overwritten with shutil.copy2().
    """
    if keep_sub_path:
        rel_parent = src.parent.resolve().relative_to(watch_root.resolve())
        dest_dir = output_root / rel_parent
    else:
        dest_dir = output_root

    ensure_output_directory(dest_dir, output_policy)
    copied = 0

    for item in src.parent.iterdir():
        if not item.is_file():
            continue
        if is_video_file(item):
            continue

        dst = dest_dir / item.name

        # Defensive guard when source/output layouts overlap.
        try:
            if item.resolve() == dst.resolve():
                continue
        except OSError:
            pass

        shutil.copy2(item, dst)
        apply_output_file_mode(dst, output_file_mode)
        copied += 1

    return copied


def file_fingerprint(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def scan_watch_videos(
    watch_root: Path,
    output_root: Path,
    temp_root: Path | None = None,
) -> list[Path]:
    """
    Recursively scan watch_root for video files using os.walk().

    IMPORTANT:
    output_root is excluded ONLY when output_root is a child of watch_root.

    Example that must remain valid:
        watch  = /mnt/user/download/complete
        output = /mnt/user/download

    Here output is the PARENT of watch, so excluding everything under output
    would incorrectly exclude the entire watch tree.

    os.walk() is also preferable for long-running polling on Unraid's /mnt/user
    FUSE shares because it lets us prune a child output directory before
    descending into it.
    """
    watch_root = watch_root.resolve()
    output_root = output_root.resolve()

    output_inside_watch = (
        output_root != watch_root
        and path_is_within(output_root, watch_root)
    )

    temp_inside_watch = False
    resolved_temp_root: Path | None = None

    if temp_root is not None:
        resolved_temp_root = temp_root.resolve()

        if resolved_temp_root == watch_root:
            raise RuntimeError(
                "--temp-dir 不能与 --watch 指向同一个目录"
            )

        temp_inside_watch = path_is_within(
            resolved_temp_root,
            watch_root,
        )

    files: list[Path] = []

    def onerror(exc: OSError) -> None:
        log_event("WARN", "Scan", str(exc), error=True)

    for root, dirs, names in os.walk(
        watch_root,
        topdown=True,
        followlinks=False,
        onerror=onerror,
    ):
        root_path = Path(root).resolve()

        if output_inside_watch or temp_inside_watch:
            kept_dirs = []

            for dirname in dirs:
                candidate = (root_path / dirname).resolve()

                if (
                    output_inside_watch
                    and path_is_within(candidate, output_root)
                ):
                    continue

                if (
                    temp_inside_watch
                    and resolved_temp_root is not None
                    and path_is_within(candidate, resolved_temp_root)
                ):
                    continue

                kept_dirs.append(dirname)

            dirs[:] = kept_dirs

        for name in names:
            path = root_path / name

            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            if is_video_file(path):
                try:
                    files.append(path.resolve())
                except OSError:
                    continue

    files.sort(key=lambda p: str(p).lower())
    return files


def parse_01(value: str) -> int:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("仅允许 0 或 1")
    return int(value)




def expected_other_file_destination(
    src_file: Path,
    *,
    watch_root: Path,
    output_root: Path,
    keep_sub_path: bool,
) -> Path:
    if keep_sub_path:
        rel_parent = src_file.parent.resolve().relative_to(
            watch_root.resolve()
        )
        dest_dir = output_root / rel_parent
    else:
        dest_dir = output_root

    return dest_dir / src_file.name


def directory_cleanup_ready(
    source_dir: Path,
    *,
    watch_root: Path,
    output_root: Path,
    keep_sub_path: bool,
    settled: dict[Path, tuple[int, int]],
    queued: dict[Path, tuple[int, int]],
) -> tuple[bool, str]:
    """
    Decide whether a source directory may be deleted as one transaction.

    Safety rules:
      1. No child directories may remain. They are handled independently first.
      2. Every direct child video must have a settled fingerprint matching its
         current file and its expected output MP4 must exist.
      3. No direct child video may still be queued/running.
      4. Every direct child non-video file must already exist at the expected
         output location.
    """
    if not source_dir.exists():
        return False, "目录已不存在"

    try:
        children = list(source_dir.iterdir())
    except OSError as exc:
        return False, f"无法读取目录: {exc}"

    child_dirs = [p for p in children if p.is_dir()]
    if child_dirs:
        return False, f"仍有 {len(child_dirs)} 个子目录"

    videos = [p.resolve() for p in children if is_video_file(p)]

    for video in videos:
        try:
            current_fp = file_fingerprint(video)
        except OSError as exc:
            return False, f"无法读取视频状态: {video.name}: {exc}"

        if queued.get(video) == current_fp:
            return False, f"视频仍在队列/转码中: {video.name}"

        if settled.get(video) != current_fp:
            return False, f"视频尚未成功完成: {video.name}"

        expected_video = watch_output_for_file(
            video,
            watch_root,
            output_root,
            keep_sub_path,
        )
        if not expected_video.is_file():
            return False, f"视频输出不存在: {expected_video}"

    for other in children:
        if not other.is_file():
            continue
        if is_video_file(other):
            continue

        expected_other = expected_other_file_destination(
            other,
            watch_root=watch_root,
            output_root=output_root,
            keep_sub_path=keep_sub_path,
        )
        if not expected_other.is_file():
            return False, f"非视频文件尚未复制: {other.name}"

    # Require at least one direct video so an unrelated all-non-video directory
    # is never deleted merely because a neighboring watch event happened.
    if not videos:
        return False, "目录中没有直接视频文件"

    return True, "ready"


def cleanup_completed_source_directories(
    candidate_dirs: set[Path],
    *,
    watch_root: Path,
    output_root: Path,
    keep_sub_path: bool,
    settled: dict[Path, tuple[int, int]],
    queued: dict[Path, tuple[int, int]],
) -> int:
    """
    Delete fully completed source directories, deepest paths first.

    Deleting deepest-first means that when child directories have already been
    completed and removed, their parent can become eligible on the same pass.
    """
    deleted = 0

    # Include ancestors up to, but never including, watch_root. This lets a
    # parent directory become removable after its child directories finish.
    expanded: set[Path] = set()
    watch_root = watch_root.resolve()

    for directory in candidate_dirs:
        current = directory.resolve()

        while current != watch_root:
            if not path_is_within(current, watch_root):
                break

            expanded.add(current)
            current = current.parent

    for source_dir in sorted(
        expanded,
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not source_dir.exists():
            continue

        ready, reason = directory_cleanup_ready(
            source_dir,
            watch_root=watch_root,
            output_root=output_root,
            keep_sub_path=keep_sub_path,
            settled=settled,
            queued=queued,
        )

        if not ready:
            continue

        shutil.rmtree(source_dir)
        deleted += 1
        log_event("INFO", "Delete dir", str(source_dir))

        # Purge no-longer-relevant settled entries inside the deleted subtree.
        for path in list(settled):
            if path_is_within(path, source_dir):
                settled.pop(path, None)

    return deleted



def probe_video_info(
    ffmpeg: str,
    src: Path,
) -> tuple[float | None, int, int]:
    """
    Probe duration and the first video stream's coded width/height using
    ffmpeg itself, so ffprobe is not required.
    """
    p = run_capture([
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i", str(src),
    ])

    duration: float | None = None

    m = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        p.stdout,
    )
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        duration = (
            hours * 3600
            + minutes * 60
            + seconds
        )

    # Match resolution only on a Video stream line.
    width = height = 0

    for line in p.stdout.splitlines():
        if "Stream #" not in line or "Video:" not in line:
            continue

        # Typical:
        # Video: hevc (...), yuv420p(...), 1920x1080 [SAR ...]
        matches = re.findall(
            r"(?<![0-9A-Fa-f])"
            r"(\d{2,5})x(\d{2,5})"
            r"(?![0-9A-Fa-f])",
            line,
        )

        if matches:
            width, height = map(int, matches[-1])
            break

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"无法探测视频分辨率: {src}\n\n"
            f"{p.stdout}"
        )

    return duration, width, height


def probe_duration(ffmpeg: str, src: Path) -> float | None:
    """Compatibility helper for callers that only need duration."""
    duration, _width, _height = probe_video_info(
        ffmpeg,
        src,
    )
    return duration



def parse_ffmpeg_time(value: str) -> float:
    """Parse HH:MM:SS.microseconds emitted by FFmpeg -progress."""
    try:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"

    total = int(seconds + 0.5)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_bytes(size: int | float | None) -> str:
    """Compact binary-size formatter for progress/DONE output."""
    if size is None:
        return "--"

    try:
        value = float(size)
    except (TypeError, ValueError):
        return "--"

    if value < 0:
        return "--"

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0

    if unit == "B":
        return f"{int(value)}B"
    if value >= 100:
        return f"{value:.0f}{unit}"
    if value >= 10:
        return f"{value:.1f}{unit}"
    return f"{value:.2f}{unit}"


def progress_bar(percent: float | None, width: int = 20) -> str:
    if percent is None:
        return "[" + "?" * width + "]"

    percent = max(0.0, min(100.0, percent))
    done = int(width * percent / 100.0)
    if done >= width:
        return "[" + "=" * width + "]"

    return "[" + "=" * done + ">" + "." * max(0, width - done - 1) + "]"


@dataclass
class _ProgressJob:
    index: int
    src: Path
    duration: float | None
    started: float
    source_size: int


class ProgressReporter:
    """NBMiner-style periodic status table for concurrent workers."""

    _MIN_WINDOW = 1.0

    def __init__(
        self,
        total: int,
        workers: int,
        interval: float = 2.0,
        gpu: int = 0,
    ):
        self.total = total
        self.workers = max(1, workers)
        self.interval = max(0.25, interval)
        self.gpu = gpu

        self.lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.closed_jobs: set[int] = set()
        self.last_snapshot = 0.0

        self.active: dict[int, _ProgressJob] = {}
        self.latest: dict[int, dict[str, str]] = {}
        self.history: dict[int, deque[tuple[float, float, int | None]]] = {}

    def set_total(self, total: int) -> None:
        with self.lock:
            self.total = max(self.total, total)

    def start(
        self,
        slot: int,
        index: int,
        src: Path,
        duration: float | None,
    ) -> None:
        try:
            source_size = src.stat().st_size
        except OSError:
            source_size = 0

        job = _ProgressJob(
            index=index,
            src=src,
            duration=duration,
            started=time.monotonic(),
            source_size=source_size,
        )

        with self.lock:
            self.active[slot] = job
            self.latest.pop(slot, None)
            self.history[slot] = deque([(job.started, 0.0, 0)])

        duration_text = (
            format_duration(duration)
            if duration is not None
            else "unknown"
        )
        log_event(
            "INFO",
            "New job",
            f"{src.name} -> W{slot} | duration {duration_text}",
        )

    @staticmethod
    def _state_seconds(state: dict[str, str]) -> float:
        raw = state.get("out_time_us")
        if raw:
            try:
                return int(raw) / 1_000_000.0
            except ValueError:
                pass

        raw = state.get("out_time")
        if raw:
            return parse_ffmpeg_time(raw)

        raw = state.get("out_time_ms")
        if raw:
            try:
                return int(raw) / 1_000_000.0
            except ValueError:
                pass

        return 0.0

    @staticmethod
    def _state_frames(state: dict[str, str]) -> int | None:
        raw = state.get("frame", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _truncate(name: str, width: int) -> str:
        if len(name) <= width:
            return name
        return name[:width] if width <= 3 else name[: width - 3] + "..."

    @staticmethod
    def _estimate_ratio(
        state: dict[str, str],
        out_seconds: float,
        duration: float | None,
        source_size: int,
    ) -> float | None:
        if not duration or duration <= 0 or source_size <= 0 or out_seconds <= 0:
            return None

        progress = out_seconds / duration
        if progress < 0.02:
            return None

        raw_size = state.get("total_size")
        if not raw_size or raw_size in {"N/A", "-1"}:
            return None

        try:
            output_size = int(raw_size)
        except ValueError:
            return None

        estimated_input = source_size * progress
        if output_size < 0 or estimated_input <= 0:
            return None

        return output_size / estimated_input

    @staticmethod
    def _terminal_width() -> int:
        try:
            return shutil.get_terminal_size(fallback=(120, 24)).columns
        except Exception:
            return 120

    def _interval_window(
        self,
        slot: int,
    ) -> tuple[float, float, int | None] | None:
        """Wallclock seconds, encoded seconds and frames of one display window."""
        history = self.history.get(slot)
        if not history:
            return None

        end_time, end_seconds, end_frames = history[-1]
        target = end_time - max(self.interval, self._MIN_WINDOW)
        start_time, start_seconds, start_frames = history[0]
        for sample in reversed(history):
            if sample[0] <= target:
                start_time, start_seconds, start_frames = sample
                break

        elapsed = end_time - start_time
        if elapsed < self._MIN_WINDOW:
            return None

        gained = end_seconds - start_seconds
        if end_frames is not None and start_frames is not None:
            frames = end_frames - start_frames
        else:
            frames = None

        if gained < 0 or (frames is not None and frames < 0):
            return None

        return elapsed, gained, frames

    def _row(
        self,
        slot: int,
        job: _ProgressJob,
        state: dict[str, str],
        now: float,
        file_width: int,
        fps: float | None,
        speed: float | None,
    ) -> str:
        out_seconds = self._state_seconds(state)
        elapsed = max(0.0, now - job.started)

        percent = (
            min(100.0, out_seconds / job.duration * 100.0)
            if job.duration and job.duration > 0
            else None
        )
        ratio = self._estimate_ratio(
            state,
            out_seconds,
            job.duration,
            job.source_size,
        )

        progress_text = f"{percent:6.1f}%" if percent is not None else "    ?.?%"
        ratio_text = f"{ratio * 100:6.1f}%" if ratio is not None else "     --"
        fps_text = f"{fps:.1f}" if fps is not None else "--"
        speed_text = f"{speed:.2f}x" if speed is not None else "--"
        video_text = (
            f"{format_duration(out_seconds)}/"
            f"{format_duration(job.duration)}"
        )

        return (
            f"{slot:>2} "
            f"{f'{job.index}/{self.total}':>7} "
            f"{progress_text:>7} "
            f"{video_text:>17} "
            f"{format_duration(elapsed):>8} "
            f"{fps_text:>6} "
            f"{speed_text:>7} "
            f"{ratio_text:>7} "
            f"{self._truncate(job.src.name, file_width)}"
        )

    def _build_table(
        self,
        active_slots: list[int],
        completed: int,
        now: float,
    ) -> list[str]:
        header = (
            f"{'W':>2} {'JOB':>7} {'PROG':>7} {'VIDEO':>17} "
            f"{'TIME':>8} {'FPS':>6} {'SPEED':>7} {'RATIO':>7} FILE"
        )
        terminal_width = max(90, self._terminal_width())
        fixed_width = len(header) - len("FILE")
        file_width = max(12, min(52, terminal_width - fixed_width))
        separator_width = min(terminal_width, fixed_width + file_width)

        rows = [
            f"{clock_text()} " + "-" * max(8, separator_width - 9),
            header,
        ]
        speeds: list[float] = []

        for slot in active_slots:
            job = self.active[slot]
            state = self.latest[slot]
            window = self._interval_window(slot)

            fps = None
            speed = None
            if window is not None:
                elapsed, gained, frames = window
                fps = frames / elapsed if frames is not None else None
                speed = gained / elapsed

            rows.append(
                self._row(slot, job, state, now, file_width, fps, speed)
            )
            if speed is not None:
                speeds.append(speed)

        running = len(active_slots)
        queue = max(
            0,
            self.total - completed - self.failed - self.skipped - running,
        )
        avg_speed = f"{sum(speeds) / len(speeds):.2f}x" if speeds else "--"

        rows.extend([
            "-" * separator_width,
            (
                f" GPU {self.gpu} | Running {running}/{self.workers} | "
                f"Done {completed} | Queue {queue} | Avg Speed {avg_speed}"
            ),
            "-" * separator_width,
        ])
        return rows

    def update(
        self,
        slot: int,
        index: int,
        src: Path,
        duration: float | None,
        state: dict[str, str],
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        lines = None

        with self.lock:
            job = self.active.get(slot)
            if job is None or job.index != index:
                return

            self.latest[slot] = dict(state)
            history = self.history.setdefault(slot, deque())
            history.append(
                (now, self._state_seconds(state), self._state_frames(state))
            )
            horizon = now - (max(self.interval, self._MIN_WINDOW) + 2.0)
            while len(history) > 2 and history[0][0] < horizon:
                history.popleft()

            if not force and now - self.last_snapshot < self.interval:
                return

            slots = sorted(self.active)
            if not slots or any(slot_id not in self.latest for slot_id in slots):
                return

            lines = self._build_table(slots, self.completed, now)
            self.last_snapshot = now

        safe_print_block(lines)

    def _close(
        self,
        slot: int,
        index: int,
        outcome: str,
    ) -> tuple[_ProgressJob | None, int]:
        with self.lock:
            job = self.active.get(slot)
            if job is not None and job.index == index:
                self.active.pop(slot, None)
                self.latest.pop(slot, None)
                self.history.pop(slot, None)
            else:
                job = None

            if index not in self.closed_jobs:
                self.closed_jobs.add(index)
                if outcome == "done":
                    self.completed += 1
                elif outcome == "failed":
                    self.failed += 1
                elif outcome == "skipped":
                    self.skipped += 1

            return job, self.completed

    def finish(
        self,
        slot: int,
        index: int,
        src: Path,
        duration: float | None,
        dst: Path | None = None,
    ) -> None:
        finished = time.monotonic()
        job, completed = self._close(slot, index, "done")

        elapsed = (
            max(0.0, finished - job.started)
            if job is not None
            else None
        )
        source_size = job.source_size if job is not None else 0

        try:
            output_size = dst.stat().st_size if dst is not None else 0
        except OSError:
            output_size = 0

        ratio = output_size / source_size if source_size > 0 else None
        ratio_text = f"{ratio * 100:.1f}%" if ratio is not None else "--"
        size_text = (
            f"{format_bytes(source_size)} -> {format_bytes(output_size)}"
            if source_size > 0 and output_size > 0
            else "--"
        )

        log_event(
            "DONE",
            f"W{slot}",
            f"{src.name} | {size_text} | Ratio {ratio_text} | "
            f"Time {format_duration(elapsed)} | Done {completed}/{self.total}",
        )

    def mark_failed(self, slot: int, index: int) -> None:
        self._close(slot, index, "failed")

    def mark_skipped(self, slot: int, index: int) -> None:
        self._close(slot, index, "skipped")



def build_ffmpeg_command(
    ffmpeg: str,
    src: Path,
    temp_dst: Path,
    encoder: str,
    gpu: int,
    preset: str,
    video_filter: str,
    cq: float | None,
    bitrate: str | None,
    crop: str | None,
    crop_decoder: str | None,
) -> list[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-stats_period", "0.5",
        "-progress", "pipe:1",
        "-loglevel", "info",
        "-xerror",

        "-init_hw_device", f"cuda=gpu:{gpu}",
        "-filter_hw_device", "gpu",
    ]

    if crop is not None:
        if crop_decoder is None:
            raise RuntimeError("内部错误: crop 已启用但没有 CUVID decoder")

        # Decoder-side GPU crop. CUVID exposes both `gpu` and `crop` decoder
        # options. scale_cuda below acts as a strict CUDA-frame gate: if the
        # decoder were to return system-memory frames, the job fails instead of
        # silently falling back to a CPU filter.
        cmd += [
            "-c:v", crop_decoder,
            "-gpu", str(gpu),
            "-crop", crop,
        ]
    else:
        # Normal path: automatic codec detection through FFmpeg's CUDA hwaccel.
        cmd += [
            "-hwaccel", "cuda",
            "-hwaccel_device", "gpu",
            "-hwaccel_output_format", "cuda",
        ]

    cmd += [
        # FFmpeg enables input autorotation by default. Disable it so rotation
        # is deterministic and remains inside our explicit CUDA filter path.
        # Use --rotate when a physical rotation is desired.
        "-noautorotate",
        "-i", str(src),
        "-map", "0:v:0",
        "-map", "0:a?",

        "-vf", video_filter,
        "-noautoscale",
        "-c:v", encoder,
        "-gpu", str(gpu),
        "-preset", preset,
    ]

    if cq is not None:
        cmd += ["-rc", "vbr", "-cq:v", f"{cq:g}", "-b:v", "0"]
    else:
        assert bitrate is not None
        cmd += ["-rc", "vbr", "-b:v", bitrate]

    cmd += [
        "-fps_mode:v", "vfr",
        "-c:a", "copy",
        "-map_metadata", "0",
        "-map_chapters", "0",
    ]

    if encoder == "hevc_nvenc":
        cmd += ["-tag:v", "hvc1"]

    cmd += ["-f", "mp4", "-y", str(temp_dst)]
    return cmd


def format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    import shlex
    return shlex.join(cmd)



def subprocess_group_kwargs() -> dict:
    """
    Start each FFmpeg in its own process group/session.

    Linux/POSIX: start_new_session=True, so os.killpg(pid, ...) can stop the
    entire process tree.

    Windows: CREATE_NEW_PROCESS_GROUP is used where available.
    """
    if os.name == "nt":
        return {
            "creationflags": getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
        }

    return {"start_new_session": True}


def terminate_process_tree(
    process: subprocess.Popen,
    *,
    force: bool = False,
) -> None:
    """Terminate one FFmpeg process tree on Windows or POSIX."""
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            # Python stdlib does not provide killpg on Windows. FFmpeg is the
            # direct child we created, so terminate/kill the child itself.
            if force:
                process.kill()
            else:
                process.terminate()
        else:
            # With start_new_session=True, child PID is also its process-group
            # ID. This kills FFmpeg plus any descendants it may create.
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(process.pid, sig)

    except ProcessLookupError:
        pass

    except Exception:
        # Defensive fallback.
        try:
            if process.poll() is None:
                process.kill() if force else process.terminate()
        except Exception:
            pass


def terminate_all_ffmpeg() -> None:
    """Stop the whole batch and terminate every active FFmpeg process tree."""
    _STOP_EVENT.set()

    with _PROCESS_LOCK:
        processes = list(_ACTIVE_PROCESSES)

    for process in processes:
        terminate_process_tree(process, force=False)

    # Short grace period, then hard-kill anything still alive.
    deadline = time.monotonic() + 1.0
    for process in processes:
        if process.poll() is not None:
            continue

        try:
            process.wait(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except Exception:
            pass

    for process in processes:
        if process.poll() is None:
            terminate_process_tree(process, force=True)


def shutdown_executor_after_abort(
    executor: ThreadPoolExecutor,
    futures,
    *,
    timeout: float = 2.0,
) -> int:
    """
    Stop child processes and give already-running worker threads a short,
    bounded window to unwind.

    Returns the number of worker futures still not finished at the end of the
    grace window. Python's ThreadPoolExecutor threads are non-daemon and are
    joined by concurrent.futures during normal interpreter shutdown, even when
    executor.shutdown(wait=False) was used. A stuck worker can therefore leave
    the Python PID visible after the program has printed its exit log.

    The __main__ finalizer below uses os._exit() after a caught termination
    signal, so any genuinely stuck worker cannot keep the service process alive.
    """
    future_list = list(futures)

    for future in future_list:
        future.cancel()

    terminate_all_ffmpeg()

    not_done = set()
    if future_list:
        _done, not_done = wait(
            future_list,
            timeout=max(0.0, timeout),
        )

    executor.shutdown(
        wait=False,
        cancel_futures=True,
    )

    if not_done:
        log_event(
            "WARN",
            "Shutdown",
            f"{len(not_done)} worker(s) still active after "
            f"{timeout:g}s; process finalizer will force exit",
            error=True,
        )
    else:
        log_event(
            "INFO",
            "Shutdown",
            "all worker threads drained",
        )

    return len(not_done)


def run_ffmpeg(
    cmd: list[str],
    progress_callback=None,
) -> tuple[int, str]:
    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt

    """
    Run FFmpeg and parse -progress pipe:1 in real time.

    stderr is drained in a separate thread so FFmpeg cannot block on a full
    stderr pipe. stderr remains buffered per job and is only printed on error.
    """
    spawn_cmd = linux_parent_death_wrap(cmd)

    process = subprocess.Popen(
        spawn_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **subprocess_group_kwargs(),
    )

    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)

    # Ctrl+C could arrive in the tiny window between the pre-spawn check and
    # process registration. Kill the just-created process in that case.
    if _STOP_EVENT.is_set():
        terminate_process_tree(process, force=True)

    stderr_lines: list[str] = []

    def drain_stderr():
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(
        target=drain_stderr,
        name=f"ffmpeg-stderr-{process.pid}",
        daemon=True,
    )
    stderr_thread.start()

    try:
        assert process.stdout is not None
        state: dict[str, str] = {}

        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            state[key] = value

            if key == "progress":
                if progress_callback is not None:
                    progress_callback(dict(state))
                state.clear()

        rc = process.wait()
        stderr_thread.join(timeout=5.0)

        return rc, "".join(stderr_lines)

    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)



def make_temp_output_path(
    dst: Path,
    temp_dir: Path | None,
) -> Path:
    if temp_dir is None:
        return dst.with_name(f"{dst.stem}.part.mp4")

    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{dst.stem}.{uuid.uuid4().hex}.part.mp4"


def commit_temp_output(
    temp_dst: Path,
    dst: Path,
    output_policy: OutputDirPolicy,
) -> None:
    ensure_output_directory(dst.parent, output_policy)

    try:
        temp_dst.replace(dst)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    staging = dst.with_name(
        f".{dst.name}.{uuid.uuid4().hex}.staging"
    )

    try:
        shutil.copy2(temp_dst, staging)
        staging.replace(dst)
        temp_dst.unlink(missing_ok=True)
    except Exception:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise



def transcode_one(
    ffmpeg: str,
    src: Path,
    dst: Path,
    encoder: str,
    gpu: int,
    preset: str,
    resolution_limit: tuple[int | None, int | None] | None,
    rotate: int,
    pad: tuple[int, int] | None,
    pad_x: str,
    pad_y: str,
    pad_color: str,
    cq: float | None,
    bitrate: str | None,
    crop: str | None,
    crop_decoder: str | None,
    overwrite: bool,
    temp_dir: Path | None = None,
    output_policy: OutputDirPolicy = OutputDirPolicy(),
    output_file_mode: int | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_index: int = 1,
    worker_slot: int = 1,
) -> str:
    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt

    if src.resolve() == dst.resolve():
        raise RuntimeError(f"拒绝覆盖输入文件本身: {src}")

    # Default mode writes *.part.mp4 next to the final output, so the
    # destination directory must exist before FFmpeg starts.
    #
    # With --temp-dir, defer creation of the final output directory until
    # commit_temp_output() after FFmpeg succeeds. This prevents interrupted
    # watch jobs from leaving empty output/sub/... directories behind.
    if temp_dir is None:
        ensure_output_directory(dst.parent, output_policy)

    if dst.exists() and not overwrite:
        return "skipped"

    # With --overwrite, deliberately keep the old destination in place while
    # FFmpeg writes the temporary output. commit_temp_output() replaces it only
    # after the new file has completed successfully.

    temp_dst = make_temp_output_path(dst, temp_dir)
    if temp_dst.exists():
        temp_dst.unlink()

    duration, src_width, src_height = probe_video_info(
        ffmpeg,
        src,
    )

    (
        scale_filter,
        scale_input,
        scale_output,
        needs_resize,
    ) = build_scale_filter_for_source(
        src_width,
        src_height,
        resolution_limit,
        crop,
    )

    video_filter = build_video_filter_chain(
        scale_filter,
        rotate=rotate,
        pad=pad,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_color=pad_color,
    )

    resize_text = (
        f"{scale_input[0]}x{scale_input[1]} -> "
        f"{scale_output[0]}x{scale_output[1]}"
        if needs_resize
        else f"{scale_input[0]}x{scale_input[1]} unchanged"
    )

    log_event(
        "INFO",
        f"W{worker_slot} Scale",
        f"job {job_index} | {resize_text} | {src.name}",
    )

    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt

    if progress_reporter is not None:
        progress_reporter.start(
            worker_slot,
            job_index,
            src,
            duration,
        )

    cmd = build_ffmpeg_command(
        ffmpeg,
        src,
        temp_dst,
        encoder,
        gpu,
        preset,
        video_filter,
        cq,
        bitrate,
        crop,
        crop_decoder,
    )

    def on_progress(state: dict[str, str]) -> None:
        if progress_reporter is not None:
            progress_reporter.update(
                worker_slot,
                job_index,
                src,
                duration,
                state,
                force=(state.get("progress") == "end"),
            )

    rc, log = run_ffmpeg(cmd, progress_callback=on_progress)

    if rc != 0:
        try:
            temp_dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"FFmpeg 转码失败，退出码 {rc}: {src}\n"
            "没有执行任何 CPU 解码/CPU 缩放 fallback。\n\n"
            f"命令:\n{format_cmd(cmd)}\n\n"
            f"FFmpeg 日志:\n{log}"
        )

    if not temp_dst.exists() or temp_dst.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg 返回成功，但输出为空: {temp_dst}\n\n"
            f"FFmpeg 日志:\n{log}"
        )

    commit_temp_output(temp_dst, dst, output_policy)
    apply_output_file_mode(dst, output_file_mode)

    if progress_reporter is not None:
        progress_reporter.finish(
            worker_slot,
            job_index,
            src,
            duration,
            dst,
        )

    return "ok"



@dataclass(frozen=True)
class TranscodeConfig:
    """Resolved, immutable configuration shared by batch and watch modes."""

    args: argparse.Namespace
    ffmpeg: str
    encoder: str
    resolution_limit: tuple[int | None, int | None] | None
    crop: str | None
    pad: tuple[int, int] | None
    pad_x: str
    pad_y: str
    temp_root: Path | None
    input_path: Path
    output_policy: OutputDirPolicy
    output_file_mode: int | None

    @property
    def watch_mode(self) -> bool:
        return self.args.watch is not None

    @property
    def quality_text(self) -> str:
        if self.args.cq is not None:
            return f"CQ {self.args.cq:g} / {self.args.preset.upper()}"
        return f"{self.args.bitrate} / {self.args.preset.upper()}"

    def run_job(
        self,
        reporter: ProgressReporter,
        index: int,
        src: Path,
        dst: Path,
        worker_slot: int,
        crop_decoder: str | None,
    ) -> tuple[int, Path, Path, str]:
        if _STOP_EVENT.is_set():
            raise KeyboardInterrupt

        result = transcode_one(
            ffmpeg=self.ffmpeg,
            src=src,
            dst=dst,
            encoder=self.encoder,
            gpu=self.args.gpu,
            preset=self.args.preset,
            resolution_limit=self.resolution_limit,
            rotate=self.args.rotate,
            pad=self.pad,
            pad_x=self.pad_x,
            pad_y=self.pad_y,
            pad_color=self.args.pad_color,
            cq=self.args.cq,
            bitrate=self.args.bitrate,
            crop=self.crop,
            crop_decoder=crop_decoder,
            overwrite=self.args.overwrite,
            temp_dir=self.temp_root,
            output_policy=self.output_policy,
            output_file_mode=self.output_file_mode,
            progress_reporter=reporter,
            job_index=index,
            worker_slot=worker_slot,
        )
        return index, src, dst, result


@dataclass
class WatchState:
    """Mutable state owned by one long-running watch session."""

    observed: dict[Path, tuple[int, int, float]] = field(default_factory=dict)
    settled: dict[Path, tuple[int, int]] = field(default_factory=dict)
    failed_fingerprints: dict[Path, tuple[int, int]] = field(default_factory=dict)
    queued: dict[Path, tuple[int, int]] = field(default_factory=dict)

    destination_sources: dict[Path, Path] = field(default_factory=dict)
    claimed_destinations: set[Path] = field(default_factory=set)

    pending: list[
        tuple[int, Path, Path, tuple[int, int], str | None]
    ] = field(default_factory=list)

    cleanup_candidates: set[Path] = field(default_factory=set)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    discovered: int = 0
    ok: int = 0
    skipped: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "严格 NVIDIA 视频转码: NVIDIA decode -> CUDA filters -> NVENC。\n"
            "输出容器固定为 MP4；仅转码第一个视频流，所有音轨 stream-copy；"
            "字幕/附件当前不复制，章节和全局 metadata 尝试保留。\n"
            "若音轨无法直接封装进 MP4，任务失败，不做音频转码 fallback。\n"
            "禁用 FFmpeg 自动 rotation metadata；旋转仅由 --rotate 显式执行。\n"
            "SIGINT/SIGTERM/SIGHUP/SIGQUIT 会触发完整清理；检测到 Unraid "
            "User Scripts 时，nv_transcode 自身会绑定父 shell 的死亡 SIGTERM；"
            "Linux 下 FFmpeg 也有父进程死亡保护；可为脚本新建的 output "
            "目录指定 UID/GID/mode，并单独指定 output 文件 mode。\n"
            "视频硬件链路不可用时直接失败，不做 CPU decode/scale fallback。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "示例：\n"
            "  普通目录: script.py /video -o /out --cq 28 -r 1080p --recursive\n"
            "  Watch:    script.py --watch /in --output /out --temp-dir /tmp/transcode "
            "--cq 28 -r 1080p --workers 4 --keep-sub-path --copy-other-files --keep-src 0"
        ),
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="普通批处理模式的输入视频文件或文件夹；与 --watch 二选一",
    )
    parser.add_argument(
        "--watch",
        help=(
            "持续监控目录（递归包含所有子目录）。\n"
            "当前已有视频会先处理；之后新视频文件稳定后自动转码。"
        ),
    )
    parser.add_argument(
        "--ffmpeg",
        help=(
            "FFmpeg 二进制路径或命令名。\n"
            "Windows: --ffmpeg D:\\ffmpeg\\bin\\ffmpeg.exe\n"
            "Linux:   --ffmpeg /usr/local/bin/ffmpeg\n"
            "省略时依次查找 PATH、脚本目录；仍未找到则从 BtbN/FFmpeg-Builds "
            "latest release 自动下载匹配系统/架构的 static GPL build 到脚本目录，"
            "并执行 SHA-256 校验。"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        help=(
            "输出路径（输出容器固定为 MP4）。\n"
            "普通单文件: 可指定 output.mp4 或输出目录。\n"
            "普通目录: 指定输出目录，每个输入输出为 <stem>.mp4；"
            "--recursive 时保留相对目录结构。\n"
            "省略时: 单文件输出 *.nvenc.mp4；目录输出到同级 <目录名>_nvenc。\n"
            "watch 模式必须显式指定 --output DIR。"
        ),
    )
    parser.add_argument(
        "--temp-dir",
        help=(
            "FFmpeg 转码临时文件目录。\n"
            "省略时保持当前行为：*.part.mp4 写在最终输出目录旁边。\n"
            "指定后，最终 output 子目录延迟到转码成功提交时才创建。\n"
            "跨文件系统提交时会先在目标目录写隐藏 .staging 文件，再原子替换最终文件。\n"
            "例如 Unraid: --temp-dir /mnt/cache/transcode-temp"
        ),
    )

    linux_output = parser.add_argument_group("Linux output 目录")
    linux_output.add_argument(
        "--output-uid",
        type=parse_nonnegative_int,
        default=None,
        help=(
            "仅 Linux：脚本新创建的 output 目录设置为该 UID。"
            "默认不修改属主；已有目录永不修改。"
        ),
    )
    linux_output.add_argument(
        "--output-gid",
        type=parse_nonnegative_int,
        default=None,
        help=(
            "仅 Linux：脚本新创建的 output 目录设置为该 GID。"
            "默认不修改属组；已有目录永不修改。"
        ),
    )
    linux_output.add_argument(
        "--output-mode",
        type=parse_output_mode,
        default=None,
        metavar="MODE",
        help=(
            "仅 Linux：脚本新创建的 output 目录权限，例如 775、0775、0750。"
            "按精确 chmod mode 应用，不受 umask 影响；已有目录永不修改。"
        ),
    )
    linux_output.add_argument(
        "--output-file-mode",
        type=parse_output_mode,
        default=None,
        metavar="MODE",
        help=(
            "仅 Linux：output 文件权限，例如 664、0664、0644。"
            "作用于最终转码文件和 --copy-other-files 复制的文件；"
            "按精确 chmod mode 应用，不受源文件权限或 umask 影响。"
        ),
    )

    watch = parser.add_argument_group("watch 模式")
    watch.add_argument(
        "--keep-src",
        type=parse_01,
        choices=(0, 1),
        default=1,
        help=(
            "仅 watch 模式。1=保留源文件（默认）；0=成功后删除。\n"
            "配合 --copy-other-files 时，0 使用目录级事务删除："
            "同目录全部视频成功且其它文件复制完成后才删除整个源目录。\n"
            "watch 根目录本身永不删除；根目录中的视频会在成功后逐个删除。"
        ),
    )
    watch.add_argument(
        "--copy-other-files",
        nargs="?",
        const=1,
        default=0,
        type=parse_01,
        choices=(0, 1),
        help=(
            "仅 watch 模式。成功转码后复制源视频所在目录的直接非视频文件；"
            "不递归复制子目录。\n"
            "可写 --copy-other-files 或 --copy-other-files 1；默认 0。\n"
            "keep-sub-path=0 时不同源目录的同名附属文件可能互相覆盖。"
        ),
    )
    watch.add_argument(
        "--keep-sub-path",
        nargs="?",
        const=1,
        default=0,
        type=parse_01,
        choices=(0, 1),
        help=(
            "仅 watch 模式。保持相对子目录结构。\n"
            "例如 watch/sub/a.mkv -> output/sub/a.mp4。默认 0（扁平输出）。\n"
            "建议有重复文件名的目录使用 1；同一运行中若不同输入映射到同一输出名，"
            "脚本会拒绝该冲突而不会静默覆盖。"
        ),
    )
    watch.add_argument(
        "--watch-interval",
        type=float,
        default=2.0,
        help="仅 watch 模式：目录扫描间隔，默认 2 秒。",
    )
    watch.add_argument(
        "--watch-status-interval",
        type=float,
        default=30.0,
        help="仅 watch 模式：状态日志间隔，默认 30 秒；0=关闭。",
    )
    watch.add_argument(
        "--stable-seconds",
        type=float,
        default=5.0,
        help="仅 watch 模式：文件 size/mtime 连续不变多少秒后才入队，默认 5 秒。",
    )

    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument(
        "--cq",
        type=float,
        help=(
            "NVENC VBR-CQ 目标质量，例如 28。数值越小通常质量越高；"
            "0 表示 NVENC 自动。允许范围按所选编码器和当前 FFmpeg build 启动时检查。"
        ),
    )
    rate.add_argument(
        "--bitrate",
        type=validate_bitrate,
        help="NVENC VBR 目标视频码率，例如 2500k、4M、6.5M。",
    )

    video = parser.add_argument_group("视频处理")
    video.add_argument(
        "-r", "--resolution",
        default="source",
        help=(
            "缩放阶段的分辨率上限，默认 source；只允许缩小，不主动放大。\n"
            "限制作用在 crop 之后、rotate/pad 之前；pad 可使最终画布大于该上限。\n"
            "  source    不额外缩放（crop/rotate/pad 仍照常生效）\n"
            "  720p      最大高度720，保持宽高比\n"
            "  1080p     最大高度1080，保持宽高比\n"
            "  1280x720  最大1280x720，保持宽高比并完整装入该范围\n"
            "  1280x-2   最大宽度1280，高度不限，保持宽高比\n"
            "  高度单独限制请直接使用 720p/1080p 等写法"
        ),
    )
    video.add_argument(
        "--crop",
        help=(
            "GPU 裁剪，格式 TOP:BOTTOM:LEFT:RIGHT。\n"
            "例如 --crop 140:140:0:0 表示上下各裁 140 像素。\n"
            "启用后自动选择对应 *_cuvid decoder；不支持则直接报错。"
        ),
    )
    video.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help=(
            "GPU 物理旋转角度，顺时针计：0/90/180/270，默认 0。\n"
            "使用 transpose_cuda。脚本禁用 FFmpeg 自动 rotation metadata；"
            "需要旋转请显式指定。"
        ),
    )
    video.add_argument(
        "--pad",
        help=(
            "GPU 补边后的画布尺寸，例如 --pad 1920x1080。\n"
            "使用 pad_cuda；默认居中、黑色；画布不能小于进入 pad 的图像尺寸。"
        ),
    )
    video.add_argument(
        "--pad-x",
        default="center",
        help="仅 --pad 生效：X 位置，整数或 center，默认 center。",
    )
    video.add_argument(
        "--pad-y",
        default="center",
        help="仅 --pad 生效：Y 位置，整数或 center，默认 center。",
    )
    video.add_argument(
        "--pad-color",
        type=validate_pad_color,
        default="black",
        help="仅 --pad 生效：颜色，默认 black；例如 white、0x202020、black@0.5。",
    )
    video.add_argument(
        "--codec",
        choices=sorted(ENCODERS),
        default="hevc",
        help="输出视频编码器，默认 hevc；可选 h264/hevc/av1。",
    )
    video.add_argument(
        "--preset",
        choices=("p1", "p2", "p3", "p4", "p5", "p6", "p7"),
        default="p5",
        help="NVENC preset：p1 最快，p7 最高质量/最慢；默认 p5。",
    )
    video.add_argument("--gpu", type=int, default=0, help="NVIDIA GPU 索引，默认 0")
    video.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发 FFmpeg 任务数，默认 1；例如 --workers 3",
    )
    video.add_argument(
        "--progress-interval",
        type=float,
        default=2.0,
        help="状态表最小刷新间隔，必须 >=0.25 秒；默认 2 秒。",
    )

    batch = parser.add_argument_group("普通批处理")
    batch.add_argument(
        "--recursive",
        action="store_true",
        help="仅普通目录批处理：递归扫描子目录；watch 始终递归。",
    )

    common = parser.add_argument_group("通用行为")
    common.add_argument(
        "--overwrite",
        action="store_true",
        help="已有输出默认跳过；开启后新文件成功完成时原子替换旧输出。",
    )
    common.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "单个文件失败后继续；默认任一失败就终止其它正在运行的 FFmpeg。"
            "watch 中失败文件同一 size/mtime 不会反复重试；源文件变化后可再次处理。"
        ),
    )

    return parser


def option_was_supplied(argv: list[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in argv)


def parse_config(
    parser: argparse.ArgumentParser,
) -> TranscodeConfig:
    args = parser.parse_args()
    argv = sys.argv[1:]
    watch_mode = args.watch is not None

    if bool(args.input) == watch_mode:
        parser.error("必须且只能指定一个：普通 input 或 --watch DIR")
    if watch_mode and not args.output:
        parser.error("--watch 模式必须指定 --output DIR")

    numeric_checks = (
        (args.watch_interval > 0, "--watch-interval 必须 > 0"),
        (args.watch_status_interval >= 0, "--watch-status-interval 必须 >= 0"),
        (args.stable_seconds >= 0, "--stable-seconds 必须 >= 0"),
        (args.gpu >= 0, "--gpu 必须 >= 0"),
        (args.workers >= 1, "--workers 必须 >= 1"),
        (args.progress_interval >= 0.25, "--progress-interval 必须 >= 0.25"),
        (args.cq is None or args.cq >= 0, "--cq 必须 >= 0"),
    )
    for valid, message in numeric_checks:
        if not valid:
            parser.error(message)

    watch_only = (
        "--keep-src",
        "--copy-other-files",
        "--keep-sub-path",
        "--watch-interval",
        "--watch-status-interval",
        "--stable-seconds",
    )
    if not watch_mode:
        supplied = [name for name in watch_only if option_was_supplied(argv, name)]
        if supplied:
            parser.error("以下参数仅支持 --watch 模式: " + ", ".join(supplied))
    elif args.recursive:
        parser.error("--watch 已始终递归扫描，不需要也不接受 --recursive")

    try:
        resolution_limit = parse_resolution(args.resolution)
        crop = parse_crop(args.crop)
        pad = parse_pad(args.pad)
        pad_x = parse_pad_position(args.pad_x, "x")
        pad_y = parse_pad_position(args.pad_y, "y")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    output_policy = OutputDirPolicy(
        uid=args.output_uid,
        gid=args.output_gid,
        mode=args.output_mode,
    )

    if (
        output_policy.enabled
        or args.output_file_mode is not None
    ) and platform.system() != "Linux":
        parser.error(
            "--output-uid / --output-gid / --output-mode / "
            "--output-file-mode 仅支持 Linux"
        )

    if pad is None:
        orphan_pad = [
            name for name in ("--pad-x", "--pad-y", "--pad-color")
            if option_was_supplied(argv, name)
        ]
        if orphan_pad:
            parser.error("以下参数只有配合 --pad 才有效: " + ", ".join(orphan_pad))

    temp_root = None
    if args.temp_dir:
        temp_root = Path(args.temp_dir).expanduser().resolve()
        if temp_root.exists() and not temp_root.is_dir():
            parser.error(f"--temp-dir 不是目录: {temp_root}")
        temp_root.mkdir(parents=True, exist_ok=True)

    input_value = args.watch if watch_mode else args.input
    input_path = Path(input_value).expanduser().resolve()

    if not watch_mode and input_path.is_file() and args.recursive:
        parser.error("单文件输入不接受 --recursive")

    if not watch_mode and input_path.is_dir() and args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not output.is_dir():
            parser.error(
                "普通目录输入时 --output 必须是目录；"
                f"当前路径是文件: {output}"
            )

    return TranscodeConfig(
        args=args,
        ffmpeg="",
        encoder=ENCODERS[args.codec],
        resolution_limit=resolution_limit,
        crop=crop,
        pad=pad,
        pad_x=pad_x,
        pad_y=pad_y,
        temp_root=temp_root,
        input_path=input_path,
        output_policy=output_policy,
        output_file_mode=args.output_file_mode,
    )


def resolve_runtime(config: TranscodeConfig) -> TranscodeConfig:
    ffmpeg = find_ffmpeg(config.args.ffmpeg)

    if config.args.cq is not None:
        cq_range = probe_encoder_cq_range(ffmpeg, config.encoder)
        if cq_range is not None:
            cq_min, cq_max = cq_range
            if not (cq_min <= config.args.cq <= cq_max):
                raise RuntimeError(
                    f"当前 FFmpeg 的 {config.encoder} 要求 --cq 位于 "
                    f"{cq_min:g}..{cq_max:g}；收到 {config.args.cq:g}"
                )

    print_banner()
    preflight(
        ffmpeg,
        config.encoder,
        need_pad=config.pad is not None,
        need_rotate=config.args.rotate != 0,
    )

    return TranscodeConfig(
        args=config.args,
        ffmpeg=ffmpeg,
        encoder=config.encoder,
        resolution_limit=config.resolution_limit,
        crop=config.crop,
        pad=config.pad,
        pad_x=config.pad_x,
        pad_y=config.pad_y,
        temp_root=config.temp_root,
        input_path=config.input_path,
        output_policy=config.output_policy,
        output_file_mode=config.output_file_mode,
    )


def log_runtime_guards(
    installed_signals: list[str],
    parent_guard_enabled: bool,
    guarded_parent_pid: int | None,
) -> None:
    log_event("INFO", "Signals", ", ".join(installed_signals))

    if parent_guard_enabled:
        log_event(
            "INFO",
            "Parent guard",
            f"Unraid User Scripts detected | PPID {guarded_parent_pid} | "
            "PR_SET_PDEATHSIG=SIGTERM",
        )
    elif sys.platform.startswith("linux"):
        log_event(
            "INFO",
            "Parent guard",
            "not enabled (not launched by Unraid User Scripts)",
        )

    if sys.platform.startswith("linux"):
        log_event(
            "INFO",
            "Child guard",
            "FFmpeg PR_SET_PDEATHSIG=SIGTERM",
        )


def log_common_config(
    config: TranscodeConfig,
    input_label: str,
    input_path: Path,
    output_root: Path,
) -> None:
    args = config.args
    log_event("INFO", input_label, str(input_path))
    log_event("INFO", "Output", str(output_root))
    log_event(
        "INFO",
        "Temp",
        str(config.temp_root) if config.temp_root is not None else "output directory",
    )
    log_event("INFO", "GPU", str(args.gpu))
    log_event("INFO", "Codec", f"{args.codec.upper()} NVENC")
    log_event("INFO", "Quality", config.quality_text)
    log_event("INFO", "MaxRes", args.resolution)
    log_event("INFO", "Workers", str(args.workers))
    if config.output_policy.enabled:
        log_event(
            "INFO",
            "Output dirs",
            config.output_policy.describe(),
        )
    if config.output_file_mode is not None:
        log_event(
            "INFO",
            "Output files",
            f"mode={config.output_file_mode:04o}",
        )


def validate_unique_destinations(
    jobs: list[tuple[Path, Path]],
) -> None:
    owners: dict[Path, Path] = {}
    collisions: list[tuple[Path, Path, Path]] = []

    for src, dst in jobs:
        key = dst.resolve()
        owner = owners.get(key)
        if owner is None:
            owners[key] = src
        elif owner != src:
            collisions.append((key, owner, src))

    if not collisions:
        return

    details = "\n".join(
        f"  {dst}: {first.name} <-> {second.name}"
        for dst, first, second in collisions
    )
    raise RuntimeError(
        "多个输入文件映射到了同一个输出路径；请调整输入文件名/目录结构：\n"
        + details
    )


def build_crop_decoder_map(
    config: TranscodeConfig,
    jobs: list[tuple[Path, Path]],
) -> dict[Path, str]:
    if config.crop is None or not jobs:
        return {}

    log_event("INFO", "Crop", "probing CUVID decoders")
    available = get_available_decoders(config.ffmpeg)
    decoder_map: dict[Path, str] = {}
    summary: dict[str, int] = {}

    for src, _dst in jobs:
        _codec, decoder = resolve_cuvid_crop_decoder(
            config.ffmpeg,
            src,
            available,
        )
        decoder_map[src] = decoder
        summary[decoder] = summary.get(decoder, 0) + 1

    log_event(
        "INFO",
        "Crop",
        ", ".join(f"{name} x{count}" for name, count in sorted(summary.items())),
    )
    return decoder_map


def log_job_failure(
    reporter: ProgressReporter,
    worker_slot: int,
    index: int,
    total: int,
    src: Path,
    exc: Exception,
    failed: list[tuple[Path, str]],
) -> None:
    failed.append((src, str(exc)))
    reporter.mark_failed(worker_slot, index)
    log_event(
        "FAIL",
        "Job",
        f"{index}/{total} | {src.name} | {exc}",
        error=True,
    )


def run_batch_mode(config: TranscodeConfig) -> int:
    args = config.args
    files = collect_inputs(config.input_path, args.recursive)
    output_root = resolve_output_root(config.input_path, args.output)

    log_common_config(config, "Input", config.input_path, output_root)
    log_event("INFO", "Files", str(len(files)))

    jobs: list[tuple[Path, Path]] = []
    skipped = 0

    for src in files:
        dst = output_for_file(
            src,
            config.input_path,
            args.output,
            output_root,
            args.recursive,
            config.output_policy,
        )
        if src.resolve() == dst.resolve():
            raise RuntimeError(
                f"输入和输出映射到同一路径: {src}. 请指定其它 --output。"
            )
        if dst.exists() and not args.overwrite:
            log_event("INFO", "Skip", f"{src.name} | output exists")
            skipped += 1
            continue
        jobs.append((src, dst))

    validate_unique_destinations(jobs)
    crop_decoders = build_crop_decoder_map(config, jobs)

    total = len(jobs)
    ok = 0
    failed: list[tuple[Path, str]] = []
    reporter = ProgressReporter(
        total=total,
        workers=args.workers,
        interval=args.progress_interval,
        gpu=args.gpu,
    )

    def do_job(index: int, src: Path, dst: Path, slot: int):
        return config.run_job(
            reporter,
            index,
            src,
            dst,
            slot,
            crop_decoders.get(src),
        )

    if args.workers == 1:
        for index, (src, dst) in enumerate(jobs, 1):
            try:
                _i, _s, _d, result = do_job(index, src, dst, 1)
                if result == "ok":
                    ok += 1
                else:
                    skipped += 1
                    reporter.mark_skipped(1, index)
            except KeyboardInterrupt:
                terminate_all_ffmpeg()
                raise
            except Exception as exc:
                log_job_failure(reporter, 1, index, total, src, exc, failed)
                if not args.continue_on_error:
                    break
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        in_flight = {}
        next_job = 0
        aborted = False

        def submit(slot: int) -> bool:
            nonlocal next_job
            if _STOP_EVENT.is_set() or next_job >= total:
                return False

            index = next_job + 1
            src, dst = jobs[next_job]
            next_job += 1
            future = executor.submit(do_job, index, src, dst, slot)
            in_flight[future] = (index, src, slot)
            return True

        try:
            for slot in range(1, min(args.workers, total) + 1):
                submit(slot)

            while in_flight:
                future = next(as_completed(tuple(in_flight)))
                index, src, slot = in_flight.pop(future)

                try:
                    _i, _s, _d, result = future.result()
                    if result == "ok":
                        ok += 1
                    else:
                        skipped += 1
                        reporter.mark_skipped(slot, index)
                except KeyboardInterrupt:
                    aborted = True
                    terminate_all_ffmpeg()
                    raise
                except Exception as exc:
                    log_job_failure(
                        reporter,
                        slot,
                        index,
                        total,
                        src,
                        exc,
                        failed,
                    )
                    if not args.continue_on_error:
                        aborted = True
                        terminate_all_ffmpeg()
                        break

                if not aborted and not _STOP_EVENT.is_set():
                    submit(slot)

        except KeyboardInterrupt:
            aborted = True
            terminate_all_ffmpeg()
            raise
        finally:
            if aborted or _STOP_EVENT.is_set():
                shutdown_executor_after_abort(
                    executor,
                    tuple(in_flight),
                    timeout=2.0,
                )
            else:
                executor.shutdown(wait=True)

    log_event(
        "INFO",
        "Summary",
        f"success={ok} skipped={skipped} failed={len(failed)}",
    )
    for src, message in failed:
        log_event("FAIL", "Summary", f"{src} | {message}", error=True)

    return 1 if failed else 0


def prepare_watch_paths(
    config: TranscodeConfig,
) -> tuple[Path, Path]:
    args = config.args
    watch_root = config.input_path

    if not watch_root.is_dir():
        raise RuntimeError(f"--watch 目录不存在: {watch_root}")

    output_root = Path(args.output).expanduser().resolve()
    if output_root == watch_root:
        raise RuntimeError("--output 不能与 --watch 指向同一个目录")
    if output_root.exists() and not output_root.is_dir():
        raise RuntimeError(f"--output 必须是目录: {output_root}")

    ensure_output_directory(output_root, config.output_policy)

    if config.temp_root == watch_root:
        raise RuntimeError("--temp-dir 不能与 --watch 指向同一个目录")

    return watch_root, output_root


def log_watch_config(
    config: TranscodeConfig,
    watch_root: Path,
    output_root: Path,
) -> None:
    args = config.args
    log_common_config(config, "Watch", watch_root, output_root)
    log_event(
        "INFO",
        "Source",
        f"keep={args.keep_src} copy-other={args.copy_other_files} "
        f"sub-path={args.keep_sub_path}",
    )

    if config.crop is not None or args.rotate or config.pad is not None:
        log_event(
            "INFO",
            "Filters",
            f"crop={config.crop or 'off'} rotate={args.rotate} "
            f"pad={config.pad or 'off'}",
        )

    if path_is_within(output_root, watch_root):
        relation = "output is inside watch; output subtree excluded"
    elif path_is_within(watch_root, output_root):
        relation = "output is parent of watch; input remains enabled"
    else:
        relation = "watch/output are separate"
    log_event("INFO", "Watch path", relation)

    if (
        config.temp_root is not None
        and path_is_within(config.temp_root, watch_root)
    ):
        log_event(
            "INFO",
            "Temp path",
            "temp-dir is inside watch; temp subtree excluded",
        )

    log_event(
        "INFO",
        "Watch",
        f"started | scan {args.watch_interval:g}s | "
        f"stable {args.stable_seconds:g}s | Ctrl+C to exit",
    )


def handle_watch_success(
    config: TranscodeConfig,
    state: WatchState,
    watch_root: Path,
    output_root: Path,
    index: int,
    src: Path,
    fp: tuple[int, int],
) -> None:
    args = config.args
    state.settled[src] = fp
    state.failed_fingerprints.pop(src, None)
    state.cleanup_candidates.add(src.parent.resolve())

    if args.copy_other_files:
        copied = copy_sibling_non_video_files(
            src,
            watch_root=watch_root,
            output_root=output_root,
            keep_sub_path=bool(args.keep_sub_path),
            output_policy=config.output_policy,
            output_file_mode=config.output_file_mode,
        )
        log_event(
            "INFO",
            "Copy",
            f"job {index} | {copied} non-video file(s)",
        )

    if args.keep_src == 0:
        if not args.copy_other_files:
            src.unlink()
            log_event("INFO", "Delete", f"job {index} | {src}")
            state.settled.pop(src, None)
        elif src.parent.resolve() == watch_root:
            src.unlink()
            log_event(
                "INFO",
                "Delete",
                f"job {index} | {src} (watch-root file)",
            )
            state.settled.pop(src, None)
        else:
            cleanup_completed_source_directories(
                state.cleanup_candidates,
                watch_root=watch_root,
                output_root=output_root,
                keep_sub_path=bool(args.keep_sub_path),
                settled=state.settled,
                queued=state.queued,
            )

    state.ok += 1


def queue_watch_files(
    config: TranscodeConfig,
    state: WatchState,
    reporter: ProgressReporter,
    watch_root: Path,
    output_root: Path,
    available_decoders: set[str],
    now: float,
) -> int:
    args = config.args
    current_paths: set[Path] = set()
    scanned = scan_watch_videos(
        watch_root,
        output_root,
        config.temp_root,
    )

    for src in scanned:
        current_paths.add(src)

        try:
            fp = file_fingerprint(src)
        except (FileNotFoundError, PermissionError):
            continue

        if state.queued.get(src) == fp:
            continue

        if state.failed_fingerprints.get(src) == fp:
            continue
        state.failed_fingerprints.pop(src, None)

        if state.settled.get(src) == fp:
            dst = watch_output_for_file(
                src,
                watch_root,
                output_root,
                bool(args.keep_sub_path),
            )
            if dst.exists():
                continue
            state.settled.pop(src, None)

        previous = state.observed.get(src)
        if previous is None or previous[:2] != fp:
            state.observed[src] = (fp[0], fp[1], now)
            previous = state.observed[src]
            if args.stable_seconds > 0:
                continue

        unchanged_for = now - previous[2]
        if unchanged_for < args.stable_seconds:
            continue

        dst = watch_output_for_file(
            src,
            watch_root,
            output_root,
            bool(args.keep_sub_path),
        )
        dst_key = dst.resolve()

        owner = state.destination_sources.get(dst_key)
        if owner is not None and owner != src:
            message = (
                "输出路径冲突：不同输入映射到同一个目标文件。"
                f" current={src} owner={owner} dst={dst}. "
                "请使用 --keep-sub-path 1 或调整文件名。"
            )
            state.failed.append((src, message))
            state.failed_fingerprints[src] = fp
            state.observed.pop(src, None)
            log_event("FAIL", "Collision", message, error=True)

            if not args.continue_on_error:
                terminate_all_ffmpeg()
                raise RuntimeError(message)
            continue

        state.destination_sources[dst_key] = src
        if dst_key in state.claimed_destinations:
            continue

        if dst.exists() and not args.overwrite:
            log_event(
                "INFO",
                "Skip",
                f"{src.name} | output exists: {dst}",
            )
            state.settled[src] = fp
            state.failed_fingerprints.pop(src, None)
            state.observed.pop(src, None)
            state.skipped += 1
            continue

        crop_decoder = None
        if config.crop is not None:
            try:
                codec, crop_decoder = resolve_cuvid_crop_decoder(
                    config.ffmpeg,
                    src,
                    available_decoders,
                )
                log_event(
                    "INFO",
                    "Decode",
                    f"{src.name} | codec={codec} | decoder={crop_decoder}",
                )
            except Exception as exc:
                state.failed.append((src, str(exc)))
                state.failed_fingerprints[src] = fp
                state.observed.pop(src, None)
                log_event("FAIL", "Watch", f"{src} | {exc}", error=True)

                if not args.continue_on_error:
                    terminate_all_ffmpeg()
                    raise
                continue

        state.discovered += 1
        index = state.discovered
        reporter.set_total(index)

        state.queued[src] = fp
        state.claimed_destinations.add(dst_key)
        state.observed.pop(src, None)
        state.cleanup_candidates.add(src.parent.resolve())
        state.pending.append((index, src, dst, fp, crop_decoder))

        log_event(
            "INFO",
            "Queue",
            f"job {index} | {src.name} | stable {unchanged_for:.1f}s",
        )

    for old_path in list(state.observed):
        if old_path not in current_paths:
            state.observed.pop(old_path, None)

    return len(scanned)


def run_watch_mode(config: TranscodeConfig) -> int:
    args = config.args
    watch_root, output_root = prepare_watch_paths(config)
    log_watch_config(config, watch_root, output_root)

    available_decoders = (
        get_available_decoders(config.ffmpeg)
        if config.crop is not None
        else set()
    )

    state = WatchState()
    reporter = ProgressReporter(
        total=0,
        workers=args.workers,
        interval=args.progress_interval,
        gpu=args.gpu,
    )
    executor = ThreadPoolExecutor(max_workers=args.workers)
    in_flight = {}
    free_slots = list(range(1, args.workers + 1))

    last_scan = 0.0
    last_status = 0.0
    last_scan_count: int | None = None

    def submit_pending() -> None:
        while (
            state.pending
            and free_slots
            and not _STOP_EVENT.is_set()
        ):
            slot = free_slots.pop(0)
            index, src, dst, fp, decoder = state.pending.pop(0)
            future = executor.submit(
                config.run_job,
                reporter,
                index,
                src,
                dst,
                slot,
                decoder,
            )
            in_flight[future] = (slot, index, src, dst, fp)

    try:
        while not _STOP_EVENT.is_set():
            now = time.monotonic()

            if now - last_scan >= args.watch_interval:
                scan_count = queue_watch_files(
                    config,
                    state,
                    reporter,
                    watch_root,
                    output_root,
                    available_decoders,
                    now,
                )
                if last_scan_count is None:
                    log_event("INFO", "Scan", f"initial {scan_count} video(s)")
                elif scan_count != last_scan_count:
                    log_event(
                        "INFO",
                        "Scan",
                        f"{scan_count} video(s), previous {last_scan_count}",
                    )
                last_scan_count = scan_count
                last_scan = now

            if (
                args.watch_status_interval > 0
                and now - last_status >= args.watch_status_interval
            ):
                log_event(
                    "INFO",
                    "Watch",
                    f"scanned={last_scan_count or 0} "
                    f"observing={len(state.observed)} "
                    f"pending={len(state.pending)} "
                    f"running={len(in_flight)} "
                    f"settled={len(state.settled)} "
                    f"failed={len(state.failed_fingerprints)}",
                )
                last_status = now

            submit_pending()

            for future in [f for f in tuple(in_flight) if f.done()]:
                slot, index, src, dst, fp = in_flight.pop(future)
                free_slots.append(slot)
                free_slots.sort()
                state.queued.pop(src, None)
                state.claimed_destinations.discard(dst.resolve())

                try:
                    _i, _s, _d, result = future.result()
                    if result == "ok":
                        handle_watch_success(
                            config,
                            state,
                            watch_root,
                            output_root,
                            index,
                            src,
                            fp,
                        )
                    else:
                        state.skipped += 1
                        state.settled[src] = fp
                        state.failed_fingerprints.pop(src, None)
                        reporter.mark_skipped(slot, index)

                except KeyboardInterrupt:
                    raise

                except Exception as exc:
                    state.failed.append((src, str(exc)))
                    state.failed_fingerprints[src] = fp
                    reporter.mark_failed(slot, index)
                    log_event(
                        "FAIL",
                        f"W{slot}",
                        f"job {index} | {src.name} | {exc}",
                        error=True,
                    )
                    if not args.continue_on_error:
                        terminate_all_ffmpeg()
                        raise

            if (
                args.keep_src == 0
                and args.copy_other_files
                and state.cleanup_candidates
            ):
                cleanup_completed_source_directories(
                    state.cleanup_candidates,
                    watch_root=watch_root,
                    output_root=output_root,
                    keep_sub_path=bool(args.keep_sub_path),
                    settled=state.settled,
                    queued=state.queued,
                )

            time.sleep(min(0.25, args.watch_interval))

    finally:
        if _STOP_EVENT.is_set():
            shutdown_executor_after_abort(
                executor,
                tuple(in_flight),
                timeout=2.0,
            )
        else:
            executor.shutdown(wait=True)

    return 0


def graceful_exit_code(exc: KeyboardInterrupt) -> int:
    _STOP_EVENT.set()

    if isinstance(exc, ShutdownSignal):
        signum = exc.signum
        log_event(
            "WARN",
            "Signal",
            f"{_signal_name(signum)}: stopping all FFmpeg jobs",
            error=True,
        )
        return 128 + signum

    log_event(
        "WARN",
        "Abort",
        "Ctrl+C: stopping all FFmpeg jobs",
        error=True,
    )
    return 130


def main() -> int:
    global _RECEIVED_SIGNAL

    _STOP_EVENT.clear()
    _RECEIVED_SIGNAL = None

    installed_signals = install_signal_handlers()
    parent_guard_enabled, guarded_parent_pid = enable_self_parent_death_guard()

    save_terminal_state()
    atexit.register(restore_terminal_state)

    parser = build_parser()
    config = parse_config(parser)

    try:
        config = resolve_runtime(config)

        log_runtime_guards(
            installed_signals,
            parent_guard_enabled,
            guarded_parent_pid,
        )

        if config.watch_mode:
            return run_watch_mode(config)
        return run_batch_mode(config)

    except KeyboardInterrupt as exc:
        exit_code = graceful_exit_code(exc)
        terminate_all_ffmpeg()
        restore_terminal_state()
        return exit_code

    except Exception as exc:
        log_event("FAIL", "Fatal", str(exc), error=True)
        terminate_all_ffmpeg()
        restore_terminal_state()
        return 1


def finalize_process_exit(exit_code: int) -> None:
    """
    Complete process termination after main() has performed normal cleanup.

    When shutdown was initiated by a signal, do not use SystemExit: CPython's
    concurrent.futures atexit hook waits for ThreadPoolExecutor worker threads,
    including a worker stuck in third-party/native code. That can leave the
    nv_transcode Python PID alive even though main() already returned.

    os._exit() is intentionally used only for signal-driven shutdown, and only
    after:
      * all tracked FFmpeg/probe processes have been terminated
      * workers received a bounded drain window
      * terminal state was restored
      * all log output has been flushed
    """
    if _RECEIVED_SIGNAL is None:
        raise SystemExit(exit_code)

    try:
        log_event(
            "INFO",
            "Exit",
            f"pid={os.getpid()} code={exit_code}",
        )
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


if __name__ == "__main__":
    internal_rc = maybe_run_internal_mode()
    if internal_rc is not None:
        raise SystemExit(internal_rc)

    finalize_process_exit(main())
