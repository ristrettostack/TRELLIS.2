# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,xformers,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec,wheel-dir: -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
XFORMERS=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
WHEEL_DIR=""
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --wheel-dir) WHEEL_DIR="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention (Ampere+/sm80+ GPUs only - see note below on Turing/T4)"
    echo "  --xformers              Install xformers (attention backend with Turing/T4 support, sm75+)"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    echo "  --wheel-dir DIR         Cache/reuse built wheels for nvdiffrast, nvdiffrec, cumesh,"
    echo "                          o-voxel, and flexgemm in DIR instead of compiling every run."
    echo "                          First run with an empty/new DIR builds and caches; later runs"
    echo "                          (even in a fresh Kaggle/Colab session) reusing the same DIR"
    echo "                          just pip-install the cached .whl - no recompiling."
    echo "                          On Kaggle: point DIR at /kaggle/working/wheels the first time,"
    echo "                          then turn that folder into a Kaggle Dataset and point DIR at"
    echo "                          the mounted /kaggle/input/<dataset> path on later runs."
    echo "                          NOTE: wheels are compiled for a specific GPU arch (sm75 for T4,"
    echo "                          etc) - only reuse a wheel dir across runs on the same GPU type."
    return
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    exit 1
fi

# Detect compute capability so we can warn (not block) when flash-attn is requested
# on a GPU it doesn't support. flash-attn's PyPI/official wheels only support
# Ampere/Ada/Hopper (compute capability >= 8.0); on Turing cards like the T4
# (compute capability 7.5) it installs "successfully" but raises at the first
# kernel call. trellis2's own attention config already auto-falls-back to
# xformers/sdpa at runtime if this happens, but it's cheaper to just tell the
# user up front and point them at --xformers instead, which officially supports
# Turing (sm75+) via its Flash and CUTLASS operators.
GPU_CC=""
if [ "$PLATFORM" = "cuda" ] && command -v nvidia-smi > /dev/null; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
fi

# Build-once, reuse-forever helper for the source-compiled extensions (nvdiffrast,
# nvdiffrec, cumesh, o-voxel, flexgemm). Each of these currently does a plain
# `pip install <src_dir> --no-build-isolation`, which recompiles CUDA code from
# scratch on every single run - the main cost of the ~30min setup on an ephemeral
# environment like Kaggle. When --wheel-dir is given, this instead:
#   1. Looks in WHEEL_DIR for an already-built wheel matching the package -> if
#      found, just `pip install`s it directly (seconds, no compiling).
#   2. Otherwise builds a wheel via `pip wheel` (compiles once), installs it, and
#      - if WHEEL_DIR is writable - saves it into WHEEL_DIR so the *next* run
#      (even a fresh Kaggle session, as long as it reuses the same WHEEL_DIR/
#      dataset) can skip straight to step 1.
# When --wheel-dir is not given at all, behavior is unchanged (always compiles).
install_or_build_wheel() {
    local name="$1" pkg_glob="$2" src_path="$3" extra_pip_args="$4"

    if [ -z "$WHEEL_DIR" ] ; then
        pip install "$src_path" --no-build-isolation $extra_pip_args
        return
    fi

    mkdir -p "$WHEEL_DIR" 2>/dev/null
    existing_wheel=$(find "$WHEEL_DIR" -iname "${pkg_glob}*.whl" 2>/dev/null | head -n1)
    if [ -n "$existing_wheel" ] ; then
        echo "[$name] Found cached wheel, installing directly (no compile): $existing_wheel"
        pip install "$existing_wheel" $extra_pip_args
        return
    fi

    if [ -w "$WHEEL_DIR" ] ; then
        echo "[$name] No cached wheel in $WHEEL_DIR - building once and caching for next time..."
        pip wheel "$src_path" --no-build-isolation --no-deps -w "$WHEEL_DIR"
        built_wheel=$(find "$WHEEL_DIR" -iname "${pkg_glob}*.whl" 2>/dev/null | head -n1)
        if [ -n "$built_wheel" ] ; then
            pip install "$built_wheel" $extra_pip_args
        else
            echo "[$name] Warning: 'pip wheel' didn't produce a ${pkg_glob}*.whl file (check the"
            echo "[$name] package's normalized name) - installing directly without caching instead."
            pip install "$src_path" --no-build-isolation $extra_pip_args
        fi
    else
        # WHEEL_DIR is read-only (e.g. a mounted Kaggle Dataset input) and had no
        # matching wheel - can't cache here, just build+install for this run.
        echo "[$name] No cached wheel in read-only $WHEEL_DIR - compiling for this run only"
        echo "[$name] (build a wheel into a writable --wheel-dir once, then re-upload it as a"
        echo "[$name] dataset, to avoid recompiling $name in future runs)."
        pip install "$src_path" --no-build-isolation $extra_pip_args
    fi
}

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        # Compare using plain bash integer arithmetic on the major version (e.g. "7.5" -> 7)
        # rather than `bc`, which isn't guaranteed to be present on minimal/slim images.
        GPU_CC_MAJOR="${GPU_CC%%.*}"
        if [ -n "$GPU_CC_MAJOR" ] && [ "$GPU_CC_MAJOR" -lt 8 ] 2>/dev/null ; then
            echo "[FLASHATTN] Warning: detected GPU compute capability $GPU_CC (e.g. T4 = 7.5)."
            echo "[FLASHATTN] flash-attn's official wheels only support Ampere+ (sm80+) and will"
            echo "[FLASHATTN] fail at runtime on this GPU, even though the pip install itself succeeds."
            echo "[FLASHATTN] Installing anyway since --flash-attn was requested, but consider using"
            echo "[FLASHATTN] --xformers instead, or setting ATTN_BACKEND=xformers / ATTN_BACKEND=sdpa."
        fi
        pip install flash-attn==2.7.3
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$XFORMERS" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        # Installed unpinned, from the same PyTorch cu124 wheel index used for torch
        # itself above, so pip resolves an xformers build matching the already-installed
        # torch==2.6.0+cu124 instead of silently upgrading/downgrading torch.
        pip install xformers --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install xformers --index-url https://download.pytorch.org/whl/rocm6.2.4
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NEW_ENV" = true ] ; then
    conda create -n trellis2 python=3.10
    conda activate trellis2
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    pip install kornia timm
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        install_or_build_wheel "NVDIFFRAST" "nvdiffrast" /tmp/extensions/nvdiffrast
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        install_or_build_wheel "NVDIFFREC" "nvdiffrec" /tmp/extensions/nvdiffrec
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    install_or_build_wheel "CUMESH" "cumesh" /tmp/extensions/CuMesh
fi

if [ "$FLEXGEMM" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    # matches either a "flexgemm*" or "flex_gemm*" distribution name, since the
    # importable module name is `flex_gemm` but the wheel's distribution name
    # (used in the .whl filename) isn't guaranteed to match exactly
    install_or_build_wheel "FLEXGEMM" "flex*gemm" /tmp/extensions/FlexGEMM
fi

if [ "$OVOXEL" = true ] ; then
    mkdir -p /tmp/extensions
    cp -r o-voxel /tmp/extensions/o-voxel
    # NOTE: o-voxel/pyproject.toml originally pinned `cumesh` and `flex_gemm` as
    # direct git+https URLs, which made pip re-clone and recompile both from
    # source on every o-voxel install even when already-built local copies
    # existed. Fixed upstream in trellis2/o-voxel/pyproject.toml by pinning them
    # as plain names instead, so pip resolves against whatever's already
    # installed (cumesh/flex_gemm run earlier in this script). If you're on a
    # copy of the repo that still has the old git-URL pins, either patch
    # o-voxel/pyproject.toml directly or add --no-deps to the install below.
    install_or_build_wheel "O-VOXEL" "o_voxel" /tmp/extensions/o-voxel
fi