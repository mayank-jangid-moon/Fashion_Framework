#!/bin/bash

# OpenPose Wrapper Script
# This script ensures OpenPose runs from the correct directory with proper library paths

set -e

# Parse command line arguments
OPENPOSE_BIN=""
IMAGE_DIR=""
JSON_DIR=""
IMAGES_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --openpose_bin)
            OPENPOSE_BIN="$2"
            shift 2
            ;;
        --image_dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --write_json)
            JSON_DIR="$2"
            shift 2
            ;;
        --write_images)
            IMAGES_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$OPENPOSE_BIN" ] || [ -z "$IMAGE_DIR" ] || [ -z "$JSON_DIR" ] || [ -z "$IMAGES_DIR" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 --openpose_bin <path> --image_dir <path> --write_json <path> --write_images <path> [other_args...]"
    exit 1
fi

# Check if OpenPose binary exists
if [ ! -f "$OPENPOSE_BIN" ]; then
    echo "ERROR: OpenPose binary not found: $OPENPOSE_BIN"
    exit 1
fi

# Get the OpenPose root directory (should be 4 levels up from the binary)
# e.g., openpose/build/examples/openpose/openpose.bin -> openpose/
OPENPOSE_ROOT=$(dirname $(dirname $(dirname $(dirname "$OPENPOSE_BIN"))))
OPENPOSE_ROOT=$(realpath "$OPENPOSE_ROOT")

echo "OpenPose binary: $OPENPOSE_BIN"
echo "OpenPose root directory: $OPENPOSE_ROOT"
echo "Working directory will be: $OPENPOSE_ROOT"

# Convert paths to absolute paths
IMAGE_DIR_ABS=$(realpath "$IMAGE_DIR")
JSON_DIR_ABS=$(realpath "$JSON_DIR")
IMAGES_DIR_ABS=$(realpath "$IMAGES_DIR")

# Create output directories
mkdir -p "$JSON_DIR_ABS"
mkdir -p "$IMAGES_DIR_ABS"

# Get the relative path to the binary from the OpenPose root
BINARY_REL_PATH=$(realpath --relative-to="$OPENPOSE_ROOT" "$OPENPOSE_BIN")

echo "Converting paths to absolute:"
echo "  Image dir: $IMAGE_DIR_ABS"
echo "  JSON dir: $JSON_DIR_ABS"
echo "  Images dir: $IMAGES_DIR_ABS"

# Change to OpenPose root directory
cd "$OPENPOSE_ROOT"

# Set library path for OpenPose
export LD_LIBRARY_PATH="$OPENPOSE_ROOT/build/src/openpose:$OPENPOSE_ROOT/build/caffe/lib:$LD_LIBRARY_PATH"

echo "Library path: $LD_LIBRARY_PATH"

# Build command as array to handle spaces properly
OPENPOSE_CMD=(
    "./$BINARY_REL_PATH"
    "--image_dir" "$IMAGE_DIR_ABS"
    "--write_json" "$JSON_DIR_ABS"
    "--write_images" "$IMAGES_DIR_ABS"
)

# Add extra arguments if any
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    OPENPOSE_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running OpenPose command: ${OPENPOSE_CMD[*]}"
echo "Current directory: $(pwd)"

# Execute OpenPose with proper array expansion
exec "${OPENPOSE_CMD[@]}"
