#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set default container name
container_tag="puma-challenge-baseline-track1"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    container_tag="$1"
fi

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$container_tag")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $container_tag"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(date -d "$build_timestamp" +"%Y%m%d_%H%M%S")

# Set the output filename with timestamp and build information
output_filename="${SCRIPT_DIR}/${container_tag}_${formatted_build_info}.tar.gz"

# 进度提示函数
progress_indicator() {
    while :; do
        printf "."
        sleep 2
    done
}

echo "[1/3] Starting Docker container export: $container_tag"
echo -n "Progress: "

# 启动后台进度指示
progress_indicator &
PROGRESS_PID=$!
disown

# 保存并压缩容器
docker save "$container_tag" | gzip -vc > "$output_filename"

# 停止进度指示
kill $PROGRESS_PID
echo -e "\n[2/3] Compression completed"

# 最终文件校验
echo "[3/3] Verifying output file:"
ls -lh "$output_filename" | awk '{print "Size:", $5, "File:", $NF}'

echo "Successfully saved to:"
echo "$output_filename"