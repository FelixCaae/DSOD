#!/bin/bash
# production_runner.sh

set -euo pipefail  # 严格的错误处理
# 配置
readonly LOG_DIR="./logs"
readonly LOCK_FILE="/tmp/script_runner.lock"

# 初始化
setup() {
    mkdir -p "$LOG_DIR"
    
    # 检查锁文件，防止重复执行
    if [[ -f "$LOCK_FILE" ]]; then
        echo "错误: 另一个实例正在运行"
        exit 1
    fi
    touch "$LOCK_FILE"
    
    # 清理函数
    cleanup() {
        rm -f "$LOCK_FILE"
    }
    trap cleanup EXIT
}

# 主执行
main() {
    setup
    
    local scripts=("./configs/def-detr-base/city2foggy/teaching_standard_dino_w04.sh" "./configs/def-detr-base/city2foggy/teaching_standard_dino_w06.sh")
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    for script in "${scripts[@]}"; do
        local log_file="$LOG_DIR/${script%.sh}_${timestamp}.log"
        
        echo "执行: $script -> 日志: $log_file"
        
        # 带超时执行（30分钟超时）
        timeout 1800 bash "$script" > "$log_file" 2>&1
        
        if [[ $? -eq 0 ]]; then
            echo "✓ $script 完成"
        else
            echo "✗ $script 失败或超时"
            exit 1
        fi
    done
    
    echo "所有脚本执行成功"
}

main "$@"