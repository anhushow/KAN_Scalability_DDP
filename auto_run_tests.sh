#!/bin/bash

# ==============================================================================
# KAN并行训练性能测试自动化脚本
# 功能：批量提交所有30个测试，智能管理作业队列
# 作者：KAN Performance Testing Suite
# 版本：2.0
# ==============================================================================


SCRIPT_DIR="test_scripts"
RESULTS_DIR="results"
LOGS_DIR="logs"
MAX_CONCURRENT_JOBS=8          # 最大并发作业数
SUBMIT_INTERVAL=30             # 提交间隔（秒）
MONITOR_INTERVAL=60            # 监控间隔（秒）
MAX_RETRIES=2                  # 最大重试次数
# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color


log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


declare -A TEST_CONFIGS=(
    # 强扩展性测试（第一批）- 6个
    ["strong_scaling_n1g1"]="priority=1,desc=1GPU基准测试,time=120"
    ["strong_scaling_n1g2"]="priority=1,desc=2GPU单节点强扩展,time=120"
    ["strong_scaling_n2g2"]="priority=1,desc=2GPU双节点强扩展,time=150"
    ["strong_scaling_n2g4"]="priority=1,desc=4GPU双节点强扩展,time=150"
    ["strong_scaling_n4g4"]="priority=1,desc=4GPU四节点强扩展,time=180"
    ["strong_scaling_n4g8"]="priority=1,desc=8GPU四节点强扩展,time=180"

    # 弱扩展性测试（第二批）- 6个
    ["weak_scaling_n1g1"]="priority=2,desc=1GPU弱扩展基准,time=120"
    ["weak_scaling_n1g2"]="priority=2,desc=2GPU单节点弱扩展,time=120"
    ["weak_scaling_n2g2"]="priority=2,desc=2GPU双节点弱扩展,time=150"
    ["weak_scaling_n2g4"]="priority=2,desc=4GPU双节点弱扩展,time=150"
    ["weak_scaling_n4g4"]="priority=2,desc=4GPU四节点弱扩展,time=180"
    ["weak_scaling_n4g8"]="priority=2,desc=8GPU四节点弱扩展,time=180"

    # 通信开销测试（第三批）- 5个
    ["communication_intra_n1g2"]="priority=3,desc=节点内通信基准,time=120"
    ["communication_inter_n2g2"]="priority=3,desc=2GPU跨节点通信,time=150"
    ["communication_inter_n2g4"]="priority=3,desc=4GPU跨节点通信,time=150"
    ["communication_inter_n4g4"]="priority=3,desc=4GPU四节点间通信,time=180"
    ["communication_inter_n4g8"]="priority=3,desc=8GPU四节点间通信,time=200"

    # 模型扩展性测试（第四批）- 4个
    ["model_scaling_small_n2g4"]="priority=4,desc=小型模型(~1K参数)4GPU,time=150"
    ["model_scaling_medium_n2g4"]="priority=4,desc=中型模型(~4K参数)4GPU,time=150"
    ["model_scaling_large_n2g4"]="priority=4,desc=大型模型(~16K参数)4GPU,time=180"
    ["model_scaling_xlarge_n2g4"]="priority=4,desc=超大型模型(~64K参数)4GPU,time=200"
)


setup_directories() {
    log "创建必要目录..."
    mkdir -p "$RESULTS_DIR" "$LOGS_DIR" test_output

    
    log "检查并设置脚本执行权限..."
    chmod +x test_scripts/*.sh 2>/dev/null || true
    chmod +x auto_run_tests.sh 2>/dev/null || true

   
    cat > test_status.txt << 'EOF'
# KAN性能测试状态跟踪
# 格式: 测试名称 状态 作业ID 提交时间 完成时间
EOF

    success "目录创建完成，脚本权限已设置"
}


get_queue_count() {
    squeue -u $USER --noheader | wc -l
}


get_completed_count() {
    
    find "$RESULTS_DIR" -name "*_performance.json" 2>/dev/null | wc -l
}


get_test_config() {
    local test_name=$1
    local key=$2
    echo "${TEST_CONFIGS[$test_name]}" | grep -o "${key}=[^,]*" | cut -d'=' -f2
}


get_resource_level() {
    local test_name=$1
   
    if [[ $test_name =~ n1g1 ]]; then
        echo "1"  
    elif [[ $test_name =~ n1g2 ]]; then
        echo "2"
    elif [[ $test_name =~ n2g2 ]]; then
        echo "3"
    elif [[ $test_name =~ n1g4 ]]; then
        echo "4"
    elif [[ $test_name =~ n2g4 ]]; then
        echo "5"
    elif [[ $test_name =~ n1g8 ]]; then
        echo "6"
    elif [[ $test_name =~ n2g8 ]]; then
        echo "7"
    elif [[ $test_name =~ n4g8 ]]; then
        echo "8"  
    else
        echo "5"  
    fi
}


smart_submit_pending_tests() {
    local max_submissions=${1:-3}  
    local submitted_count=0

    
    local temp_file=$(mktemp)
    local counter_file=$(mktemp)
    echo "0" > "$counter_file"

    
    while read -r line; do
        if [[ $line =~ ^[^#].* ]]; then
            local test_name=$(echo "$line" | cut -d' ' -f1)
            local status=$(echo "$line" | cut -d' ' -f2)

            if [ "$status" == "PENDING" ]; then
                local resource_level=$(get_resource_level "$test_name")
                echo "$resource_level $test_name" >> "$temp_file"
            fi
        fi
    done < test_status.txt

    
    local sorted_file=$(mktemp)
    sort -n "$temp_file" > "$sorted_file"

    while read resource_level test_name; do
        local current_count=$(cat "$counter_file")
        if [ $current_count -ge $max_submissions ]; then
            break
        fi

        
        if [ $(get_queue_count) -lt $MAX_CONCURRENT_JOBS ]; then
            local test_priority=$(get_test_config "$test_name" "priority")
            local test_desc=$(get_test_config "$test_name" "desc")

            log "智能重试 (资源等级$resource_level): $test_name"

            if submit_test "$test_name" "$test_priority" "$test_desc" "true"; then
                echo $((current_count + 1)) > "$counter_file"
                sleep $SUBMIT_INTERVAL
            fi
        else
            break  
        fi
    done < "$sorted_file"

    submitted_count=$(cat "$counter_file")

    rm -f "$temp_file" "$counter_file" "$sorted_file"

    if [ $submitted_count -gt 0 ]; then
        log "智能提交完成: 本轮提交了 $submitted_count 个测试"
    fi

    return $submitted_count
}


is_test_completed() {
    local test_name=$1
    
    [[ -f "$RESULTS_DIR/${test_name}_performance.json" ]] \
    || [[ -f "$RESULTS_DIR/${test_name}_single_performance.json" ]] \
    || [[ -f "$RESULTS_DIR/${test_name}_ddp_performance.json" ]]
}




is_test_submitted() {
    local test_name=$1
    grep -q "^$test_name \(SUBMITTED\|PENDING\|RETRY\)" test_status.txt 2>/dev/null
}


get_test_status() {
    local test_name=$1
    if is_test_completed "$test_name"; then
        echo "COMPLETED"
    elif grep -q "^$test_name SUBMITTED" test_status.txt 2>/dev/null; then
        echo "SUBMITTED"
    elif grep -q "^$test_name PENDING" test_status.txt 2>/dev/null; then
        echo "PENDING"
    elif grep -q "^$test_name RETRY" test_status.txt 2>/dev/null; then
        echo "RETRY"
    elif grep -q "^$test_name FAILED" test_status.txt 2>/dev/null; then
        echo "FAILED"
    else
        echo "NOT_STARTED"
    fi
}


submit_test() {
    local test_name=$1
    local priority=$2
    local desc=$3
    local is_retry=${4:-false}

    if is_test_completed "$test_name"; then
        success "测试 $test_name 已完成，跳过"
        return 0
    fi

    if is_test_submitted "$test_name" && [ "$is_retry" != "true" ]; then
        warning "测试 $test_name 已提交，跳过"
        return 0
    fi

    
    if [ ! -f "$SCRIPT_DIR/${test_name}.sh" ]; then
        error "测试脚本不存在: $SCRIPT_DIR/${test_name}.sh"
        return 1
    fi

    
    local job_id=$(sbatch --parsable "$SCRIPT_DIR/${test_name}.sh" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$job_id" ]; then
        success "已提交测试: $test_name (作业ID: $job_id)"
        log "  描述: $desc"
        log "  优先级: $priority"

        
        if grep -q "^$test_name " test_status.txt 2>/dev/null; then
            
            sed -i "s/^$test_name .*/$test_name SUBMITTED $job_id $(date '+%Y-%m-%d %H:%M:%S') -/" test_status.txt
        else
            
            echo "$test_name SUBMITTED $job_id $(date '+%Y-%m-%d %H:%M:%S') -" >> test_status.txt
        fi

        return 0
    else
        warning "提交失败: $test_name (资源不足或队列满)"

        # 标记为PENDING状态，等待重试
        if grep -q "^$test_name " test_status.txt 2>/dev/null; then
            sed -i "s/^$test_name .*/$test_name PENDING - $(date '+%Y-%m-%d %H:%M:%S') -/" test_status.txt
        else
            echo "$test_name PENDING - $(date '+%Y-%m-%d %H:%M:%S') -" >> test_status.txt
        fi

        return 1
    fi
}


batch_submit_tests() {
    local priority=$1
    local batch_name=$2

    log "开始提交$batch_name (优先级: $priority)"

    local submitted_count=0
    local pending_count=0
    local skipped_count=0

    
    local temp_file=$(mktemp)

    for test_name in "${!TEST_CONFIGS[@]}"; do
        local test_priority=$(get_test_config "$test_name" "priority")

        if [ "$test_priority" == "$priority" ]; then
            local resource_level=$(get_resource_level "$test_name")
            echo "$resource_level $test_name" >> "$temp_file"
        fi
    done

    
    local sorted_file=$(mktemp)
    sort -n "$temp_file" > "$sorted_file"

    while read resource_level test_name; do
        local test_priority=$(get_test_config "$test_name" "priority")
        local test_desc=$(get_test_config "$test_name" "desc")

        
        while [ $(get_queue_count) -ge $MAX_CONCURRENT_JOBS ]; do
            log "队列已满 ($(get_queue_count)/$MAX_CONCURRENT_JOBS)，等待$MONITOR_INTERVAL 秒..."
            sleep $MONITOR_INTERVAL
        done

        
        if submit_test "$test_name" "$test_priority" "$test_desc"; then
            ((submitted_count++))
            sleep $SUBMIT_INTERVAL
        else
            
            if is_test_completed "$test_name"; then
                ((skipped_count++))
            else
                ((pending_count++))
            fi
        fi
    done < "$sorted_file"

    rm -f "$temp_file" "$sorted_file"

    success "$batch_name 提交完成: 成功提交 $submitted_count 个，等待重试 $pending_count 个，已完成跳过 $skipped_count 个"
}


monitor_jobs() {
    local running_jobs=$(get_queue_count)
    local completed_tests=$(get_completed_count)
    local total_tests=${#TEST_CONFIGS[@]}

    
    local submitted_count=$(grep -c "SUBMITTED" test_status.txt 2>/dev/null || echo "0")
    local pending_count=$(grep -c "PENDING" test_status.txt 2>/dev/null || echo "0")
    local failed_count=$(grep -c "FAILED" test_status.txt 2>/dev/null || echo "0")
    local not_started_count=0

    
    for test_name in "${!TEST_CONFIGS[@]}"; do
        if [ "$(get_test_status "$test_name")" == "NOT_STARTED" ]; then
            ((not_started_count++))
        fi
    done

    
    clear
    echo "================================================================="
    echo "           KAN并行训练性能测试 - 实时监控"
    echo "================================================================="
    echo "当前时间: $(date)"
    echo "运行中作业: $running_jobs"
    echo "已完成测试: $completed_tests/$total_tests"
    echo "完成率: $(echo "scale=1; $completed_tests * 100 / $total_tests" | bc -l 2>/dev/null || echo "0")%"
    echo "================================================================="
    echo "测试状态统计:"
    echo "  已完成: $completed_tests"
    echo "  运行中: $submitted_count"
    echo "  等待重试: $pending_count"
    echo "  失败: $failed_count"
    echo "  未开始: $not_started_count"
    echo "================================================================="

    
    echo "作业队列状态:"
    squeue -u $USER --format="%.10i %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "无运行作业"

    echo "================================================================="

    
    echo "最近完成的测试:"
    ls -t "$RESULTS_DIR"/*_performance.json "$RESULTS_DIR"/*_single_performance.json "$RESULTS_DIR"/*_ddp_performance.json 2>/dev/null | head -5 | while read file; do
        local test_name=$(basename "$file" | sed 's/_single_performance.json\|_ddp_performance.json\|_performance.json//')
        local file_time=$(stat -c %y "$file" 2>/dev/null | cut -d'.' -f1)
        echo "  * $test_name ($file_time)"
    done

    echo "================================================================="

    
    echo "等待重试的测试:"
    if [ "$pending_count" -gt 0 ]; then
        grep "PENDING" test_status.txt 2>/dev/null | while read line; do
            local test_name=$(echo "$line" | cut -d' ' -f1)
            echo "  ~ $test_name"
        done
    else
        echo "  暂无等待重试的测试"
    fi

    echo "================================================================="

    
    check_and_retry_failed_jobs
}


check_and_retry_failed_jobs() {
    local retry_count=0

    
    while read -r line; do
        if [[ $line =~ ^[^#].* ]]; then
            local test_name=$(echo "$line" | cut -d' ' -f1)
            local status=$(echo "$line" | cut -d' ' -f2)
            local job_id=$(echo "$line" | cut -d' ' -f3)

            if [ "$status" == "SUBMITTED" ]; then
                # 检查作业是否还在队列中
                if ! squeue -j "$job_id" &>/dev/null; then
                    # 作业已完成，检查是否成功
                    if is_test_completed "$test_name"; then
                        # 更新状态为完成
                        sed -i "s/^$test_name SUBMITTED/$test_name COMPLETED/" test_status.txt
                        success "测试完成: $test_name"
                    else
                        # 作业失败，标记为PENDING等待重试
                        sed -i "s/^$test_name SUBMITTED/$test_name PENDING/" test_status.txt
                        warning "测试失败，标记为等待重试: $test_name"
                    fi
                fi
            fi
        fi
    done < test_status.txt

    
    smart_submit_pending_tests 5  
    retry_count=$?

    if [ $retry_count -gt 0 ]; then
        log "本轮智能重试了 $retry_count 个PENDING测试"
    fi
}

retry_failed_tests() {
    log "检查并重试失败的测试..."
    
    local retry_count=0
    
    while read -r line; do
        if [[ $line =~ ^[^#].* ]]; then
            local test_name=$(echo "$line" | cut -d' ' -f1)
            local status=$(echo "$line" | cut -d' ' -f2)
            
            if [ "$status" == "FAILED" ]; then
                local test_priority=$(get_test_config "$test_name" "priority")
                local test_desc=$(get_test_config "$test_name" "desc")
                
                log "重试失败的测试: $test_name"
                
                # 等待队列空间
                while [ $(get_queue_count) -ge $MAX_CONCURRENT_JOBS ]; do
                    sleep $MONITOR_INTERVAL
                done
                
                # 重新提交
                if submit_test "$test_name" "$test_priority" "$test_desc"; then
                    ((retry_count++))
                    # 更新状态
                    sed -i "s/^$test_name FAILED/$test_name RETRY/" test_status.txt
                    sleep $SUBMIT_INTERVAL
                fi
            fi
        fi
    done < test_status.txt
    
    if [ $retry_count -gt 0 ]; then
        success "重试了$retry_count 个失败的测试"
    else
        log "没有需要重试的测试"
    fi
}


generate_final_report() {
    log "生成最终测试报告..."
    
    local report_file="test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
================================================================
KAN并行训练性能测试 - 最终报告
================================================================
测试完成时间: $(date)
总测试数量: ${#TEST_CONFIGS[@]}
完成测试数量: $(get_completed_count)
成功率: $(echo "scale=1; $(get_completed_count) * 100 / ${#TEST_CONFIGS[@]}" | bc -l 2>/dev/null || echo "0")%

================================================================
测试结果统计:
================================================================
EOF
    
    # 按优先级分组统计
    for priority in 1 2 3 4; do
        local batch_name=""
        case $priority in
            1) batch_name="强扩展性测试" ;;
            2) batch_name="弱扩展性测试" ;;
            3) batch_name="通信开销测试" ;;
            4) batch_name="模型扩展性测试" ;;
        esac
        
        echo "" >> "$report_file"
        echo "$batch_name:" >> "$report_file"
        echo "----------------------------------------" >> "$report_file"
        
        for test_name in "${!TEST_CONFIGS[@]}"; do
            local test_priority=$(get_test_config "$test_name" "priority")
            local test_desc=$(get_test_config "$test_name" "desc")
            
            if [ "$test_priority" == "$priority" ]; then
                if is_test_completed "$test_name"; then
                    echo "  + $test_name - $test_desc" >> "$report_file"
                else
                    echo "  - $test_name - $test_desc" >> "$report_file"
                fi
            fi
        done
    done
    
    cat >> "$report_file" << EOF

================================================================
详细测试状态:
================================================================
EOF
    
    
    cat test_status.txt >> "$report_file"
    
    cat >> "$report_file" << EOF

================================================================
文件位置:
================================================================
结果文件: $RESULTS_DIR/
日志文件: $LOGS_DIR/
测试状态: test_status.txt
测试报告: $report_file

================================================================
后续操作建议:
================================================================
1. 生成结果汇总: python summarize_results.py
2. 生成性能图表: 自动生成在当前目录
3. 打包结果文件: tar -czf results_backup.tar.gz results/ logs/
4. 检查失败测试: grep FAILED test_status.txt
5. 重新运行失败测试: 手动提交相应的脚本文件
================================================================
快速性能查看命令:
================================================================
# 查看训练时间对比
grep "总训练时间:" results/*_summary.txt | sort

# 查看并行效率对比  
grep "并行效率:" results/*_summary.txt | sort

# 查看通信开销对比
grep "通信开销:" results/*_summary.txt | sort

# 查看GPU利用率:
grep "GPU利用率:" results/*_summary.txt | sort

================================================================
EOF
    
    success "最终报告已生成: $report_file"
}


show_help() {
    cat << EOF
KAN并行训练性能测试自动化脚本
用法: $0 [选项]

选项:
  -h, --help              显示帮助信息
  -j, --max-jobs NUM      设置最大并发作业数 (默认: 8)
  -i, --interval NUM      设置提交间隔秒数 (默认: 30)
  -m, --monitor NUM       设置监控间隔秒数 (默认: 60)
  -r, --retry             重试失败的测试
  -s, --status            显示当前测试状态
  -c, --check             检查环境和文件

示例:
  $0                      # 运行所有测试
  $0 -j 6 -i 45          # 最多6个并发作业，提交间隔45秒
  $0 -r                  # 重试失败的测试
  $0 -s                  # 显示测试状态
  $0 -c                  # 检查环境
EOF
}


check_environment() {
    log "检查环境和文件..."
    
    local issues=0
    
    
    if [ ! -d "$SCRIPT_DIR" ]; then
        error "测试脚本目录不存在: $SCRIPT_DIR"
        ((issues++))
    else
        local script_count=$(ls -1 "$SCRIPT_DIR"/*.sh 2>/dev/null | wc -l)
        success "测试脚本目录存在，包含$script_count 个脚本"
    fi
    
    
    for file in simple_main.py simple_kan.py simple_datasets.py summarize_results.py; do
        if [ ! -f "$file" ]; then
            error "必要文件不存在: $file"
            ((issues++))
        else
            success "文件存在: $file"
        fi
    done
    
    
    if ! command -v sbatch &> /dev/null; then
        error "SLURM命令不可用: sbatch"
        ((issues++))
    else
        success "SLURM命令可用"
    fi
    
    if ! command -v squeue &> /dev/null; then
        error "SLURM命令不可用: squeue"
        ((issues++))
    else
        success "SLURM命令可用"
    fi
    
    
    if ! command -v python &> /dev/null; then
        error "Python命令不可用"
        ((issues++))
    else
        success "Python命令可用: $(python --version)"
    fi
    
    if [ $issues -eq 0 ]; then
        success "环境检查通过，可以开始测试"
        return 0
    else
        error "环境检查失败，发现 $issues 个问题"
        return 1
    fi
}


show_status() {
    log "显示当前测试状态..."
    
    local running_jobs=$(get_queue_count)
    local completed_tests=$(get_completed_count)
    local total_tests=${#TEST_CONFIGS[@]}
    
    echo "================================================================="
    echo "           KAN并行训练性能测试 - 当前状态"
    echo "================================================================="
    echo "当前时间: $(date)"
    echo "运行中作业: $running_jobs"
    echo "已完成测试: $completed_tests/$total_tests"
    echo "完成率: $(echo "scale=1; $completed_tests * 100 / $total_tests" | bc -l 2>/dev/null || echo "0")%"
    echo "================================================================="
    
    if [ -f test_status.txt ]; then
        echo "测试状态详情:"
        grep -v "^#" test_status.txt | while read line; do
            if [ -n "$line" ]; then
                local test_name=$(echo "$line" | cut -d' ' -f1)
                local status=$(echo "$line" | cut -d' ' -f2)
                local job_id=$(echo "$line" | cut -d' ' -f3)
                
                case $status in
                    "SUBMITTED") echo "  >> $test_name (作业ID: $job_id)" ;;
                    "COMPLETED") echo "  + $test_name" ;;
                    "FAILED") echo "  - $test_name" ;;
                    "RETRY") echo "  >> $test_name (重试中)" ;;
                    *) echo "  ? $test_name ($status)" ;;
                esac
            fi
        done
    else
        echo "测试状态文件不存在"
    fi
    
    echo "================================================================="
}


cleanup() {
    log "清理临时文件..."
    # 这里可以添加清理代码
    log "清理完成"
}


main() {
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -j|--max-jobs)
                MAX_CONCURRENT_JOBS="$2"
                shift 2
                ;;
            -i|--interval)
                SUBMIT_INTERVAL="$2"
                shift 2
                ;;
            -m|--monitor)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            -r|--retry)
                log "重试模式"
                retry_failed_tests
                exit 0
                ;;
            -s|--status)
                show_status
                exit 0
                ;;
            -c|--check)
                check_environment
                exit $?
                ;;
            *)
                error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo "================================================================="
    echo "           KAN并行训练性能测试自动化脚本"
    echo "================================================================="
    echo "总测试数量: ${#TEST_CONFIGS[@]}"
    echo "最大并发作业: $MAX_CONCURRENT_JOBS"
    echo "提交间隔: $SUBMIT_INTERVAL 秒"
    echo "监控间隔: $MONITOR_INTERVAL 秒"
    echo "================================================================="
    
    
    if ! check_environment; then
        exit 1
    fi
    
    
    trap cleanup EXIT
    
    
    setup_directories
    
    
    batch_submit_tests 1 "强扩展性测试"
    batch_submit_tests 2 "弱扩展性测试"
    batch_submit_tests 3 "通信开销测试"
    batch_submit_tests 4 "模型扩展性测试"
    
    log "初始批次提交完成，开始持续监控直到所有测试完成..."

    
    local start_time=$(date +%s)
    local timeout_seconds=36000  # 10小时
    local max_end_time=$((start_time + timeout_seconds))

    
    while true; do
        local current_time=$(date +%s)
        local completed_tests=$(get_completed_count)
        local total_tests=${#TEST_CONFIGS[@]}

        
        if [ $current_time -gt $max_end_time ]; then
            local elapsed_hours=$(echo "scale=1; ($current_time - $start_time) / 3600" | bc -l)
            error "测试超时！已运行 ${elapsed_hours} 小时，超过10小时限制"
            error "当前进度: $completed_tests/$total_tests 已完成"
            error "自动停止测试流程"
            break
        fi

        if [ $completed_tests -eq $total_tests ]; then
            local elapsed_hours=$(echo "scale=1; ($current_time - $start_time) / 3600" | bc -l)
            success "所有 $total_tests 个测试已完成！总耗时: ${elapsed_hours} 小时"
            break
        fi

        local elapsed_hours=$(echo "scale=1; ($current_time - $start_time) / 3600" | bc -l)
        local remaining_hours=$(echo "scale=1; ($max_end_time - $current_time) / 3600" | bc -l)
        log "当前进度: $completed_tests/$total_tests 已完成，已运行 ${elapsed_hours} 小时，剩余 ${remaining_hours} 小时"
        monitor_jobs

        
        sleep 30
    done
    
    
    generate_final_report
    
    success "自动化测试流程完成！"
    log "查看详细报告: cat test_report_*.txt"
    log "生成结果汇总: python summarize_results.py"
    log "检查测试状态: $0 -s"
}

# 脚本入口
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi 