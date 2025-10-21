"""
SRT Timestamp Overlap Fixer
修复 WhisperX 生成的 SRT 字幕时间戳重叠问题

策略：调整前一条字幕的结束时间，确保不与下一条重叠
"""

import srt
from datetime import timedelta
from typing import List


def fix_srt_overlaps(srt_content: str, gap_ms: int = 50) -> tuple[str, dict]:
    """
    修复 SRT 字幕中的时间戳重叠问题
    
    Args:
        srt_content: SRT 字幕文本内容
        gap_ms: 字幕间隙（毫秒），默认 50ms
        
    Returns:
        tuple: (修复后的SRT内容, 统计信息字典)
    """
    # 解析 SRT
    try:
        subtitles = list(srt.parse(srt_content))
    except Exception as e:
        print(f"[SRT Fixer] Parse error: {e}")
        return srt_content, {"error": str(e)}
    
    if len(subtitles) == 0:
        return srt_content, {"total": 0, "fixed": 0, "overlaps": []}
    
    # 统计信息
    stats = {
        "total": len(subtitles),
        "fixed": 0,
        "overlaps": []
    }
    
    gap_delta = timedelta(milliseconds=gap_ms)
    
    # 遍历字幕，检测并修复重叠
    for i in range(len(subtitles) - 1):
        current = subtitles[i]
        next_sub = subtitles[i + 1]
        
        # 检测重叠：当前结束时间 > 下一条开始时间
        if current.end > next_sub.start:
            overlap_ms = int((current.end - next_sub.start).total_seconds() * 1000)
            
            # 记录重叠信息
            stats["overlaps"].append({
                "index": i + 1,
                "overlap_ms": overlap_ms,
                "original_end": str(current.end),
                "next_start": str(next_sub.start)
            })
            
            # 修复：调整当前字幕的结束时间
            # 设置为下一条开始时间 - gap
            new_end = next_sub.start - gap_delta
            
            # 确保结束时间不早于开始时间
            if new_end <= current.start:
                # 如果调整后结束时间过早，至少保持 100ms 的最小持续时间
                new_end = current.start + timedelta(milliseconds=100)
                # 如果还是会重叠，则设置为下一条开始时间（0间隙）
                if new_end > next_sub.start:
                    new_end = next_sub.start
            
            current.end = new_end
            stats["fixed"] += 1
    
    # 生成修复后的 SRT 内容
    fixed_content = srt.compose(subtitles)
    
    return fixed_content, stats


def fix_srt_file(input_path: str, output_path: str = None, gap_ms: int = 50) -> dict:
    """
    修复 SRT 文件中的时间戳重叠
    
    Args:
        input_path: 输入 SRT 文件路径
        output_path: 输出文件路径（None 则覆盖原文件）
        gap_ms: 字幕间隙（毫秒）
        
    Returns:
        dict: 统计信息
    """
    # 读取文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复重叠
    fixed_content, stats = fix_srt_overlaps(content, gap_ms)
    
    # 写入文件
    output = output_path or input_path
    with open(output, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    return stats


def print_fix_report(stats: dict, verbose: bool = False):
    """打印修复报告"""
    if "error" in stats:
        print(f"[SRT Fixer] ❌ Error: {stats['error']}")
        return
    
    print(f"[SRT Fixer] 📊 Total subtitles: {stats['total']}")
    print(f"[SRT Fixer] ✅ Fixed overlaps: {stats['fixed']}")
    
    if verbose and stats['fixed'] > 0:
        print(f"[SRT Fixer] 📋 Overlap details:")
        for overlap in stats['overlaps'][:5]:  # 只显示前5个
            print(f"  - Subtitle #{overlap['index']}: {overlap['overlap_ms']}ms overlap")
        if len(stats['overlaps']) > 5:
            print(f"  ... and {len(stats['overlaps']) - 5} more")


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python srt_timestamp_fixer.py <input.srt> [output.srt] [gap_ms]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    gap_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    print(f"[SRT Fixer] Processing: {input_file}")
    print(f"[SRT Fixer] Gap setting: {gap_ms}ms")
    
    stats = fix_srt_file(input_file, output_file, gap_ms)
    print_fix_report(stats, verbose=True)
    
    print(f"[SRT Fixer] ✨ Done! Output: {output_file or input_file}")

