#!/bin/bash
echo "🔄 重试推送..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo "🌐 访问您的仓库: https://github.com/wqh11144/ComfyUI-WhisperX-No-Pyannote"
    echo ""
    echo "📊 仓库内容:"
    echo "   - 移除了 pyannote.audio 依赖"
    echo "   - 移除了多说话人识别功能"
    echo "   - 修复了循环导入和语法错误"
    echo "   - 保留了核心转录和翻译功能"
else
    echo ""
    echo "❌ 推送失败，请确保仓库已创建"
fi
