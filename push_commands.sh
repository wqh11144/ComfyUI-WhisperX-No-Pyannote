#!/bin/bash

# 您的 GitHub 用户名
USERNAME="wqh11144"

# 仓库名称（可修改）
REPO_NAME="${1:-ComfyUI-WhisperX-No-Pyannote}"

echo "🚀 开始推送到 GitHub..."
echo "   用户: $USERNAME"
echo "   仓库: $REPO_NAME"
echo ""

# 移除旧的远程仓库
echo "�� 移除旧的远程配置..."
git remote remove origin 2>/dev/null || true

# 添加新的远程仓库
echo "➕ 添加新的远程仓库..."
git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"

# 显示远程配置
echo "📋 远程仓库配置:"
git remote -v
echo ""

# 推送代码
echo "🚀 推送到 GitHub..."
git push -u origin main

echo ""
echo "✅ 完成！"
echo "🌐 访问您的仓库: https://github.com/$USERNAME/$REPO_NAME"

