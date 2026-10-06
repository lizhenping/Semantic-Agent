@echo off
chcp 65001 >nul
title SemanticSQL 合成任务（关闭此窗口 = 停止任务）
cd /d D:\学术\tkde\semantic-agent
echo ============================================
echo  SemanticSQL 正式合成（断点续跑，可反复执行）
echo  关闭本窗口即停止；重新双击本文件即从断点继续
echo ============================================
bash start_synthesis.sh
echo.
echo 任务结束（或已停止）。按任意键关闭窗口。
pause >nul
