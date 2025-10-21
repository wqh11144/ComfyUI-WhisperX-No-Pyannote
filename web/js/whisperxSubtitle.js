import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'

app.registerExtension({
    name: "WhisperX.SubtitleDownload",
    async setup() {
        console.log("[WhisperX Extension] Loading subtitle download extension...");
        
        // 添加自定义样式
        const style = document.createElement("style");
        style.innerHTML = `
            .whisperx-subtitle-download {
                display: inline-block;
                padding: 6px 12px;
                margin: 4px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                border: none;
            }
            .whisperx-subtitle-download:hover {
                background-color: #45a049;
            }
            .whisperx-subtitle-container {
                padding: 8px;
                margin-top: 4px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
        `;
        document.head.appendChild(style);
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[WhisperX Extension] Registering node:", nodeData?.name);
        
        if (nodeData?.name == "WhisperX") {
            console.log("[WhisperX Extension] Hooking into WhisperX node");
            
            // 监听工作流开始执行
            const originalOnExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function() {
                if (originalOnExecutionStart) {
                    originalOnExecutionStart.apply(this, arguments);
                }
                
                // 隐藏下载按钮
                if (this.subtitleWidget) {
                    this.subtitleWidget.hidden = true;
                    console.log("[WhisperX] Download button hidden on execution start");
                }
                
                // 清空之前的字幕文件信息
                this.subtitle_files = null;
            };
            
            nodeType.prototype.onExecuted = function (data) {
                console.log("[WhisperX] onExecuted called!");
                console.log("[WhisperX] Output data:", data);
                console.log("[WhisperX] data.subtitle:", data.subtitle);
                
                // 处理字幕文件下载
                if (data.subtitle && data.subtitle.length > 0) {
                    console.log("[WhisperX] Subtitle files detected:", data.subtitle);
                    
                    // 存储字幕文件信息供右键菜单使用
                    this.subtitle_files = data.subtitle.map(fileInfo => {
                        // 拼接完整的下载 URL
                        const downloadUrl = api.apiURL('/view?' + new URLSearchParams({
                            filename: fileInfo.filename,
                            type: fileInfo.type,
                            subfolder: fileInfo.subfolder || ""
                        }));
                        
                        console.log(`[WhisperX] Subtitle download URL: ${downloadUrl}`);
                        console.log(`[WhisperX] File info:`, fileInfo);
                        
                        return {
                            filename: fileInfo.filename,
                            url: downloadUrl
                        };
                    });
                    
                    // 创建或显示下载按钮
                    if (!this.subtitleWidget) {
                        this.subtitleWidget = this.addWidget("button", "📥 Download Subtitles", "click", () => {
                            if (this.subtitle_files && this.subtitle_files.length > 0) {
                                // 下载所有字幕文件
                                this.subtitle_files.forEach(file => {
                                    const a = document.createElement("a");
                                    a.href = file.url;
                                    a.download = file.filename;
                                    document.body.append(a);
                                    a.click();
                                    setTimeout(() => a.remove(), 100);
                                });
                            }
                        });
                    } else {
                        // 显示已存在的按钮
                        this.subtitleWidget.hidden = false;
                    }
                    
                    console.log("[WhisperX] Download button shown");
                    
                    // 刷新节点显示
                    this.setDirtyCanvas(true, true);
                }
            };
            
            // 添加右键菜单选项
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }
                
                // 如果有字幕文件，添加下载选项
                if (this.subtitle_files && this.subtitle_files.length > 0) {
                    options.unshift(
                        {
                            content: "📥 Download Subtitle Files",
                            disabled: true,  // 作为标题
                        }
                    );
                    
                    this.subtitle_files.forEach((file, index) => {
                        options.unshift({
                            content: `  📄 ${file.filename}`,
                            callback: () => {
                                console.log(`[WhisperX] Downloading: ${file.filename}`);
                                const a = document.createElement("a");
                                a.href = file.url;
                                a.download = file.filename;
                                document.body.append(a);
                                a.click();
                                requestAnimationFrame(() => a.remove());
                            },
                        });
                    });
                    
                    options.unshift(null); // 分隔线
                }
                
                return options;
            };
        }
    }
});

