import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'

app.registerExtension({
    name: "WhisperX.SubtitleDownload",
    async setup() {
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
        
        // 监听 API 事件，在队列历史中添加下载按钮
        api.addEventListener("executed", ({ detail }) => {
            const { node, output } = detail;
            if (output?.subtitle && output.subtitle.length > 0) {
                console.log("[WhisperX] Queue output detected:", output);
                // 下载信息已经通过 UI 返回，浏览器会自动处理
            }
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "WhisperX") {
            nodeType.prototype.onExecuted = function (data) {
                console.log("[WhisperX] Output data:", data);
                
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
                    
                    // 在节点上添加下载按钮小部件
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
                    }
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

