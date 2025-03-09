# 工具库

## 编码转换工具

### Shell 版本

`convert.sh`: 将指定目录中的文件从 GB2312 转换为 UTF-8

```bash
sudo chmod +x convert.sh
./convert.sh /path/to/your/directory
```

### Python 版本

`convert.py`: 一个更强大的编码转换工具，支持多种编码格式（如 GBK、Big5 等）转换为 UTF-8

#### 功能特点

-   自动检测文件编码
-   支持指定文件扩展名过滤
-   支持二进制文件检测和跳过
-   支持文件备份
-   可配置编码检测置信度阈值

#### 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行脚本：

```bash
python convert.py <目录路径> [选项]
```

#### 可用选项

-   `-e, --extensions`: 指定要处理的文件扩展名，例如：`.txt .csv`
-   `-c, --confidence`: 设置编码检测的最小置信度阈值（0-1.0），默认 0.5
-   `-b, --backup`: 转换前创建.bak 备份文件
-   `-sb, --skip-binary`: 跳过可能的二进制文件

#### 使用示例

```bash
# 转换指定目录下的所有文件
python convert.py /path/to/directory

# 只转换txt和csv文件，并创建备份
python convert.py /path/to/directory -e .txt .csv -b

# 设置更高的置信度阈值并跳过二进制文件
python convert.py /path/to/directory -c 0.8 -sb
```
