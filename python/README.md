# Python编码转换工具

`convert.py`是一个功能强大的编码转换工具，用于将文件从各种编码格式（如GBK、Big5等）转换为UTF-8。

## 功能特点

- 自动检测文件编码
- 支持指定文件扩展名过滤
- 支持二进制文件检测和跳过
- 支持文件备份
- 可配置编码检测置信度阈值

## 安装依赖

在使用前，请先安装所需的依赖包：

```bash
pip install -r requirements.txt
```

或者使用uv安装（如果已配置）：

```bash
uv pip install -r requirements.txt
```

## 使用方法

基本用法：

```bash
python convert.py <目录路径> [选项]
```

### 可用选项

- `-e, --extensions`: 指定要处理的文件扩展名，例如：`.txt .csv`
- `-c, --confidence`: 设置编码检测的最小置信度阈值（0-1.0），默认0.5
- `-b, --backup`: 转换前创建.bak备份文件
- `-sb, --skip-binary`: 跳过可能的二进制文件

### 使用示例

```bash
# 转换指定目录下的所有文件
python convert.py /path/to/directory

# 只转换txt和csv文件，并创建备份
python convert.py /path/to/directory -e .txt .csv -b

# 设置更高的置信度阈值并跳过二进制文件
python convert.py /path/to/directory -c 0.8 -sb
```

## 工作原理

1. 工具使用`chardet`库自动检测文件的原始编码
2. 根据配置的置信度阈值判断检测结果是否可靠
3. 如果需要，创建文件备份
4. 将文件内容从原始编码转换为UTF-8
5. 保存转换后的内容到原文件（或者新文件，如果指定了备份）

## 注意事项

- 对于无法可靠检测编码的文件，工具会自动跳过
- 建议在处理重要文件时使用备份选项
- 二进制文件检测功能可能不是100%准确，请谨慎使用