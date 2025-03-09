import os
import argparse
import sys
from chardet import detect

def is_binary_content(content):
    """判断文件是否为二进制文件"""
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
    return bool(content.translate(bytes.maketrans(b"", b"", textchars)))

def backup_file(file_path):
    """备份文件"""
    backup_path = file_path + '.bak'
    os.rename(file_path, backup_path)
    print(f"已创建备份: {backup_path}")

def convert_encoding(file_path, target_encoding='utf-8', confidence_threshold=0.5, skip_binary=False, backup=False):
    """转换文件编码"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        if skip_binary and is_binary_content(content):
            print(f"跳过二进制文件: {file_path}")
            return

        result = detect(content)
        encoding = result['encoding']
        confidence = result['confidence']

        if confidence < confidence_threshold:
            print(f"跳过 {file_path}: 编码检测置信度过低 ({confidence:.2f} < {confidence_threshold})")
            return

        if backup:
            backup_file(file_path)

        if encoding != target_encoding:
            new_content = content.decode(encoding, errors='ignore').encode(target_encoding, errors='ignore')
            with open(file_path, 'wb') as f:
                f.write(new_content)
            print(f"成功转换文件编码: {file_path} -> {target_encoding.upper()}")
        else:
            print(f"文件已经是目标编码: {file_path}")

    except Exception as e:
        print(f"处理 {file_path} 时出错: {type(e).__name__} - {e}")

def process_directory(directory, extensions=None, confidence_threshold=0.5, backup=False, skip_binary=False):
    """遍历目录并处理文件"""
    if extensions:
        extensions = set(extensions)
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue
            file_path = os.path.join(root, file)
            convert_encoding(file_path, confidence_threshold=confidence_threshold, skip_binary=skip_binary, backup=backup)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文件编码转换工具（GBK/Big5等 → UTF-8）",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("directory", help="需要处理的目录路径（支持相对路径）")
    parser.add_argument("-e", "--extensions", nargs='+', help="指定要处理的文件扩展名（例如 .txt .csv）")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="编码检测的最小置信度阈值（0-1.0）")
    parser.add_argument("-b", "--backup", action="store_true", help="转换前创建.bak备份文件")
    parser.add_argument("-sb", "--skip-binary", action="store_true", help="跳过可能的二进制文件")

    args = parser.parse_args()
    target_dir = os.path.abspath(args.directory)

    if not os.path.exists(target_dir):
        print(f"错误: 目录不存在 {target_dir}")
        sys.exit(1)

    print("\n[当前配置]")
    print(f"目标目录: {target_dir}")
    print(f"文件类型: {args.extensions or '全部文件'}")
    print(f"置信阈值: {args.confidence:.1f}")
    print(f"备份文件: {'是' if args.backup else '否'}")
    print(f"跳过二进制: {'是' if args.skip_binary else '否'}")
    print("=" * 50 + "\n")

    process_directory(target_dir, args.extensions, args.confidence, args.backup, args.skip_binary)