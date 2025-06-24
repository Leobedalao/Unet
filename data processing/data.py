import os
import shutil


def copy_and_rename_files(source_dir):
    # 基础路径配置
    dataset_dir = r'D:\pyproject\Denoise for mc -3d unet\dataset'

    # 初始化目标配置
    patterns = ['1e5_test', '1e6_test', '1e7_test', '1e8_test', '1e9_test']
    target_dirs = {
        p.split('_')[0]: os.path.join(dataset_dir, p.split('_')[0])
        for p in patterns
    }
    counters = {key: 1 for key in target_dirs.keys()}

    # 创建所有目标文件夹
    for d in target_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        current_dir = os.path.basename(root)
        category = None

        # 匹配目录模式
        for pattern in patterns:
            if pattern in current_dir:
                category = pattern.split('_')[0]
                break

        if not category:
            continue

        # 源文件路径
        src_file = os.path.join(root, 'voxel_dose_mean.mha')
        if not os.path.isfile(src_file):
            continue

        # 目标路径处理
        target_dir = target_dirs[category]
        new_filename = f"voxel_dose_mean_{counters[category]}.mha"
        dest_path = os.path.join(target_dir, new_filename)

        # 执行复制
        try:
            shutil.copy2(src_file, dest_path)
            print(f"Success: {src_file} -> {dest_path}")
            counters[category] += 1
        except Exception as e:
            print(f"Error copying {src_file}: {str(e)}")


#if __name__ == '__main__':
 #   source_directory = input("请输入源目录路径：").strip()
  #  copy_and_rename_files(source_directory)